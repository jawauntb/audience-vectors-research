"""Open AlexNet layer-5 memorability probe on BOLD Moments.

This is the smaller-model counterpart to the TRIBE/V-JEPA feature-space tests.
It asks whether a transparent AlexNet conv5 representation contains a compact
linear memorability direction, then summarizes where that direction lives in the
open layer tensor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

DEFAULT_ALEXNET_DIR = Path("data/raw/algonauts2021/alexnet")
DEFAULT_ANNOTATIONS = Path("data/raw/bold_moments/annotations.json")
DEFAULT_JSON = Path("data/reports/alexnet_memorability_probe.json")
DEFAULT_MD = Path("data/reports/alexnet_memorability_probe.md")


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)

    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata(np.asarray(a, dtype=np.float64))
    rb = _rankdata(np.asarray(b, dtype=np.float64))
    return float(np.corrcoef(ra, rb)[0, 1])


def _load_scores(annotations: Path, video_ids: list[str]) -> np.ndarray:
    payload = json.loads(annotations.read_text())
    scores = []
    missing = []
    for video_id in video_ids:
        key = video_id.zfill(4)
        row = payload.get(key)
        if row is None or "memorability_score" not in row:
            missing.append(key)
            continue
        scores.append(float(row["memorability_score"]))
    if missing:
        raise ValueError(f"missing memorability scores for {len(missing)} videos")
    return np.asarray(scores, dtype=np.float32)


def _load_features(alexnet_dir: Path, space: str) -> tuple[np.ndarray, list[str]]:
    video_ids_path = alexnet_dir / "video_ids.json"
    if video_ids_path.exists():
        video_ids = [str(x) for x in json.loads(video_ids_path.read_text())]
    else:
        video_ids = [f"{i:04d}" for i in range(1, 1103)]

    if space == "pca":
        train = np.load(alexnet_dir / "pca_100" / "train_layer_5.npy")
        test = np.load(alexnet_dir / "pca_100" / "test_layer_5.npy")
        features = np.concatenate([train, test], axis=0)
    elif space == "raw":
        features = np.load(alexnet_dir / "layer_5_all.npy")
    else:
        raise ValueError(f"unknown feature space: {space}")

    if features.shape[0] != len(video_ids):
        raise ValueError(
            f"feature rows {features.shape[0]} != video ids {len(video_ids)}"
        )
    return np.asarray(features, dtype=np.float32), video_ids


def _fit_direction(
    features: np.ndarray, scores: np.ndarray, top_frac: float
) -> np.ndarray:
    order = np.argsort(scores)
    n_each = max(3, int(round(len(scores) * top_frac)))
    neg = features[order[:n_each]].mean(axis=0)
    pos = features[order[-n_each:]].mean(axis=0)
    direction = pos - neg
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("direction has near-zero norm")
    return np.asarray(direction / norm, dtype=np.float32)


def _ablate(features: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return features - np.outer(features @ direction, direction)


def _folds(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    chunks = np.array_split(perm, k)
    folds = []
    for i, test in enumerate(chunks):
        train = np.concatenate([chunk for j, chunk in enumerate(chunks) if j != i])
        folds.append((train, test))
    return folds


def _cv_probe(
    features: np.ndarray,
    scores: np.ndarray,
    *,
    folds: int,
    seed: int,
    top_frac: float,
    random_ablation_runs: int,
) -> dict[str, Any]:
    fold_rows = []
    baseline_preds = np.zeros(len(scores), dtype=np.float32)
    ablated_preds = np.zeros(len(scores), dtype=np.float32)
    rng = np.random.default_rng(seed + 10_000)
    random_rhos = []

    for fold_idx, (train_i, test_i) in enumerate(
        _folds(len(scores), folds, seed), start=1
    ):
        scaler = StandardScaler()
        train_x = np.asarray(scaler.fit_transform(features[train_i]), dtype=np.float32)
        test_x = np.asarray(scaler.transform(features[test_i]), dtype=np.float32)
        train_y = scores[train_i]
        test_y = scores[test_i]

        direction = _fit_direction(train_x, train_y, top_frac)
        pred = test_x @ direction
        baseline_preds[test_i] = pred
        baseline_rho = _spearman(pred, test_y)

        train_res = _ablate(train_x, direction)
        test_res = _ablate(test_x, direction)
        direction2 = _fit_direction(train_res, train_y, top_frac)
        pred_res = test_res @ direction2
        ablated_preds[test_i] = pred_res
        ablated_rho = _spearman(pred_res, test_y)

        fold_random = []
        for _ in range(random_ablation_runs):
            random_direction = rng.normal(size=train_x.shape[1]).astype(np.float32)
            random_direction /= max(float(np.linalg.norm(random_direction)), 1e-12)
            train_random = _ablate(train_x, random_direction)
            test_random = _ablate(test_x, random_direction)
            direction_random = _fit_direction(train_random, train_y, top_frac)
            fold_random.append(_spearman(test_random @ direction_random, test_y))
        random_rhos.extend(fold_random)

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_i)),
                "n_test": int(len(test_i)),
                "baseline_rho": float(baseline_rho),
                "ablated_rho": float(ablated_rho),
                "cos_v_v2": float(np.dot(direction, direction2)),
                "random_ablation_mean_rho": float(np.mean(fold_random)),
                "random_ablation_std_rho": float(np.std(fold_random)),
            }
        )

    baseline_cv = _spearman(baseline_preds, scores)
    ablated_cv = _spearman(ablated_preds, scores)
    random_arr = np.asarray(random_rhos, dtype=np.float32)
    z_vs_random = (
        float((ablated_cv - random_arr.mean()) / max(float(random_arr.std()), 1e-12))
        if random_arr.size
        else None
    )
    return {
        "folds": fold_rows,
        "baseline_cv_rho": float(baseline_cv),
        "ablated_cv_rho": float(ablated_cv),
        "delta_after_ablation": float(ablated_cv - baseline_cv),
        "random_ablation_mean_rho": float(random_arr.mean())
        if random_arr.size
        else None,
        "random_ablation_std_rho": float(random_arr.std()) if random_arr.size else None,
        "z_ablated_vs_random": z_vs_random,
    }


def _global_direction_summary(
    raw_features: np.ndarray,
    scores: np.ndarray,
    *,
    top_frac: float,
    top_k: int,
) -> dict[str, Any]:
    scaler = StandardScaler()
    scaled = np.asarray(scaler.fit_transform(raw_features), dtype=np.float32)
    direction = _fit_direction(scaled, scores, top_frac)
    tensor = direction.reshape(256, 6, 6)
    channel_energy = np.sqrt((tensor**2).sum(axis=(1, 2)))
    spatial_energy = np.sqrt((tensor**2).sum(axis=0))
    top_channels = [
        {
            "channel": int(idx),
            "energy": float(channel_energy[idx]),
            "signed_mean": float(tensor[idx].mean()),
        }
        for idx in np.argsort(channel_energy)[::-1][:top_k]
    ]
    return {
        "top_channels": top_channels,
        "spatial_energy_6x6": spatial_energy.astype(float).round(6).tolist(),
        "channel_energy_gini": _gini(channel_energy),
        "top_10_channel_energy_share": float(
            channel_energy[np.argsort(channel_energy)[::-1][:10]].sum()
            / max(channel_energy.sum(), 1e-12)
        ),
    }


def _gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if np.allclose(x.sum(), 0):
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def _markdown(payload: dict[str, Any]) -> str:
    pca = payload["pca_probe"]
    raw = payload["raw_probe"]
    summary = payload["raw_direction_summary"]
    top_channels = ", ".join(
        f"{row['channel']} ({row['energy']:.3f})" for row in summary["top_channels"][:8]
    )
    return "\n".join(
        [
            "# AlexNet Layer-5 Memorability Probe",
            "",
            f"- Videos: {payload['n_videos']} BOLD Moments clips.",
            f"- Scores: human memorability from `{payload['annotations']}`.",
            f"- PCA layer-5 baseline CV rho: **{pca['baseline_cv_rho']:+.3f}**.",
            f"- PCA after v_mem ablation CV rho: **{pca['ablated_cv_rho']:+.3f}**.",
            f"- PCA random-ablation mean rho: **{pca['random_ablation_mean_rho']:+.3f}** +/- {pca['random_ablation_std_rho']:.3f}.",
            f"- Raw conv5 baseline CV rho: **{raw['baseline_cv_rho']:+.3f}**.",
            f"- Raw conv5 after v_mem ablation CV rho: **{raw['ablated_cv_rho']:+.3f}**.",
            f"- Raw top-channel energy share (top 10/256): **{summary['top_10_channel_energy_share']:.3f}**.",
            f"- Raw channel-energy Gini: **{summary['channel_energy_gini']:.3f}**.",
            f"- Top raw conv5 channels by direction energy: {top_channels}.",
            "",
            "Interpretation: AlexNet is an open sanity-check model, not a replacement for TRIBE.",
            "A positive baseline means memorability is partially visible in a transparent convnet layer;",
            "the ablation/random-ablation contrast tells us whether that signal is direction-specific",
            "inside this smaller representation.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alexnet-dir", type=Path, default=DEFAULT_ALEXNET_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-frac", type=float, default=0.30)
    parser.add_argument("--random-ablation-runs", type=int, default=40)
    parser.add_argument("--top-k-channels", type=int, default=12)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    pca_features, video_ids = _load_features(args.alexnet_dir, "pca")
    raw_features, raw_video_ids = _load_features(args.alexnet_dir, "raw")
    if raw_video_ids != video_ids:
        raise ValueError("raw and PCA video id order differs")
    scores = _load_scores(args.annotations, video_ids)

    print(f"[alexnet-probe] videos={len(video_ids)}")
    print("[alexnet-probe] probing PCA layer_5")
    pca_probe = _cv_probe(
        pca_features,
        scores,
        folds=args.folds,
        seed=args.seed,
        top_frac=args.top_frac,
        random_ablation_runs=args.random_ablation_runs,
    )
    print("[alexnet-probe] probing raw conv5")
    raw_probe = _cv_probe(
        raw_features,
        scores,
        folds=args.folds,
        seed=args.seed,
        top_frac=args.top_frac,
        random_ablation_runs=args.random_ablation_runs,
    )
    summary = _global_direction_summary(
        raw_features,
        scores,
        top_frac=args.top_frac,
        top_k=args.top_k_channels,
    )

    payload = {
        "n_videos": int(len(video_ids)),
        "alexnet_dir": str(args.alexnet_dir),
        "annotations": str(args.annotations),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "top_frac": float(args.top_frac),
        "random_ablation_runs_per_fold": int(args.random_ablation_runs),
        "pca_probe": pca_probe,
        "raw_probe": raw_probe,
        "raw_direction_summary": summary,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    args.out_md.write_text(_markdown(payload))

    print(
        "[alexnet-probe] PCA rho "
        f"{pca_probe['baseline_cv_rho']:+.3f} -> {pca_probe['ablated_cv_rho']:+.3f}"
    )
    print(
        "[alexnet-probe] raw rho "
        f"{raw_probe['baseline_cv_rho']:+.3f} -> {raw_probe['ablated_cv_rho']:+.3f}"
    )
    print(f"[alexnet-probe] wrote {args.out_json}")
    print(f"[alexnet-probe] wrote {args.out_md}")


if __name__ == "__main__":
    main()
