"""Forward-pass patching probe for the AlexNet conv5 memorability direction.

This is a tighter mechanistic sanity check than offline feature ablation:

1. Fit the human-memorability direction in standardized AlexNet conv5 space.
2. During the AlexNet forward pass, patch conv5 activations frame-by-frame.
3. Continue the forward pass through fc6/fc7/fc8.
4. Test whether memorability remains linearly recoverable downstream.

The intervention is still inside AlexNet, not TRIBE, and AlexNet is not a video
generator. The point is to test whether the learned conv5 direction behaves like
an actionable activation-space feature in a fully open network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from alexnet_memorability_probe import (  # noqa: E402
    _fit_direction,
    _folds,
    _load_scores,
    _rankdata,
)
from extract_boldmoments_alexnet_layer5 import (  # noqa: E402
    DEFAULT_DEVKIT_ALEXNET,
    DEFAULT_VIDEO_DIR,
    _choose_device,
    _load_model,
    _preprocess_frames,
    _sample_frames,
    _video_paths,
)

DEFAULT_ALEXNET_DIR = Path("data/raw/algonauts2021/alexnet")
DEFAULT_ANNOTATIONS = Path("data/raw/bold_moments/annotations.json")
DEFAULT_JSON = Path("data/reports/alexnet_forward_patch_probe.json")
DEFAULT_MD = Path("data/reports/alexnet_forward_patch_probe.md")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata(np.asarray(a, dtype=np.float64))
    rb = _rankdata(np.asarray(b, dtype=np.float64))
    return float(np.corrcoef(ra, rb)[0, 1])


def _load_raw_conv5(alexnet_dir: Path) -> tuple[np.ndarray, list[str]]:
    features = np.load(alexnet_dir / "layer_5_all.npy")
    video_ids = [
        str(x) for x in json.loads((alexnet_dir / "video_ids.json").read_text())
    ]
    if features.shape[0] != len(video_ids):
        raise ValueError(
            f"conv5 rows {features.shape[0]} != video ids {len(video_ids)}"
        )
    return np.asarray(features, dtype=np.float32), video_ids


def _cv_linear_readout(
    features: np.ndarray,
    scores: np.ndarray,
    *,
    folds: int,
    seed: int,
    top_frac: float,
) -> dict[str, Any]:
    preds = np.zeros(len(scores), dtype=np.float32)
    fold_rows = []
    for fold_idx, (train_i, test_i) in enumerate(
        _folds(len(scores), folds, seed), start=1
    ):
        scaler = StandardScaler()
        train_x = np.asarray(scaler.fit_transform(features[train_i]), dtype=np.float32)
        test_x = np.asarray(scaler.transform(features[test_i]), dtype=np.float32)
        direction = _fit_direction(train_x, scores[train_i], top_frac)
        fold_pred = test_x @ direction
        preds[test_i] = fold_pred
        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_i)),
                "n_test": int(len(test_i)),
                "rho": _spearman(fold_pred, scores[test_i]),
            }
        )
    return {
        "cv_rho": _spearman(preds, scores),
        "folds": fold_rows,
    }


def _make_patch_stats(
    raw_conv5: np.ndarray, scores: np.ndarray, *, top_frac: float
) -> dict[str, np.ndarray]:
    scaler = StandardScaler()
    scaled = np.asarray(scaler.fit_transform(raw_conv5), dtype=np.float32)
    direction = _fit_direction(scaled, scores, top_frac)
    return {
        "mean": np.asarray(scaler.mean_, dtype=np.float32),
        "scale": np.asarray(scaler.scale_, dtype=np.float32),
        "direction": direction,
    }


def _patch_conv5(
    conv5: torch.Tensor,
    stats: dict[str, torch.Tensor],
    *,
    mode: str,
    alpha: float,
    random_direction: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = conv5.shape
    flat = conv5.reshape(shape[0], -1)
    z = (flat - stats["mean"]) / stats["scale"]
    direction = random_direction if random_direction is not None else stats["direction"]
    if mode == "ablate":
        z = z - (z @ direction).unsqueeze(1) * direction.unsqueeze(0)
    elif mode == "add":
        z = z + alpha * direction.unsqueeze(0)
    elif mode == "subtract":
        z = z - alpha * direction.unsqueeze(0)
    elif mode == "none":
        pass
    else:
        raise ValueError(f"unknown patch mode: {mode}")
    patched = z * stats["scale"] + stats["mean"]
    return patched.reshape(shape)


def _forward_from_conv5(
    model: Any,
    conv5: torch.Tensor,
) -> dict[str, np.ndarray]:
    flat = conv5.reshape(conv5.shape[0], 256 * 6 * 6)
    fc6 = model.fc6(flat)
    fc7 = model.fc7(fc6)
    logits = model.fc8(fc7)
    return {
        "conv5": conv5.detach().cpu().numpy().reshape(conv5.shape[0], -1),
        "fc6": fc6.detach().cpu().numpy(),
        "fc7": fc7.detach().cpu().numpy(),
        "logits": logits.detach().cpu().numpy(),
    }


@torch.inference_mode()
def _extract_variants(
    model: Any,
    videos: list[Path],
    *,
    patch_stats_np: dict[str, np.ndarray],
    device: torch.device,
    n_frames: int,
    alpha: float,
    random_patches: int,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    patch_stats = {
        key: torch.from_numpy(value).to(device) for key, value in patch_stats_np.items()
    }
    rng = np.random.default_rng(seed)
    random_dirs = []
    for _ in range(random_patches):
        d = rng.normal(size=patch_stats_np["direction"].shape).astype(np.float32)
        d /= max(float(np.linalg.norm(d)), 1e-12)
        random_dirs.append(torch.from_numpy(d).to(device))

    variant_parts: dict[str, dict[str, list[np.ndarray]]] = {
        "baseline": {"conv5": [], "fc6": [], "fc7": [], "logits": []},
        "ablate": {"conv5": [], "fc6": [], "fc7": [], "logits": []},
        "add": {"conv5": [], "fc6": [], "fc7": [], "logits": []},
        "subtract": {"conv5": [], "fc6": [], "fc7": [], "logits": []},
    }
    for i in range(random_patches):
        variant_parts[f"random_ablate_{i:02d}"] = {
            "conv5": [],
            "fc6": [],
            "fc7": [],
            "logits": [],
        }

    checks = []
    for video in tqdm(videos, desc="[alexnet-forward-patch]"):
        frames = _sample_frames(video, n_frames)
        batch = _preprocess_frames(frames).to(device)
        conv1 = model.conv1(batch)
        conv2 = model.conv2(conv1)
        conv3 = model.conv3(conv2)
        conv4 = model.conv4(conv3)
        conv5 = model.conv5(conv4)

        variants: dict[str, torch.Tensor] = {
            "baseline": conv5,
            "ablate": _patch_conv5(conv5, patch_stats, mode="ablate", alpha=alpha),
            "add": _patch_conv5(conv5, patch_stats, mode="add", alpha=alpha),
            "subtract": _patch_conv5(conv5, patch_stats, mode="subtract", alpha=alpha),
        }
        for i, random_direction in enumerate(random_dirs):
            variants[f"random_ablate_{i:02d}"] = _patch_conv5(
                conv5,
                patch_stats,
                mode="ablate",
                alpha=alpha,
                random_direction=random_direction,
            )

        baseline_mean = conv5.detach().cpu().numpy().reshape(conv5.shape[0], -1).mean(0)
        checks.append(
            {
                "video_id": video.stem,
                "baseline_conv5_projection": float(
                    ((baseline_mean - patch_stats_np["mean"]) / patch_stats_np["scale"])
                    @ patch_stats_np["direction"]
                ),
            }
        )

        for variant_name, variant_conv5 in variants.items():
            out = _forward_from_conv5(model, variant_conv5)
            for layer, arr in out.items():
                variant_parts[variant_name][layer].append(
                    np.asarray(arr.mean(axis=0), dtype=np.float32)
                )

    variant_arrays: dict[str, dict[str, np.ndarray]] = {}
    for variant_name, layer_parts in variant_parts.items():
        variant_arrays[variant_name] = {
            layer: np.stack(parts).astype(np.float32)
            for layer, parts in layer_parts.items()
        }
    return variant_arrays, checks


def _summarize_variants(
    variants: dict[str, dict[str, np.ndarray]],
    scores: np.ndarray,
    *,
    folds: int,
    seed: int,
    top_frac: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant_name, layers in variants.items():
        if variant_name.startswith("random_ablate_"):
            continue
        summary[variant_name] = {
            layer: _cv_linear_readout(
                features,
                scores,
                folds=folds,
                seed=seed,
                top_frac=top_frac,
            )
            for layer, features in layers.items()
        }

    random_rows: list[dict[str, Any]] = []
    for variant_name, layers in variants.items():
        if not variant_name.startswith("random_ablate_"):
            continue
        row: dict[str, Any] = {"variant": variant_name}
        for layer, features in layers.items():
            row[layer] = _cv_linear_readout(
                features,
                scores,
                folds=folds,
                seed=seed,
                top_frac=top_frac,
            )["cv_rho"]
        random_rows.append(row)
    if random_rows:
        random_summary: dict[str, Any] = {
            layer: {
                "mean_cv_rho": float(np.mean([row[layer] for row in random_rows])),
                "std_cv_rho": float(np.std([row[layer] for row in random_rows])),
            }
            for layer in ["conv5", "fc6", "fc7", "logits"]
        }
        random_summary["runs"] = random_rows
        summary["random_ablate"] = random_summary
    return summary


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = []
    for layer in ["conv5", "fc6", "fc7", "logits"]:
        baseline = summary["baseline"][layer]["cv_rho"]
        ablate = summary["ablate"][layer]["cv_rho"]
        add = summary["add"][layer]["cv_rho"]
        subtract = summary["subtract"][layer]["cv_rho"]
        random_mean = summary.get("random_ablate", {}).get(layer, {}).get("mean_cv_rho")
        random_std = summary.get("random_ablate", {}).get(layer, {}).get("std_cv_rho")
        rows.append(
            "| {layer} | {baseline:+.3f} | {ablate:+.3f} | {add:+.3f} | "
            "{subtract:+.3f} | {random} |".format(
                layer=layer,
                baseline=baseline,
                ablate=ablate,
                add=add,
                subtract=subtract,
                random=(
                    f"{random_mean:+.3f} +/- {random_std:.3f}"
                    if random_mean is not None and random_std is not None
                    else "n/a"
                ),
            )
        )
    return "\n".join(
        [
            "# AlexNet Forward-Pass Conv5 Patching Probe",
            "",
            f"- Videos: {payload['n_videos']} BOLD Moments clips.",
            f"- Frames per video: {payload['n_frames']}.",
            "- Patch: standardized conv5 direction learned from BMD human memorability.",
            f"- Alpha for add/subtract patches: {payload['alpha']}.",
            "",
            "| layer readout | baseline CV rho | conv5 v_mem ablated | conv5 +alpha v_mem | conv5 -alpha v_mem | random conv5 ablation |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "Interpretation: this is a true forward-pass intervention inside AlexNet,",
            "because conv5 is patched before fc6/fc7/fc8 are computed. It is still not",
            "TRIBE-internal patching and does not generate new videos; it asks whether",
            "the compact conv5 memorability direction behaves like an actionable feature",
            "in a fully open network.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--alexnet-dir", type=Path, default=DEFAULT_ALEXNET_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--devkit-alexnet", type=Path, default=DEFAULT_DEVKIT_ALEXNET)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("data/raw/algonauts2021/alexnet.pth")
    )
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-frac", type=float, default=0.30)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--random-patches", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    raw_conv5, video_ids = _load_raw_conv5(args.alexnet_dir)
    if args.limit:
        raw_conv5 = raw_conv5[: args.limit]
        video_ids = video_ids[: args.limit]
    scores = _load_scores(args.annotations, video_ids)
    videos = _video_paths(args.video_dir, args.limit)
    if [video.stem for video in videos] != video_ids:
        raise ValueError("video order differs from saved AlexNet feature order")

    patch_stats = _make_patch_stats(raw_conv5, scores, top_frac=args.top_frac)
    device = _choose_device(args.device)
    model = _load_model(args.devkit_alexnet, args.checkpoint, device)
    print(
        f"[alexnet-forward-patch] videos={len(videos)} device={device} "
        f"frames/video={args.n_frames}"
    )

    variants, projection_checks = _extract_variants(
        model,
        videos,
        patch_stats_np=patch_stats,
        device=device,
        n_frames=args.n_frames,
        alpha=args.alpha,
        random_patches=args.random_patches,
        seed=args.seed,
    )
    summary = _summarize_variants(
        variants,
        scores,
        folds=args.folds,
        seed=args.seed,
        top_frac=args.top_frac,
    )

    payload = {
        "n_videos": int(len(videos)),
        "video_dir": str(args.video_dir),
        "alexnet_dir": str(args.alexnet_dir),
        "annotations": str(args.annotations),
        "n_frames": int(args.n_frames),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "top_frac": float(args.top_frac),
        "alpha": float(args.alpha),
        "random_patches": int(args.random_patches),
        "summary": summary,
        "projection_checks_head": projection_checks[:20],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    args.out_md.write_text(_markdown(payload))
    print(f"[alexnet-forward-patch] wrote {args.out_json}")
    print(f"[alexnet-forward-patch] wrote {args.out_md}")


if __name__ == "__main__":
    main()
