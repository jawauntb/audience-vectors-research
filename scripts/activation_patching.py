"""Activation patching: causal test for the memorability direction.

For each fold:
  1. Baseline: train memorability direction v on train, project test
     features onto v, report Spearman vs BMD memorability.
  2. Directional ablation: remove v from features (project out the v
     component), train a NEW direction v2 on ablated train features,
     project ablated test features onto v2, report Spearman.
  3. If baseline ρ >> ablated ρ, the original v captured the unique
     memorability signal — a causal claim, not just a correlational one.

Usage:
    uv run python scripts/activation_patching.py --features-dir data/features/tribe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from audience_vectors.config import get_config


def spearmanr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]), 0.0


def _load_bmd() -> dict[str, float]:
    p = Path("./data/raw/bold_moments/annotations.json")
    with p.open() as fh:
        ann = json.load(fh)
    return {
        f"bmd_vid_idx{eid}": float(e["memorability_score"])
        for eid, e in ann.items()
        if "memorability_score" in e
    }


def _load_feature(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if "frames" in payload.files:
        arr = np.asarray(payload["frames"], dtype=np.float32)
        return arr.mean(axis=0) if arr.ndim == 2 else arr
    return np.asarray(payload["embedding"], dtype=np.float32)


def _collect(features_dir: Path, scored: list[tuple[str, float]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids, feats, scores = [], [], []
    for sid, score in scored:
        f = features_dir / f"{sid}.npz"
        if not f.exists():
            continue
        feats.append(_load_feature(f))
        scores.append(score)
        ids.append(sid)
    return np.stack(feats), np.asarray(scores, dtype=np.float32), ids


def _train_direction(features: np.ndarray, scores: np.ndarray, top_k_frac: float = 0.30) -> np.ndarray:
    order = np.argsort(scores)
    n_each = max(3, int(len(scores) * top_k_frac))
    neg = features[order[:n_each]].mean(axis=0)
    pos = features[order[-n_each:]].mean(axis=0)
    direction = pos - neg
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-12 else direction


def _ablate(features: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the direction component from every feature row."""
    projections = features @ direction
    return features - np.outer(projections, direction)


def _kfold_indices(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = n if i == k - 1 else start + fold_size
        test = perm[start:end]
        train = np.concatenate([perm[:start], perm[end:]])
        folds.append((train, test))
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k-frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = get_config()
    cfg.paths.ensure()
    segments_path = cfg.paths.training / "segments.parquet"
    bmd = _load_bmd()
    segs = pl.read_parquet(segments_path).to_dicts()
    scored = [(s["sample_id"], bmd[s["video_id"]]) for s in segs if s["video_id"] in bmd]
    print(f"[patching] features={args.features_dir} scored_total={len(scored)}")

    features, scores, _ = _collect(args.features_dir, scored)
    print(f"[patching] loaded n={len(features)} dim={features.shape[1]}")

    fold_results = []
    for fold_idx, (train_i, test_i) in enumerate(_kfold_indices(len(features), args.folds, args.seed)):
        train_x, test_x = features[train_i], features[test_i]
        train_y, test_y = scores[train_i], scores[test_i]

        v = _train_direction(train_x, train_y, args.top_k_frac)
        baseline_scores = test_x @ v
        baseline_rho, _ = spearmanr(baseline_scores, test_y)

        train_x_ablated = _ablate(train_x, v)
        test_x_ablated = _ablate(test_x, v)
        v2 = _train_direction(train_x_ablated, train_y, args.top_k_frac)
        ablated_scores = test_x_ablated @ v2
        ablated_rho, _ = spearmanr(ablated_scores, test_y)

        cos_v_v2 = float(np.dot(v, v2))

        fold_results.append({
            "fold": fold_idx + 1,
            "n_test": int(len(test_i)),
            "baseline_rho": float(baseline_rho),
            "ablated_rho": float(ablated_rho),
            "cos_v_v2": cos_v_v2,
            "destruction_pct": float(100 * (1 - ablated_rho / baseline_rho)) if baseline_rho > 1e-6 else None,
        })

    print()
    print("Fold | n_test | baseline ρ | ablated ρ | cos(v,v2) | signal destroyed")
    print("-" * 75)
    for f in fold_results:
        d = f"{f['destruction_pct']:.1f}%" if f['destruction_pct'] is not None else "n/a"
        print(f"  {f['fold']:>2} | {f['n_test']:>6} | "
              f"{f['baseline_rho']:>+9.3f} | {f['ablated_rho']:>+8.3f} | "
              f"{f['cos_v_v2']:>+8.3f} | {d:>16}")

    mean_baseline = np.mean([f["baseline_rho"] for f in fold_results])
    mean_ablated = np.mean([f["ablated_rho"] for f in fold_results])
    mean_cos = np.mean([f["cos_v_v2"] for f in fold_results])
    mean_destr = 100 * (1 - mean_ablated / mean_baseline) if mean_baseline > 1e-6 else None

    print("-" * 75)
    print(f"MEAN          | {mean_baseline:>+9.3f} | {mean_ablated:>+8.3f} | "
          f"{mean_cos:>+8.3f} | {mean_destr:>15.1f}%")

    print()
    print("Interpretation:")
    print(f"  - Baseline direction v predicts BMD memorability at ρ={mean_baseline:+.3f}")
    print(f"  - After ablating v from features and re-training direction v2:")
    print(f"      ρ drops to {mean_ablated:+.3f} ({mean_destr:.0f}% of signal destroyed)")
    print(f"  - cos(v, v2)={mean_cos:+.3f} (near zero by construction; orthogonal to v)")
    if mean_destr is not None and mean_destr > 75:
        print(f"  - VERDICT: v captures the UNIQUE memorability direction in TRIBE features")
        print(f"             (>{75}% of predictive signal removed by ablating v alone).")
    else:
        print(f"  - VERDICT: multiple memorability-correlated directions exist;")
        print(f"             v is not the unique signal.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "features_dir": str(args.features_dir),
            "n_total": len(features),
            "folds": fold_results,
            "mean_baseline_rho": float(mean_baseline),
            "mean_ablated_rho": float(mean_ablated),
            "mean_cos_v_v2": float(mean_cos),
            "mean_destruction_pct": float(mean_destr) if mean_destr is not None else None,
        }
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\n[done] wrote {args.output}")


if __name__ == "__main__":
    main()
