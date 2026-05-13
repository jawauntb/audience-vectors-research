"""Compare every memorability predictor we have against BMD ground truth.

  - BMD human memorability_score (ground truth)
  - Zero-shot Gemini synthetic memorability (no training)
  - V-JEPA contrastive vector projection (trained on top/bottom split)

Reports Spearman correlation, plus the difference at the held-out band.

Usage:
    uv run python scripts/eval_baseline_comparison.py \\
        --vector data/models/vectors/<vector_id>.npz \\
        --features-dir data/features/vjepa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audience_vectors.activations import project_features


def _spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    rx = sorted(range(n), key=lambda i: xs[i])
    ry = sorted(range(n), key=lambda i: ys[i])
    rank_x = [0] * n
    rank_y = [0] * n
    for r, i in enumerate(rx):
        rank_x[i] = r
    for r, i in enumerate(ry):
        rank_y[i] = r
    mx = sum(rank_x) / n
    my = sum(rank_y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rank_x, rank_y))
    dx = (sum((a - mx) ** 2 for a in rank_x)) ** 0.5
    dy = (sum((b - my) ** 2 for b in rank_y)) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def _load_bmd_mem() -> dict[str, float]:
    bmd_path = Path("./data/raw/bold_moments/annotations.json")
    if not bmd_path.exists():
        return {}
    with bmd_path.open() as fh:
        ann = json.load(fh)
    return {
        f"bmd_vid_idx{eid}": float(e["memorability_score"])
        for eid, e in ann.items()
        if "memorability_score" in e
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vector",
        type=Path,
        required=True,
        help="Path to .npz contrastive direction (from train_audience_vectors.py).",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("./data/features/vjepa"),
        help="Directory of per-segment .npz feature files.",
    )
    parser.add_argument(
        "--gemini-labels",
        type=Path,
        default=Path("./data/labels/synthetic_gemini.parquet"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import polars as pl  # noqa: PLC0415

    bmd_mem = _load_bmd_mem()
    if not bmd_mem:
        raise SystemExit("BMD ground truth not found")

    # Gemini scores per segment.
    gem_df = pl.read_parquet(args.gemini_labels) if args.gemini_labels.exists() else None
    gem_per_seg: dict[str, float] = {}
    if gem_df is not None:
        for r in gem_df.iter_rows(named=True):
            if isinstance(r["scores"], dict) and "memorability" in r["scores"]:
                gem_per_seg[r["segment_id"]] = float(r["scores"]["memorability"])

    # Contrastive direction.
    payload = np.load(args.vector, allow_pickle=False)
    direction = payload["direction"]
    meta_path = args.vector.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    train_ids = set(meta.get("positive_ids", []) + meta.get("negative_ids", []))

    # Build aligned (sample_id, bmd_mem, gemini_mem, vjepa_proj) for every segment that has features.
    rows: list[tuple[str, float, float | None, float | None, bool]] = []
    for feat_path in sorted(args.features_dir.glob("*.npz")):
        sample_id = feat_path.stem
        video_id = sample_id.rsplit("_seg_", 1)[0]
        gt = bmd_mem.get(video_id)
        if gt is None:
            continue
        feat = np.load(feat_path, allow_pickle=False)
        if "embedding" in feat.files:
            vec = np.asarray(feat["embedding"], dtype=np.float32)
            proj = project_features(vec, direction)
        elif "frames" in feat.files:
            proj = project_features(np.asarray(feat["frames"], dtype=np.float32), direction)
        else:
            proj = None
        gem = gem_per_seg.get(sample_id)
        rows.append((sample_id, gt, gem, proj, sample_id in train_ids))

    # Overall numbers (all segments with features + BMD label).
    gem_all = [r[2] for r in rows if r[2] is not None]
    bmd_all_with_gem = [r[1] for r in rows if r[2] is not None]
    proj_all = [r[3] for r in rows if r[3] is not None]
    bmd_all_with_proj = [r[1] for r in rows if r[3] is not None]

    # Held-out: segments not in the training tails.
    holdout = [r for r in rows if not r[4]]
    holdout_bmd_gem = [(r[1], r[2]) for r in holdout if r[2] is not None]
    holdout_bmd_proj = [(r[1], r[3]) for r in holdout if r[3] is not None]

    print(f"All segments with BMD label + V-JEPA features: n={len(rows)}")
    print(f"  ρ(BMD, Gemini)         = {_spearman(bmd_all_with_gem, gem_all):+.3f}   "
          f"n_paired={len(gem_all)}")
    print(f"  ρ(BMD, V-JEPA proj)    = {_spearman(bmd_all_with_proj, proj_all):+.3f}   "
          f"n_paired={len(proj_all)}")

    print(f"\nHeld-out (not in training tails): n={len(holdout)}")
    if holdout_bmd_gem:
        ys, xs = zip(*holdout_bmd_gem)
        print(f"  ρ(BMD, Gemini)         = {_spearman(list(ys), list(xs)):+.3f}   "
              f"n_paired={len(holdout_bmd_gem)}")
    if holdout_bmd_proj:
        ys, xs = zip(*holdout_bmd_proj)
        print(f"  ρ(BMD, V-JEPA proj)    = {_spearman(list(ys), list(xs)):+.3f}   "
              f"n_paired={len(holdout_bmd_proj)}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Baseline comparison — BMD memorability vs predictors",
            "",
            f"- Vector: `{args.vector.name}`",
            f"- Features dir: `{args.features_dir}`",
            f"- Training tails IDs: {len(train_ids)}",
            "",
            "## All segments",
            "",
            "| predictor | Spearman ρ | n |",
            "|---|---:|---:|",
            f"| zero-shot Gemini memorability | "
            f"{_spearman(bmd_all_with_gem, gem_all):+.3f} | {len(gem_all)} |",
            f"| V-JEPA contrastive projection | "
            f"{_spearman(bmd_all_with_proj, proj_all):+.3f} | {len(proj_all)} |",
            "",
            "## Held-out (excluding training tails)",
            "",
            "| predictor | Spearman ρ | n |",
            "|---|---:|---:|",
        ]
        if holdout_bmd_gem:
            ys, xs = zip(*holdout_bmd_gem)
            lines.append(
                f"| zero-shot Gemini memorability | "
                f"{_spearman(list(ys), list(xs)):+.3f} | {len(holdout_bmd_gem)} |"
            )
        if holdout_bmd_proj:
            ys, xs = zip(*holdout_bmd_proj)
            lines.append(
                f"| V-JEPA contrastive projection | "
                f"{_spearman(list(ys), list(xs)):+.3f} | {len(holdout_bmd_proj)} |"
            )
        args.output.write_text("\n".join(lines) + "\n")
        print(f"\n[done] wrote {args.output}")


if __name__ == "__main__":
    main()
