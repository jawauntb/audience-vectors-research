"""5-fold cross-validation + bootstrap CIs for one feature space.

Replaces the single-split point estimate in `train_audience_vectors.py`
with a proper statistical report:

  - 5 fold contrastive-direction trainings
  - Per-fold held-out Spearman with bootstrap 95% CI
  - Mean ± stdev across folds
  - Random-baseline distribution (200 random unit vectors)

Usage:
    uv run python scripts/eval_cv.py --features-dir data/features/vjepa
    uv run python scripts/eval_cv.py --features-dir data/features/tribe \\
        --output data/reports/cv_tribe.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audience_vectors.activations import (
    cross_validate_contrastive,
    random_baseline_spearman,
    summary_to_dict,
)
from audience_vectors.config import get_config


def _load_bmd() -> dict[str, float]:
    p = Path("./data/raw/bold_moments/annotations.json")
    if not p.exists():
        raise SystemExit("missing BMD annotations")
    with p.open() as fh:
        ann = json.load(fh)
    return {
        f"bmd_vid_idx{eid}": float(e["memorability_score"])
        for eid, e in ann.items()
        if "memorability_score" in e
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-k-frac", type=float, default=0.30)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = get_config()
    cfg.paths.ensure()

    import polars as pl  # noqa: PLC0415

    segments_path = cfg.paths.training / "segments.parquet"
    if not segments_path.exists():
        raise SystemExit(f"missing {segments_path}")
    bmd_lookup = _load_bmd()
    segs = pl.read_parquet(segments_path).to_dicts()
    scored: list[tuple[str, float]] = []
    for s in segs:
        video_id = s["video_id"]
        if video_id in bmd_lookup:
            scored.append((s["sample_id"], bmd_lookup[video_id]))

    print(f"[cv] features={args.features_dir}  segments_with_gt={len(scored)}  folds={args.folds}")
    summary = cross_validate_contrastive(
        scored_segments=scored,
        features_dir=args.features_dir,
        k=args.folds,
        top_k_frac=args.top_k_frac,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    print("\nPer-fold Spearman vs BMD memorability:")
    for f in summary.folds:
        print(f"  fold {f.fold_idx + 1}/{args.folds}  "
              f"n_test={f.n_test:>4}  "
              f"ρ={f.spearman:+.3f}  "
              f"[{f.bootstrap_ci_low:+.3f}, {f.bootstrap_ci_high:+.3f}]")
    print(f"\nMean ± stdev: {summary.mean_spearman:+.3f} ± {summary.stdev_spearman:.3f}")
    print(f"Median:       {summary.median_spearman:+.3f}")
    print(f"Range over folds: [{summary.overall_ci_low:+.3f}, {summary.overall_ci_high:+.3f}]")

    print(f"\n[null] random-direction baseline (n={args.random_trials} trials):")
    mean_r, stdev_r, abs_95 = random_baseline_spearman(
        scored_segments=scored,
        features_dir=args.features_dir,
        n_trials=args.random_trials,
        seed=args.seed,
    )
    print(f"  random ρ:   {mean_r:+.3f} ± {stdev_r:.3f}  "
          f"|ρ|@95th = {abs_95:+.3f}")

    significantly_beats_random = abs(summary.mean_spearman) > abs_95
    print(f"\n→ Contrastive vector {'BEATS' if significantly_beats_random else 'DOES NOT BEAT'} "
          f"random-direction baseline (|mean_ρ|={abs(summary.mean_spearman):.3f} "
          f"vs |ρ|@95th={abs_95:.3f})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Cross-validated contrastive vector: `{args.features_dir.name}`",
            "",
            f"- Features dir: `{args.features_dir}`",
            f"- Segments with BMD ground truth: **n={len(scored)}**",
            f"- Folds: **{args.folds}**",
            f"- Bootstrap resamples per fold: {args.bootstrap_resamples}",
            f"- Random-direction baseline trials: {args.random_trials}",
            "",
            "## Per-fold results",
            "",
            "| fold | n_test | Spearman ρ | 95% CI |",
            "|---:|---:|---:|---|",
        ]
        for f in summary.folds:
            lines.append(
                f"| {f.fold_idx + 1} | {f.n_test} | "
                f"{f.spearman:+.3f} | "
                f"[{f.bootstrap_ci_low:+.3f}, {f.bootstrap_ci_high:+.3f}] |"
            )
        lines += [
            "",
            "## Summary",
            "",
            f"- **Mean Spearman**: {summary.mean_spearman:+.3f} ± {summary.stdev_spearman:.3f}",
            f"- **Median**: {summary.median_spearman:+.3f}",
            f"- **Range over folds**: [{summary.overall_ci_low:+.3f}, {summary.overall_ci_high:+.3f}]",
            "",
            "## Random-direction null distribution",
            "",
            f"- Mean ρ: {mean_r:+.3f} ± {stdev_r:.3f}",
            f"- |ρ| at 95th percentile: {abs_95:+.3f}",
            "",
            (f"**Verdict:** contrastive direction {'**beats**' if significantly_beats_random else 'does NOT beat'} "
             f"random-direction baseline (|mean_ρ|={abs(summary.mean_spearman):.3f} "
             f"vs |ρ|@95th={abs_95:.3f})."),
        ]
        args.output.write_text("\n".join(lines) + "\n")
        json_path = args.output.with_suffix(".json")
        json_path.write_text(json.dumps(summary_to_dict(summary), indent=2))
        print(f"\n[done] wrote {args.output} (+ JSON {json_path.name})")


if __name__ == "__main__":
    main()
