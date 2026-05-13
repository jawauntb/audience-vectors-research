"""Compare synthetic VLM labels against ground-truth human labels.

Currently joins Gemini synthetic labels (`data/labels/synthetic_gemini.parquet`)
against BOLD Moments human memorability scores (from the annotations.json
on disk). Reports Spearman correlations for each VLM axis against
`memorability_score`, plus a small summary table.

Usage:
    uv run python scripts/eval_label_correlation.py
    uv run python scripts/eval_label_correlation.py --output data/reports/vlm_vs_human.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audience_vectors.config import get_config


def _spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    rx = sorted(range(n), key=lambda i: xs[i])
    ry = sorted(range(n), key=lambda i: ys[i])
    rank_x = [0] * n
    rank_y = [0] * n
    for rank, i in enumerate(rx):
        rank_x[i] = rank
    for rank, i in enumerate(ry):
        rank_y[i] = rank
    mx = sum(rank_x) / n
    my = sum(rank_y) / n
    num = sum((rx_ - mx) * (ry_ - my) for rx_, ry_ in zip(rank_x, rank_y))
    dx = (sum((rx_ - mx) ** 2 for rx_ in rank_x)) ** 0.5
    dy = (sum((ry_ - my) ** 2 for ry_ in rank_y)) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Path to synthetic labels parquet (default: data/labels/synthetic_gemini.parquet).",
    )
    parser.add_argument(
        "--bmd-annotations",
        type=Path,
        default=None,
        help="Path to BMD annotations.json (default: $BOLD_MOMENTS_ROOT/annotations.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write a markdown report here (default: stdout only).",
    )
    args = parser.parse_args()

    import polars as pl  # noqa: PLC0415

    cfg = get_config()
    labels_path = args.labels or (cfg.paths.labels / "synthetic_gemini.parquet")
    bmd_path = args.bmd_annotations or (Path("./data/raw/bold_moments/annotations.json"))
    if not labels_path.exists():
        raise SystemExit(f"labels parquet missing: {labels_path}")
    if not bmd_path.exists():
        raise SystemExit(f"BMD annotations missing: {bmd_path}")

    labels_df = pl.read_parquet(labels_path)
    with bmd_path.open() as fh:
        ann = json.load(fh)
    mem_lookup = {f"bmd_vid_idx{eid}": e["memorability_score"] for eid, e in ann.items()}

    real_mem: list[float] = []
    axis_values: dict[str, list[float]] = {}
    for row in labels_df.iter_rows(named=True):
        video_id = row["segment_id"].rsplit("_seg_", 1)[0]
        real = mem_lookup.get(video_id)
        scores = row["scores"]
        if real is None or not isinstance(scores, dict):
            continue
        real_mem.append(real)
        for axis, value in scores.items():
            axis_values.setdefault(axis, []).append(value)

    n = len(real_mem)
    if n == 0:
        raise SystemExit("no overlap between labels and BMD ground truth")

    rows: list[tuple[str, float, float, float]] = []
    for axis, ys in sorted(axis_values.items()):
        rho = _spearman(real_mem, ys)
        rows.append((axis, rho, min(ys), max(ys)))
    rows.sort(key=lambda r: -abs(r[1]))

    print(f"n={n}  ground_truth=memorability_score (BMD/Memento)")
    print(f"{'axis':<22} {'spearman_rho':>14} {'min':>8} {'max':>8}")
    for axis, rho, lo, hi in rows:
        print(f"{axis:<22} {rho:>+14.3f} {lo:>8.2f} {hi:>8.2f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# VLM synthetic labels vs BOLD Moments human memorability",
            "",
            f"- Samples: **n={n}**",
            f"- Ground truth: `memorability_score` from BMD `annotations.json`",
            "- VLM: zero-shot Gemini 2.5 Flash via `audience_vectors.labeling.GeminiLabeler`",
            "",
            "| axis | Spearman ρ | min | max |",
            "|---|---:|---:|---:|",
        ]
        for axis, rho, lo, hi in rows:
            lines.append(f"| {axis} | {rho:+.3f} | {lo:.2f} | {hi:.2f} |")
        args.output.write_text("\n".join(lines) + "\n")
        print(f"\n[done] wrote {args.output}")


if __name__ == "__main__":
    main()
