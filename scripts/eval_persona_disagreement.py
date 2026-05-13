"""Analyze persona-conditioned labels: where do personas agree vs disagree?

Per-segment cross-persona stdev tells us which clips have "audience
consensus" (low stdev) vs "audience-polarizing" (high stdev). The
polarizing clips are the most informative for downstream contrastive
audience-vector extraction — that's where personas project differently
on the same stimulus.

Also reports per-axis disagreement, and correlations between persona
clusters across segments.

Usage:
    uv run python scripts/eval_persona_disagreement.py
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import polars as pl  # noqa: PLC0415

    cfg = get_config()
    labels_path = args.labels or (cfg.paths.labels / "synthetic_persona_gemini.parquet")
    if not labels_path.exists():
        raise SystemExit(f"missing {labels_path}")

    df = pl.read_parquet(labels_path)
    n_total = len(df)
    persona_ids = sorted({r["persona_id"] for r in df.iter_rows(named=True)})
    segment_ids = sorted({r["segment_id"] for r in df.iter_rows(named=True)})
    print(f"n_labels={n_total} personas={len(persona_ids)} segments={len(segment_ids)}\n")

    # Build (segment, axis) -> {persona: score}
    by_seg_axis: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    axes: set[str] = set()
    for r in df.iter_rows(named=True):
        s = r["scores"]
        if not isinstance(s, dict):
            continue
        for axis, val in s.items():
            by_seg_axis[(r["segment_id"], axis)][r["persona_id"]] = val
            axes.add(axis)

    # Per-segment cross-persona stdev — averaged across all axes.
    seg_disagreement: list[tuple[str, float, float]] = []
    for seg in segment_ids:
        stdevs = []
        means = []
        for axis in axes:
            vals = list(by_seg_axis[(seg, axis)].values())
            if len(vals) > 1:
                stdevs.append(statistics.stdev(vals))
                means.append(statistics.mean(vals))
        if stdevs:
            seg_disagreement.append((seg, statistics.mean(means), statistics.mean(stdevs)))
    seg_disagreement.sort(key=lambda x: -x[2])

    print("Most audience-polarizing segments (high cross-persona stdev):")
    for seg, m, s in seg_disagreement[:5]:
        print(f"  {seg}   mean={m:.3f}  stdev={s:.3f}")
    print("\nMost audience-consensus segments (low cross-persona stdev):")
    for seg, m, s in seg_disagreement[-5:]:
        print(f"  {seg}   mean={m:.3f}  stdev={s:.3f}")

    # Per-axis disagreement.
    print("\nPer-axis avg cross-persona stdev (which axes split audiences most):")
    axis_stdev: list[tuple[str, float]] = []
    for axis in sorted(axes):
        all_stdevs = []
        for seg in segment_ids:
            vals = list(by_seg_axis[(seg, axis)].values())
            if len(vals) > 1:
                all_stdevs.append(statistics.stdev(vals))
        if all_stdevs:
            axis_stdev.append((axis, statistics.mean(all_stdevs)))
    for axis, s in sorted(axis_stdev, key=lambda x: -x[1]):
        print(f"  {axis:<22} {s:.3f}")

    # Persona-pair Spearman on memorability across segments.
    # High correlation = personas with similar taste; low = distinct viewer types.
    print("\nPersona-pair Spearman correlations on memorability:")
    persona_vectors = {
        pid: [by_seg_axis[(seg, "memorability")].get(pid, 0.0) for seg in segment_ids]
        for pid in persona_ids
    }
    for i, pa in enumerate(persona_ids):
        for pb in persona_ids[i + 1:]:
            rho = _spearman(persona_vectors[pa], persona_vectors[pb])
            print(f"  {pa:<28} vs {pb:<28} ρ={rho:+.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Persona disagreement on BMD clips",
            "",
            f"- n_labels={n_total}, personas={len(persona_ids)}, segments={len(segment_ids)}",
            "",
            "## Most audience-polarizing clips (high cross-persona stdev)",
            "",
            "| segment | mean | stdev |",
            "|---|---:|---:|",
        ]
        for seg, m, s in seg_disagreement[:5]:
            lines.append(f"| `{seg}` | {m:.3f} | {s:.3f} |")
        lines += ["", "## Per-axis cross-persona stdev", "",
                  "| axis | mean stdev |", "|---|---:|"]
        for axis, s in sorted(axis_stdev, key=lambda x: -x[1]):
            lines.append(f"| {axis} | {s:.3f} |")
        args.output.write_text("\n".join(lines) + "\n")
        print(f"\n[done] wrote {args.output}")


if __name__ == "__main__":
    main()
