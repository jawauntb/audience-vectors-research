"""Per-persona Spearman correlations between Gemini predictions and BMD ground truth.

Where the global VLM scores at +0.51 against BMD memorability (see
vlm_vs_human.md), each persona-conditioned variant is a different
audience hypothesis. This script shows how each persona's predictions
align with the global human memorability score.

If a persona's predictions correlate strongly, that persona's taste
overlaps with the BMD subject pool. If a persona is uncorrelated, that
persona has tastes orthogonal to the human study — which is the actual
signal we want for audience-vector decomposition.

Usage:
    uv run python scripts/eval_persona_vs_bmd.py --output data/reports/persona_vs_bmd.md
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import polars as pl  # noqa: PLC0415

    cfg = get_config()
    persona_labels_path = cfg.paths.labels / "synthetic_persona_gemini.parquet"
    bmd_path = Path("./data/raw/bold_moments/annotations.json")
    if not persona_labels_path.exists() or not bmd_path.exists():
        raise SystemExit("missing persona labels or BMD annotations")

    df = pl.read_parquet(persona_labels_path)
    with bmd_path.open() as fh:
        ann = json.load(fh)
    mem_lookup = {f"bmd_vid_idx{eid}": e["memorability_score"] for eid, e in ann.items()}

    # Group: persona_id -> {segment_id: {axis: score}}
    by_persona: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in df.iter_rows(named=True):
        if not isinstance(r["scores"], dict):
            continue
        by_persona[r["persona_id"]][r["segment_id"]] = dict(r["scores"])

    axes = sorted({a for segs in by_persona.values() for s in segs.values() for a in s})
    personas = sorted(by_persona)

    print(f"personas: {len(personas)}  axes: {len(axes)}  segments_total: ")
    print(f"\n{'persona':<28} {'n':>4}  " + "  ".join(f"{ax[:10]:>10}" for ax in axes))
    print("-" * (32 + 12 * len(axes)))

    rows: list[tuple[str, int, dict[str, float]]] = []
    for persona in personas:
        seg_scores = by_persona[persona]
        # Build aligned arrays of (gt, pred) for each axis.
        real: list[float] = []
        per_axis_pred: dict[str, list[float]] = {a: [] for a in axes}
        for seg_id, scores in seg_scores.items():
            video_id = seg_id.rsplit("_seg_", 1)[0]
            gt = mem_lookup.get(video_id)
            if gt is None:
                continue
            real.append(gt)
            for a in axes:
                per_axis_pred[a].append(float(scores.get(a, 0.0)))
        per_axis_rho: dict[str, float] = {
            a: _spearman(real, per_axis_pred[a]) for a in axes
        }
        rows.append((persona, len(real), per_axis_rho))
        print(f"{persona:<28} {len(real):>4}  " +
              "  ".join(f"{per_axis_rho[a]:>+10.3f}" for a in axes))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Per-persona Spearman correlation vs BMD human memorability",
            "",
            "Each cell is Spearman ρ between persona-conditioned Gemini predictions",
            "for that axis and BMD's human memorability_score, computed across the",
            "20 sample BMD clips.",
            "",
            "| persona | n | " + " | ".join(axes) + " |",
            "|---|---:|" + "---:|" * len(axes),
        ]
        for persona, n, per_axis in rows:
            lines.append(
                f"| {persona} | {n} | "
                + " | ".join(f"{per_axis[a]:+.3f}" for a in axes)
                + " |"
            )
        args.output.write_text("\n".join(lines) + "\n")
        print(f"\n[done] wrote {args.output}")


if __name__ == "__main__":
    main()
