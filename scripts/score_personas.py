"""Drive per-persona scoring on every TRIBE clip.

For each persona:
  1. Train a contrastive direction on TRIBE features using that persona's
     memorability scores (top 30% vs bottom 30%).
  2. Project EVERY clip's features onto the direction → per-persona score.

Then build a long-format CSV (segment_id × persona) and a summary markdown
with per-persona top/bottom clips and high-disagreement clips.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import polars as pl


def _load_feature(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if "frames" in payload.files:
        arr = np.asarray(payload["frames"], dtype=np.float32)
        return arr.mean(axis=0) if arr.ndim == 2 else arr
    return np.asarray(payload["embedding"], dtype=np.float32)


def _direction(features: np.ndarray, scores: np.ndarray, top_k_frac: float = 0.30) -> np.ndarray:
    order = np.argsort(scores)
    n_each = max(3, int(len(scores) * top_k_frac))
    neg = features[order[:n_each]].mean(axis=0)
    pos = features[order[-n_each:]].mean(axis=0)
    d = pos - neg
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else d


def _load_bmd() -> dict[str, float]:
    p = Path("./data/raw/bold_moments/annotations.json")
    with p.open() as fh:
        ann = json.load(fh)
    return {
        f"bmd_vid_idx{eid}": float(e["memorability_score"])
        for eid, e in ann.items()
        if "memorability_score" in e
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features/tribe"))
    parser.add_argument("--persona-file", type=Path, default=Path("data/labels/synthetic_persona_haiku_clean.parquet"))
    parser.add_argument("--axis", default="memorability")
    parser.add_argument("--top-k-frac", type=float, default=0.30)
    parser.add_argument("--csv-out", type=Path, default=Path("data/reports/persona_scores.csv"))
    parser.add_argument("--report-out", type=Path, default=Path("data/reports/persona_driving.md"))
    args = parser.parse_args()

    df = pl.read_parquet(args.persona_file)
    scores_struct = df.select("scores").unnest("scores")
    if args.axis not in scores_struct.columns:
        raise SystemExit(f"axis {args.axis!r} not found in {scores_struct.columns}")

    rows = df.with_columns(scores_struct[args.axis].alias("_score")).select(
        ["persona_id", "segment_id", "_score"]
    ).to_dicts()

    by_persona: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["_score"] is None:
            continue
        by_persona.setdefault(r["persona_id"], {})[r["segment_id"]] = float(r["_score"])

    # Load ALL segment features (the universe we want to score over)
    feature_files = sorted(args.features_dir.glob("*.npz"))
    print(f"[score] loading {len(feature_files)} TRIBE features")
    all_features: dict[str, np.ndarray] = {}
    for f in feature_files:
        sid = f.stem
        all_features[sid] = _load_feature(f)
    all_ids = sorted(all_features.keys())
    feat_mat = np.stack([all_features[s] for s in all_ids])

    print(f"[score] training {len(by_persona)} persona directions on axis '{args.axis}'")
    directions: dict[str, np.ndarray] = {}
    for persona_id, seg_scores in by_persona.items():
        feats = []
        scores_list = []
        for sid, s in seg_scores.items():
            if sid not in all_features:
                continue
            feats.append(all_features[sid])
            scores_list.append(s)
        if len(feats) < 10:
            continue
        directions[persona_id] = _direction(
            np.stack(feats), np.asarray(scores_list, dtype=np.float32), args.top_k_frac,
        )

    # Also compute the GLOBAL BMD direction for reference
    bmd = _load_bmd()
    bmd_feats = []
    bmd_scores_list = []
    for sid in all_ids:
        vid = sid.split("_seg_")[0]
        if vid in bmd:
            bmd_feats.append(all_features[sid])
            bmd_scores_list.append(bmd[vid])
    if bmd_feats:
        directions["BMD_human_global"] = _direction(
            np.stack(bmd_feats), np.asarray(bmd_scores_list, dtype=np.float32), args.top_k_frac,
        )

    persona_ids = sorted([p for p in directions if p != "BMD_human_global"])
    columns = persona_ids + ["BMD_human_global"]
    score_mat = np.stack([feat_mat @ directions[p] for p in columns], axis=1)

    # CSV
    csv_rows: list[dict] = []
    for i, sid in enumerate(all_ids):
        row = {"segment_id": sid}
        for j, col in enumerate(columns):
            row[col] = round(float(score_mat[i, j]), 4)
        vid = sid.split("_seg_")[0]
        row["bmd_human_memorability"] = round(bmd.get(vid, float("nan")), 4) if vid in bmd else None
        csv_rows.append(row)
    out_df = pl.DataFrame(csv_rows)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(args.csv_out)
    print(f"[csv] wrote {args.csv_out} ({len(csv_rows)} clips × {len(columns)+2} cols)")

    # Summary report
    lines: list[str] = [
        f"# Persona-driven scoring on TRIBE features",
        "",
        f"Axis: **{args.axis}** · Personas: **{len(persona_ids)}** · Clips: **{len(all_ids)}**",
        "",
        "## Top-5 clips per persona (highest predicted score)",
        "",
    ]
    for p in persona_ids:
        order = np.argsort(-score_mat[:, columns.index(p)])
        top = [all_ids[order[k]] for k in range(5)]
        lines.append(f"- **{p}**: " + ", ".join(f"`{s}`" for s in top))
    lines += [
        "",
        "## Top-5 clips by BMD human memorability (for reference)",
        "",
    ]
    bmd_col = columns.index("BMD_human_global")
    order = np.argsort(-score_mat[:, bmd_col])
    lines.append("- " + ", ".join(f"`{all_ids[order[k]]}`" for k in range(5)))

    # High-disagreement clips: cross-persona stdev
    persona_only = score_mat[:, : len(persona_ids)]
    persona_stdev = persona_only.std(axis=1)
    order_div = np.argsort(-persona_stdev)
    lines += [
        "",
        "## Most audience-polarizing clips (highest cross-persona stdev)",
        "",
        "| segment_id | cross-persona stdev | BMD human ρ |",
        "|---|---|---|",
    ]
    for k in range(10):
        idx = order_div[k]
        bmd_val = score_mat[idx, bmd_col]
        lines.append(f"| `{all_ids[idx]}` | {persona_stdev[idx]:.3f} | {bmd_val:+.3f} |")

    # Audience-orthogonal clips: high persona variance, low correlation with BMD
    # For each clip, persona scores - BMD score; high abs delta = persona drives different signal
    delta = persona_only.mean(axis=1) - score_mat[:, bmd_col]
    persona_minus_bmd = np.argsort(-np.abs(delta))
    lines += [
        "",
        "## Audience-orthogonal clips (mean persona score far from BMD global score)",
        "",
        "| segment_id | mean(persona) | BMD global | delta |",
        "|---|---|---|---|",
    ]
    for k in range(10):
        idx = persona_minus_bmd[k]
        lines.append(
            f"| `{all_ids[idx]}` | {persona_only[idx].mean():+.3f} | "
            f"{score_mat[idx, bmd_col]:+.3f} | {delta[idx]:+.3f} |"
        )

    # Cross-persona Spearman correlation
    lines += [
        "",
        "## Cross-persona Spearman correlation (on TRIBE-projected scores)",
        "",
    ]
    ranks = np.argsort(np.argsort(score_mat[:, : len(persona_ids)], axis=0), axis=0)
    corrs = np.corrcoef(ranks.T)
    off = corrs[~np.eye(len(persona_ids), dtype=bool)]
    lines.append(
        f"- Off-diagonal Spearman: mean = {off.mean():+.3f}, "
        f"median = {float(np.median(off)):+.3f}, "
        f"range [{off.min():+.3f}, {off.max():+.3f}]"
    )

    args.report_out.write_text("\n".join(lines) + "\n")
    print(f"[md] wrote {args.report_out}")

    # JSON sidecar for HTML paper
    json_out = args.report_out.with_suffix(".json")
    json_out.write_text(json.dumps({
        "axis": args.axis,
        "persona_ids": persona_ids,
        "all_columns": columns,
        "segments": all_ids,
        "scores": score_mat.tolist(),
        "directions_norm_check": {p: float(np.linalg.norm(directions[p])) for p in columns},
    }))
    print(f"[json] wrote {json_out}")


if __name__ == "__main__":
    main()
