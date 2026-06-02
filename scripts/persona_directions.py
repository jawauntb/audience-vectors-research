"""Per-persona contrastive directions on video features.

For each persona:
  1. Use that persona's predicted `memorability` (or any axis) as the score.
  2. Train a contrastive direction on feature activations using top-K vs bottom-K
     of THAT persona's scores.
  3. Report cosine similarity between personas' directions:
     - High |cos| → personas share an axis, even if the signed cosine is negative.
     - Low |cos|  → personas are closer to independent axes.

Signed cosine means are not an orthogonality test because sign-flipped directions
are the same axis with opposite polarity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def _load_feature(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if "frames" in payload.files:
        arr = np.asarray(payload["frames"], dtype=np.float32)
        return arr.mean(axis=0) if arr.ndim == 2 else arr
    return np.asarray(payload["embedding"], dtype=np.float32)


def _direction(
    features: np.ndarray, scores: np.ndarray, top_k_frac: float = 0.30
) -> np.ndarray:
    order = np.argsort(scores)
    n_each = max(3, int(len(scores) * top_k_frac))
    neg = features[order[:n_each]].mean(axis=0)
    pos = features[order[-n_each:]].mean(axis=0)
    d = pos - neg
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else d


def _effective_rank(cos: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(cos)
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(eigvals.sum())
    if total <= 0:
        return 0.0
    probs = eigvals / total
    probs = probs[probs > 0]
    return float(np.exp(-np.sum(probs * np.log(probs))))


def _load_persona_scores(persona_file: Path, axis: str) -> dict[str, dict[str, float]]:
    df = pl.read_parquet(persona_file)
    scores = df.select("scores").unnest("scores")
    if axis not in scores.columns:
        raise SystemExit(f"axis {axis!r} not in persona scores: {scores.columns}")
    out: dict[str, dict[str, float]] = {}
    rows = (
        df.with_columns(scores[axis].alias("_score"))
        .select(["persona_id", "segment_id", "_score"])
        .to_dicts()
    )
    for r in rows:
        if r["_score"] is None:
            continue
        out.setdefault(r["persona_id"], {})[r["segment_id"]] = float(r["_score"])
    return out


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path, default=Path("data/features/tribe")
    )
    parser.add_argument(
        "--persona-file",
        type=Path,
        default=Path("data/labels/synthetic_persona_haiku_clean.parquet"),
    )
    parser.add_argument("--axis", default="memorability")
    parser.add_argument("--top-k-frac", type=float, default=0.30)
    parser.add_argument(
        "--output", type=Path, default=Path("data/reports/persona_directions.md")
    )
    args = parser.parse_args()

    if not args.persona_file.exists():
        # fallback
        alt = Path("data/labels/synthetic_persona_haiku.parquet")
        if alt.exists():
            args.persona_file = alt
        else:
            raise SystemExit(f"persona file not found: {args.persona_file}")

    by_persona = _load_persona_scores(args.persona_file, args.axis)
    print(f"[persona-dir] axis={args.axis} personas={len(by_persona)}")

    directions: dict[str, np.ndarray] = {}
    sizes: dict[str, int] = {}
    for persona_id, seg_scores in by_persona.items():
        feats, scores = [], []
        for seg, s in seg_scores.items():
            p = args.features_dir / f"{seg}.npz"
            if not p.exists():
                continue
            feats.append(_load_feature(p))
            scores.append(s)
        if len(feats) < 10:
            print(
                f"  skip {persona_id}: only n={len(feats)} segments with TRIBE features"
            )
            continue
        feats_arr = np.stack(feats)
        scores_arr = np.asarray(scores, dtype=np.float32)
        directions[persona_id] = _direction(feats_arr, scores_arr, args.top_k_frac)
        sizes[persona_id] = len(feats)

    persona_ids = sorted(directions.keys())
    print(
        f"\n  trained {len(persona_ids)} directions on {args.features_dir} features (n_min={min(sizes.values()) if sizes else 0})"
    )

    print(f"\nCosine similarity matrix (persona directions on '{args.axis}'):")
    print("  Off-diagonal close to +1.0 → personas collapse onto a shared axis.")
    print("  Off-diagonal close to -1.0 → same axis with opposite polarity.")
    print("  Off-diagonal near 0       → closer to independent axes.\n")

    cos = np.zeros((len(persona_ids), len(persona_ids)))
    for i, a in enumerate(persona_ids):
        for j, b in enumerate(persona_ids):
            cos[i, j] = float(np.dot(directions[a], directions[b]))

    header = "  " + "  ".join(p[:18].rjust(18) for p in persona_ids)
    print(header)
    for i, a in enumerate(persona_ids):
        row = "  ".join(f"{cos[i, j]:+.3f}".rjust(18) for j in range(len(persona_ids)))
        print(f"{a[:18].ljust(18)}  {row}")

    off = cos[~np.eye(len(persona_ids), dtype=bool)]
    mean_abs = float(np.mean(np.abs(off)))
    median_abs = float(np.median(np.abs(off)))
    erank = _effective_rank(cos)
    print(
        f"\nOff-diagonal stats: signed_mean={off.mean():+.3f}  signed_median={np.median(off):+.3f}  "
        f"mean_abs={mean_abs:.3f}  median_abs={median_abs:.3f}  "
        f"min={off.min():+.3f}  max={off.max():+.3f}  effective_rank={erank:.2f}/{len(persona_ids)}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Per-persona contrastive directions",
        "",
        f"- Features: `{args.features_dir}`",
        f"- Persona axis: **`{args.axis}`**",
        f"- Personas: {len(persona_ids)}",
        f"- top_k_frac: {args.top_k_frac}",
        f"- Signed off-diagonal cosine similarity: mean={off.mean():+.3f}, median={np.median(off):+.3f}",
        f"- Unsigned overlap: mean |cos|={mean_abs:.3f}, median |cos|={median_abs:.3f}",
        f"- Effective rank: {erank:.2f} / {len(persona_ids)}",
        "",
        "## Cosine similarity matrix",
        "",
        "| | " + " | ".join(persona_ids) + " |",
        "|---|" + "---|" * len(persona_ids),
    ]
    for i, a in enumerate(persona_ids):
        row_cells = [f"{cos[i, j]:+.3f}" for j in range(len(persona_ids))]
        lines.append(f"| **{a}** | " + " | ".join(row_cells) + " |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- Off-diagonal entries near **+1.0** mean two personas have the same",
        "  contrastive direction in activation space.",
        "- Off-diagonal entries near **−1.0** mean the same latent axis with",
        "  opposite polarity; squared projection would treat them as the same",
        "  axis.",
        "- Off-diagonal entries near **0** are the actual orthogonality signal.",
        "  Use mean |cos| and effective rank, not signed mean alone, to judge",
        "  whether personas decompose audience response into independent axes.",
    ]
    args.output.write_text("\n".join(lines) + "\n")
    print(f"\n[done] wrote {args.output}")

    json_out = args.output.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            {
                "axis": args.axis,
                "persona_ids": persona_ids,
                "cosine_matrix": cos.tolist(),
                "sizes": sizes,
                "off_diagonal": {
                    "mean": float(off.mean()),
                    "median": float(np.median(off)),
                    "mean_abs": mean_abs,
                    "median_abs": median_abs,
                    "min": float(off.min()),
                    "max": float(off.max()),
                    "effective_rank": erank,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
