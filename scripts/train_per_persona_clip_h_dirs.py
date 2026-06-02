"""Train per-persona contrastive directions in CLIP-ViT-H space directly from
BMD CLIP-ViT-H embeddings + Haiku persona memorability scores. These directions
will be used for per-persona α-steering on SVD."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    print("[per-persona-clip-h] loading data")
    clip = np.load("data/features/clip_image_h_embeddings.npz", allow_pickle=False)
    embeds = np.asarray(clip["embeddings"], dtype=np.float32)
    sids = list(np.asarray(clip["sample_ids"]).tolist())
    sid_to_idx = {s: i for i, s in enumerate(sids)}

    persona_df = pl.read_parquet("data/labels/synthetic_persona_haiku_clean.parquet")
    scores_struct = persona_df.select("scores").unnest("scores")
    rows = persona_df.with_columns(
        scores_struct["memorability"].alias("_s")
    ).select(["persona_id", "segment_id", "_s"]).to_dicts()
    by_persona: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["_s"] is None: continue
        by_persona.setdefault(r["persona_id"], {})[r["segment_id"]] = float(r["_s"])

    directions = {}
    for p, seg_scores in by_persona.items():
        pairs = [(sid_to_idx[s], v) for s, v in seg_scores.items() if s in sid_to_idx]
        if len(pairs) < 50: continue
        idxs = np.asarray([i for i, _ in pairs])
        pscs = np.asarray([s for _, s in pairs], dtype=np.float32)
        o = np.argsort(pscs); ne = int(len(pscs) * 0.30)
        E = embeds[idxs]
        d = E[o[-ne:]].mean(axis=0) - E[o[:ne]].mean(axis=0)
        d /= (np.linalg.norm(d) + 1e-12)
        directions[p] = d
        print(f"  {p}: dim={d.shape[0]}, norm=1.0")

    # Cosine matrix to confirm structure
    names = sorted(directions)
    V = np.stack([directions[n] for n in names])
    cos = V @ V.T
    mask = ~np.eye(len(names), dtype=bool)
    off = cos[mask]
    print(f"\nPersona CLIP-ViT-H direction cosines (off-diagonal):")
    print(f"  mean   = {off.mean():+.3f}")
    print(f"  median = {np.median(off):+.3f}")
    print(f"  range  = [{off.min():+.3f}, {off.max():+.3f}]")

    out = Path("data/reports/persona_clip_h_directions.npz")
    np.savez_compressed(out, **{f"v__{n}": directions[n] for n in names})
    Path("data/reports/persona_clip_h_directions.json").write_text(json.dumps({
        "personas": names,
        "dim": int(V.shape[1]),
        "off_diag_cosine": {
            "mean": float(off.mean()),
            "median": float(np.median(off)),
            "min": float(off.min()),
            "max": float(off.max()),
        },
    }, indent=2))
    print(f"\n[done] saved {out}")


if __name__ == "__main__":
    main()
