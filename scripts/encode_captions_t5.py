"""Encode all 1022 BMD Gemini captions through CogVideoX's T5-XXL encoder.

Saves `(sample_id, t5_mean_embedding)` to data/features/t5xxl_captions.npz.
These are needed to train the TRIBE → T5 adapter (next step).

The CogVideoX text encoder is T5-XXL (4096 dim). We mean-pool over the token
dimension to get a single per-caption vector.

Uses Modal `.map()` for parallel encoding across multiple warm containers.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    sys.path.insert(0, "src")
    import modal
    from audience_vectors.modal_app.functions.cogvideox_generator import CogVideoXGenerator

    # Load Gemini captions, unique per segment, filter to those we have TRIBE features for
    g = pl.read_parquet("data/labels/synthetic_gemini.parquet").unique(subset=["segment_id"])
    g = g.filter(pl.col("reason").is_not_null())
    captions = []
    sample_ids = []
    feat_dir = Path("data/features/tribe")
    for r in g.iter_rows(named=True):
        sid = r["segment_id"]
        if not (feat_dir / f"{sid}.npz").exists():
            continue
        sample_ids.append(sid)
        captions.append(r["reason"])
    print(f"[t5] encoding {len(captions)} captions through CogVideoX T5-XXL")

    Generator = modal.Cls.from_name("audience-vectors-dev", "CogVideoXGenerator")
    gen = Generator()

    # Use .map for parallel dispatch — Modal fans out to as many warm containers as available
    print("[t5] dispatching .map …")
    results = list(gen.predict_text_embedding.map(captions))
    print(f"[t5] received {len(results)} embeddings")

    embeds = np.stack([np.asarray(r["embedding"], dtype=np.float32) for r in results])
    out = Path("data/features/t5xxl_captions.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        embeddings=embeds,
        sample_ids=np.asarray(sample_ids),
    )
    print(f"[done] wrote {out} ({embeds.shape})")


if __name__ == "__main__":
    main()
