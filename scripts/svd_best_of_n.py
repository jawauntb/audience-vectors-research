"""Best-of-N test-time compute baseline on SVD.

For each of K seed images, generate N variants at α=0 (no steering, just
different random seeds), push through TRIBE, project on v_mem. Report:
  - spread of TRIBE projections within each seed (variance worth picking from)
  - "best-of-N" score: max over N - mean over N (the lift)
  - whether v_mem-ranking finds the most-memorable variant

If the spread is meaningful, best-of-N is a cheap production trick:
generate cheaply, score with the brain direction, ship the winner.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SEEDS = [
    "vid_idx0150.mp4",   # cleanly steerable in our sweep (ρ=+0.9)
    "vid_idx0250.mp4",   # cleanly steerable (ρ=+1.0)
    "vid_idx0850.mp4",   # cleanly steerable (ρ=+1.0)
    "vid_idx0001.mp4",   # baseline
    "vid_idx0450.mp4",   # resisted steering in our sweep (ρ=-0.3)
]
N_PER_SEED = 10


def _extract_frame(mp4_path: Path) -> bytes:
    import imageio  # noqa: PLC0415
    reader = imageio.get_reader(str(mp4_path))
    frame = reader.get_data(0)
    reader.close()
    img = Image.fromarray(frame).resize((1024, 576))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated/svd_best_of_n"))
    parser.add_argument("--num-inference-steps", type=int, default=25)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bmd_dir = Path("data/raw/bold_moments/videos")
    seed_bytes = {}
    for clip in SEEDS:
        p = bmd_dir / clip
        if not p.exists():
            print(f"  [skip] {clip} not found"); continue
        seed_bytes[clip] = _extract_frame(p)
    print(f"[best-of-N] {len(seed_bytes)} seeds × N={N_PER_SEED} variants = {len(seed_bytes) * N_PER_SEED} clips")

    import modal
    Generator = modal.Cls.from_name("audience-vectors-dev", "SVDGenerator")
    gen = Generator()

    pending = []
    for ci, (clip, img_bytes) in enumerate(seed_bytes.items()):
        clip_stem = clip.replace(".mp4", "")
        for n in range(N_PER_SEED):
            label = f"{clip_stem}_n{n:02d}"
            fc = gen.generate.spawn(
                img_bytes,
                steering_vector=None,
                alpha=0.0,
                num_inference_steps=args.num_inference_steps,
                seed=10000 + ci * 100 + n,  # different seed per variant
                output_label=label,
            )
            pending.append((clip_stem, n, label, fc))
    print(f"[best-of-N] spawned {len(pending)} jobs")

    for clip_stem, n, label, fc in pending:
        try:
            vb = fc.get(timeout=20 * 60)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {label}: {exc!r}"); continue
        if vb:
            (args.out_dir / f"{label}.mp4").write_bytes(vb)
            print(f"  ✓ {label} ({len(vb)/1024:.0f} KB)")

    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
