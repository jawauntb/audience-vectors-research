"""SVD-XT smoke test: 1 seed image × 5 α values, generate via Modal,
download, push through TRIBE, project on v_mem (TRIBE space).

Uses v_mem_clip_h (the adapter-derived direction in CLIP-ViT-H space) as the
steering vector. SVD takes a CLIP-image embedding for cross-attention
conditioning, and we add α × v_mem_clip_h to that embedding BEFORE the U-Net
sees it.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _extract_frame(mp4_path: Path) -> bytes:
    import imageio  # noqa: PLC0415
    reader = imageio.get_reader(str(mp4_path))
    frame = reader.get_data(0)
    reader.close()
    img = Image.fromarray(frame).resize((1024, 576))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-clip", default="bmd_vid_idx0001.mp4",
                        help="BMD clip to use as the seed image (first frame).")
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated/svd_smoke"))
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load("data/reports/adapter_tribe_to_clip_h.pt", weights_only=False, map_location="cpu")
    v_mem = np.asarray(ckpt["v_mem_clip_h_via_adapter"], dtype=np.float32)
    print(f"[svd-smoke] v_mem dim={v_mem.shape[0]} cos_alignment={ckpt['cos_alignment']:+.4f}")

    seed_path = Path("data/raw/bold_moments/videos") / args.seed_clip
    if not seed_path.exists():
        raise SystemExit(f"seed clip missing: {seed_path}")
    print(f"[svd-smoke] extracting first frame from {seed_path}")
    image_bytes = _extract_frame(seed_path)

    alphas = [-2.0, -1.0, 0.0, +1.0, +2.0]

    import modal
    Generator = modal.Cls.from_name("audience-vectors-dev", "SVDGenerator")
    gen = Generator()

    v_list = v_mem.tolist()
    print(f"[svd-smoke] spawning {len(alphas)} jobs (concurrency = up to 20 H100s)")
    pending = []
    for ai, alpha in enumerate(alphas):
        label = f"smoke_a{ai}"
        fc = gen.generate.spawn(
            image_bytes,
            steering_vector=v_list,
            alpha=alpha,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            output_label=label,
        )
        pending.append((label, alpha, fc))

    print("[svd-smoke] awaiting results...")
    for label, alpha, fc in pending:
        try:
            video_bytes = fc.get(timeout=20 * 60)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {label} (α={alpha:+.1f}): {exc!r}")
            continue
        if video_bytes:
            out = args.out_dir / f"{label}.mp4"
            out.write_bytes(video_bytes)
            print(f"  ✓ {label} (α={alpha:+.1f})  {len(video_bytes)/1024:.1f} KB")

    manifest = args.out_dir / "manifest.json"
    manifest.write_text(json.dumps([
        {"label": l, "alpha": a} for l, a, _ in pending
    ], indent=2))
    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
