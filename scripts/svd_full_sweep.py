"""Full SVD steering sweep: 10 diverse seed images × 5 alphas = 50 clips.

Each seed image is the first frame of a different BMD clip, sampled across
the memorability range for diversity. Uses v_mem_clip_h_via_adapter as the
steering vector. Generates in parallel on Modal SVD (max_containers=20).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SEED_CLIPS = [
    "vid_idx0001.mp4", "vid_idx0050.mp4", "vid_idx0150.mp4",
    "vid_idx0250.mp4", "vid_idx0350.mp4", "vid_idx0450.mp4",
    "vid_idx0550.mp4", "vid_idx0650.mp4", "vid_idx0750.mp4",
    "vid_idx0850.mp4",
]
ALPHAS = [-5.0, -2.5, 0.0, +2.5, +5.0]


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
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated/svd_sweep"))
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load("data/reports/adapter_tribe_to_clip_h.pt", weights_only=False, map_location="cpu")
    v_mem = np.asarray(ckpt["v_mem_clip_h_via_adapter"], dtype=np.float32)
    print(f"[sweep] v_mem cos_alignment={ckpt['cos_alignment']:+.4f}")

    bmd_dir = Path("data/raw/bold_moments/videos")
    seed_bytes = {}
    for clip in SEED_CLIPS:
        p = bmd_dir / clip
        if not p.exists():
            print(f"  [skip] {clip} not found")
            continue
        seed_bytes[clip] = _extract_frame(p)
    print(f"[sweep] {len(seed_bytes)} seed images extracted")

    import modal
    Generator = modal.Cls.from_name("audience-vectors-dev", "SVDGenerator")
    gen = Generator()

    v_list = v_mem.tolist()
    pending = []
    for ci, (clip, img_bytes) in enumerate(seed_bytes.items()):
        clip_stem = clip.replace(".mp4", "")
        for ai, alpha in enumerate(ALPHAS):
            label = f"{clip_stem}_a{ai}"
            fc = gen.generate.spawn(
                img_bytes,
                steering_vector=v_list,
                alpha=alpha,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed + ci,
                output_label=label,
            )
            pending.append((clip_stem, alpha, label, fc))
    print(f"[sweep] spawned {len(pending)} jobs")

    for clip_stem, alpha, label, fc in pending:
        try:
            vb = fc.get(timeout=20 * 60)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {label}: {exc!r}"); continue
        if vb:
            (args.out_dir / f"{label}.mp4").write_bytes(vb)
            print(f"  ✓ {label}  α={alpha:+.1f}  {len(vb)/1024:.0f} KB")

    manifest = args.out_dir / "manifest.json"
    manifest.write_text(json.dumps([
        {"clip": c, "alpha": a, "label": l} for c, a, l, _ in pending
    ], indent=2))
    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
