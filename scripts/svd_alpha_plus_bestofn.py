"""α-steering + best-of-N composition test.

For each of 3 seeds, generate N=10 variants at α=+10 (each variant gets a
different seed so within-seed variance is preserved despite all being steered).
We already have N=10 at α=0 from the original best-of-N. Compare the
within-seed lift (best − median) of the α=+10 set vs the α=0 set.

If α-steering and best-of-N compound, the α=+10 set should have a higher
overall mean AND comparable within-seed spread, giving a combined lift that
exceeds either alone."""

from __future__ import annotations

import argparse, io
from pathlib import Path
import numpy as np
from PIL import Image


SEEDS = ["vid_idx0150.mp4", "vid_idx0250.mp4", "vid_idx0850.mp4"]
N = 10
ALPHA = +10.0


def _extract_frame(mp4_path: Path) -> bytes:
    import imageio
    reader = imageio.get_reader(str(mp4_path))
    frame = reader.get_data(0); reader.close()
    img = Image.fromarray(frame).resize((1024, 576))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated/svd_alpha_bon"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    ckpt = torch.load("data/reports/adapter_tribe_to_clip_h.pt", weights_only=False, map_location="cpu")
    v_mem = np.asarray(ckpt["v_mem_clip_h_via_adapter"], dtype=np.float32).tolist()

    import modal
    Generator = modal.Cls.from_name("audience-vectors-dev", "SVDGenerator")
    gen = Generator()

    bmd_dir = Path("data/raw/bold_moments/videos")
    pending = []
    for si, seed_file in enumerate(SEEDS):
        img = _extract_frame(bmd_dir / seed_file)
        seed_stem = seed_file.replace(".mp4", "")
        for n in range(N):
            label = f"{seed_stem}_a10_n{n:02d}"
            out = args.out_dir / f"{label}.mp4"
            if out.exists(): continue
            fc = gen.generate.spawn(
                img, steering_vector=v_mem, alpha=ALPHA,
                num_inference_steps=25, seed=50000 + si * 100 + n, output_label=label,
            )
            pending.append((label, fc))
    print(f"[α+best-of-N] spawned {len(pending)} jobs at α={ALPHA}")
    for label, fc in pending:
        try:
            vb = fc.get(timeout=20 * 60)
            (args.out_dir / f"{label}.mp4").write_bytes(vb)
            print(f"  ✓ {label}", flush=True)
        except Exception as e:
            print(f"  ✗ {label}: {e}", flush=True)
    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
