"""N=20 best-of-N on 2 SVD seeds — does the +2.07 lift plateau or keep growing?"""
from __future__ import annotations
import argparse, io
from pathlib import Path
import numpy as np
from PIL import Image


def _extract_frame(mp4_path: Path) -> bytes:
    import imageio
    reader = imageio.get_reader(str(mp4_path))
    frame = reader.get_data(0)
    reader.close()
    img = Image.fromarray(frame).resize((1024, 576))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated/svd_n20"))
    parser.add_argument("--num-inference-steps", type=int, default=25)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    SEEDS = ["vid_idx0150.mp4", "vid_idx0850.mp4"]
    N = 20
    bmd_dir = Path("data/raw/bold_moments/videos")

    import modal
    Generator = modal.Cls.from_name("audience-vectors-dev", "SVDGenerator")
    gen = Generator()
    pending = []
    for ci, clip in enumerate(SEEDS):
        img = _extract_frame(bmd_dir / clip)
        stem = clip.replace(".mp4", "")
        for n in range(N):
            label = f"{stem}_n{n:02d}"
            out = args.out_dir / f"{label}.mp4"
            if out.exists():
                continue
            fc = gen.generate.spawn(
                img, steering_vector=None, alpha=0.0,
                num_inference_steps=args.num_inference_steps,
                seed=20000 + ci * 100 + n, output_label=label,
            )
            pending.append((label, fc))
    print(f"[bigger-N] spawned {len(pending)} jobs")
    for label, fc in pending:
        try:
            vb = fc.get(timeout=20 * 60)
            (args.out_dir / f"{label}.mp4").write_bytes(vb)
            print(f"  ✓ {label}", flush=True)
        except Exception as e:
            print(f"  ✗ {label}: {e}")
    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
