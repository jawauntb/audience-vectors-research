"""Per-persona α-steering on SVD. For each persona direction (in CLIP-ViT-H space)
generate K seed × M alphas. We'll then TRIBE-eval everything and build a
(persona-steered × persona-scored) grid."""

from __future__ import annotations

import argparse, io
from pathlib import Path
import numpy as np
from PIL import Image


SEEDS = ["vid_idx0150.mp4", "vid_idx0250.mp4", "vid_idx0850.mp4"]
ALPHAS = [-5.0, 0.0, +5.0]  # 3 alphas × 12 personas × 3 seeds = 108 clips


def _extract_frame(mp4_path: Path) -> bytes:
    import imageio
    reader = imageio.get_reader(str(mp4_path))
    frame = reader.get_data(0); reader.close()
    img = Image.fromarray(frame).resize((1024, 576))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated/svd_per_persona"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # load per-persona CLIP-ViT-H directions
    d = np.load("data/reports/persona_clip_h_directions.npz", allow_pickle=False)
    persona_dirs = {k.replace("v__", ""): np.asarray(d[k], dtype=np.float32) for k in d.files}
    print(f"[per-persona] loaded {len(persona_dirs)} persona directions in CLIP-ViT-H space")

    import modal
    Generator = modal.Cls.from_name("audience-vectors-dev", "SVDGenerator")
    gen = Generator()

    bmd_dir = Path("data/raw/bold_moments/videos")
    seed_bytes = {s: _extract_frame(bmd_dir / s) for s in SEEDS}
    print(f"[per-persona] extracted {len(seed_bytes)} seed images")

    pending = []
    for si, (seed_file, img) in enumerate(seed_bytes.items()):
        seed_stem = seed_file.replace(".mp4", "")
        for persona_name, v in persona_dirs.items():
            for ai, alpha in enumerate(ALPHAS):
                label = f"{seed_stem}__{persona_name}__a{ai}"
                out = args.out_dir / f"{label}.mp4"
                if out.exists():
                    continue
                fc = gen.generate.spawn(
                    img, steering_vector=v.tolist(), alpha=alpha,
                    num_inference_steps=25, seed=42000 + si * 100, output_label=label,
                )
                pending.append((label, alpha, fc))
    print(f"[per-persona] spawned {len(pending)} jobs")

    for label, alpha, fc in pending:
        try:
            vb = fc.get(timeout=20 * 60)
            (args.out_dir / f"{label}.mp4").write_bytes(vb)
            print(f"  ✓ {label} α={alpha:+.1f}", flush=True)
        except Exception as e:
            print(f"  ✗ {label}: {e}", flush=True)

    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
