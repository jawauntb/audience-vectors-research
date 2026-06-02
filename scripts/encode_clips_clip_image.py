"""Encode BMD clip frames through a CLIP image encoder for image-space steering.

For each clip, sample 4 evenly-spaced frames, encode each with CLIP-ViT-L-14
image encoder, and mean-pool to a single 768-dim embedding per clip.

These will be paired with TRIBE features to train a TRIBE → CLIP-image adapter,
then a direction in CLIP-image space could be used to steer image-conditioned
T2V models (e.g. AnimateDiff, Stable Video Diffusion).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


def _frames_from_mp4(path: Path, n_frames: int = 4) -> list[Image.Image]:
    """Sample n_frames evenly-spaced PIL images from an mp4."""
    import imageio  # noqa: PLC0415
    reader = imageio.get_reader(str(path))
    meta = reader.get_meta_data()
    n_total = reader.count_frames() if hasattr(reader, "count_frames") else (
        int(meta.get("fps", 24) * meta.get("duration", 3.0))
    )
    if n_total <= 0:
        # Walk to estimate length
        n_total = 0
        for _ in reader:
            n_total += 1
        reader = imageio.get_reader(str(path))
    indices = np.linspace(0, max(0, n_total - 1), num=n_frames, dtype=int)
    out: list[Image.Image] = []
    for idx in indices:
        frame = reader.get_data(int(idx))
        out.append(Image.fromarray(frame))
    reader.close()
    return out


def main() -> None:
    print("[clip] loading CLIP-ViT-L-14")
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    model_id = "openai/clip-vit-large-patch14"
    proc = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(device).eval()
    print(f"[clip] device={device}")

    bmd_dir = Path("data/raw/bold_moments/videos")
    if not bmd_dir.exists():
        raise SystemExit(f"BMD videos missing at {bmd_dir} — needed for image encoding")
    mp4s = sorted(bmd_dir.glob("*.mp4"))
    print(f"[clip] {len(mp4s)} BMD clips found")

    out_dir = Path("data/features/clip_image")
    out_dir.mkdir(parents=True, exist_ok=True)

    embeds: list[np.ndarray] = []
    sids: list[str] = []
    for i, mp4 in enumerate(mp4s):
        sid = f"bmd_{mp4.stem}_seg_0000"
        out_path = out_dir / f"{sid}.npz"
        if out_path.exists():
            d = np.load(out_path)
            embeds.append(np.asarray(d["embedding"], dtype=np.float32))
            sids.append(sid)
            continue
        try:
            frames = _frames_from_mp4(mp4, n_frames=4)
        except Exception as exc:  # noqa: BLE001
            print(f"  [clip] ✗ {mp4.name}: {exc}")
            continue
        inputs = proc(images=frames, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs).image_embeds  # (4, 768)
        emb = out.mean(dim=0).float().cpu().numpy()
        emb /= (np.linalg.norm(emb) + 1e-12)
        np.savez_compressed(out_path, embedding=emb.astype(np.float32))
        embeds.append(emb)
        sids.append(sid)
        if (i + 1) % 50 == 0:
            print(f"  [clip] {i+1}/{len(mp4s)} encoded")

    print(f"[clip] all encoded → {len(embeds)} clips")
    np.savez_compressed(
        "data/features/clip_image_embeddings.npz",
        embeddings=np.stack(embeds),
        sample_ids=np.asarray(sids),
    )


if __name__ == "__main__":
    main()
