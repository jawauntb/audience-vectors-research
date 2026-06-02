"""Re-encode BMD clip frames with CLIP-ViT-H-14 (the encoder SVD uses)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


def _frames_from_mp4(path: Path, n_frames: int = 4) -> list[Image.Image]:
    import imageio  # noqa: PLC0415
    reader = imageio.get_reader(str(path))
    n_total = 0
    for _ in reader:
        n_total += 1
    reader = imageio.get_reader(str(path))
    indices = np.linspace(0, max(0, n_total - 1), num=n_frames, dtype=int)
    out: list[Image.Image] = []
    for idx in indices:
        out.append(Image.fromarray(reader.get_data(int(idx))))
    reader.close()
    return out


def main() -> None:
    print("[clip-h] loading CLIP-ViT-H-14")
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    proc = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(device).eval()
    print(f"[clip-h] device={device}")

    bmd_dir = Path("data/raw/bold_moments/videos")
    mp4s = sorted(bmd_dir.glob("*.mp4"))
    print(f"[clip-h] {len(mp4s)} clips")

    out_dir = Path("data/features/clip_image_h")
    out_dir.mkdir(parents=True, exist_ok=True)

    embeds, sids = [], []
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
            print(f"  ✗ {mp4.name}: {exc}")
            continue
        inputs = proc(images=frames, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs).image_embeds  # (4, 1024)
        emb = out.mean(dim=0).float().cpu().numpy()
        emb /= (np.linalg.norm(emb) + 1e-12)
        np.savez_compressed(out_path, embedding=emb.astype(np.float32))
        embeds.append(emb)
        sids.append(sid)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(mp4s)} encoded")

    np.savez_compressed(
        "data/features/clip_image_h_embeddings.npz",
        embeddings=np.stack(embeds),
        sample_ids=np.asarray(sids),
    )
    print(f"[done] {len(embeds)} clips → CLIP-ViT-H embeddings (dim=1024)")


if __name__ == "__main__":
    main()
