"""TRIBE → CLIP-ViT-H-14 image-embedding adapter (1024-dim, the encoder SVD uses).

Same recipe as `train_tribe_to_clip_adapter.py` (which used ViT-L 768d).
Outputs `data/reports/adapter_tribe_to_clip_h.pt` with the trained adapter
plus v_mem_clip_h directions (native + adapter-derived).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _bmd() -> dict[str, float]:
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    return {f"bmd_vid_idx{e}": float(a["memorability_score"])
            for e, a in ann.items() if "memorability_score" in a}


def _load_tribe(path: Path) -> np.ndarray:
    p = np.load(path, allow_pickle=False)
    arr = np.asarray(p["frames"], dtype=np.float32)
    return arr.mean(axis=0) if arr.ndim == 2 else arr


class Adapter(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    print("[adapter-clip-h] loading data")
    clip_data = np.load("data/features/clip_image_h_embeddings.npz", allow_pickle=False)
    clip_embeds = np.asarray(clip_data["embeddings"], dtype=np.float32)
    clip_sids = list(np.asarray(clip_data["sample_ids"]).tolist())
    bmd = _bmd()
    feat_dir = Path("data/features/tribe")

    rows = []
    for i, sid in enumerate(clip_sids):
        vid = sid.split("_seg_")[0]
        if vid not in bmd:
            continue
        f = feat_dir / f"{sid}.npz"
        if not f.exists():
            continue
        rows.append({"sid": sid, "tribe": _load_tribe(f),
                     "clip": clip_embeds[i], "mem": bmd[vid]})

    X = np.stack([r["tribe"] for r in rows]).astype(np.float32)
    Y = np.stack([r["clip"] for r in rows]).astype(np.float32)
    mems = np.asarray([r["mem"] for r in rows], dtype=np.float32)
    print(f"  X={X.shape}  Y={Y.shape}  mems={mems.shape}")

    order = np.argsort(mems); ne = int(len(mems) * 0.30)
    low_idx, high_idx = order[:ne], order[-ne:]
    y_high_mean = Y[high_idx].mean(axis=0)
    y_low_mean = Y[low_idx].mean(axis=0)
    diff = y_high_mean - y_low_mean
    cos_means = float(np.dot(y_high_mean, y_low_mean) /
                      (np.linalg.norm(y_high_mean) * np.linalg.norm(y_low_mean) + 1e-12))
    pct = 100 * np.linalg.norm(diff) / np.linalg.norm(y_high_mean)
    print(f"\n[clip-h-space] cos(high-mean, low-mean) = {cos_means:.4f}")
    print(f"[clip-h-space] ||diff|| / ||mean||      = {pct:.2f}%")
    print(f"  (T5 was 1.2%; CLIP-ViT-L was 22.7%)\n")

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[adapter-clip-h] device={device}")
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(rows))
    split = int(0.9 * len(rows))
    tr, va = perm[:split], perm[split:]
    adapter = Adapter(in_dim=X.shape[1], hidden=1024, out_dim=Y.shape[1]).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(X[tr], device=device); Yt = torch.tensor(Y[tr], device=device)
    Xv = torch.tensor(X[va], device=device); Yv = torch.tensor(Y[va], device=device)
    cos_loss = nn.CosineEmbeddingLoss()
    target_t = torch.ones(len(tr), device=device)

    epochs = 200; batch = 64
    for ep in range(epochs):
        idx = torch.randperm(len(tr), device=device)
        adapter.train()
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            pred = adapter(Xt[b])
            loss = 0.5 * cos_loss(pred, Yt[b], target_t[b]) + 0.5 * ((pred - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            adapter.eval()
            with torch.no_grad():
                pv = adapter(Xv)
                v_cos = nn.functional.cosine_similarity(pv, Yv).mean().item()
                v_mse = ((pv - Yv) ** 2).mean().item()
                print(f"  ep {ep:3d}  loss={loss.item():.4f}  val_cos={v_cos:.4f}  val_mse={v_mse:.4f}")

    adapter.eval()
    v_mem_tribe = X[high_idx].mean(axis=0) - X[low_idx].mean(axis=0)
    v_mem_tribe /= (np.linalg.norm(v_mem_tribe) + 1e-12)
    v_mem_clip_h_native = diff / (np.linalg.norm(diff) + 1e-12)
    with torch.no_grad():
        hi = adapter(torch.tensor(X[high_idx], device=device)).mean(dim=0).cpu().numpy()
        lo = adapter(torch.tensor(X[low_idx], device=device)).mean(dim=0).cpu().numpy()
    v_mem_clip_h_via_adapter = hi - lo
    v_mem_clip_h_via_adapter /= (np.linalg.norm(v_mem_clip_h_via_adapter) + 1e-12)
    cos_alignment = float(np.dot(v_mem_clip_h_native, v_mem_clip_h_via_adapter))

    print(f"\n[result] cos(native, adapter-derived) = {cos_alignment:+.4f}")
    print(f"         (compare: T5 = -0.187; ViT-L = +0.938; MiniLM (§6.2) = +0.936)")

    out = Path("data/reports/adapter_tribe_to_clip_h.pt")
    torch.save({
        "state_dict": adapter.state_dict(),
        "config": {"in_dim": int(X.shape[1]), "hidden": 1024, "out_dim": int(Y.shape[1])},
        "v_mem_tribe": v_mem_tribe,
        "v_mem_clip_h_native": v_mem_clip_h_native,
        "v_mem_clip_h_via_adapter": v_mem_clip_h_via_adapter,
        "cos_alignment": cos_alignment,
        "cos_means_high_low": cos_means,
        "diff_pct_of_norm": float(pct),
        "encoder": "CLIP-ViT-H-14 image (laion2B-s32B-b79K)",
    }, out)
    Path("data/reports/adapter_tribe_to_clip_h.json").write_text(json.dumps({
        "n_clips": int(len(rows)),
        "cos_high_low_means": float(cos_means),
        "diff_pct": float(pct),
        "cos_alignment": float(cos_alignment),
    }, indent=2))
    print(f"[done] saved {out}")


if __name__ == "__main__":
    main()
