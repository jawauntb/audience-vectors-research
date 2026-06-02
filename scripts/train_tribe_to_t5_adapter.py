"""Train TRIBE → T5-XXL adapter for brain-direction-conditioned generation.

Mirrors `scripts/approach_b_clip_adapter.py` but to 4096-dim T5 embeddings
(CogVideoX's text encoder space). Produces:
  - data/reports/adapter_tribe_to_t5.pt — trained MLP weights
  - v_mem_t5 — the brain-derived memorability direction translated into T5 space
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
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    print("[adapter-T5] loading data")
    t5 = np.load("data/features/t5xxl_captions.npz", allow_pickle=False)
    t5_embeds = np.asarray(t5["embeddings"], dtype=np.float32)
    t5_ids = list(np.asarray(t5["sample_ids"]).tolist())
    print(f"  T5 embeds: {t5_embeds.shape}")

    bmd = _bmd()
    feat_dir = Path("data/features/tribe")

    rows = []
    for i, sid in enumerate(t5_ids):
        vid = sid.split("_seg_")[0]
        if vid not in bmd:
            continue
        f = feat_dir / f"{sid}.npz"
        if not f.exists():
            continue
        rows.append({"sid": sid, "i": i, "tribe": _load_tribe(f), "t5": t5_embeds[i], "mem": bmd[vid]})

    X = np.stack([r["tribe"] for r in rows]).astype(np.float32)
    Y = np.stack([r["t5"] for r in rows]).astype(np.float32)
    mems = np.asarray([r["mem"] for r in rows], dtype=np.float32)
    print(f"  X (TRIBE): {X.shape}, Y (T5): {Y.shape}, mems: {mems.shape}")

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[adapter-T5] device={device}")

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(rows))
    split = int(0.9 * len(rows))
    tr, va = perm[:split], perm[split:]

    adapter = Adapter(in_dim=X.shape[1], hidden=2048, out_dim=Y.shape[1]).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=5e-4, weight_decay=1e-4)

    Xt = torch.tensor(X[tr], device=device)
    Yt = torch.tensor(Y[tr], device=device)
    Xv = torch.tensor(X[va], device=device)
    Yv = torch.tensor(Y[va], device=device)

    cos_loss = nn.CosineEmbeddingLoss()
    target_t = torch.ones(len(tr), device=device)
    target_v = torch.ones(len(va), device=device)  # noqa: F841 unused-but-typing

    epochs = 200
    batch = 64
    for ep in range(epochs):
        idx = torch.randperm(len(tr), device=device)
        adapter.train()
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            pred = adapter(Xt[b])
            l_cos = cos_loss(pred, Yt[b], target_t[b])
            l_mse = ((pred - Yt[b]) ** 2).mean()
            loss = 0.5 * l_cos + 0.5 * l_mse
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            adapter.eval()
            with torch.no_grad():
                pv = adapter(Xv)
                v_cos = nn.functional.cosine_similarity(pv, Yv).mean().item()
                v_mse = ((pv - Yv) ** 2).mean().item()
                print(f"  ep {ep:3d}  train_loss={loss.item():.4f}  val_cos={v_cos:.4f}  val_mse={v_mse:.4f}")

    adapter.eval()

    # Direction: top-30% vs bot-30% memorable
    order = np.argsort(mems)
    n_each = int(len(mems) * 0.30)
    low_idx, high_idx = order[:n_each], order[-n_each:]

    # Brain-space direction (sanity check, same as before)
    v_mem_tribe = X[high_idx].mean(axis=0) - X[low_idx].mean(axis=0)
    v_mem_tribe /= (np.linalg.norm(v_mem_tribe) + 1e-12)

    # T5-native direction
    v_mem_t5_native = Y[high_idx].mean(axis=0) - Y[low_idx].mean(axis=0)
    v_mem_t5_native /= (np.linalg.norm(v_mem_t5_native) + 1e-12)

    # Adapter-derived T5-space direction
    with torch.no_grad():
        hi = adapter(torch.tensor(X[high_idx], device=device)).mean(dim=0).cpu().numpy()
        lo = adapter(torch.tensor(X[low_idx], device=device)).mean(dim=0).cpu().numpy()
    v_mem_t5_via_adapter = hi - lo
    v_mem_t5_via_adapter /= (np.linalg.norm(v_mem_t5_via_adapter) + 1e-12)

    cos_alignment = float(np.dot(v_mem_t5_native, v_mem_t5_via_adapter))
    print(f"\n[result] cos(v_mem_t5_native, v_mem_t5_via_adapter) = {cos_alignment:+.4f}")

    # Save
    out = Path("data/reports/adapter_tribe_to_t5.pt")
    torch.save({
        "state_dict": adapter.state_dict(),
        "config": {"in_dim": int(X.shape[1]), "hidden": 2048, "out_dim": int(Y.shape[1])},
        "v_mem_tribe": v_mem_tribe,
        "v_mem_t5_native": v_mem_t5_native,
        "v_mem_t5_via_adapter": v_mem_t5_via_adapter,
        "cos_alignment": cos_alignment,
        "sample_ids": [r["sid"] for r in rows],
        "encoder": "T5-XXL (CogVideoX-5b)",
    }, out)
    print(f"[done] saved {out}")

    sidecar = Path("data/reports/adapter_tribe_to_t5.json")
    sidecar.write_text(json.dumps({
        "n_clips": len(rows),
        "tribe_dim": int(X.shape[1]),
        "t5_dim": int(Y.shape[1]),
        "cos_alignment": cos_alignment,
    }, indent=2))


if __name__ == "__main__":
    main()
