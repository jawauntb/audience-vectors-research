"""Approach B: TRIBE → CLIP/text-embedding adapter for steering.

Pipeline:
  1. For every BMD clip we have BOTH TRIBE features (20484-dim) AND a Gemini-
     produced text description.
  2. Encode the text description with a sentence/CLIP-style text encoder →
     fixed-dim embedding e_text.
  3. Train an MLP adapter: f(TRIBE) ≈ e_text. (regression to the text manifold)
  4. Compute v_mem_TRIBE: contrastive direction in TRIBE space (mem top-30 vs bot-30).
  5. Compute v_mem_text = f(v_mem_TRIBE) — the memorability direction in text-embedding space.
  6. Validate: project all caption embeddings onto v_mem_text, measure AUC vs memorability.
  7. Steering demo: take a base prompt, add α · v_mem_text to its embedding,
     report the nearest neighbors (concepts) of the steered embedding.

This is the feasible version of "brain-derived directions can steer
a text-conditioned generative model" — we derive the steering direction;
plugging it into a real T2V generator is the remaining step (out of scope
for compute now, but the adapter + direction we save are the interface).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def _bmd() -> dict[str, float]:
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    return {
        f"bmd_vid_idx{eid}": float(e["memorability_score"])
        for eid, e in ann.items() if "memorability_score" in e
    }


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
    print("[B] loading data")
    g = pl.read_parquet("data/labels/synthetic_gemini.parquet").unique(subset=["segment_id"])
    g = g.filter(pl.col("reason").is_not_null())

    bmd = _bmd()
    feat_dir = Path("data/features/tribe")
    rows: list[dict] = []
    for r in g.iter_rows(named=True):
        sid = r["segment_id"]
        vid = sid.split("_seg_")[0]
        f = feat_dir / f"{sid}.npz"
        if not f.exists() or vid not in bmd:
            continue
        rows.append({"sid": sid, "caption": r["reason"], "mem": bmd[vid], "feature_path": f})
    print(f"[B] {len(rows)} clips with TRIBE+caption+BMD")

    print("[B] encoding captions (all-MiniLM-L6-v2, dim=384)")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    captions = [r["caption"] for r in rows]
    emb = model.encode(captions, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    out_dim = emb.shape[1]
    print(f"[B] caption embeddings: {emb.shape}")

    print("[B] loading TRIBE features")
    X = np.stack([_load_tribe(r["feature_path"]) for r in rows]).astype(np.float32)
    mems = np.asarray([r["mem"] for r in rows], dtype=np.float32)
    in_dim = X.shape[1]
    print(f"[B] X.shape = {X.shape}")

    # ===== train adapter =====
    print("[B] training adapter TRIBE → text-embedding")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[B] device = {device}")
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(rows))
    split = int(0.9 * len(rows))
    train_idx = perm[:split]
    val_idx = perm[split:]

    adapter = Adapter(in_dim=in_dim, hidden=1024, out_dim=out_dim).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)

    Xt = torch.tensor(X[train_idx], device=device)
    Yt = torch.tensor(emb[train_idx], device=device)
    Xv = torch.tensor(X[val_idx], device=device)
    Yv = torch.tensor(emb[val_idx], device=device)

    cos_loss = nn.CosineEmbeddingLoss()
    target = torch.ones(len(train_idx), device=device)
    target_v = torch.ones(len(val_idx), device=device)

    epochs = 200
    batch = 64
    for ep in range(epochs):
        idx = torch.randperm(len(train_idx), device=device)
        adapter.train()
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            pred = adapter(Xt[b])
            l_cos = cos_loss(pred, Yt[b], target[b])
            l_mse = ((pred - Yt[b]) ** 2).mean()
            loss = 0.5 * l_cos + 0.5 * l_mse
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            adapter.eval()
            with torch.no_grad():
                pv = adapter(Xv)
                v_cos = nn.functional.cosine_similarity(pv, Yv).mean().item()
                v_mse = ((pv - Yv) ** 2).mean().item()
                print(f"[B] epoch {ep:3d}  train_loss={loss.item():.4f}  "
                      f"val_cos={v_cos:.4f}  val_mse={v_mse:.4f}")

    adapter.eval()

    # ===== compute v_mem in both spaces =====
    print("[B] computing memorability directions")
    order = np.argsort(mems)
    n_each = int(len(rows) * 0.30)
    low_idx, high_idx = order[:n_each], order[-n_each:]
    v_mem_tribe = X[high_idx].mean(axis=0) - X[low_idx].mean(axis=0)
    v_mem_tribe /= (np.linalg.norm(v_mem_tribe) + 1e-12)
    v_mem_text = emb[high_idx].mean(axis=0) - emb[low_idx].mean(axis=0)
    v_mem_text /= (np.linalg.norm(v_mem_text) + 1e-12)

    # adapter-mapped direction: apply on a +ve and -ve set, take difference
    with torch.no_grad():
        v_high_adapted = adapter(torch.tensor(X[high_idx], device=device)).mean(dim=0).cpu().numpy()
        v_low_adapted = adapter(torch.tensor(X[low_idx], device=device)).mean(dim=0).cpu().numpy()
    v_mem_text_via_adapter = v_high_adapted - v_low_adapted
    v_mem_text_via_adapter /= (np.linalg.norm(v_mem_text_via_adapter) + 1e-12)

    cos_adapter_vs_text = float(np.dot(v_mem_text, v_mem_text_via_adapter))
    print(f"[B] cos(v_mem_text, v_mem_text_via_adapter) = {cos_adapter_vs_text:+.3f}")

    # ===== validate: does text direction predict BMD? =====
    proj_text = emb @ v_mem_text
    proj_adapter = emb @ v_mem_text_via_adapter

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    aucs_text, aucs_adapter = [], []
    for tr, te in kf.split(emb):
        # fit fresh direction per fold to avoid leakage
        o = np.argsort(mems[tr])
        ne = int(len(tr) * 0.30)
        v_f = emb[tr][o[-ne:]].mean(axis=0) - emb[tr][o[:ne]].mean(axis=0)
        v_f /= (np.linalg.norm(v_f) + 1e-12)
        med = np.median(mems[te])
        bin_lab = (mems[te] > med).astype(int)
        aucs_text.append(roc_auc_score(bin_lab, emb[te] @ v_f))

        # adapter-based direction on this fold
        with torch.no_grad():
            hi = adapter(torch.tensor(X[tr][o[-ne:]], device=device)).mean(dim=0).cpu().numpy()
            lo = adapter(torch.tensor(X[tr][o[:ne]], device=device)).mean(dim=0).cpu().numpy()
        v_a = hi - lo
        v_a /= (np.linalg.norm(v_a) + 1e-12)
        aucs_adapter.append(roc_auc_score(bin_lab, emb[te] @ v_a))

    print(f"[B] caption-direction AUC (text-native):       {np.mean(aucs_text):.3f} ± {np.std(aucs_text):.3f}")
    print(f"[B] caption-direction AUC (adapter-derived):   {np.mean(aucs_adapter):.3f} ± {np.std(aucs_adapter):.3f}")

    # ===== steering: nearest captions to v_mem_text and to a steered example =====
    sims_text_only = emb @ v_mem_text
    rank = np.argsort(-sims_text_only)
    closest_captions_to_direction = [
        {"caption": rows[i]["caption"][:140], "mem": float(rows[i]["mem"]), "proj": float(sims_text_only[i])}
        for i in rank[:5]
    ]
    farthest_captions_to_direction = [
        {"caption": rows[i]["caption"][:140], "mem": float(rows[i]["mem"]), "proj": float(sims_text_only[i])}
        for i in rank[-5:]
    ]

    out = {
        "n_clips": len(rows),
        "device": str(device),
        "text_dim": out_dim,
        "cos_adapter_vs_text_direction": cos_adapter_vs_text,
        "text_direction_auc_mean": float(np.mean(aucs_text)),
        "text_direction_auc_std": float(np.std(aucs_text)),
        "adapter_direction_auc_mean": float(np.mean(aucs_adapter)),
        "adapter_direction_auc_std": float(np.std(aucs_adapter)),
        "closest_captions_to_direction": closest_captions_to_direction,
        "farthest_captions_to_direction": farthest_captions_to_direction,
    }
    Path("data/reports/approach_b.json").write_text(json.dumps(out, indent=2))

    # save the trained adapter + directions for downstream use
    torch.save({
        "state_dict": adapter.state_dict(),
        "config": {"in_dim": in_dim, "hidden": 1024, "out_dim": out_dim},
        "v_mem_tribe": v_mem_tribe,
        "v_mem_text": v_mem_text,
        "v_mem_text_via_adapter": v_mem_text_via_adapter,
        "embeddings": emb,
        "sample_ids": [r["sid"] for r in rows],
        "mems": mems,
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
    }, "data/reports/approach_b_adapter.pt")
    print("[done] wrote approach_b.json + approach_b_adapter.pt")


if __name__ == "__main__":
    main()
