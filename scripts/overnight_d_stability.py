"""(D) v_mem subsampling stability — train v_mem on different random 80%
subsets of BMD, measure pairwise cosine across versions.

A high mean pairwise cosine (e.g., >0.95) is direct evidence that the
direction is a stable property of TRIBE-aligned representation, not a
quirk of specific clip selection. Pairs with §5.3 ablation:
ablation shows compactness, stability shows reproducibility.
"""

from __future__ import annotations

import json, numpy as np
from pathlib import Path


def fit_v(X, y):
    o = np.argsort(y); ne = int(len(y) * 0.30)
    v = X[o[-ne:]].mean(axis=0) - X[o[:ne]].mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def main() -> None:
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    bmd = {f"bmd_vid_idx{e}": float(a["memorability_score"])
           for e, a in ann.items() if "memorability_score" in a}

    feats, mems = [], []
    for f in sorted(Path("data/features/tribe").glob("bmd_vid_idx*.npz")):
        sid = f.stem; vid = sid.split("_seg_")[0]
        if vid not in bmd: continue
        arr = np.asarray(np.load(f, allow_pickle=False)["frames"], dtype=np.float32)
        if arr.ndim != 2: continue
        feats.append(arr.mean(axis=0)); mems.append(bmd[vid])
    X = np.stack(feats); y = np.asarray(mems, dtype=np.float32)
    N = len(y)

    n_runs = 50
    frac = 0.80
    rng = np.random.default_rng(0)
    vs = []
    for r in range(n_runs):
        idx = rng.choice(N, size=int(frac * N), replace=False)
        v = fit_v(X[idx], y[idx])
        vs.append(v)
    V = np.stack(vs)  # (n_runs, D)

    cos = V @ V.T
    upper = []
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            upper.append(cos[i, j])
    upper = np.asarray(upper, dtype=np.float32)

    print(f"[stab] {n_runs} runs × {int(frac*N)}/N clips")
    print(f"[stab] pairwise cosine: mean = {upper.mean():.4f}, "
          f"min = {upper.min():.4f}, max = {upper.max():.4f}, std = {upper.std():.4f}")

    # Also: train on disjoint halves and measure cosine. Maximally adversarial.
    disjoint_cos = []
    rng_d = np.random.default_rng(20260515)
    for r in range(20):
        perm = rng_d.permutation(N)
        h1, h2 = perm[:N//2], perm[N//2:]
        v1 = fit_v(X[h1], y[h1])
        v2 = fit_v(X[h2], y[h2])
        disjoint_cos.append(float(v1 @ v2))
    disjoint_cos = np.asarray(disjoint_cos, dtype=np.float32)
    print(f"[stab] disjoint-halves cosine (n=20 pairs): "
          f"mean = {disjoint_cos.mean():.4f}, "
          f"min = {disjoint_cos.min():.4f}, max = {disjoint_cos.max():.4f}")

    out = {
        "n_clips": int(N),
        "n_runs_subsample": n_runs,
        "subsample_fraction": frac,
        "pairwise_cos_mean": float(upper.mean()),
        "pairwise_cos_std": float(upper.std()),
        "pairwise_cos_min": float(upper.min()),
        "pairwise_cos_max": float(upper.max()),
        "disjoint_halves_cos_mean": float(disjoint_cos.mean()),
        "disjoint_halves_cos_min": float(disjoint_cos.min()),
        "disjoint_halves_cos_max": float(disjoint_cos.max()),
        "n_disjoint_pairs": int(len(disjoint_cos)),
    }
    Path("data/reports/vmem_stability.json").write_text(json.dumps(out, indent=2))
    print("[stab] done — wrote data/reports/vmem_stability.json")


if __name__ == "__main__":
    main()
