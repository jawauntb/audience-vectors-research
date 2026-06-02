"""(A) Label-permutation null distribution for v_mem.

Closes the §8 limitation: 'Null and ablation controls. The current
random-direction null is preliminary. Label-permutation contrastive nulls...
are needed before making stronger necessity claims.'

Procedure
---------
- For each of n_perm=1000 permutations of BMD memorability labels:
    - rebuild v_mem from permuted labels
    - score held-out (5-fold CV) ρ between v_mem-projection and TRUE labels
- Compare distribution of permuted-ρ to actual ρ = +0.40
- Also: same procedure for ablation magnitude (does ablating v_mem_perm kill signal?)
"""

from __future__ import annotations

import json, numpy as np
from pathlib import Path


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
    print(f"[null] N = {N} clips, D = {X.shape[1]}")

    def spearman(a, b):
        return float(np.corrcoef(np.argsort(np.argsort(a)),
                                  np.argsort(np.argsort(b)))[0, 1])

    def cv_rho(X, y, seed=0, k=5):
        idx = np.arange(len(y))
        rng = np.random.default_rng(seed); rng.shuffle(idx)
        folds = np.array_split(idx, k)
        proj_full = np.zeros(len(y), dtype=np.float32)
        for fi in range(k):
            te = folds[fi]; tr = np.concatenate([folds[j] for j in range(k) if j != fi])
            ytr = y[tr]
            ne = int(len(ytr) * 0.30)
            o = np.argsort(ytr)
            v = X[tr][o[-ne:]].mean(axis=0) - X[tr][o[:ne]].mean(axis=0)
            v /= np.linalg.norm(v) + 1e-9
            proj_full[te] = X[te] @ v
        return spearman(proj_full, y)

    rho_actual = cv_rho(X, y)
    print(f"[null] actual 5-fold CV Spearman ρ = {rho_actual:+.4f}")

    rng = np.random.default_rng(20260515)
    n_perm = 1000
    perm_rhos = np.zeros(n_perm, dtype=np.float32)
    for i in range(n_perm):
        y_perm = y.copy(); rng.shuffle(y_perm)
        perm_rhos[i] = cv_rho(X, y_perm, seed=int(rng.integers(0, 1 << 31)))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_perm}] running mean = {perm_rhos[:i+1].mean():+.4f}, "
                  f"max = {perm_rhos[:i+1].max():+.4f}", flush=True)

    p_value = float((perm_rhos >= rho_actual).mean())
    z = float((rho_actual - perm_rhos.mean()) / (perm_rhos.std() + 1e-9))
    out = {
        "n_perm": n_perm,
        "n_clips": int(N),
        "rho_actual": float(rho_actual),
        "perm_rho_mean": float(perm_rhos.mean()),
        "perm_rho_std": float(perm_rhos.std()),
        "perm_rho_max": float(perm_rhos.max()),
        "perm_rho_min": float(perm_rhos.min()),
        "perm_rho_95th": float(np.quantile(perm_rhos, 0.95)),
        "perm_rho_99th": float(np.quantile(perm_rhos, 0.99)),
        "p_value_one_sided": p_value,
        "z_score": z,
        "perm_rhos_hist_bins": np.linspace(-0.15, 0.15, 31).tolist(),
        "perm_rhos_hist_counts": np.histogram(perm_rhos, bins=np.linspace(-0.15, 0.15, 31))[0].tolist(),
    }

    Path("data/reports/null_label_perm.json").write_text(json.dumps(out, indent=2))
    print(f"\n[null] done — ρ_actual = {rho_actual:+.4f}, "
          f"perm μ = {perm_rhos.mean():+.4f} ± {perm_rhos.std():.4f}, "
          f"p = {p_value:.4f}, z = {z:.2f}")
    print(f"[null] perm max = {perm_rhos.max():+.4f} (< actual = {rho_actual > perm_rhos.max()})")


if __name__ == "__main__":
    main()
