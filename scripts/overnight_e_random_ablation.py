"""(E) Random-direction ablation control — does ablating a random direction
do anything? Or is v_mem specifically the load-bearing one?

Procedure
---------
1. Compute v_mem CV ρ (= +0.40 baseline)
2. For n_runs=200:
    - sample a random unit vector v_random in R^D
    - ablate v_random from X
    - retrain v_mem on residual, compute CV ρ
3. Compare distribution to actual v_mem ablation drop (should be near 0)

This is the dual of (A) the label-permutation null. (A) asks: would
contrastive on permuted labels give comparable ρ?  This (E) asks:
is the ablation killing all signal because v_mem is special, or would
ANY direction kill signal?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def spearman(a, b):
    return float(
        np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
    )


def cv_proj_rho(X, y, k=5, seed=0):
    idx = np.arange(len(y))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    proj = np.zeros(len(y), dtype=np.float32)
    for fi in range(k):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(k) if j != fi])
        ytr = y[tr]
        ne = int(len(ytr) * 0.30)
        o = np.argsort(ytr)
        v = X[tr][o[-ne:]].mean(axis=0) - X[tr][o[:ne]].mean(axis=0)
        v /= np.linalg.norm(v) + 1e-9
        proj[te] = X[te] @ v
    return spearman(proj, y)


def fit_v(X, y):
    o = np.argsort(y)
    ne = int(len(y) * 0.30)
    v = X[o[-ne:]].mean(axis=0) - X[o[:ne]].mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def ablate(X, v):
    return X - (X @ v)[:, None] * v[None, :]


def kfold_indices(n, k=5, seed=0):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for fi in range(k):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(k) if j != fi])
        yield tr, te


def cv_rho_after_direction_ablation(X, y, direction_fn, k=5, seed=0):
    proj = np.zeros(len(y), dtype=np.float32)
    for tr, te in kfold_indices(len(y), k=k, seed=seed):
        X_tr, X_te = X[tr], X[te]
        y_tr = y[tr]
        v_ablate = direction_fn(X_tr, y_tr)
        X_tr_res = ablate(X_tr, v_ablate)
        X_te_res = ablate(X_te, v_ablate)
        v_res = fit_v(X_tr_res, y_tr)
        proj[te] = X_te_res @ v_res
    return spearman(proj, y)


def main() -> None:
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    bmd = {
        f"bmd_vid_idx{e}": float(a["memorability_score"])
        for e, a in ann.items()
        if "memorability_score" in a
    }

    feats, mems = [], []
    for f in sorted(Path("data/features/tribe").glob("bmd_vid_idx*.npz")):
        sid = f.stem
        vid = sid.split("_seg_")[0]
        if vid not in bmd:
            continue
        arr = np.asarray(np.load(f, allow_pickle=False)["frames"], dtype=np.float32)
        if arr.ndim != 2:
            continue
        feats.append(arr.mean(axis=0))
        mems.append(bmd[vid])
    X = np.stack(feats)
    y = np.asarray(mems, dtype=np.float32)
    N, D = X.shape
    print(f"[rab] N={N}, D={D}")

    # Baseline: pre-ablation ρ
    rho_pre = cv_proj_rho(X, y)
    print(f"[rab] pre-ablation ρ = {rho_pre:+.4f}")

    # v_mem ablation (actual), learned inside each CV fold to avoid label leakage.
    rho_vmem = cv_rho_after_direction_ablation(X, y, fit_v)
    print(f"[rab] v_mem-ablation ρ = {rho_vmem:+.4f} (Δ {rho_vmem - rho_pre:+.4f})")

    # Random direction ablations
    rng = np.random.default_rng(20260515)
    n_runs = 200
    rho_random = np.zeros(n_runs, dtype=np.float32)
    for r in range(n_runs):
        v_rand = rng.standard_normal(D).astype(np.float32)
        v_rand /= np.linalg.norm(v_rand) + 1e-9
        rho_random[r] = cv_rho_after_direction_ablation(
            X,
            y,
            lambda _X, _y, v=v_rand: v,
            seed=int(rng.integers(0, 1 << 31)),
        )
        if (r + 1) % 25 == 0:
            print(
                f"  [{r+1}/{n_runs}] mean random-ablation ρ = {rho_random[:r+1].mean():+.4f}, "
                f"min = {rho_random[:r+1].min():+.4f}",
                flush=True,
            )

    z_vs_random = (rho_vmem - rho_random.mean()) / (rho_random.std() + 1e-9)
    p_value = float((rho_random <= rho_vmem).mean())
    print(f"\n[rab] v_mem ablation killed ρ to {rho_vmem:+.4f}")
    print(
        f"      random ablations: mean = {rho_random.mean():+.4f} ± {rho_random.std():.4f}"
    )
    print(f"      z = {z_vs_random:.2f}, one-sided p (random ≤ v_mem) = {p_value:.4f}")

    out = {
        "n_clips": int(N),
        "feature_dim": int(D),
        "rho_pre_ablation": float(rho_pre),
        "rho_vmem_ablation": float(rho_vmem),
        "fold_safe_vmem_ablation": True,
        "n_random": n_runs,
        "rho_random_ablation_mean": float(rho_random.mean()),
        "rho_random_ablation_std": float(rho_random.std()),
        "rho_random_ablation_min": float(rho_random.min()),
        "rho_random_ablation_max": float(rho_random.max()),
        "z_vs_random": float(z_vs_random),
        "p_one_sided_random_le_vmem": p_value,
    }
    Path("data/reports/random_ablation_null.json").write_text(json.dumps(out, indent=2))
    print("[rab] done — wrote data/reports/random_ablation_null.json")


if __name__ == "__main__":
    main()
