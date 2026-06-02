"""(C) Multi-direction memorability extraction — how dominant is axis 1?

Procedure
---------
1. Train v_mem_1 on train-fold X via contrastive top-30% vs bottom-30%.
2. Project to residual: X_1 = X - <X, v_mem_1> v_mem_1.
3. Train v_mem_2 on residual the same way. Force orthogonality.
4. Iterate up to K=10 directions.
5. For each direction k: 5-fold CV ρ between projection and labels.
6. Report the spectrum — do directions 2,3,... carry residual signal,
   or does signal die after k=1?

Addresses a gap left by the §5.3 ablation: that result shows the first
contrastive direction is load-bearing for the same procedure. This explicit
multi-direction extraction asks whether additional residual axes carry
meaningful held-out predictive power.
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


def fit_orthogonal_direction(X_res, y, directions):
    v = fit_v(X_res, y)
    for v_prev in directions:
        v = v - (v @ v_prev) * v_prev
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    return v / n


def cv_multi_direction(X, y, K=10, kfold=5, seed=0):
    pred_per_direction = np.zeros((K, len(y)), dtype=np.float32)
    cumulative_pred = np.zeros((K, len(y)), dtype=np.float32)
    direction_counts = np.zeros(K, dtype=np.int32)

    for tr, te in kfold_indices(len(y), k=kfold, seed=seed):
        X_tr, X_te = X[tr], X[te]
        y_tr = y[tr]
        X_tr_res = X_tr.copy()
        X_te_res = X_te.copy()
        directions = []
        train_proj_cols = []
        test_proj_cols = []

        for ki in range(K):
            v = fit_orthogonal_direction(X_tr_res, y_tr, directions)
            if v is None:
                break
            directions.append(v)
            direction_counts[ki] += 1

            pred_per_direction[ki, te] = X_te_res @ v
            train_proj_cols.append(X_tr @ v)
            test_proj_cols.append(X_te @ v)

            A_tr = np.stack(train_proj_cols, axis=1)
            A_te = np.stack(test_proj_cols, axis=1)
            w = np.linalg.lstsq(A_tr, y_tr, rcond=None)[0]
            cumulative_pred[ki, te] = A_te @ w

            X_tr_res = ablate(X_tr_res, v)
            X_te_res = ablate(X_te_res, v)

    rho_per_direction = []
    cumulative_rho = []
    for ki in range(K):
        if direction_counts[ki] == kfold:
            rho_per_direction.append(spearman(pred_per_direction[ki], y))
            cumulative_rho.append(spearman(cumulative_pred[ki], y))
        else:
            break
    return rho_per_direction, cumulative_rho


def fit_full_directions_for_diagnostics(X, y, K):
    directions = []
    X_res = X.copy()
    for _ in range(K):
        v = fit_orthogonal_direction(X_res, y, directions)
        if v is None:
            break
        directions.append(v)
        X_res = ablate(X_res, v)
    return directions


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
    print(f"[multi] N={len(y)} clips, D={X.shape[1]}")

    K = 10
    rhos, cumulative_rhos = cv_multi_direction(X, y, K=K)
    for k, (rho_k, cum_rho) in enumerate(zip(rhos, cumulative_rhos, strict=False)):
        print(
            f"  k={k+1}: fold-safe residual-axis ρ = {rho_k:+.4f}, "
            f"cumulative ρ_top{k+1} = {cum_rho:+.4f}",
            flush=True,
        )

    directions = fit_full_directions_for_diagnostics(X, y, K=len(rhos))

    # Cross-direction cosines (should be near 0 by construction)
    if len(directions) > 1:
        D = np.stack(directions)
        cos_mat = D @ D.T
        np.fill_diagonal(cos_mat, 0.0)
        max_off = float(np.abs(cos_mat).max())
        print(
            f"\n[multi] max off-diagonal cosine across {len(directions)} directions: {max_off:.6f}"
        )

    out = {
        "n_clips": int(len(y)),
        "K": K,
        "n_directions_found": len(directions),
        "rho_per_direction": [float(r) for r in rhos],
        "cumulative_rho": [float(r) for r in cumulative_rhos],
        "fold_safe": True,
        "max_off_diagonal_cosine": max_off if len(directions) > 1 else 0.0,
        "interpretation": (
            "rho_per_direction[k] is the 5-fold CV Spearman ρ between BMD-mem and "
            "the projection on the k-th orthogonal contrastive memorability direction "
            "in the residual space after ablating directions 1..k-1. "
            "cumulative_rho[k] is the CV ρ obtained by linearly combining the first "
            "k+1 directions. If rho_per_direction[k] decays sharply, memorability is "
            "compact (low intrinsic dimension)."
        ),
    }
    Path("data/reports/multi_direction.json").write_text(json.dumps(out, indent=2))
    print("\n[multi] done — wrote data/reports/multi_direction.json")


if __name__ == "__main__":
    main()
