"""(B) Nonlinear residual probes — does ANY non-linear model recover memorability
after directional ablation?

Closes §8 limitation: 'Linear readout limitation. The ablation result concerns
signal recoverable by the same contrastive linear procedure. Nonlinear probes,
ridge models, or alternate residual readouts may recover additional memorability
information.'

Compares ρ between predicted and true memorability for:
  1. Linear (contrastive direction) — baseline, our published number
  2. Ridge regression — closed-form linear with regularization
  3. Kernel-ridge — RBF, captures local nonlinearity
  4. MLP (2-layer) — expressive nonlinear probe
  5. Random forest — different inductive bias

Trained on FULL X and RESIDUAL X (post-ablation of v_mem) — if the residual
ρ is ~0 across ALL probe families, that's strong evidence of compactness.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def spearman(a, b):
    return float(
        np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
    )


def cv_predict_linear(X, y, k=5, seed=0):
    """Contrastive linear baseline."""
    idx = np.arange(len(y))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    pred = np.zeros(len(y), dtype=np.float32)
    for fi in range(k):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(k) if j != fi])
        ytr = y[tr]
        ne = int(len(ytr) * 0.30)
        o = np.argsort(ytr)
        v = X[tr][o[-ne:]].mean(axis=0) - X[tr][o[:ne]].mean(axis=0)
        v /= np.linalg.norm(v) + 1e-9
        pred[te] = X[te] @ v
    return pred


def cv_predict_model(X, y, model_fn, k=5, seed=0):
    idx = np.arange(len(y))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    pred = np.zeros(len(y), dtype=np.float32)
    for fi in range(k):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(k) if j != fi])
        m = model_fn()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def train_direction(X, y):
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


def pca_model(estimator):
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=256, random_state=0),
        estimator,
    )


def cv_predict_fold_safe(X, y, model_fn, residual=False, k=5, seed=0):
    """Cross-validated predictions with all preprocessing fit inside each fold.

    When residual=True, v_mem is learned on that fold's training rows only, then
    projected out of both train and test rows before fitting the probe.
    """
    pred = np.zeros(len(y), dtype=np.float32)
    for tr, te in kfold_indices(len(y), k=k, seed=seed):
        X_tr, X_te = X[tr], X[te]
        y_tr = y[tr]
        if residual:
            v_mem = train_direction(X_tr, y_tr)
            X_tr = ablate(X_tr, v_mem)
            X_te = ablate(X_te, v_mem)

        model = model_fn()
        model.fit(X_tr, y_tr)
        pred[te] = model.predict(X_te)
    return pred


def cv_predict_contrastive_fold_safe(X, y, residual=False, k=5, seed=0):
    pred = np.zeros(len(y), dtype=np.float32)
    for tr, te in kfold_indices(len(y), k=k, seed=seed):
        X_tr, X_te = X[tr], X[te]
        y_tr = y[tr]
        if residual:
            v_mem = train_direction(X_tr, y_tr)
            X_tr = ablate(X_tr, v_mem)
            X_te = ablate(X_te, v_mem)
        v = train_direction(X_tr, y_tr)
        pred[te] = X_te @ v
    return pred


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
    print(f"[nlin] N={len(y)} clips, D={X.shape[1]}")

    print("[nlin] fold-safe residual mode: v_mem, scaler, and PCA fit within each fold")

    results = {"full": {}, "residual": {}}

    # === FULL space ===
    print("\n=== Probing FULL space (v_mem signal intact) ===")
    # Linear contrastive
    pred_lin = cv_predict_contrastive_fold_safe(X, y)
    rho_lin = spearman(pred_lin, y)
    print(f"  linear contrastive ρ = {rho_lin:+.4f}")
    results["full"]["linear"] = rho_lin

    # Ridge on PCA
    pred_ridge = cv_predict_fold_safe(
        X,
        y,
        lambda: pca_model(Ridge(alpha=10.0)),
    )
    rho_ridge = spearman(pred_ridge, y)
    print(f"  ridge (PCA-256) ρ = {rho_ridge:+.4f}")
    results["full"]["ridge"] = rho_ridge

    # Kernel ridge RBF
    pred_krr = cv_predict_fold_safe(
        X,
        y,
        lambda: pca_model(KernelRidge(alpha=1, kernel="rbf", gamma=0.01)),
    )
    rho_krr = spearman(pred_krr, y)
    print(f"  kernel-ridge (RBF, PCA-256) ρ = {rho_krr:+.4f}")
    results["full"]["kernel_ridge"] = rho_krr

    # MLP
    pred_mlp = cv_predict_fold_safe(
        X,
        y,
        lambda: pca_model(
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                max_iter=300,
                random_state=0,
                early_stopping=True,
                validation_fraction=0.1,
            )
        ),
    )
    rho_mlp = spearman(pred_mlp, y)
    print(f"  MLP (128,64, PCA-256) ρ = {rho_mlp:+.4f}")
    results["full"]["mlp"] = rho_mlp

    # Random forest
    pred_rf = cv_predict_fold_safe(
        X,
        y,
        lambda: pca_model(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                random_state=0,
                n_jobs=-1,
            )
        ),
    )
    rho_rf = spearman(pred_rf, y)
    print(f"  random forest (PCA-256) ρ = {rho_rf:+.4f}")
    results["full"]["random_forest"] = rho_rf

    # === RESIDUAL space (v_mem ablated) ===
    print("\n=== Probing RESIDUAL space (v_mem ablated) ===")
    pred_lin_r = cv_predict_contrastive_fold_safe(X, y, residual=True)
    rho_lin_r = spearman(pred_lin_r, y)
    print(f"  linear contrastive ρ = {rho_lin_r:+.4f}")
    results["residual"]["linear"] = rho_lin_r

    pred_ridge_r = cv_predict_fold_safe(
        X,
        y,
        lambda: pca_model(Ridge(alpha=10.0)),
        residual=True,
    )
    rho_ridge_r = spearman(pred_ridge_r, y)
    print(f"  ridge (PCA-256) ρ = {rho_ridge_r:+.4f}")
    results["residual"]["ridge"] = rho_ridge_r

    pred_krr_r = cv_predict_fold_safe(
        X,
        y,
        lambda: pca_model(KernelRidge(alpha=1, kernel="rbf", gamma=0.01)),
        residual=True,
    )
    rho_krr_r = spearman(pred_krr_r, y)
    print(f"  kernel-ridge (RBF, PCA-256) ρ = {rho_krr_r:+.4f}")
    results["residual"]["kernel_ridge"] = rho_krr_r

    pred_mlp_r = cv_predict_fold_safe(
        X,
        y,
        lambda: pca_model(
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                max_iter=300,
                random_state=0,
                early_stopping=True,
                validation_fraction=0.1,
            )
        ),
        residual=True,
    )
    rho_mlp_r = spearman(pred_mlp_r, y)
    print(f"  MLP (128,64, PCA-256) ρ = {rho_mlp_r:+.4f}")
    results["residual"]["mlp"] = rho_mlp_r

    pred_rf_r = cv_predict_fold_safe(
        X,
        y,
        lambda: pca_model(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                random_state=0,
                n_jobs=-1,
            )
        ),
        residual=True,
    )
    rho_rf_r = spearman(pred_rf_r, y)
    print(f"  random forest (PCA-256) ρ = {rho_rf_r:+.4f}")
    results["residual"]["random_forest"] = rho_rf_r

    # Drops
    print("\n=== Drop in ρ after v_mem ablation ===")
    drops = {}
    for m in results["full"]:
        d = results["full"][m] - results["residual"][m]
        drops[m] = float(d)
        print(
            f"  {m:18s}: {results['full'][m]:+.4f} → {results['residual'][m]:+.4f}  (Δ {d:+.4f})"
        )

    out = {
        "n_clips": int(len(y)),
        "feature_dim_full": int(X.shape[1]),
        "feature_dim_pca": 256,
        "fold_safe": True,
        "pca_note": "StandardScaler and PCA-256 are fit within each CV train fold.",
        "rho_full": results["full"],
        "rho_residual": results["residual"],
        "rho_drop": drops,
    }
    Path("data/reports/nonlinear_probes.json").write_text(json.dumps(out, indent=2))
    print("\n[nlin] done — wrote data/reports/nonlinear_probes.json")


if __name__ == "__main__":
    main()
