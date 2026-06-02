"""(F) Per-time-bin v_mem direction analysis.

TRIBE outputs (T, 20484) — we've been mean-pooling over T. This asks:
  - Train v_mem separately on EACH time bin
  - Are early-bin v_mems vs late-bin v_mems pointing in the same direction?
  - Is there a temporal evolution of "what is memorable"?

If the per-bin v_mems are highly correlated (cos > 0.8): memorability is a
stable feature across the clip. If they diverge (cos < 0.5): different
moments contribute different aspects. Complements §6.10 (time-series profile
of a single fixed v_mem) by giving the dual analysis: time-varying v_mem.
"""

from __future__ import annotations

import json, numpy as np
from pathlib import Path


def fit_v(X, y):
    o = np.argsort(y); ne = int(len(y) * 0.30)
    v = X[o[-ne:]].mean(axis=0) - X[o[:ne]].mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                             np.argsort(np.argsort(b)))[0, 1])


def main() -> None:
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    bmd = {f"bmd_vid_idx{e}": float(a["memorability_score"])
           for e, a in ann.items() if "memorability_score" in a}

    raw, mems = {}, {}
    for f in sorted(Path("data/features/tribe").glob("bmd_vid_idx*.npz")):
        sid = f.stem; vid = sid.split("_seg_")[0]
        if vid not in bmd: continue
        arr = np.asarray(np.load(f, allow_pickle=False)["frames"], dtype=np.float32)
        if arr.ndim != 2: continue
        raw[sid] = arr; mems[sid] = bmd[vid]

    # Distribution of T
    Tc = {}
    for s, a in raw.items(): Tc[a.shape[0]] = Tc.get(a.shape[0], 0) + 1
    print(f"[temp] T distribution: {sorted(Tc.items())}")
    T_main = max(Tc, key=Tc.get)
    sids = [s for s in raw if raw[s].shape[0] == T_main]
    print(f"[temp] using T={T_main}, n_clips={len(sids)}")

    # Per-bin v_mem
    per_bin = {}
    y = np.asarray([mems[s] for s in sids], dtype=np.float32)
    for t in range(T_main):
        Xt = np.stack([raw[s][t] for s in sids])
        vt = fit_v(Xt, y)
        per_bin[t] = vt

    # Pairwise cosines across time bins
    cos_mat = np.zeros((T_main, T_main), dtype=np.float32)
    for i in range(T_main):
        for j in range(T_main):
            cos_mat[i, j] = float(per_bin[i] @ per_bin[j])
    print(f"\n[temp] cross-bin cosine matrix:")
    for i in range(T_main):
        print("  " + " ".join(f"{cos_mat[i,j]:+.3f}" for j in range(T_main)))

    # Compare to mean-pooled v_mem
    X_mean = np.stack([raw[s].mean(axis=0) for s in sids])
    v_mean = fit_v(X_mean, y)
    bin_to_mean = {t: float(per_bin[t] @ v_mean) for t in range(T_main)}
    print(f"\n[temp] per-bin v cosine to mean-pooled v_mem: {bin_to_mean}")

    # Per-bin predictive CV ρ
    per_bin_rho = {}
    for t in range(T_main):
        Xt = np.stack([raw[s][t] for s in sids])
        # 5-fold CV
        idx = np.arange(len(sids))
        rng = np.random.default_rng(0); rng.shuffle(idx)
        folds = np.array_split(idx, 5)
        proj = np.zeros(len(sids), dtype=np.float32)
        for fi in range(5):
            te = folds[fi]; tr = np.concatenate([folds[j] for j in range(5) if j != fi])
            vt = fit_v(Xt[tr], y[tr])
            proj[te] = Xt[te] @ vt
        per_bin_rho[t] = spearman(proj, y)
    print(f"\n[temp] per-bin CV ρ:")
    for t, r in per_bin_rho.items():
        print(f"  t={t}: ρ = {r:+.4f}")

    out = {
        "T": int(T_main),
        "n_clips_aligned": len(sids),
        "cross_bin_cosine_matrix": cos_mat.tolist(),
        "mean_off_diagonal_cosine": float(np.mean([cos_mat[i,j]
                                                    for i in range(T_main)
                                                    for j in range(T_main) if i!=j])),
        "bin_to_mean_pooled_cosine": bin_to_mean,
        "per_bin_cv_rho": per_bin_rho,
        "mean_pooled_cv_rho_reference": 0.40,
    }
    Path("data/reports/temporal_v_mem.json").write_text(json.dumps(out, indent=2))
    print("[temp] done — wrote data/reports/temporal_v_mem.json")


if __name__ == "__main__":
    main()
