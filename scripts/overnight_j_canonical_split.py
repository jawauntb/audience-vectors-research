"""(J) Canonical BMD train/test split (1000/102).

The BMD authors' intended evaluation: train on the canonical training set,
report on the canonical test set. Currently the paper reports 5-fold CV
across all 1022 clips with features. This adds the canonical split number
so the result is directly comparable to BMD literature.
"""

from __future__ import annotations

import json, numpy as np
from pathlib import Path


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                             np.argsort(np.argsort(b)))[0, 1])


def main() -> None:
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    bmd_score = {f"bmd_vid_idx{e}": float(a["memorability_score"])
                 for e, a in ann.items() if "memorability_score" in a}
    bmd_set   = {f"bmd_vid_idx{e}": a.get("set", "unknown") for e, a in ann.items()}

    feats, mems, sets, sids = [], [], [], []
    for f in sorted(Path("data/features/tribe").glob("bmd_vid_idx*.npz")):
        sid = f.stem; vid = sid.split("_seg_")[0]
        if vid not in bmd_score: continue
        arr = np.asarray(np.load(f, allow_pickle=False)["frames"], dtype=np.float32)
        if arr.ndim != 2: continue
        feats.append(arr.mean(axis=0)); mems.append(bmd_score[vid])
        sets.append(bmd_set[vid]); sids.append(sid)
    X = np.stack(feats); y = np.asarray(mems, dtype=np.float32)
    sets = np.asarray(sets)

    train_mask = sets == "train"
    test_mask = sets == "test"
    print(f"[canon] train: {train_mask.sum()}, test: {test_mask.sum()}, total with feat: {len(y)}")

    # Train v_mem on canonical train, test on canonical test
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    o = np.argsort(ytr); ne = int(len(ytr) * 0.30)
    v_mem = Xtr[o[-ne:]].mean(axis=0) - Xtr[o[:ne]].mean(axis=0)
    v_mem /= np.linalg.norm(v_mem) + 1e-9
    rho_test = spearman(Xte @ v_mem, yte)
    print(f"[canon] canonical test Spearman ρ = {rho_test:+.4f} (n={test_mask.sum()})")

    # Also Pearson for completeness
    pred = Xte @ v_mem
    pearson_r = float(np.corrcoef(pred, yte)[0, 1])
    print(f"[canon] canonical test Pearson r  = {pearson_r:+.4f}")

    # Bootstrap CI on canonical test ρ
    rng = np.random.default_rng(20260515)
    n_boot = 10000
    Nte = test_mask.sum()
    boots = np.zeros(n_boot, dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, Nte, size=Nte)
        boots[b] = spearman((Xte @ v_mem)[idx], yte[idx])
    ci_lo, ci_hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    print(f"[canon] 95% bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # Also: random-direction null on the canonical test split
    rng2 = np.random.default_rng(0)
    rand_rhos = []
    for _ in range(1000):
        v_rand = rng2.standard_normal(X.shape[1]).astype(np.float32)
        v_rand /= np.linalg.norm(v_rand) + 1e-9
        rand_rhos.append(spearman(Xte @ v_rand, yte))
    rand_rhos = np.asarray(rand_rhos)
    z_random = (rho_test - rand_rhos.mean()) / (rand_rhos.std() + 1e-9)
    print(f"[canon] random-direction null on canonical test: μ = {rand_rhos.mean():+.4f}, "
          f"σ = {rand_rhos.std():.4f}")
    print(f"[canon] z vs random null: {z_random:.2f}")

    out = {
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "canonical_test_spearman": float(rho_test),
        "canonical_test_pearson": pearson_r,
        "ci_95_low": ci_lo,
        "ci_95_high": ci_hi,
        "random_direction_null_mean": float(rand_rhos.mean()),
        "random_direction_null_std": float(rand_rhos.std()),
        "z_vs_random": float(z_random),
    }
    Path("data/reports/canonical_split.json").write_text(json.dumps(out, indent=2))
    print("[canon] done — wrote data/reports/canonical_split.json")


if __name__ == "__main__":
    main()
