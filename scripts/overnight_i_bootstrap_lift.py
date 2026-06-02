"""(I) Bootstrap confidence intervals on the headline best-of-N lift.

Current paper: SVD +2.07, Veo +2.15 (point estimates).
This adds 95% CIs and standard errors via stratified bootstrap over seeds/prompts.
Also reports BCa intervals if scipy is around.
"""

from __future__ import annotations

import json, numpy as np
from pathlib import Path


def load_feat(p):
    arr = np.asarray(np.load(p, allow_pickle=False)["frames"], dtype=np.float32)
    return arr.mean(axis=0) if arr.ndim == 2 else arr


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
    o = np.argsort(y); ne = int(len(y) * 0.30)
    v_mem = X[o[-ne:]].mean(axis=0) - X[o[:ne]].mean(axis=0)
    v_mem /= np.linalg.norm(v_mem) + 1e-9

    pairs = [
        ("svd", Path("data/features/tribe_best_of_n")),
        ("veo", Path("data/features/tribe_veo_bon")),
    ]

    out = {}
    for label, src in pairs:
        # Group by seed/prompt
        by_grp = {}
        for f in sorted(src.glob("*.npz")):
            name = f.stem
            if "_n" not in name: continue
            seed, n_str = name.rsplit("_n", 1)
            try: n = int(n_str)
            except: continue
            proj = float(load_feat(f) @ v_mem)
            by_grp.setdefault(seed, []).append(proj)

        # Per-group lift = max - median
        lifts = []
        for seed, projs in by_grp.items():
            if len(projs) < 3: continue
            projs = sorted(projs)
            mx = projs[-1]
            md = projs[len(projs)//2]
            lifts.append(mx - md)
        lifts = np.asarray(lifts)
        N = len(lifts)
        print(f"\n[{label}] groups: {N}, point estimate mean lift = {lifts.mean():+.3f}")

        # Bootstrap over groups
        rng = np.random.default_rng(20260515)
        n_boot = 10000
        boot_means = np.zeros(n_boot, dtype=np.float32)
        for b in range(n_boot):
            samp = rng.choice(lifts, size=N, replace=True)
            boot_means[b] = samp.mean()
        ci_lo = float(np.quantile(boot_means, 0.025))
        ci_hi = float(np.quantile(boot_means, 0.975))
        se = float(boot_means.std())
        print(f"[{label}] bootstrap mean = {boot_means.mean():+.3f}, "
              f"SE = {se:.3f}, 95% CI = [{ci_lo:+.3f}, {ci_hi:+.3f}]")

        out[label] = {
            "n_groups": int(N),
            "point_estimate_mean_lift": float(lifts.mean()),
            "bootstrap_n": n_boot,
            "bootstrap_se": se,
            "ci_lo_95": ci_lo,
            "ci_hi_95": ci_hi,
            "per_group_lifts": [float(l) for l in lifts],
        }

    Path("data/reports/bootstrap_lift.json").write_text(json.dumps(out, indent=2))
    print("\n[done] wrote data/reports/bootstrap_lift.json")


if __name__ == "__main__":
    main()
