"""(K) Cross-domain v_mem transfer WITHIN BMD.

We don't have a second video memorability dataset overnight. But BMD has
rich scene/action tags. We can simulate cross-dataset by:
  - Train v_mem on indoor scenes, test on outdoor scenes
  - Train on action-heavy clips, test on static clips
  - Train on a random 50% scene set, test on the other 50%

If v_mem transfers cleanly across these subdomains, it's evidence the
direction is not just a BMD-content-specific quirk.
"""

from __future__ import annotations

import json, numpy as np
from pathlib import Path


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                             np.argsort(np.argsort(b)))[0, 1])


def fit_v(X, y):
    o = np.argsort(y); ne = int(len(y) * 0.30)
    v = X[o[-ne:]].mean(axis=0) - X[o[:ne]].mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def main() -> None:
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    bmd_score = {f"bmd_vid_idx{e}": float(a["memorability_score"])
                 for e, a in ann.items() if "memorability_score" in a}
    bmd_meta = {f"bmd_vid_idx{e}": a for e, a in ann.items()}

    INDOOR_SCENES = {"living_room", "kitchen", "bedroom", "playroom", "office",
                     "gymnasium/indoor", "library", "classroom", "bathroom",
                     "dining_room", "hospital_room", "restaurant"}
    OUTDOOR_SCENES = {"park", "street", "yard", "lake/natural", "beach", "mountain",
                      "forest", "field", "garden", "playground", "highway",
                      "river", "ocean", "desert", "snowfield"}

    feats, mems, sids, vids = [], [], [], []
    for f in sorted(Path("data/features/tribe").glob("bmd_vid_idx*.npz")):
        sid = f.stem; vid = sid.split("_seg_")[0]
        if vid not in bmd_score: continue
        arr = np.asarray(np.load(f, allow_pickle=False)["frames"], dtype=np.float32)
        if arr.ndim != 2: continue
        feats.append(arr.mean(axis=0)); mems.append(bmd_score[vid])
        sids.append(sid); vids.append(vid)
    X = np.stack(feats); y = np.asarray(mems, dtype=np.float32)
    print(f"[xdom] N total = {len(y)}")

    # Build masks
    indoor_mask = []
    outdoor_mask = []
    for v in vids:
        scenes = set(bmd_meta[v].get("scenes", []))
        indoor_mask.append(bool(scenes & INDOOR_SCENES))
        outdoor_mask.append(bool(scenes & OUTDOOR_SCENES))
    indoor_mask = np.asarray(indoor_mask)
    outdoor_mask = np.asarray(outdoor_mask)
    print(f"[xdom] indoor: {indoor_mask.sum()}, outdoor: {outdoor_mask.sum()}, "
          f"overlap: {(indoor_mask & outdoor_mask).sum()}")

    # Make them disjoint for clean transfer
    indoor_only = indoor_mask & ~outdoor_mask
    outdoor_only = outdoor_mask & ~indoor_mask
    print(f"[xdom] indoor-only: {indoor_only.sum()}, outdoor-only: {outdoor_only.sum()}")

    results = {}

    # Indoor -> Outdoor
    v_indoor = fit_v(X[indoor_only], y[indoor_only])
    proj_outdoor = X[outdoor_only] @ v_indoor
    rho_io = spearman(proj_outdoor, y[outdoor_only])
    print(f"\n[xdom] train INDOOR -> test OUTDOOR: ρ = {rho_io:+.4f} "
          f"(n_train={indoor_only.sum()}, n_test={outdoor_only.sum()})")
    results["indoor_to_outdoor"] = {"rho": rho_io, "n_train": int(indoor_only.sum()),
                                     "n_test": int(outdoor_only.sum())}

    # Outdoor -> Indoor
    v_outdoor = fit_v(X[outdoor_only], y[outdoor_only])
    proj_indoor = X[indoor_only] @ v_outdoor
    rho_oi = spearman(proj_indoor, y[indoor_only])
    print(f"[xdom] train OUTDOOR -> test INDOOR: ρ = {rho_oi:+.4f}")
    results["outdoor_to_indoor"] = {"rho": rho_oi, "n_train": int(outdoor_only.sum()),
                                     "n_test": int(indoor_only.sum())}

    # Cosine between the two domain-specific v_mems
    cos_io = float(v_indoor @ v_outdoor)
    print(f"[xdom] cos(v_indoor, v_outdoor) = {cos_io:+.4f}")
    results["v_indoor_v_outdoor_cos"] = cos_io

    # Now action vs static (use action tag presence)
    has_action = []
    for v in vids:
        actions = bmd_meta[v].get("actions", [])
        has_action.append(len(actions) >= 1)
    has_action = np.asarray(has_action)
    no_action = ~has_action
    print(f"\n[xdom] with-action: {has_action.sum()}, without-action: {no_action.sum()}")

    if no_action.sum() > 30:
        v_action = fit_v(X[has_action], y[has_action])
        proj_static = X[no_action] @ v_action
        rho_as = spearman(proj_static, y[no_action])
        v_static = fit_v(X[no_action], y[no_action])
        cos_as = float(v_action @ v_static)
        print(f"[xdom] train ACTION -> test STATIC: ρ = {rho_as:+.4f}")
        print(f"[xdom] cos(v_action, v_static) = {cos_as:+.4f}")
        results["action_to_static"] = {"rho": rho_as,
                                        "n_train": int(has_action.sum()),
                                        "n_test": int(no_action.sum()),
                                        "v_cos": cos_as}

    # Random 50/50 split for reference
    rng = np.random.default_rng(0)
    n_runs = 20
    rhos_random = []
    cos_random = []
    for _ in range(n_runs):
        perm = rng.permutation(len(y))
        h1, h2 = perm[:len(y)//2], perm[len(y)//2:]
        v1 = fit_v(X[h1], y[h1])
        v2 = fit_v(X[h2], y[h2])
        rhos_random.append(spearman(X[h2] @ v1, y[h2]))
        cos_random.append(float(v1 @ v2))
    rhos_random = np.asarray(rhos_random); cos_random = np.asarray(cos_random)
    print(f"\n[xdom] random 50/50 split (n=20): ρ = {rhos_random.mean():+.4f} "
          f"± {rhos_random.std():.4f}, cos = {cos_random.mean():+.4f} ± {cos_random.std():.4f}")
    results["random_50_50"] = {"rho_mean": float(rhos_random.mean()),
                               "rho_std": float(rhos_random.std()),
                               "v_cos_mean": float(cos_random.mean()),
                               "v_cos_std": float(cos_random.std()),
                               "n_runs": n_runs}

    Path("data/reports/cross_domain.json").write_text(json.dumps(results, indent=2))
    print("\n[xdom] done — wrote data/reports/cross_domain.json")


if __name__ == "__main__":
    main()
