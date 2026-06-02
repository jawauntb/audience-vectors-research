"""ICA decomposition of TRIBE features — SAE substitute for n=1022.

Run FastICA on the 1022 × 20484 matrix to extract K independent components.
For each component:
  - compute its "loading" per clip (the ICA source)
  - rank components by Spearman correlation with BMD memorability
  - save top-K components for downstream contrastive direction analysis

The idea: instead of one global memorability direction, get many interpretable
brain-aligned features and find which ones drive the memorability signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import FastICA


def _bmd() -> dict[str, float]:
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    return {f"bmd_vid_idx{e}": float(a["memorability_score"])
            for e, a in ann.items() if "memorability_score" in a}


def _load_tribe(path: Path) -> np.ndarray:
    p = np.load(path, allow_pickle=False)
    arr = np.asarray(p["frames"], dtype=np.float32)
    return arr.mean(axis=0) if arr.ndim == 2 else arr


def main() -> None:
    print("[ica] loading TRIBE features")
    feat_dir = Path("data/features/tribe")
    bmd = _bmd()
    feats, scores, ids = [], [], []
    for f in sorted(feat_dir.glob("bmd_vid_idx*.npz")):
        sid = f.stem
        vid = sid.split("_seg_")[0]
        if vid not in bmd:
            continue
        feats.append(_load_tribe(f))
        scores.append(bmd[vid])
        ids.append(sid)
    X = np.stack(feats).astype(np.float32)
    mems = np.asarray(scores, dtype=np.float32)
    print(f"[ica] X={X.shape}, mems={mems.shape}")

    K = 100
    print(f"[ica] fitting FastICA with {K} components")
    ica = FastICA(n_components=K, random_state=42, max_iter=600, tol=1e-3,
                  whiten="unit-variance")
    sources = ica.fit_transform(X)  # (n_samples, K) — per-clip activation of each component
    print(f"[ica] sources={sources.shape}")
    components = ica.components_  # (K, n_features) — spatial weights of each component
    mixing = ica.mixing_  # (n_features, K)
    print(f"[ica] components={components.shape}")

    # For each ICA component: Spearman correlation between its per-clip source value and BMD memorability
    print("\n[ica] ranking components by predictive power on BMD memorability")
    rho_per_comp = []
    for k in range(K):
        src = sources[:, k]
        rho = float(np.corrcoef(np.argsort(np.argsort(src)),
                                np.argsort(np.argsort(mems)))[0, 1])
        rho_per_comp.append((k, rho))
    rho_per_comp.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'rank':>4} {'component':>10} {'Spearman ρ vs BMD':>22} {'|ρ|':>8}")
    print("-" * 55)
    for i, (k, rho) in enumerate(rho_per_comp[:20]):
        print(f"{i+1:>4} {k:>10} {rho:>+22.3f} {abs(rho):>8.3f}")

    # Save
    np.savez_compressed(
        "data/features/ica_tribe.npz",
        sources=sources,
        components=components,
        mems=mems,
        sample_ids=np.asarray(ids),
        ranked_components=np.asarray([k for k, _ in rho_per_comp]),
        ranked_rhos=np.asarray([r for _, r in rho_per_comp]),
    )

    payload = {
        "n_clips": int(len(ids)),
        "n_components": K,
        "ranked": [{"rank": i + 1, "component": int(k), "rho": float(r)}
                   for i, (k, r) in enumerate(rho_per_comp[:20])],
        "best_rho": float(rho_per_comp[0][1]),
        "best_component": int(rho_per_comp[0][0]),
    }
    Path("data/reports/ica_tribe.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[done] best single ICA component: #{rho_per_comp[0][0]}  ρ={rho_per_comp[0][1]:+.3f}")
    print(f"        (compare: global contrastive v_mem at +0.40)")


if __name__ == "__main__":
    main()
