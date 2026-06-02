"""BMD fMRI joint analysis — sub-01 pilot.

What this does
--------------
Downloads sub-01's per-clip beta estimates from OpenNeuro (ds005165, BMD),
resamples from fsaverage (163842 verts/hemi) to fsaverage5 (10242 verts/hemi)
which matches the TRIBE-prediction space (20484 total), and computes:

  1. v_mem_measured = contrastive direction on MEASURED brain betas
  2. cos(v_mem_measured, v_mem_TRIBE) — does TRIBE's predicted direction
     point in the same direction as a real-brain direction?
  3. Does v_mem_measured predict BMD memorability ρ comparably to TRIBE?

The third test is the strongest validation: if measured-brain v_mem gets
ρ ≈ +0.40 too, the framework's compactness claim is grounded in real
neural data, not just in TRIBE's predictions.

Disk requirements
-----------------
~8 GB peak per subject (4 GB per hemi). Script can free per-hemi after
extracting v_mem_measured. Run with --keep-files to retain raw betas.

Per-subject runtime estimate
----------------------------
- download: ~5-10 min on a fast connection
- load + resample: ~1-2 min
- v_mem compute: <30s

Usage
-----
    .venv/bin/python scripts/bmd_fmri_pilot.py --subject 01
    .venv/bin/python scripts/bmd_fmri_pilot.py --subject 01 --keep-files
"""

from __future__ import annotations

import argparse
import json
import pickle
import urllib.request
from pathlib import Path

import numpy as np

S3_BASE = (
    "https://s3.amazonaws.com/openneuro.org/ds005165/"
    "derivatives/versionB/fsaverage/GLM"
)


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 1024 * 1024:
        print(
            f"  [cache] {out_path.name} already exists ({out_path.stat().st_size/1e6:.0f} MB)"
        )
        return
    print(f"  [get] {url}")
    print(f"        -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    sz = out_path.stat().st_size
    print(f"        downloaded {sz/1e6:.0f} MB")


def get_fsavg5_to_fsavg_indices() -> tuple[np.ndarray, np.ndarray]:
    """Return (idx_lh, idx_rh) — the indices in fsaverage (163842 verts/hemi)
    that correspond to fsaverage5 (10242 verts/hemi).

    The standard FreeSurfer ico-decimation convention is that fsaverage5 vertices
    are the *first* 10242 vertices in fsaverage when the mesh is laid out using
    the standard sphere.reg registration. We verify this against nilearn's
    fsaverage5 mesh.
    """
    from nilearn.datasets import fetch_surf_fsaverage
    from nilearn.surface import load_surf_mesh

    f5 = fetch_surf_fsaverage("fsaverage5")
    fa = fetch_surf_fsaverage("fsaverage")

    def coords(path):
        m = load_surf_mesh(path)
        return m.coordinates if hasattr(m, "coordinates") else m[0]

    coords_5l = coords(f5.sphere_left)
    coords_al = coords(fa.sphere_left)
    coords_5r = coords(f5.sphere_right)
    coords_ar = coords(fa.sphere_right)
    print(f"  fsaverage5 verts (L/R): {coords_5l.shape[0]}, {coords_5r.shape[0]}")
    print(f"  fsaverage  verts (L/R): {coords_al.shape[0]}, {coords_ar.shape[0]}")
    n5 = coords_5l.shape[0]
    # Verify first-N convention (FreeSurfer ico subdivision)
    ok_l = np.allclose(coords_5l, coords_al[:n5], atol=1e-4)
    ok_r = np.allclose(coords_5r, coords_ar[:n5], atol=1e-4)
    if ok_l and ok_r:
        print("  [ok] fsaverage5 = fsaverage[:10242] (standard ico convention)")
        return np.arange(n5), np.arange(n5)
    # Fallback: nearest-neighbor on sphere
    print("  [warn] vertex layout differs; using nearest-neighbor mapping")
    from scipy.spatial import cKDTree  # type: ignore[reportAttributeAccessIssue]

    tree_l = cKDTree(coords_al)
    tree_r = cKDTree(coords_ar)
    _, idx_l = tree_l.query(coords_5l)
    _, idx_r = tree_r.query(coords_5r)
    return idx_l, idx_r


def load_betas(pkl_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load BMD organized_betas pickle. Returns (betas, clip_ids).

    BMD format (from blahner/BOLDMomentsDataset starter code):
      organized_betas[clip_id] = ndarray shape (n_repetitions, n_vertices)
    We average across repetitions.

    OpenNeuro ds005165's prepared pickle can also be:
      (ndarray shape (n_clips, n_repetitions, n_vertices), clip_id_list)
    """
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict):
        clip_ids = sorted(d.keys())
        first = d[clip_ids[0]]
        if isinstance(first, np.ndarray):
            if first.ndim == 2:
                betas = np.stack([d[c].mean(axis=0) for c in clip_ids])
            else:
                betas = np.stack([d[c] for c in clip_ids])
        else:
            raise RuntimeError(f"unexpected dict value type: {type(first)}")
    elif isinstance(d, np.ndarray):
        betas = d
        clip_ids = [f"clip_{i:04d}" for i in range(d.shape[0])]
        print("  [warn] file is ndarray (no clip IDs), using sequential")
    elif (
        isinstance(d, tuple)
        and len(d) == 2
        and isinstance(d[0], np.ndarray)
        and isinstance(d[1], list)
    ):
        arr, clip_ids_raw = d
        clip_ids = [str(c) for c in clip_ids_raw]
        if arr.ndim == 3:
            betas = arr.mean(axis=1)
        elif arr.ndim == 2:
            betas = arr
        else:
            raise RuntimeError(f"unexpected tuple array shape: {arr.shape}")
    else:
        raise RuntimeError(f"unexpected pickle type: {type(d)}")
    return betas, clip_ids


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="01")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/bmd_fmri"))
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Don't delete raw 8 GB pkls after extracting v_mem_measured",
    )
    args = parser.parse_args()
    sub = args.subject
    sub_dir = args.out_dir / f"sub-{sub}"
    print(f"=== BMD fMRI pilot — subject {sub} ===")

    # === 1. Download both hemispheres ===
    print(f"\n[1] Downloading sub-{sub} training betas (~8 GB)...")
    pkl_l = sub_dir / f"sub-{sub}_organized_betas_task-train_hemi-left_normalized.pkl"
    pkl_r = sub_dir / f"sub-{sub}_organized_betas_task-train_hemi-right_normalized.pkl"
    download(f"{S3_BASE}/sub-{sub}/prepared_betas/{pkl_l.name}", pkl_l)
    download(f"{S3_BASE}/sub-{sub}/prepared_betas/{pkl_r.name}", pkl_r)

    # === 2. Get the fsaverage → fsaverage5 index mapping ===
    print("\n[2] Computing fsaverage → fsaverage5 vertex mapping...")
    idx_l, idx_r = get_fsavg5_to_fsavg_indices()

    # === 3. Load betas + downsample ===
    print("\n[3] Loading + downsampling to fsaverage5...")
    betas_l, clip_ids = load_betas(pkl_l)
    print(
        f"  left:  shape {betas_l.shape}, dtype {betas_l.dtype}, "
        f"n_clips={len(clip_ids)}, n_verts={betas_l.shape[-1]}"
    )
    betas_l = betas_l[:, idx_l]
    print(f"  left downsampled to {betas_l.shape}")
    betas_r, clip_ids_r = load_betas(pkl_r)
    print(f"  right: shape {betas_r.shape}, dtype {betas_r.dtype}")
    betas_r = betas_r[:, idx_r]
    assert (
        clip_ids == clip_ids_r
    ), f"clip ID mismatch L vs R: {clip_ids[:3]} vs {clip_ids_r[:3]}"

    # Concat L+R = 20484 verts (TRIBE space)
    betas = np.concatenate([betas_l, betas_r], axis=1).astype(np.float32)
    print(f"  combined shape: {betas.shape} (expected (n_clips, 20484))")

    # Free
    del betas_l, betas_r

    # === 4. Map clip IDs to BMD memorability scores ===
    print("\n[4] Aligning with BMD memorability + TRIBE features...")
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    bmd_score = {
        f"bmd_vid_idx{e}": float(a["memorability_score"])
        for e, a in ann.items()
        if "memorability_score" in a
    }

    # BMD clip IDs in betas might be like "vid_idx0001" or numeric or filenames
    # Try common patterns
    print(f"  sample BMD clip IDs from betas: {clip_ids[:3]}")
    aligned = []
    for i, cid in enumerate(clip_ids):
        s = str(cid)
        # Try to match
        keys = [s, f"bmd_{s}"]
        if s.isdigit():
            keys.append(f"bmd_vid_idx{int(s):04d}")
        if s.startswith("vid") and s[3:].isdigit():
            keys.append(f"bmd_vid_idx{int(s[3:]):04d}")
        if s.startswith("vid_idx"):
            keys.append(f"bmd_{s}")
        for key in keys:
            if key and key in bmd_score:
                aligned.append((i, key, bmd_score[key]))
                break
    print(f"  aligned {len(aligned)} / {len(clip_ids)} clips with BMD scores")
    if len(aligned) < 100:
        print("  [warn] poor alignment — clip ID format may need adjustment")
        print(f"  betas clip_ids[:5]: {clip_ids[:5]}")
        print(f"  bmd_score keys[:5]: {list(bmd_score)[:5]}")

    if len(aligned) >= 50:
        idx_arr = np.asarray([a[0] for a in aligned])
        mem_arr = np.asarray([a[2] for a in aligned], dtype=np.float32)
        X = betas[idx_arr]

        # === 5. Train v_mem on MEASURED brain ===
        print(f"\n[5] Training v_mem_measured (contrastive on {len(aligned)} clips)...")
        o = np.argsort(mem_arr)
        ne = int(len(mem_arr) * 0.30)
        v_mem_measured = X[o[-ne:]].mean(axis=0) - X[o[:ne]].mean(axis=0)
        v_mem_measured /= np.linalg.norm(v_mem_measured) + 1e-9

        # === 6. Load v_mem_TRIBE ===
        print("\n[6] Loading v_mem_TRIBE for comparison...")
        feats, mems = [], []
        for f in sorted(Path("data/features/tribe").glob("bmd_vid_idx*.npz")):
            sid = f.stem
            vid = sid.split("_seg_")[0]
            if vid not in bmd_score:
                continue
            arr = np.asarray(np.load(f, allow_pickle=False)["frames"], dtype=np.float32)
            if arr.ndim != 2:
                continue
            feats.append(arr.mean(axis=0))
            mems.append(bmd_score[vid])
        Xt = np.stack(feats)
        yt = np.asarray(mems, dtype=np.float32)
        ot = np.argsort(yt)
        ne_t = int(len(yt) * 0.30)
        v_mem_tribe = Xt[ot[-ne_t:]].mean(axis=0) - Xt[ot[:ne_t]].mean(axis=0)
        v_mem_tribe /= np.linalg.norm(v_mem_tribe) + 1e-9
        print(f"  v_mem_TRIBE shape: {v_mem_tribe.shape}")
        print(f"  v_mem_measured shape: {v_mem_measured.shape}")

        # === 7. The headline numbers ===
        cos_match = float(v_mem_measured @ v_mem_tribe)
        print("\n[7] HEADLINE RESULTS:")
        print(f"  cos(v_mem_measured_sub{sub}, v_mem_TRIBE) = {cos_match:+.4f}")

        # 5-fold CV of measured v_mem predicting BMD memorability
        def cv_rho(X, y, k=5):
            idx = np.arange(len(y))
            rng = np.random.default_rng(0)
            rng.shuffle(idx)
            folds = np.array_split(idx, k)
            proj = np.zeros(len(y))
            for fi in range(k):
                te = folds[fi]
                tr = np.concatenate([folds[j] for j in range(k) if j != fi])
                ytr = y[tr]
                n = int(len(ytr) * 0.30)
                o = np.argsort(ytr)
                v = X[tr][o[-n:]].mean(axis=0) - X[tr][o[:n]].mean(axis=0)
                v /= np.linalg.norm(v) + 1e-9
                proj[te] = X[te] @ v
            return float(
                np.corrcoef(np.argsort(np.argsort(proj)), np.argsort(np.argsort(y)))[
                    0, 1
                ]
            )

        rho_measured = cv_rho(X, mem_arr)
        print(
            f"  5-fold CV ρ on MEASURED brain (sub-{sub}, n={len(aligned)}) = {rho_measured:+.4f}"
        )
        print("  5-fold CV ρ on TRIBE (reference, n=1022)               = +0.401")

        out = {
            "subject": sub,
            "n_clips_aligned": int(len(aligned)),
            "feature_dim": int(betas.shape[1]),
            "cos_measured_vs_tribe": cos_match,
            "cv_rho_measured": rho_measured,
            "cv_rho_tribe_reference": 0.401,
        }
        out_path = Path(f"data/reports/fmri_pilot_sub{sub}.json")
        out_path.write_text(json.dumps(out, indent=2))
        # Save measured v_mem for cross-subject aggregation
        np.savez_compressed(
            f"data/reports/fmri_pilot_sub{sub}_vmem.npz", v_mem_measured=v_mem_measured
        )
        print(f"\n[done] wrote {out_path}")

    # === 8. Cleanup ===
    if not args.keep_files:
        print("\n[8] Cleaning up raw 8 GB pkls...")
        for p in [pkl_l, pkl_r]:
            if p.exists():
                sz = p.stat().st_size / 1e6
                p.unlink()
                print(f"  rm {p.name} ({sz:.0f} MB)")
    else:
        print(f"\n[8] Keeping raw files at {sub_dir}")


if __name__ == "__main__":
    main()
