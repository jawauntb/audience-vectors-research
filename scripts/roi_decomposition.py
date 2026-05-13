"""ROI decomposition of contrastive directions on TRIBE features.

For each direction (BMD memorability + 12 personas), split the 20,484
fsaverage5 cortical vertices by Destrieux atlas parcels and compute per-region
'energy' (squared magnitude). This tells us where each direction *lives* in
cortex — i.e. which brain regions encode that viewer-response axis.

Outputs:
  - data/reports/roi_decomposition.json — per-direction × per-region table
  - data/reports/roi_decomposition.md — top-K regions per direction
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from nilearn.datasets import fetch_atlas_surf_destrieux


def _bmd() -> dict[str, float]:
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    return {f"bmd_vid_idx{e}": float(a["memorability_score"])
            for e, a in ann.items() if "memorability_score" in a}


def _load_feature(p: Path) -> np.ndarray:
    payload = np.load(p, allow_pickle=False)
    arr = np.asarray(payload["frames"], dtype=np.float32)
    return arr.mean(axis=0) if arr.ndim == 2 else arr


def _direction(features: np.ndarray, scores: np.ndarray, top_k_frac: float = 0.30) -> np.ndarray:
    order = np.argsort(scores)
    n_each = max(3, int(len(scores) * top_k_frac))
    v = features[order[-n_each:]].mean(axis=0) - features[order[:n_each]].mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def main() -> None:
    feat_dir = Path("data/features/tribe")
    ids = []
    feats = []
    for f in sorted(feat_dir.glob("bmd_vid_idx*.npz")):
        ids.append(f.stem)
        feats.append(_load_feature(f))
    X = np.stack(feats).astype(np.float32)
    print(f"[roi] loaded {X.shape[0]} TRIBE features (dim={X.shape[1]})")
    assert X.shape[1] == 20484, "expected 20484 fsaverage5 vertices"

    # BMD memorability direction
    bmd = _bmd()
    bmd_scores = np.asarray([bmd.get(sid.split("_seg_")[0], np.nan) for sid in ids], dtype=np.float32)
    keep = ~np.isnan(bmd_scores)
    v_bmd = _direction(X[keep], bmd_scores[keep])

    # Persona directions via the persona scoring file
    persona_file = Path("data/labels/synthetic_persona_haiku_clean.parquet")
    if not persona_file.exists():
        persona_file = Path("data/labels/synthetic_persona_haiku.parquet")
    df = pl.read_parquet(persona_file)
    scores_struct = df.select("scores").unnest("scores")
    rows = df.with_columns(scores_struct["memorability"].alias("_s")).select(["persona_id", "segment_id", "_s"]).to_dicts()

    by_persona: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["_s"] is None:
            continue
        by_persona.setdefault(r["persona_id"], {})[r["segment_id"]] = float(r["_s"])

    sid_to_idx = {sid: i for i, sid in enumerate(ids)}
    persona_dirs: dict[str, np.ndarray] = {}
    for persona, seg_scores in by_persona.items():
        idxs = []
        scs = []
        for sid, s in seg_scores.items():
            if sid in sid_to_idx:
                idxs.append(sid_to_idx[sid])
                scs.append(s)
        if len(idxs) < 50:
            continue
        Xp = X[np.asarray(idxs)]
        sp = np.asarray(scs, dtype=np.float32)
        persona_dirs[persona] = _direction(Xp, sp)
    print(f"[roi] persona directions: {len(persona_dirs)}")

    # Destrieux parcellation
    print("[roi] fetching Destrieux atlas (fsaverage5)…")
    atlas = fetch_atlas_surf_destrieux()
    lh = np.asarray(atlas["map_left"]).astype(int)
    rh = np.asarray(atlas["map_right"]).astype(int)
    labels = list(atlas["labels"])
    n_regions = len(labels)
    # 20484-long parcel vector: left first 10242, right next 10242
    parcels = np.concatenate([lh, rh])
    assert parcels.shape[0] == 20484
    # offset right-hemi labels so they're distinct
    rh_offset = parcels.copy()
    rh_offset[10242:] = rh + n_regions
    # We end up with up to 2*n_regions distinct labels
    all_labels = [f"L_{l}" for l in labels] + [f"R_{l}" for l in labels]

    def energy_per_region(v: np.ndarray) -> np.ndarray:
        energies = np.zeros(2 * n_regions, dtype=np.float32)
        for r in range(2 * n_regions):
            mask = rh_offset == r
            if mask.any():
                energies[r] = float((v[mask] ** 2).sum())
        return energies

    energy = {"BMD_memorability": energy_per_region(v_bmd)}
    for p, vp in persona_dirs.items():
        energy[p] = energy_per_region(vp)

    # Top-K regions per direction
    K = 10
    rankings: dict[str, list[dict]] = {}
    for name, e in energy.items():
        order = np.argsort(-e)
        top = []
        for r in order[:K]:
            top.append({"region": all_labels[r], "energy": float(e[r])})
        rankings[name] = top

    # Cosine similarity of energy distributions
    names = sorted(energy.keys())
    Emat = np.stack([energy[n] for n in names])
    Enorm = Emat / (np.linalg.norm(Emat, axis=1, keepdims=True) + 1e-12)
    energy_cos = Enorm @ Enorm.T

    # Save
    out = {
        "n_regions_per_hemi": n_regions,
        "directions": names,
        "rankings": rankings,
        "energy_cosine_matrix": energy_cos.tolist(),
    }
    Path("data/reports/roi_decomposition.json").write_text(json.dumps(out, indent=2))

    # Markdown
    md_lines = ["# ROI decomposition — where each direction lives in cortex", "",
                f"Atlas: Destrieux (fsaverage5), {n_regions} regions × 2 hemispheres", "",
                f"Energy is per-region squared magnitude of the (unit-norm) direction.",
                "Sum of energies across regions = 1 (the direction's total mass).", ""]
    for name in names:
        md_lines.append(f"## {name}")
        md_lines.append("")
        md_lines.append("| rank | region | energy |")
        md_lines.append("|---:|---|---:|")
        for i, r in enumerate(rankings[name][:K]):
            md_lines.append(f"| {i+1} | `{r['region']}` | {r['energy']:.4f} |")
        md_lines.append("")
    Path("data/reports/roi_decomposition.md").write_text("\n".join(md_lines))

    print(f"[done] wrote ROI decomposition for {len(names)} directions")
    print()
    print(f"=== BMD memorability — top 8 regions ===")
    for i, r in enumerate(rankings["BMD_memorability"][:8]):
        print(f"  {i+1:>2}. {r['region']:<40} energy={r['energy']:.4f}")


if __name__ == "__main__":
    main()
