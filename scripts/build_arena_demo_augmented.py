"""Augment arena_demo_data.json with:
  - per-tile per-frame projection on each of 13 directions (for feature A timeline)
  - per-persona top-5 ROI energy regions (for feature B brain widget)
  - 13x13 persona-vs-persona top-5 winner overlap matrix (for feature E)
"""

from __future__ import annotations

import json, numpy as np, polars as pl
from pathlib import Path


def load_frames(p):
    return np.asarray(np.load(p, allow_pickle=False)["frames"], dtype=np.float32)


def main() -> None:
    out_path = Path("data/reports/arena_demo_data.json")
    data = json.loads(out_path.read_text())

    # Build directions exactly like the arena script does
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    bmd = {f"bmd_vid_idx{e}": float(a["memorability_score"])
           for e, a in ann.items() if "memorability_score" in a}

    feats, mems, sids = [], [], []
    for f in sorted(Path("data/features/tribe").glob("bmd_vid_idx*.npz")):
        sid = f.stem; vid = sid.split("_seg_")[0]
        if vid not in bmd: continue
        arr = load_frames(f)
        if arr.ndim != 2: continue
        feats.append(arr.mean(axis=0)); mems.append(bmd[vid]); sids.append(sid)
    X = np.stack(feats); y = np.asarray(mems, dtype=np.float32)
    order = np.argsort(y); ne = int(len(y) * 0.30)
    v_global = X[order[-ne:]].mean(axis=0) - X[order[:ne]].mean(axis=0)
    v_global /= np.linalg.norm(v_global)

    persona_df = pl.read_parquet("data/labels/synthetic_persona_haiku_clean.parquet")
    ss = persona_df.select("scores").unnest("scores")
    rows = persona_df.with_columns(ss["memorability"].alias("_s")).select(
        ["persona_id", "segment_id", "_s"]).to_dicts()
    by_p = {}
    for r in rows:
        if r["_s"] is None: continue
        by_p.setdefault(r["persona_id"], {})[r["segment_id"]] = float(r["_s"])
    sid_to_idx = {s: i for i, s in enumerate(sids)}
    pdirs = {}
    for p, ssd in by_p.items():
        pairs = [(sid_to_idx[s], v) for s, v in ssd.items() if s in sid_to_idx]
        if len(pairs) < 50: continue
        idxs = np.asarray([i for i, _ in pairs]); pscs = np.asarray([s for _, s in pairs], dtype=np.float32)
        o = np.argsort(pscs); ne_p = int(len(pscs) * 0.30); Xp = X[idxs]
        d = Xp[o[-ne_p:]].mean(axis=0) - Xp[o[:ne_p]].mean(axis=0)
        pdirs[p] = d / np.linalg.norm(d)

    personas_sorted = sorted(pdirs)
    direction_names = personas_sorted + ["global"]
    direction_vecs = np.stack([pdirs[p] for p in personas_sorted] + [v_global])  # (13, 20484)

    # === FEATURE A: per-frame projections per tile (svd + veo) ===
    print("[A] computing per-tile timelines...")
    timelines = {"svd": {}, "veo": {}}
    pairs = [("svd", "data/features/tribe_best_of_n"),
             ("veo", "data/features/tribe_veo_bon")]
    for kind, src in pairs:
        for f in sorted(Path(src).glob("*.npz")):
            name = f.stem
            if "_n" not in name: continue
            seed, n_str = name.rsplit("_n", 1)
            try: n = int(n_str)
            except: continue
            arr = load_frames(f)  # (T, 20484)
            if arr.ndim != 2: continue
            # per-frame projection on each direction: (T, 13)
            proj = arr @ direction_vecs.T  # (T, 13)
            timelines[kind].setdefault(seed, {})[str(n)] = {
                "T": int(arr.shape[0]),
                "proj": proj.astype(np.float32).round(4).tolist(),  # round to save space
            }
    n_svd = sum(len(v) for v in timelines["svd"].values())
    n_veo = sum(len(v) for v in timelines["veo"].values())
    print(f"  svd: {n_svd} tiles, veo: {n_veo} tiles")

    # === FEATURE B: per-persona top ROI regions from roi_decomposition.json ===
    print("[B] extracting ROI rankings...")
    roi = json.loads(Path("data/reports/roi_decomposition.json").read_text())
    persona_rois = {}
    for d_name in direction_names:
        # rankings is keyed by direction; global = BMD_memorability
        key = "BMD_memorability" if d_name == "global" else d_name
        if key not in roi["rankings"]:
            persona_rois[d_name] = []
            continue
        top10 = roi["rankings"][key][:10]
        persona_rois[d_name] = [{"region": r["region"], "energy": float(r["energy"])} for r in top10]

    # === FEATURE E: persona-vs-persona top-5 winner overlap ===
    # For each pair (p1, p2), across all seeds: how often is p1's top-5
    # picks overlap with p2's top-5 picks?
    print("[E] computing persona-vs-persona top-5 overlap...")
    overlap = {}
    for kind in ("svd", "veo"):
        # n picks per seed per persona = ranked n by direction's score
        seeds = data["arena"][kind]
        # For each seed, sort items by persona p's score, take top-5 n's
        per_seed_top5 = {}  # seed -> {persona: set(top5_n)}
        for seed, items in seeds.items():
            per_seed_top5[seed] = {}
            n_total = len(items)
            k = min(5, max(1, n_total // 2))  # top half if fewer items
            for p in direction_names:
                ranked = sorted(items, key=lambda x: -x["scores"].get(p, 0))[:k]
                per_seed_top5[seed][p] = set(x["n"] for x in ranked)
        # For each persona pair, mean(|p1 ∩ p2| / |p1|) across seeds
        n_dir = len(direction_names)
        mat = np.zeros((n_dir, n_dir), dtype=np.float32)
        for i, p1 in enumerate(direction_names):
            for j, p2 in enumerate(direction_names):
                vals = []
                for seed in per_seed_top5:
                    s1, s2 = per_seed_top5[seed][p1], per_seed_top5[seed][p2]
                    if not s1: continue
                    vals.append(len(s1 & s2) / len(s1))
                mat[i, j] = float(np.mean(vals)) if vals else 0.0
        overlap[kind] = {
            "names": direction_names,
            "matrix": mat.round(3).tolist(),
        }
    print(f"  built {n_dir}x{n_dir} overlap matrices for svd + veo")

    data["timelines"] = timelines
    data["persona_rois"] = persona_rois
    data["overlap"] = overlap

    out_path.write_text(json.dumps(data))
    print(f"\n[done] wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
