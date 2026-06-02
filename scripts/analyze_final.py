"""Compute the final analyses for the 4 new experiment families,
write JSON summaries that the paper script + arena demo can consume."""

from __future__ import annotations

import json
import numpy as np
import polars as pl
from pathlib import Path


def load_feat(p):
    arr = np.asarray(np.load(p, allow_pickle=False)["frames"], dtype=np.float32)
    return arr.mean(axis=0) if arr.ndim == 2 else arr


def main() -> None:
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    bmd = {f"bmd_vid_idx{e}": float(a["memorability_score"])
           for e, a in ann.items() if "memorability_score" in a}

    # Train v_mem + 12 persona directions
    feats, mems, sids = [], [], []
    for f in sorted(Path("data/features/tribe").glob("bmd_vid_idx*.npz")):
        sid = f.stem; vid = sid.split("_seg_")[0]
        if vid not in bmd: continue
        feats.append(load_feat(f)); mems.append(bmd[vid]); sids.append(sid)
    X = np.stack(feats); y = np.asarray(mems, dtype=np.float32)
    order = np.argsort(y); ne = int(len(y) * 0.30)
    v_mem = X[order[-ne:]].mean(axis=0) - X[order[:ne]].mean(axis=0)
    v_mem /= np.linalg.norm(v_mem)

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

    out = {}

    # ===== (A) per-persona α-steering grid =====
    print("=== per-persona α-steering grid ===")
    src = Path("data/features/tribe_svd_per_persona")
    by_cell = {}  # (seed, persona_steered, alpha_idx) -> proj on each direction
    for f in sorted(src.glob("*.npz")):
        # naming: vid_idx0150__cinematic-cleo__a0
        parts = f.stem.split("__")
        if len(parts) != 3: continue
        seed, persona_steered, alpha_str = parts
        alpha_idx = int(alpha_str.replace("a", ""))
        vec = load_feat(f)
        scores = {p: float(vec @ vd) for p, vd in pdirs.items()}
        scores["global"] = float(vec @ v_mem)
        by_cell.setdefault((seed, persona_steered), {})[alpha_idx] = scores

    # Build 12×12 grid: persona-steered (rows) × persona-scored (cols).
    # Cell value = mean over seeds of [proj(α=+5) - proj(α=-5)] under steered direction
    personas_sorted = sorted(pdirs)
    grid = np.zeros((len(personas_sorted), len(personas_sorted) + 1), dtype=np.float32)
    seeds = set([k[0] for k in by_cell])
    for pi, persona_steered in enumerate(personas_sorted):
        for ci, persona_scored in enumerate(personas_sorted + ["global"]):
            deltas = []
            for seed in seeds:
                cell = by_cell.get((seed, persona_steered))
                if cell is None or 0 not in cell or 2 not in cell: continue
                deltas.append(cell[2].get(persona_scored, 0) - cell[0].get(persona_scored, 0))
            grid[pi, ci] = float(np.mean(deltas)) if deltas else 0.0
    print(f"  built {grid.shape} grid")

    diag_mean = float(np.mean([grid[i, i] for i in range(len(personas_sorted))]))
    off_diag = []
    for i in range(len(personas_sorted)):
        for j in range(len(personas_sorted)):
            if i != j: off_diag.append(grid[i, j])
    off_mean = float(np.mean(off_diag))
    print(f"  diagonal mean (steered-by-X scored-by-X): {diag_mean:+.3f}")
    print(f"  off-diagonal mean: {off_mean:+.3f}")
    print(f"  diagonal/off-diagonal ratio: {diag_mean / (abs(off_mean) + 1e-6):+.2f}")
    out["per_persona_steering"] = {
        "personas_steered": personas_sorted,
        "personas_scored": personas_sorted + ["global"],
        "grid": grid.tolist(),
        "diagonal_mean": diag_mean,
        "off_diagonal_mean": off_mean,
        "n_seeds_per_cell": int(len(seeds)),
    }

    # ===== (B) α + best-of-N composition =====
    print("\n=== α + best-of-N composition ===")
    src = Path("data/features/tribe_svd_alpha_bon")
    by_seed = {}
    for f in sorted(src.glob("*.npz")):
        # vid_idx0150_a10_n03
        name = f.stem
        if "_a10_n" not in name: continue
        seed, rest = name.rsplit("_a10_n", 1)
        n = int(rest)
        vec = load_feat(f)
        by_seed.setdefault(seed, []).append((n, float(vec @ v_mem)))

    a10_results = {}
    for seed, items in by_seed.items():
        projs = sorted([p for _, p in items])
        if len(projs) < 3: continue
        mn, md, mx = projs[0], projs[len(projs)//2], projs[-1]
        a10_results[seed] = {"n": len(projs), "min": float(mn), "median": float(md),
                              "max": float(mx), "lift": float(mx - md), "spread": float(mx - mn)}

    # And the OG α=0 best-of-N for same seeds
    a0_results = {}
    for seed in by_seed:
        feats = sorted(Path("data/features/tribe_best_of_n").glob(f"{seed}_n*.npz"))
        projs = sorted([float(load_feat(f) @ v_mem) for f in feats])
        if len(projs) < 3: continue
        mn, md, mx = projs[0], projs[len(projs)//2], projs[-1]
        a0_results[seed] = {"n": len(projs), "min": float(mn), "median": float(md),
                             "max": float(mx), "lift": float(mx - md), "spread": float(mx - mn)}

    print(f"  α=+10: mean median = {np.mean([r['median'] for r in a10_results.values()]):+.3f}, "
          f"mean max = {np.mean([r['max'] for r in a10_results.values()]):+.3f}, "
          f"mean lift = {np.mean([r['lift'] for r in a10_results.values()]):+.3f}")
    if a0_results:
        print(f"  α=0:   mean median = {np.mean([r['median'] for r in a0_results.values()]):+.3f}, "
              f"mean max = {np.mean([r['max'] for r in a0_results.values()]):+.3f}, "
              f"mean lift = {np.mean([r['lift'] for r in a0_results.values()]):+.3f}")
        max_a10 = np.mean([r['max'] for r in a10_results.values()])
        max_a0 = np.mean([r['max'] for r in a0_results.values()])
        print(f"  α=+10 max − α=0 max: {max_a10 - max_a0:+.3f} (compound effect)")

    out["alpha_plus_bon"] = {
        "seeds": list(by_seed.keys()),
        "alpha_10": a10_results,
        "alpha_0": a0_results,
        "mean_max_diff": float(np.mean([r['max'] for r in a10_results.values()]) -
                                np.mean([r['max'] for r in a0_results.values()])) if a0_results else None,
    }

    # ===== (C) Bigger Veo sweep =====
    print("\n=== bigger Veo sweep ===")
    src = Path("data/features/tribe_veo_bon")
    by_prompt = {}
    for f in sorted(src.glob("*.npz")):
        name = f.stem
        if "_n" not in name: continue
        p_id, n_str = name.rsplit("_n", 1)
        try: n = int(n_str)
        except: continue
        vec = load_feat(f)
        by_prompt.setdefault(p_id, []).append(float(vec @ v_mem))

    veo_results = {}
    for p, projs in by_prompt.items():
        projs = sorted(projs)
        if len(projs) < 3: continue
        mn, md, mx = projs[0], projs[len(projs)//2], projs[-1]
        veo_results[p] = {"n": len(projs), "min": float(mn), "median": float(md),
                          "max": float(mx), "lift": float(mx - md), "spread": float(mx - mn)}

    print(f"  N prompts: {len(veo_results)}")
    if veo_results:
        lifts = [r['lift'] for r in veo_results.values()]
        spreads = [r['spread'] for r in veo_results.values()]
        print(f"  mean lift: {np.mean(lifts):+.3f} (SVD was +2.07)")
        print(f"  mean spread: {np.mean(spreads):+.3f} (SVD was +3.89)")
        print(f"  all prompt lifts: {[f'{l:+.2f}' for l in lifts]}")
    out["veo_bigger_sweep"] = {
        "per_prompt": veo_results,
        "mean_lift": float(np.mean([r['lift'] for r in veo_results.values()])) if veo_results else None,
        "mean_spread": float(np.mean([r['spread'] for r in veo_results.values()])) if veo_results else None,
    }

    # ===== (D) Time-series (already done, just include the path) =====
    ts_path = Path("data/reports/time_series_memorability.json")
    if ts_path.exists():
        out["time_series"] = json.loads(ts_path.read_text())

    Path("data/reports/final_analyses.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] wrote data/reports/final_analyses.json")


if __name__ == "__main__":
    main()
