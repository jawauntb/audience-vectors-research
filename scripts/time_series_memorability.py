"""Time-series memorability analysis.

TRIBE outputs are (T, 20484): T time-bins × cortical vertices. We've been
mean-pooling over T. This script asks: where in time does the brain-direction
project highest? Does memorability concentrate at a specific moment?

For each BMD clip:
  - Project EACH time-frame separately onto v_mem
  - Build per-clip projection-vs-time profile
  - Aggregate: align profiles, find population-level patterns

Findings to look for:
  - Is the per-frame projection roughly flat (memorability is "static" feature)
    or does it spike at specific moments?
  - Do memorable clips have HIGHER PEAKS or HIGHER AVERAGES vs unmemorable?
  - Is the time-of-peak consistent across clips?
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path


def main() -> None:
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    bmd = {f"bmd_vid_idx{e}": float(a["memorability_score"])
           for e, a in ann.items() if "memorability_score" in a}

    print("[time-series] loading TRIBE features (preserving T dim)")
    raw_feats = {}
    mems_aligned = {}
    for f in sorted(Path("data/features/tribe").glob("bmd_vid_idx*.npz")):
        sid = f.stem; vid = sid.split("_seg_")[0]
        if vid not in bmd: continue
        d = np.load(f, allow_pickle=False)
        if "frames" not in d.files: continue
        arr = np.asarray(d["frames"], dtype=np.float32)  # (T, V)
        if arr.ndim != 2: continue
        raw_feats[sid] = arr
        mems_aligned[sid] = bmd[vid]

    # Build v_mem the standard way (mean-over-T then contrastive)
    sids_sorted = sorted(raw_feats)
    pooled = np.stack([raw_feats[s].mean(axis=0) for s in sids_sorted])
    mems_arr = np.asarray([mems_aligned[s] for s in sids_sorted], dtype=np.float32)
    order = np.argsort(mems_arr); ne = int(len(mems_arr) * 0.30)
    v_mem = pooled[order[-ne:]].mean(axis=0) - pooled[order[:ne]].mean(axis=0)
    v_mem /= np.linalg.norm(v_mem)

    # For each clip, compute per-time-frame projection
    print("[time-series] computing per-frame projections")
    profiles = {}  # sid -> array shape (T,)
    for sid, arr in raw_feats.items():
        profiles[sid] = arr @ v_mem  # (T,) per-time projection

    # Check time-length distribution
    T_counts = {}
    for p in profiles.values():
        T_counts[p.shape[0]] = T_counts.get(p.shape[0], 0) + 1
    print(f"  T-length distribution: {sorted(T_counts.items())}")

    # Group clips by T-length (most common). For clips with T=3 or T=4 typically.
    T_main = max(T_counts, key=T_counts.get)
    print(f"  using T={T_main} (n={T_counts[T_main]})")

    # Align: only clips with T = T_main
    aligned_sids = [s for s in profiles if profiles[s].shape[0] == T_main]
    P = np.stack([profiles[s] for s in aligned_sids])  # (N, T)
    M = np.asarray([mems_aligned[s] for s in aligned_sids], dtype=np.float32)

    # Split top/bottom 30% by BMD memorability
    o = np.argsort(M); ne_h = int(len(M) * 0.30)
    high = P[o[-ne_h:]]
    low  = P[o[:ne_h]]

    print(f"\n=== Per-frame projection profile, top vs bottom 30% by BMD memorability ===")
    print(f"  N high={high.shape[0]}, N low={low.shape[0]}")
    print(f"  {'frame':<8} {'mean(high)':>12} {'mean(low)':>12} {'Δ':>10} {'std(high)':>11} {'std(low)':>11}")
    for t in range(T_main):
        h_mean, h_std = float(high[:, t].mean()), float(high[:, t].std())
        l_mean, l_std = float(low[:, t].mean()), float(low[:, t].std())
        print(f"  t={t:<6} {h_mean:>+12.3f} {l_mean:>+12.3f} {h_mean - l_mean:>+10.3f} {h_std:>11.3f} {l_std:>11.3f}")

    # Where do clips peak? — argmax-of-time
    peak_t_high = np.argmax(high, axis=1)
    peak_t_low  = np.argmax(low, axis=1)
    print(f"\n=== Time-of-peak distribution ===")
    for t in range(T_main):
        h_pct = float((peak_t_high == t).mean()) * 100
        l_pct = float((peak_t_low == t).mean()) * 100
        print(f"  t={t}: high-mem peaks here {h_pct:5.1f}%   low-mem peaks here {l_pct:5.1f}%")

    # Per-clip variance across time (is signal flat or spiky?)
    per_clip_t_std = P.std(axis=1)
    per_clip_t_max_minus_min = P.max(axis=1) - P.min(axis=1)
    print(f"\n=== Per-clip time-variance (is memorability moment-specific?) ===")
    print(f"  mean stdev across time: {per_clip_t_std.mean():.3f}")
    print(f"  mean (max-min) across time: {per_clip_t_max_minus_min.mean():.3f}")
    print(f"  vs. mean per-clip mean projection (in BMD population): "
          f"{P.mean():.3f}")

    # Correlation: does per-frame max better predict BMD mem than per-frame mean?
    from itertools import product
    print(f"\n=== Predictive power: which time-statistic best predicts BMD memorability? ===")
    for stat_name, stat in [("mean", P.mean(axis=1)),
                              ("max",  P.max(axis=1)),
                              ("min",  P.min(axis=1)),
                              ("median", np.median(P, axis=1))]:
        r = float(np.corrcoef(np.argsort(np.argsort(stat)),
                              np.argsort(np.argsort(M)))[0, 1])
        print(f"  Spearman(projection-{stat_name}, BMD-mem) = {r:+.3f}")

    # Save
    out = {
        "T": int(T_main),
        "N_aligned": int(len(aligned_sids)),
        "high_mean_profile": [float(high[:, t].mean()) for t in range(T_main)],
        "low_mean_profile":  [float(low[:, t].mean()) for t in range(T_main)],
        "high_peak_distribution": [float((peak_t_high == t).mean()) for t in range(T_main)],
        "low_peak_distribution":  [float((peak_t_low == t).mean()) for t in range(T_main)],
        "per_clip_time_stdev_mean": float(per_clip_t_std.mean()),
        "per_clip_time_max_minus_min_mean": float(per_clip_t_max_minus_min.mean()),
        "spearman_by_stat": {
            "mean": float(np.corrcoef(np.argsort(np.argsort(P.mean(axis=1))), np.argsort(np.argsort(M)))[0,1]),
            "max":  float(np.corrcoef(np.argsort(np.argsort(P.max(axis=1))), np.argsort(np.argsort(M)))[0,1]),
            "min":  float(np.corrcoef(np.argsort(np.argsort(P.min(axis=1))), np.argsort(np.argsort(M)))[0,1]),
            "median": float(np.corrcoef(np.argsort(np.argsort(np.median(P, axis=1))), np.argsort(np.argsort(M)))[0,1]),
        },
    }
    Path("data/reports/time_series_memorability.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote data/reports/time_series_memorability.json")


if __name__ == "__main__":
    main()
