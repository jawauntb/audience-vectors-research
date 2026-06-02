"""(H) Persona direction stability + corrected overlap test.

For each of 12 personas: train the persona-conditioned memorability direction
v_p on the FULL persona-labeled set, then on 20 disjoint half-splits.
Measure pairwise cosines across splits — should be high if persona directions
are stable.

Also: report the corrected unsigned overlap statistics. Signed off-diagonal
means can cancel sign-flipped pairs even when personas share the same axis, so
mean |cos| and effective rank are the relevant decomposition checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


def fit_v(X, y):
    o = np.argsort(y)
    ne = int(len(y) * 0.30)
    v = X[o[-ne:]].mean(axis=0) - X[o[:ne]].mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def effective_rank(cos_mat):
    eigvals = np.linalg.eigvalsh(cos_mat)
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(eigvals.sum())
    if total <= 0:
        return 0.0
    probs = eigvals / total
    probs = probs[probs > 0]
    return float(np.exp(-np.sum(probs * np.log(probs))))


def main() -> None:  # noqa: C901, PLR0912
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    bmd = {
        f"bmd_vid_idx{e}": float(a["memorability_score"])
        for e, a in ann.items()
        if "memorability_score" in a
    }

    feats, sids = [], []
    for f in sorted(Path("data/features/tribe").glob("bmd_vid_idx*.npz")):
        sid = f.stem
        vid = sid.split("_seg_")[0]
        if vid not in bmd:
            continue
        arr = np.asarray(np.load(f, allow_pickle=False)["frames"], dtype=np.float32)
        if arr.ndim != 2:
            continue
        feats.append(arr.mean(axis=0))
        sids.append(sid)
    X = np.stack(feats)
    sid_to_idx = {s: i for i, s in enumerate(sids)}
    print(f"[ps] N={len(sids)}, D={X.shape[1]}")

    persona_df = pl.read_parquet("data/labels/synthetic_persona_haiku_clean.parquet")
    ss = persona_df.select("scores").unnest("scores")
    rows = (
        persona_df.with_columns(ss["memorability"].alias("_s"))
        .select(["persona_id", "segment_id", "_s"])
        .to_dicts()
    )
    by_p = {}
    for r in rows:
        if r["_s"] is None:
            continue
        by_p.setdefault(r["persona_id"], {})[r["segment_id"]] = float(r["_s"])

    # Persona pairs valid for each persona
    valid_p = {}
    for p, ssd in by_p.items():
        pairs = [(sid_to_idx[s], v) for s, v in ssd.items() if s in sid_to_idx]
        if len(pairs) < 50:
            continue
        valid_p[p] = pairs
    personas = sorted(valid_p)
    print(f"[ps] {len(personas)} valid personas")

    # === Full v_p directions ===
    full_v = {}
    for p in personas:
        pairs = valid_p[p]
        idxs = np.asarray([i for i, _ in pairs])
        pscs = np.asarray([s for _, s in pairs], dtype=np.float32)
        full_v[p] = fit_v(X[idxs], pscs)

    # === Disjoint-halves stability ===
    rng = np.random.default_rng(20260515)
    stability = {}
    for p in personas:
        pairs = valid_p[p]
        idxs = np.asarray([i for i, _ in pairs])
        pscs = np.asarray([s for _, s in pairs], dtype=np.float32)
        cos_pairs = []
        for r in range(20):
            perm = rng.permutation(len(pairs))
            h1, h2 = perm[: len(pairs) // 2], perm[len(pairs) // 2 :]
            v1 = fit_v(X[idxs[h1]], pscs[h1])
            v2 = fit_v(X[idxs[h2]], pscs[h2])
            cos_pairs.append(float(v1 @ v2))
        cos_arr = np.asarray(cos_pairs)
        stability[p] = {
            "mean": float(cos_arr.mean()),
            "min": float(cos_arr.min()),
            "max": float(cos_arr.max()),
        }
        print(
            f"  {p:24s}: disjoint-halves cos μ = {cos_arr.mean():.3f}, "
            f"range [{cos_arr.min():.3f}, {cos_arr.max():.3f}]"
        )

    means = [s["mean"] for s in stability.values()]
    print(
        f"\n[ps] across {len(personas)} personas: mean disjoint-halves cos = {np.mean(means):.3f}"
    )

    # === Overlap test ===
    # In R^D, random unit vectors have <v1, v2> ~ N(0, 1/sqrt(D)).
    D = X.shape[1]
    sigma_random = 1.0 / np.sqrt(D)
    mean_abs_random = np.sqrt(2 / np.pi) * sigma_random
    # Observed mean off-diag cosine across persona directions
    P = np.stack([full_v[p] for p in personas])
    cos_mat = P @ P.T
    off_diag = []
    for i in range(len(personas)):
        for j in range(len(personas)):
            if i != j:
                off_diag.append(cos_mat[i, j])
    off_diag = np.asarray(off_diag)
    mean_off = float(off_diag.mean())
    mean_abs_off = float(np.mean(np.abs(off_diag)))
    median_abs_off = float(np.median(np.abs(off_diag)))
    erank = effective_rank(cos_mat)
    z_random = mean_off / sigma_random
    z_abs_random = (mean_abs_off - mean_abs_random) / (
        sigma_random / np.sqrt(len(off_diag))
    )
    print(f"\n[ps] persona pairwise off-diagonal cosine: signed mean = {mean_off:+.4f}")
    print(f"     mean |cos| = {mean_abs_off:.4f}, median |cos| = {median_abs_off:.4f}")
    print(f"     effective rank = {erank:.2f}/{len(personas)}")
    print(f"     random R^{D} unit vectors: cos ~ N(0, {sigma_random:.5f})")
    print(f"     signed-mean z vs random null = {z_random:.2f}")
    print(f"     |cos| z approximation vs random null = {z_abs_random:.2f}")

    out = {
        "n_personas": len(personas),
        "feature_dim": int(D),
        "disjoint_halves_per_persona": stability,
        "mean_disjoint_halves_cos": float(np.mean(means)),
        "mean_off_diagonal_persona_cos": mean_off,
        "mean_abs_off_diagonal_persona_cos": mean_abs_off,
        "median_abs_off_diagonal_persona_cos": median_abs_off,
        "effective_rank": erank,
        "random_unit_vector_sigma": float(sigma_random),
        "random_unit_vector_mean_abs": float(mean_abs_random),
        "z_off_diag_vs_random": float(z_random),
        "z_abs_off_diag_vs_random_approx": float(z_abs_random),
    }
    Path("data/reports/persona_stability.json").write_text(json.dumps(out, indent=2))
    print("[ps] done — wrote data/reports/persona_stability.json")


if __name__ == "__main__":
    main()
