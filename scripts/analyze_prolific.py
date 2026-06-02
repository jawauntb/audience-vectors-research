"""Analyze Prolific survey responses.

Inputs: a directory of prolific_response_*.json files (one per rater) OR
        a single JSON exported from a Google Form (one row per rater).

Outputs:
  - data/reports/prolific_analysis.json with per-study stats
  - terminal: headline p-values

The three sub-studies (see prolific_survey.html):
  A. Best-of-N: did humans pick TRIBE's winner significantly more than 50%?
  B. α-steering: did humans pick α=+10 over α=−10? (1 pair, just descriptive)
  C. Persona winner: did humans pick persona_A's winner significantly more often
     than persona_B's? (5 pairs)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import stats


def load_responses(paths: list[Path]) -> list[dict]:
    """Load Prolific response JSONs."""
    out = []
    for p in paths:
        try:
            d = json.loads(p.read_text())
            if isinstance(d, dict) and "responses" in d:
                out.append(d)
            elif isinstance(d, list):
                out.extend(d)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ! skipping {p}: {e}", file=sys.stderr)
    return out


def exclude_failed_attention(rater: dict) -> bool:
    """Return True if rater passed all attention checks."""
    for r in rater["responses"]:
        if r["study"] == "attention" and not r["chose_winner"]:
            return False
    return True


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return center - half, center + half


def _bootstrap_ci(
    values: list[float], rng: np.random.Generator, n_boot: int = 20000
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def study_cluster_stats(raters: list[dict], study: str) -> dict:
    per_rater_hits = []
    per_rater_total = []
    per_pair = defaultdict(lambda: {"hits": 0, "total": 0})

    for rater in raters:
        hits = 0
        total = 0
        for resp in rater["responses"]:
            if resp["study"] != study:
                continue
            if study == "C_persona_winner":
                key = f"{resp['seed']}|{resp['persona_A']}|{resp['persona_B']}"
            else:
                key = resp["seed"]
            total += 1
            per_pair[key]["total"] += 1
            if resp["chose_winner"]:
                hits += 1
                per_pair[key]["hits"] += 1
        if total:
            per_rater_hits.append(hits)
            per_rater_total.append(total)

    per_rater_props = [
        h / t for h, t in zip(per_rater_hits, per_rater_total, strict=False)
    ]
    per_pair_props = [v["hits"] / v["total"] for v in per_pair.values() if v["total"]]
    rng = np.random.default_rng(20260518)

    rater_t = stats.ttest_1samp(per_rater_props, 0.5) if per_rater_props else None
    pair_t = stats.ttest_1samp(per_pair_props, 0.5) if per_pair_props else None
    rater_w = (
        stats.wilcoxon(np.asarray(per_rater_props) - 0.5) if per_rater_props else None
    )
    pair_w = (
        stats.wilcoxon(np.asarray(per_pair_props) - 0.5) if per_pair_props else None
    )

    n_raters_non_tie = sum(p != 0.5 for p in per_rater_props)
    n_raters_above = sum(p > 0.5 for p in per_rater_props)
    n_pairs_above = sum(p > 0.5 for p in per_pair_props)

    def pvalue(result: object | None) -> float | None:
        if result is None:
            return None
        return float(getattr(cast(Any, result), "pvalue"))

    return {
        "per_rater_mean": float(np.mean(per_rater_props)) if per_rater_props else None,
        "per_rater_sd": float(np.std(per_rater_props, ddof=1))
        if len(per_rater_props) > 1
        else None,
        "per_rater_t_p": pvalue(rater_t),
        "per_rater_wilcoxon_p": pvalue(rater_w),
        "per_rater_above_half": int(n_raters_above),
        "per_rater_non_tie": int(n_raters_non_tie),
        "per_rater_sign_p_greater": (
            float(
                stats.binomtest(
                    n_raters_above, n_raters_non_tie, p=0.5, alternative="greater"
                ).pvalue
            )
            if n_raters_non_tie
            else None
        ),
        "per_pair_mean": float(np.mean(per_pair_props)) if per_pair_props else None,
        "per_pair_sd": float(np.std(per_pair_props, ddof=1))
        if len(per_pair_props) > 1
        else None,
        "per_pair_t_p": pvalue(pair_t),
        "per_pair_wilcoxon_p": pvalue(pair_w),
        "per_pair_above_half": int(n_pairs_above),
        "per_pair_total": int(len(per_pair_props)),
        "per_pair_sign_p_greater": (
            float(
                stats.binomtest(
                    n_pairs_above, len(per_pair_props), p=0.5, alternative="greater"
                ).pvalue
            )
            if per_pair_props
            else None
        ),
        "rater_cluster_bootstrap_ci": _bootstrap_ci(per_rater_props, rng),
        "pair_cluster_bootstrap_ci": _bootstrap_ci(per_pair_props, rng),
    }


def main() -> None:  # noqa: C901, PLR0912
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--responses-dir", type=Path, default=Path("data/raw/prolific_responses")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/reports/prolific_analysis.json")
    )
    args = parser.parse_args()

    if not args.responses_dir.exists():
        print(
            f"[error] {args.responses_dir} does not exist. Run the survey first and "
            f"download responses to this dir."
        )
        return

    paths = sorted(args.responses_dir.glob("*.json"))
    print(f"[load] found {len(paths)} response files")
    raters = load_responses(paths)
    print(f"[load] {len(raters)} raters total")

    raters_pass = [r for r in raters if exclude_failed_attention(r)]
    print(f"[qc] {len(raters_pass)}/{len(raters)} passed attention checks")

    if len(raters_pass) < 10:
        print("[warn] fewer than 10 valid raters — results will be underpowered")

    # === A. Best-of-N ===
    print("\n=== A. Best-of-N (did humans pick TRIBE's winner?) ===")
    a_per_pair = defaultdict(lambda: {"hits": 0, "total": 0})
    for r in raters_pass:
        for resp in r["responses"]:
            if resp["study"] != "A_best_of_n":
                continue
            key = resp["seed"]
            a_per_pair[key]["total"] += 1
            if resp["chose_winner"]:
                a_per_pair[key]["hits"] += 1

    total_hits = sum(p["hits"] for p in a_per_pair.values())
    total_total = sum(p["total"] for p in a_per_pair.values())
    if total_total > 0:
        prop = total_hits / total_total
        ci_lo, ci_hi = wilson_ci(total_hits, total_total)
        # Binomial test vs 0.5
        pval = stats.binomtest(
            total_hits, total_total, p=0.5, alternative="two-sided"
        ).pvalue
        a_cluster = study_cluster_stats(raters_pass, "A_best_of_n")
        print(
            f"  pooled: {total_hits}/{total_total} = {prop:.3f} "
            f"(Wilson CI [{ci_lo:.3f}, {ci_hi:.3f}], binomial p = {pval:.4g})"
        )
        print(
            "  clustered: "
            f"per-rater t p = {a_cluster['per_rater_t_p']:.4g}, "
            f"per-pair t p = {a_cluster['per_pair_t_p']:.4g}, "
            f"pair-bootstrap CI = "
            f"[{a_cluster['pair_cluster_bootstrap_ci'][0]:.3f}, "
            f"{a_cluster['pair_cluster_bootstrap_ci'][1]:.3f}]"
        )
        for seed, p in a_per_pair.items():
            pr = p["hits"] / p["total"] if p["total"] else 0
            print(f"  {seed:20s}: {p['hits']}/{p['total']} = {pr:.3f}")

    # === B. α-steering (single pair, descriptive) ===
    print("\n=== B. α-steering (descriptive — n=1 pair) ===")
    b_hits = sum(
        1
        for r in raters_pass
        for resp in r["responses"]
        if resp["study"] == "B_alpha_steering" and resp["chose_winner"]
    )
    b_total = sum(
        1
        for r in raters_pass
        for resp in r["responses"]
        if resp["study"] == "B_alpha_steering"
    )
    if b_total > 0:
        b_ci_lo, b_ci_hi = wilson_ci(b_hits, b_total)
        pval_b = stats.binomtest(b_hits, b_total, p=0.5, alternative="two-sided").pvalue
        print(
            f"  picked α=+10: {b_hits}/{b_total} = {b_hits/b_total:.3f}  "
            f"(Wilson CI [{b_ci_lo:.3f}, {b_ci_hi:.3f}], binomial p = {pval_b:.4f})"
        )

    # === C. Persona winner ===
    print("\n=== C. Persona winner (did humans pick persona_A's choice?) ===")
    c_per_pair = defaultdict(lambda: {"hits": 0, "total": 0})
    for r in raters_pass:
        for resp in r["responses"]:
            if resp["study"] != "C_persona_winner":
                continue
            key = (resp["seed"], resp["persona_A"], resp["persona_B"])
            c_per_pair[key]["total"] += 1
            if resp["chose_winner"]:
                c_per_pair[key]["hits"] += 1

    for (seed, pA, pB), p in c_per_pair.items():
        pr = p["hits"] / p["total"] if p["total"] else 0
        print(f"  {seed} ({pA} vs {pB}): {p['hits']}/{p['total']} = {pr:.3f}")
    c_total_hits = sum(p["hits"] for p in c_per_pair.values())
    c_total_total = sum(p["total"] for p in c_per_pair.values())
    c_cluster = study_cluster_stats(raters_pass, "C_persona_winner")
    if c_total_total:
        c_pval = stats.binomtest(
            c_total_hits, c_total_total, p=0.5, alternative="two-sided"
        ).pvalue
        print(
            f"  pooled: {c_total_hits}/{c_total_total} = {c_total_hits / c_total_total:.3f} "
            f"(binomial p = {c_pval:.4g}; per-pair t p = {c_cluster['per_pair_t_p']:.4g})"
        )

    out = {
        "n_raters_total": len(raters),
        "n_raters_passed_attention": len(raters_pass),
        "study_A": {
            "pooled_hits": total_hits,
            "pooled_total": total_total,
            "proportion": total_hits / total_total if total_total else None,
            "wilson_ci_95": wilson_ci(total_hits, total_total) if total_total else None,
            "binomial_p_two_sided": pval if total_total else None,
            "clustered": a_cluster if total_total else None,
            "per_seed": dict(a_per_pair),
        },
        "study_B": {
            "hits": b_hits,
            "total": b_total,
            "proportion": b_hits / b_total if b_total else None,
            "wilson_ci_95": wilson_ci(b_hits, b_total) if b_total else None,
            "binomial_p": pval_b if b_total else None,
        },
        "study_C": {
            "pooled_hits": c_total_hits,
            "pooled_total": c_total_total,
            "proportion": c_total_hits / c_total_total if c_total_total else None,
            "binomial_p": c_pval if c_total_total else None,
            "clustered": c_cluster if c_total_total else None,
            "per_pair": {f"{s}|{a}|{b}": v for (s, a, b), v in c_per_pair.items()},
        },
    }
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
