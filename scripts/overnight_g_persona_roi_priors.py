"""(G) Persona ROI anatomical-prior consistency.

For each persona, we have:
  - top-K Destrieux regions ranked by direction-vector energy (from roi_decomposition.json)

For each persona, we have an a priori expectation of WHERE its top ROIs
should land based on the persona description:
  - cinematic/aesthetic personas → temporal + scene-memory regions (parahippocampal,
    occipito-temporal)
  - fast-scroll/visual-salience personas → early visual cortex (calcarine, cuneus)
  - emotional personas → ventral/temporal (fusiform face, STS) + anterior cingulate
  - technical/spec personas → parietal (intraparietal sulcus) + dorsolateral PFC
  - narrative/lore personas → temporal-parietal junction, precuneus
  - audio-driven personas → superior temporal gyrus

Quantify: for each persona, what fraction of its top-5 ROIs match the
predicted anatomical region family? Compare to a random-shuffle baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


# Each persona maps to a set of "expected" Destrieux region keyword patterns
PERSONA_EXPECTATIONS = {
    "frame-poet-cleo":      ["parieto_occipital", "temporal", "precuneus", "fusiform", "occip"],
    "swipe-king-zara":      ["calcarine", "cuneus", "lingual", "occip", "V1"],
    "tearjerker-theo":      ["fusiform", "temporal_sup", "cingul", "amygdala", "insula", "temporal"],
    "spec-sheet-sam":       ["parietal", "intraparietal", "supramarginal", "angular", "front", "parieto"],
    "ad-blocker-priya":     ["front_inf", "frontal_inf", "front_mid", "orbital", "cingul"],
    "bass-drop-reyna":      ["temporal_sup", "transverse_temporal", "heschl", "planum", "temporal"],
    "deep-dive-felix":      ["front_mid", "front_sup", "parietal", "intraparietal", "precuneus"],
    "drama-thread-nico":    ["temporal_sup", "temporal", "junction", "angular", "precuneus", "cingul"],
    "giggle-loop-mara":     ["fusiform", "temporal_sup", "front_inf", "cingul", "orbital"],
    "golden-hour-vance":    ["parieto_occipital", "fusiform", "lingual", "occip", "temporal"],
    "highlight-hunter-dex": ["motion", "MT", "parietal", "occip", "precuneus", "calcarine"],
    "lore-keeper-syd":      ["temporal_sup", "temporal", "precuneus", "angular", "junction", "parieto"],
    "global":               ["precuneus", "parieto_occipital", "temporal", "cingul"],
}


def hit_score(region_name: str, patterns: list[str]) -> bool:
    rn = region_name.lower().replace("g_", "").replace("s_", "")
    return any(p.lower() in rn for p in patterns)


def main() -> None:
    roi = json.loads(Path("data/reports/roi_decomposition.json").read_text())
    rankings = roi["rankings"]
    all_regions = []
    for rs in rankings.values():
        for r in rs:
            all_regions.append(r["region"])
    all_regions = list(set(all_regions))
    print(f"[roi-prior] {len(all_regions)} unique Destrieux regions")

    # Score each persona
    results = {}
    rng = np.random.default_rng(0)

    for persona, expected in PERSONA_EXPECTATIONS.items():
        key = "BMD_memorability" if persona == "global" else persona
        if key not in rankings:
            print(f"[roi-prior] skipping {persona} (no ranking)")
            continue
        top10 = rankings[key][:10]
        top5 = top10[:5]

        # How many of top-5 match expected patterns?
        hits_top5 = sum(1 for r in top5 if hit_score(r["region"], expected))
        hits_top10 = sum(1 for r in top10 if hit_score(r["region"], expected))

        # Random baseline: sample 5 regions randomly, count hits
        n_random = 1000
        random_hits = np.zeros(n_random, dtype=np.int32)
        for i in range(n_random):
            sample = rng.choice(all_regions, size=5, replace=False)
            random_hits[i] = sum(1 for r in sample if hit_score(r, expected))

        results[persona] = {
            "expected_patterns": expected,
            "top5_regions": [r["region"] for r in top5],
            "hits_top5": int(hits_top5),
            "hits_top10": int(hits_top10),
            "random_baseline_top5_mean": float(random_hits.mean()),
            "random_baseline_top5_p99": float(np.quantile(random_hits, 0.99)),
            "p_value": float((random_hits >= hits_top5).mean()),
        }
        p = results[persona]["p_value"]
        print(f"  {persona:22s}: top-5 hits {hits_top5}/5, random μ = {random_hits.mean():.2f}, p = {p:.3f}")

    # Aggregate
    mean_hits = np.mean([r["hits_top5"] for r in results.values()])
    mean_random = np.mean([r["random_baseline_top5_mean"] for r in results.values()])
    n_sig = sum(1 for r in results.values() if r["p_value"] < 0.05)
    print(f"\n[roi-prior] across {len(results)} personas:")
    print(f"  mean hits/5 = {mean_hits:.2f}  vs random {mean_random:.2f}")
    print(f"  {n_sig}/{len(results)} personas have p < 0.05 ROI–prior match")

    out = {
        "n_personas": len(results),
        "per_persona": results,
        "mean_hits_top5": float(mean_hits),
        "mean_random_baseline": float(mean_random),
        "n_significant_p05": int(n_sig),
    }
    Path("data/reports/persona_roi_priors.json").write_text(json.dumps(out, indent=2))
    print("[roi-prior] done — wrote data/reports/persona_roi_priors.json")


if __name__ == "__main__":
    main()
