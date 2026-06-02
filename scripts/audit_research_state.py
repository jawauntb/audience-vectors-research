"""Write a compact critical audit of the current research artifacts."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

R = Path("data/reports")


def load_json(name: str) -> dict[str, Any]:
    return json.loads((R / name).read_text())


def fnum(value: float | int | None, digits: int = 3, signed: bool = True) -> str:
    if value is None:
        return "n/a"
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(float(value))


def pct(num: float, den: float, digits: int = 1) -> str:
    return f"{100.0 * num / den:.{digits}f}%"


def main() -> None:
    cv_tribe = load_json("cv_tribe_n1022.json")
    cv_vjepa = load_json("cv_vjepa_n1026.json")
    canonical = load_json("canonical_split.json")
    random_ablation = load_json("random_ablation_null.json")
    nonlinear = load_json("nonlinear_probes.json")
    multi_direction = load_json("multi_direction.json")
    null_perm = load_json("null_label_perm.json")
    stability = load_json("vmem_stability.json")
    temporal = load_json("temporal_v_mem.json")
    cross_domain = load_json("cross_domain.json")
    fmri = load_json("fmri_pilot_sub01.json")
    alexnet = load_json("alexnet_memorability_probe.json")
    alexnet_forward = load_json("alexnet_forward_patch_probe.json")
    open_encoder = load_json("open_video_encoder_patch_probe.json")
    persona_stability = load_json("persona_stability.json")
    roi = load_json("roi_decomposition.json")
    prolific = load_json("prolific_analysis.json")
    wan_product = load_json(
        "wan22_lora_eval_fresh_picsum_24_r16_s150_bon_24x4_s12_m1p0_product_selector.json"
    )
    adapter_t5 = load_json("adapter_tribe_to_t5.json")
    adapter_clip_h = load_json("adapter_tribe_to_clip_h.json")
    axis_decode = load_json("persona_axis_decode.json")

    tribe_cv = float(cv_tribe["mean_spearman"])
    vjepa_cv = float(cv_vjepa["mean_spearman"])
    vjepa_ratio = tribe_cv / vjepa_cv if vjepa_cv else float("nan")
    baseline_warning = abs(tribe_cv - vjepa_cv) < 0.03

    a = prolific["study_A"]
    c = prolific["study_C"]
    wan_policies = wan_product["summary"]["policies"]
    product = wan_policies["base_or_gated_best_of_n"]
    raw_bon = wan_policies["raw_best_of_n"]
    single = wan_policies["single_lora"]

    alex_raw = alexnet["raw_probe"]
    alex_summary = alexnet_forward["summary"]
    open_patch = open_encoder["forward_patching"]
    roi_matrix = np.asarray(roi["energy_cosine_matrix"], dtype=float)
    roi_offdiag = roi_matrix[~np.eye(roi_matrix.shape[0], dtype=bool)]

    audit = {
        "date": str(date.today()),
        "headline_numbers": {
            "tribe_cv_rho": tribe_cv,
            "vjepa_cv_rho": vjepa_cv,
            "tribe_over_vjepa_full_cv_ratio": vjepa_ratio,
            "canonical_test_rho": canonical["canonical_test_spearman"],
            "prolific_best_of_n_preference": a["proportion"],
            "wan24_base_or_gated_best_of_4_mean_lift": product["mean"],
        },
        "critical_flags": {
            "downgrade_tribe_beats_vjepa_claim": baseline_warning,
            "persona_axes_not_orthogonal": True,
            "ablation_is_dominant_axis_not_full_causality": True,
            "generation_steering_not_directly_solved": True,
            "wan24_is_proxy_scored_not_human_validated": True,
        },
    }

    md: list[str] = [
        "# Critical Research Audit",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## Reviewer-Level Claim Status",
        "",
        "| Claim family | Current status | Evidence | Required framing |",
        "|---|---|---|---|",
        (
            "| TRIBE memorability direction predicts BMD human memorability | Strong | "
            f"5-fold CV rho {fnum(tribe_cv)} +/- {float(cv_tribe['stdev_spearman']):.3f}; "
            f"canonical split rho {fnum(canonical['canonical_test_spearman'])}, "
            f"95% CI [{fnum(canonical['ci_95_low'])}, {fnum(canonical['ci_95_high'])}] | "
            "Keep as the core prediction result. |"
        ),
        (
            "| TRIBE clearly beats V-JEPA | Downgrade | "
            f"Full-CV V-JEPA rho is {fnum(vjepa_cv)}, TRIBE/V-JEPA ratio {vjepa_ratio:.2f}x; "
            "older small/matched baselines vary by feature coverage and split | "
            "Do not headline 1.9x V-JEPA. Say TRIBE is brain-aligned and comparable to V-JEPA on full CV, with stronger interpretability/neural framing. |"
        ),
        (
            "| One dominant memorability direction | Supported but not complete | "
            f"Fold-safe linear ablation {fnum(random_ablation['rho_pre_ablation'])} -> "
            f"{fnum(random_ablation['rho_vmem_ablation'])}; random ablations leave "
            f"{fnum(random_ablation['rho_random_ablation_mean'])}; nonlinear RF residual "
            f"{fnum(nonlinear['rho_residual']['random_forest'])} | "
            "Call it dominant-axis compactness, not total one-dimensionality or proof of causal sufficiency. |"
        ),
        (
            "| Orthogonal multi-direction decomposition | Mostly negative | "
            f"Direction 1 rho {fnum(multi_direction['rho_per_direction'][0])}; "
            f"directions 2-10 mean {fnum(float(np.mean(multi_direction['rho_per_direction'][1:])))}, "
            f"range [{fnum(float(np.min(multi_direction['rho_per_direction'][1:])))}, "
            f"{fnum(float(np.max(multi_direction['rho_per_direction'][1:])))}] | "
            "Use as evidence against a simple multi-axis linear decomposition after v_mem. |"
        ),
        (
            "| Persona directions are independent axes | False as originally phrased | "
            f"Signed off-diagonal mean {fnum(persona_stability['mean_off_diagonal_persona_cos'])}; "
            f"mean abs cosine {persona_stability['mean_abs_off_diagonal_persona_cos']:.3f}; "
            f"effective rank {persona_stability['effective_rank']:.2f}/12 | "
            "Say personas compress to roughly four latent signed axes. Sign-flipped pairs share an axis. |"
        ),
        (
            "| Persona axes are interpretable | Exploratory | "
            f"Top four SVD components explain {100*axis_decode['svd_cumulative_variance_ratio'][3]:.1f}% of persona-direction variance; "
            "RSA links components to social/visual/attention/emotion constructs | "
            "Frame as a decoding hypothesis, not validated psychology. |"
        ),
        (
            "| ROI localization supports persona differences | Weak-to-moderate | "
            f"ROI energy profiles have high overlap: off-diagonal energy cosine mean {float(roi_offdiag.mean()):.3f} | "
            "Keep ROI as exploratory localization. Avoid implying clean neural modules per persona. |"
        ),
        (
            "| Measured fMRI independently supports v_mem | Promising pilot | "
            f"sub-01 cos(measured, TRIBE) {fnum(fmri['cos_measured_vs_tribe'])}; "
            f"measured CV rho {fnum(fmri['cv_rho_measured'])} | "
            "Single-subject neural grounding only; needs all subjects. |"
        ),
        (
            "| Open-model patching supports load-bearing direction | Good sanity check | "
            f"AlexNet conv5 offline {fnum(alex_raw['baseline_cv_rho'])} -> {fnum(alex_raw['ablated_cv_rho'])}; "
            f"forward fc7 {fnum(alex_summary['baseline']['fc7']['cv_rho'])} -> {fnum(alex_summary['ablate']['fc7']['cv_rho'])}; "
            f"CLIP add/subtract shifts {fnum(open_patch['add']['mean_delta_vs_baseline'])}/"
            f"{fnum(open_patch['subtract']['mean_delta_vs_baseline'])} | "
            "Valid open-model mechanism check, not TRIBE-internal patching. |"
        ),
        (
            "| Best-of-N ranking transfers to humans | Supported for pooled general memorability | "
            f"Prolific Study A {a['pooled_hits']}/{a['pooled_total']} = {pct(a['pooled_hits'], a['pooled_total'])}; "
            f"pair-cluster CI [{a['clustered']['pair_cluster_bootstrap_ci'][0]:.3f}, "
            f"{a['clustered']['pair_cluster_bootstrap_ci'][1]:.3f}] | "
            "Keep, but mention only 4/11 pairs individually significant and two reversal seeds. |"
        ),
        (
            "| Persona-pair winners validate audience decomposition | Not clean | "
            f"Study C pooled {c['pooled_hits']}/{c['pooled_total']} = {pct(c['pooled_hits'], c['pooled_total'])}; "
            f"pair-cluster CI [{c['clustered']['pair_cluster_bootstrap_ci'][0]:.3f}, "
            f"{c['clustered']['pair_cluster_bootstrap_ci'][1]:.3f}] | "
            "Useful preference signal, but not persona-matched raters. Avoid claiming true persona validation. |"
        ),
        (
            "| Direct alpha steering solves generation | No | "
            "Multi-seed alpha steering is not significant; alpha does not compound with best-of-N | "
            "Say direct steering is early/seed-dependent and currently not product-ready. |"
        ),
        (
            "| Wan LoRA gives product-useful steering | Proxy-useful, not validated | "
            f"Single LoRA improves {single['n_improved']}/24, mean {fnum(single['mean'])}; "
            f"raw best-of-4 improves {raw_bon['n_improved']}/24, mean {fnum(raw_bon['mean'])}; "
            f"base-or-gated best-of-4 improves {product['n_improved']}/24, mean {fnum(product['mean'])}, "
            f"median {fnum(product['median'])} | "
            "Present as a runnable proxy-guided selector, not proof humans find LoRA outputs more memorable. |"
        ),
    ]

    md += [
        "",
        "## Mathematical Sanity Checks",
        "",
        f"- Label-permutation null: actual rho {fnum(null_perm['rho_actual'])}, "
        f"perm mean {fnum(null_perm['perm_rho_mean'])}, std {null_perm['perm_rho_std']:.3f}, "
        f"z {null_perm['z_score']:.2f}, empirical one-sided p {null_perm['p_value_one_sided']}.",
        f"- Stability: 80% subsample pairwise cos {stability['pairwise_cos_mean']:.3f}; "
        f"disjoint-halves cos {stability['disjoint_halves_cos_mean']:.3f}.",
        f"- Temporal direction stability: mean off-diagonal time-bin cos "
        f"{temporal['mean_off_diagonal_cosine']:.3f}; per-bin rho range "
        f"[{min(temporal['per_bin_cv_rho'].values()):+.3f}, {max(temporal['per_bin_cv_rho'].values()):+.3f}].",
        f"- Content-domain transfer remains weak: indoor->outdoor rho "
        f"{fnum(cross_domain['indoor_to_outdoor']['rho'])}; outdoor->indoor "
        f"{fnum(cross_domain['outdoor_to_indoor']['rho'])}; random 50/50 "
        f"{fnum(cross_domain['random_50_50']['rho_mean'])} +/- {cross_domain['random_50_50']['rho_std']:.3f}.",
        f"- Text-space adapter is a negative: TRIBE->T5 memorability alignment "
        f"{fnum(adapter_t5['cos_alignment'])}; CLIP-H image-space alignment "
        f"{fnum(adapter_clip_h['cos_alignment'])}.",
        "",
        "## Product Direction",
        "",
        "The strongest product story is no longer direct steering. It is a navigable selector:",
        "",
        "1. Generate base plus LoRA/best-of-N candidates.",
        "2. Score each candidate with the TRIBE/BMD memorability direction.",
        "3. Gate semantic drift with CLIP seed-image/prompt preservation.",
        "4. Keep the base clip whenever the gated LoRA candidate is worse.",
        "",
        f"On 24 fresh non-BMD seeds this gives {product['n_improved']}/24 improved seeds, "
        f"mean lift {fnum(product['mean'])}, median lift {fnum(product['median'])}, "
        "and no negative seed-level regressions by policy construction.",
        "",
        "## Edits Required Before Sharing",
        "",
        "- Replace the V-JEPA/Gemini overclaim in headline, abstract, summary table, README, and site cards.",
        "- Move from 'causal ablation' language to 'dominant-axis ablation/control suite'.",
        "- Explicitly state that persona directions are signed-axis clusters, not near-orthogonal independent axes.",
        "- Add Wan LoRA as a product-demo tab with a runnable proxy-selection example.",
        "- Keep alpha steering as a negative/limited result, not as a product promise.",
    ]

    R.mkdir(parents=True, exist_ok=True)
    (R / "critical_research_audit_2026-05-25.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    (R / "critical_research_audit_2026-05-25.md").write_text("\n".join(md) + "\n")
    print("[audit] wrote data/reports/critical_research_audit_2026-05-25.md")
    print("[audit] wrote data/reports/critical_research_audit_2026-05-25.json")


if __name__ == "__main__":
    main()
