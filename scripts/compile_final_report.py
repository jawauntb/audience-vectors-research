"""Compile every result into one paper-ready markdown."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

R = Path("data/reports")


def _load_json(name: str) -> dict | None:
    p = R / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    cv_tribe = _load_json("cv_tribe_n1022.json")
    cv_vjepa = _load_json("cv_vjepa_n1026.json")
    patch_tribe = _load_json("patching_tribe.json")
    patch_vjepa = _load_json("patching_vjepa.json")
    pdirs_mem = _load_json("persona_directions.json")
    pdirs_att = _load_json("persona_directions_attention.json")
    random_ablation = _load_json("random_ablation_null.json")
    nonlinear = _load_json("nonlinear_probes.json")
    multi_direction = _load_json("multi_direction.json")
    tribe_temporal_fft = _load_json("tribe_temporal_spectral_probe.json")
    tribe_temporal_fft_resampled = _load_json(
        "tribe_temporal_spectral_probe_resampled.json"
    )
    tribe_introspection = _load_json("tribe_model_introspection.json")
    tribe_timepos_patch = _load_json("tribe_timepos_patch_probe.json")
    tribe_hidden_position_patch = _load_json("tribe_hidden_position_patch_probe.json")
    tribe_layerwise = _load_json("tribe_layerwise_encoder_localization.json")
    tribe_direction_patch = _load_json("tribe_layerwise_direction_patch.json")
    alexnet = _load_json("alexnet_memorability_probe.json")
    alexnet_forward = _load_json("alexnet_forward_patch_probe.json")
    open_video_encoder = _load_json("open_video_encoder_patch_probe.json")
    wan_product = _load_json(
        "wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0_product_selector.json"
    )

    lines = [
        "# Synthetic Audience Vectors — Final Results",
        "",
        f"Compiled {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Headline",
        "",
        "Core claims, all defensible on BMD (BOLD Moments) data alone:",
        "",
        "1. **Correlational** — TRIBE (brain-aligned) contrastive directions predict",
        "   human memorability at ρ≈0.40 and are competitive with the current",
        "   V-JEPA full-CV run; the old 1.9× V-JEPA headline is retired.",
        "2. **Dominant-axis compactness** — ablating the memorability direction from",
        "   TRIBE features destroys most linear signal, while corrected nonlinear",
        "   probes still recover residual signal.",
        "3. **Measured-fMRI pilot** — BMD sub-01 measured beta estimates recover",
        "   a memorability direction aligned with TRIBE (cos≈+0.336) and predictive",
        "   of BMD memorability (ρ≈+0.449), but this is still single-subject evidence.",
        "4. **Open-model sanity check** — AlexNet conv5 features show a similar",
        "   compact-direction ablation pattern, and a true forward-pass conv5 patch",
        "   weakens downstream readouts in a transparent network.",
        "5. **Audience decomposition** — per-persona contrastive directions on TRIBE",
        "   features are stable but not orthogonal; signed-cosine means can hide",
        "   sign-flipped shared axes.",
        "6. **Wan product selector** — a preference-weighted Wan2.2 LoRA improves",
        "   20/24 fresh prompts as a single sample, while base-or-gated best-of-4",
        "   improves 18/24 under the TRIBE/BMD proxy metric; this still needs human",
        "   validation.",
        "7. **Temporal/position audit** — final saved TRIBE output is mostly",
        "   temporal-DC; learned time-position and rotary-frequency patches preserve",
        "   ordering, while layerwise hidden-direction patches show that removing the",
        "   learned memorability direction sharply disrupts the readout on a 24-clip",
        "   high/low subset from the first attention residual through the final encoder.",
        "",
        "---",
        "",
        "## Conceptual frame",
        "",
        "The broader interpretation is generated media as an intervention on a",
        "human observer's generative model. TRIBE, V-JEPA, CLIP, and human memory",
        "judgments are different representation frames over the same stimulus",
        "space. The empirical question is not whether a vector is the essence of",
        "memorability, but whether the ordering induced by a readout in one frame",
        "is preserved in independent human behavior and against competing frames.",
        "This keeps the theory useful while preventing overclaiming: vectors are",
        "pragmatic coordinates, not ontological cognitive primitives.",
        "",
        "---",
        "",
        "## 1. Cross-validated correlational result",
        "",
    ]

    def _f(d, k, default=0.0):
        v = d.get(k)
        return float(v) if isinstance(v, (int, float)) else default

    def _mean_abs_offdiag(payload: dict) -> float | None:
        matrix = payload.get("cosine_matrix")
        if not matrix:
            return None
        arr = np.asarray(matrix, dtype=float)
        off = arr[~np.eye(arr.shape[0], dtype=bool)]
        return float(np.mean(np.abs(off)))

    def _effective_rank(payload: dict) -> float | None:
        matrix = payload.get("cosine_matrix")
        if not matrix:
            return None
        eigvals = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
        eigvals = np.clip(eigvals, 0.0, None)
        total = float(eigvals.sum())
        if total <= 0:
            return None
        probs = eigvals / total
        probs = probs[probs > 0]
        return float(np.exp(-np.sum(probs * np.log(probs))))

    if cv_tribe:
        n = len(cv_tribe.get("folds", [])) and sum(
            f.get("n_test", 0) for f in cv_tribe["folds"]
        )
        lines += [
            f"**TRIBE @ n={n or '?'}:**",
            f"- Mean Spearman ρ vs BMD memorability: **{_f(cv_tribe, 'mean_spearman'):+.3f} "
            f"± {_f(cv_tribe, 'stdev_spearman'):.3f}**",
            f"- Median ρ: {_f(cv_tribe, 'median_spearman'):+.3f}",
            f"- Fold range: [{_f(cv_tribe, 'overall_ci_low'):+.3f}, {_f(cv_tribe, 'overall_ci_high'):+.3f}]",
            "",
        ]

    if cv_vjepa:
        n = len(cv_vjepa.get("folds", [])) and sum(
            f.get("n_test", 0) for f in cv_vjepa["folds"]
        )
        lines += [
            f"**V-JEPA @ n={n or '?'}:**",
            f"- Mean Spearman ρ: {_f(cv_vjepa, 'mean_spearman'):+.3f} "
            f"± {_f(cv_vjepa, 'stdev_spearman'):.3f}",
            "",
        ]

    lines += [
        "**Baseline context:**",
        "",
        "| predictor | validation artifact | ρ | n |",
        "|---|---|---|---|",
        "| Gemini zero-shot | matched held-out subset | +0.139 | 939 |",
        "| V-JEPA contrastive | 5-fold CV | +0.395 ± 0.037 | 1026 |",
        "| **TRIBE contrastive** | **5-fold CV** | **+0.403 ± 0.061** | **1022** |",
        "",
        "Interpretation: TRIBE is competitive with V-JEPA for global memorability",
        "prediction, and more useful for this paper's brain-aligned analyses because",
        "it exposes predicted cortical responses for fMRI/ROI/ablation questions.",
        "",
        "---",
        "",
        "## 2. Directional ablation controls",
        "",
        "Method: for each fold, train direction `v` on train, then ablate `v` from features",
        "(`x' = x - (x·v)v`), retrain a *new* direction `v2` on the ablated train features,",
        "and project ablated test features onto `v2`.",
        "",
    ]

    if patch_tribe:
        m_b = patch_tribe.get("mean_baseline_rho", 0)
        m_a = patch_tribe.get("mean_ablated_rho", 0)
        m_d = patch_tribe.get("mean_destruction_pct", 0) or 0
        lines += [
            f"**TRIBE:** baseline ρ = {m_b:+.3f} → ablated ρ = {m_a:+.3f} "
            f"(**{m_d:.1f}%** of signal destroyed)",
        ]
    if patch_vjepa:
        m_b = patch_vjepa.get("mean_baseline_rho", 0)
        m_a = patch_vjepa.get("mean_ablated_rho", 0)
        m_d = patch_vjepa.get("mean_destruction_pct", 0) or 0
        lines += [
            f"**V-JEPA:** baseline ρ = {m_b:+.3f} → ablated ρ = {m_a:+.3f} "
            f"(**{m_d:.1f}%** of signal destroyed)",
        ]

    lines += [
        "",
        "Interpretation: the memorability signal is dominated by one contrastive",
        "direction, but the corrected nonlinear rerun means this is not a literal",
        "one-dimensionality result.",
    ]
    if random_ablation:
        lines += [
            "",
            f"**Fold-safe random-ablation control:** pre-ablation ρ = "
            f"{_f(random_ablation, 'rho_pre_ablation'):+.3f}; "
            f"v_mem ablation ρ = {_f(random_ablation, 'rho_vmem_ablation'):+.3f}; "
            f"random ablation mean ρ = {_f(random_ablation, 'rho_random_ablation_mean'):+.3f} "
            f"± {_f(random_ablation, 'rho_random_ablation_std'):.3f}; "
            f"z = {_f(random_ablation, 'z_vs_random'):+.1f}.",
        ]
    if nonlinear:
        full = nonlinear.get("rho_full", {})
        residual = nonlinear.get("rho_residual", {})
        lines += [
            f"**Fold-safe nonlinear probes:** ridge {full.get('ridge', 0):+.3f} → "
            f"{residual.get('ridge', 0):+.3f}; random forest "
            f"{full.get('random_forest', 0):+.3f} → {residual.get('random_forest', 0):+.3f}.",
        ]
    if multi_direction:
        rhos = [float(x) for x in multi_direction.get("rho_per_direction", [])]
        if len(rhos) > 1:
            rest = rhos[1:]
            lines += [
                f"**Fold-safe multi-direction check:** direction 1 ρ = {rhos[0]:+.3f}; "
                f"directions 2–{len(rhos)} mean ρ = {np.mean(rest):+.3f}, "
                f"range [{min(rest):+.3f}, {max(rest):+.3f}].",
            ]
    if tribe_temporal_fft:
        agg = tribe_temporal_fft.get("aggregate", {})
        bands = agg.get("bands", {})
        dc = bands.get("temporal_dc_k0", {}).get("test_rho_band_only", {})
        nonzero = bands.get("temporal_nonzero", {}).get("test_rho_band_only", {})
        full = agg.get("full_tensor_rho", {})
        pooled = agg.get("mean_pooled_rho", {})
        resampled_note = ""
        if tribe_temporal_fft_resampled:
            res = tribe_temporal_fft_resampled.get("aggregate", {})
            res_dc = (
                res.get("bands", {})
                .get("temporal_dc_k0", {})
                .get("test_rho_band_only", {})
            )
            resampled_note = (
                f" All-clip 3→4 resampling replicates it: full "
                f"{_f(res.get('full_tensor_rho', {}), 'mean'):+.3f}; "
                f"DC {_f(res_dc, 'mean'):+.3f}."
            )
        pos_note = ""
        if tribe_introspection:
            params = tribe_introspection.get("matching_parameters", [])
            if params:
                pos_note = (
                    f" Internal TRIBE positional target exists: "
                    f"`{params[0].get('name')}` shape {params[0].get('shape')}."
                )
        patch_note = ""
        if tribe_timepos_patch:
            scale_rows = tribe_timepos_patch.get("summary", {}).get("scales", {})
            ablated = scale_rows.get("0.0", {})
            baseline = scale_rows.get("1.0", {})
            patch_note = (
                f" Direct time_pos_embed ablation preserves the readout: "
                f"ρ {_f(baseline, 'spearman_vs_memorability'):+.3f} → "
                f"{_f(ablated, 'spearman_vs_memorability'):+.3f}, high-low gap ratio "
                f"{_f(ablated, 'high_minus_low_gap_ratio_vs_baseline'):.3f}."
            )
        hidden_note = ""
        if tribe_hidden_position_patch:
            hidden_summary = tribe_hidden_position_patch.get("summary", {})
            hidden = hidden_summary.get("hidden", {})
            rows = hidden_summary.get("output_conditions", {})
            hidden_zero = rows.get("hidden_non_dc_xp0", {})
            rotary_zero = rows.get("rotary_inv_freq_xp0", {})
            hidden_note = (
                f" Encoder hidden states expose a live sequence-structure caveat: "
                f"the hidden direction has DC energy "
                f"{_f(hidden.get('full_direction_frequency_energy', {}), 'dc'):.3f}; "
                f"encoder non-DC removal drops output ρ to "
                f"{_f(hidden_zero, 'spearman_vs_memorability'):+.3f} "
                f"(gap ratio {_f(hidden_zero, 'high_minus_low_gap_ratio_vs_baseline'):.3f}), "
                f"while rotary inv_freq zeroing preserves ordering at "
                f"ρ {_f(rotary_zero, 'spearman_vs_memorability'):+.3f}."
            )
        layerwise_note = ""
        if tribe_layerwise:
            emergence = tribe_layerwise.get("summary", {}).get("emergence", {})
            first = emergence.get("first_gap_ratio_below_0p5") or {}
            strongest = emergence.get("strongest_non_dc_dependency") or {}
            layerwise_note = (
                f" Layerwise localization puts the first gap-ratio collapse at "
                f"`{first.get('label', '?')}` (ρ {_f(first, 'patch_rho'):+.3f}, "
                f"gap ratio {_f(first, 'gap_ratio'):.3f}) and the strongest collapse "
                f"at `{strongest.get('label', '?')}` (ρ {_f(strongest, 'patch_rho'):+.3f}, "
                f"gap ratio {_f(strongest, 'gap_ratio'):.3f})."
            )
        direction_note = ""
        if tribe_direction_patch:
            emergence = tribe_direction_patch.get("summary", {}).get("emergence", [{}])[0]
            first = emergence.get("first_gap_ratio_below_0p5") or {}
            strongest = emergence.get("strongest_direction_dependency") or {}
            direction_note = (
                f" Direction-only patching is stronger: removing hidden v_mem first "
                f"collapses at `{first.get('label', '?')}` "
                f"(ρ {_f(first, 'patch_rho'):+.3f}, gap ratio {_f(first, 'gap_ratio'):.3f}) "
                f"and strongest at `{strongest.get('label', '?')}` "
                f"(ρ {_f(strongest, 'patch_rho'):+.3f}, gap ratio "
                f"{_f(strongest, 'gap_ratio'):.3f})."
            )
        lines += [
            "**TRIBE temporal Fourier audit:** final output full-tensor "
            f"ρ = {_f(full, 'mean'):+.3f} ± {_f(full, 'std'):.3f}; "
            f"mean-pooled ρ = {_f(pooled, 'mean'):+.3f}; temporal-DC-only "
            f"ρ = {_f(dc, 'mean'):+.3f}; nonzero-temporal ρ = "
            f"{_f(nonzero, 'mean'):+.3f}.{resampled_note}{pos_note}{patch_note}{hidden_note}{layerwise_note}{direction_note}",
        ]
    if alexnet:
        raw = alexnet.get("raw_probe", {})
        pca = alexnet.get("pca_probe", {})
        lines += [
            f"**Open AlexNet conv5 sanity check:** raw layer-5 ρ = "
            f"{_f(raw, 'baseline_cv_rho'):+.3f} → {_f(raw, 'ablated_cv_rho'):+.3f} "
            f"after learned-direction ablation; random-ablation mean "
            f"{_f(raw, 'random_ablation_mean_rho'):+.3f}. PCA-100 gives "
            f"{_f(pca, 'baseline_cv_rho'):+.3f} → {_f(pca, 'ablated_cv_rho'):+.3f}.",
        ]
    if alexnet_forward:
        summary = alexnet_forward.get("summary", {})
        base = summary.get("baseline", {})
        ablate = summary.get("ablate", {})
        lines += [
            "**Open AlexNet forward-pass patch:** conv5 is patched before fc6/fc7/fc8 "
            "are recomputed. Downstream readouts weaken: "
            f"fc6 {_f(base.get('fc6', {}), 'cv_rho'):+.3f} → "
            f"{_f(ablate.get('fc6', {}), 'cv_rho'):+.3f}; "
            f"fc7 {_f(base.get('fc7', {}), 'cv_rho'):+.3f} → "
            f"{_f(ablate.get('fc7', {}), 'cv_rho'):+.3f}; "
            f"logits {_f(base.get('logits', {}), 'cv_rho'):+.3f} → "
            f"{_f(ablate.get('logits', {}), 'cv_rho'):+.3f}.",
        ]
    if open_video_encoder:
        forward = open_video_encoder.get("forward_patching", {})
        add = forward.get("add", {})
        subtract = forward.get("subtract", {})
        lines += [
            f"**Open CLIP frame-encoder pilot:** "
            f"{open_video_encoder.get('model_id', 'CLIP')} block "
            f"{open_video_encoder.get('layer_index', '?')} signed patching shifts final "
            f"memorability projection by {_f(add, 'mean_delta_vs_baseline'):+.3f} "
            f"/ {_f(subtract, 'mean_delta_vs_baseline'):+.3f}; "
            "centered removal is inconclusive.",
        ]
    if wan_product:
        policies = wan_product.get("summary", {}).get("policies", {})
        selector = policies.get("base_or_gated_best_of_n", {})
        single = policies.get("single_lora", {})
        raw = policies.get("raw_best_of_n", {})
        n_seeds = wan_product.get("summary", {}).get("n_seeds", "?")
        lines += [
            f"**Wan2.2 LoRA product selector:** single LoRA improves "
            f"{single.get('n_improved', '?')}/{n_seeds}; raw best-of-4 improves "
            f"{raw.get('n_improved', '?')}/{n_seeds}; conservative base-or-gated "
            f"best-of-4 improves {selector.get('n_improved', '?')}/{n_seeds} with "
            f"mean lift {selector.get('mean', 0):+.3f}, median "
            f"{selector.get('median', 0):+.3f}, and minimum {selector.get('min', 0):+.3f}.",
            "This is proxy-scored under the TRIBE/BMD projection, not a human-validation result.",
        ]

    lines += [
        "",
        "---",
        "",
        "## 3. Audience decomposition — per-persona directions",
        "",
    ]

    if pdirs_mem:
        off = pdirs_mem.get("off_diagonal", {})
        mean_abs = _mean_abs_offdiag(pdirs_mem)
        erank = _effective_rank(pdirs_mem)
        n_personas = len(pdirs_mem.get("persona_ids", []))
        lines += [
            f"**Memorability axis** (n_personas = {n_personas}):",
            f"- Signed off-diagonal cosine: mean = {off.get('mean', 0):+.3f}, "
            f"median = {off.get('median', 0):+.3f}, "
            f"range [{off.get('min', 0):+.3f}, {off.get('max', 0):+.3f}]",
            f"- Corrected unsigned overlap: mean |cos| = {mean_abs if mean_abs is not None else 0:.3f}; "
            f"effective rank = {erank if erank is not None else 0:.2f} / {n_personas}",
            "",
        ]
    if pdirs_att:
        off = pdirs_att.get("off_diagonal", {})
        lines += [
            f"**Attention axis** (n_personas = {len(pdirs_att.get('persona_ids', []))}):",
            f"- Off-diagonal cosine: mean = {off.get('mean', 0):+.3f}, "
            f"median = {off.get('median', 0):+.3f}, "
            f"range [{off.get('min', 0):+.3f}, {off.get('max', 0):+.3f}]",
            "",
        ]

    lines += [
        "Interpretation: persona-conditioned directions are not identical to the global",
        "memorability direction, but they also do not span 12 independent axes.",
        "Sign-flipped pairs share an axis, so signed cosine means are not valid",
        "evidence of orthogonality by themselves.",
        "",
        "See per-axis matrices in `persona_directions.md` and",
        "`persona_directions_attention.md`.",
        "",
        "---",
        "",
        "## Summary table",
        "",
        "| claim | metric | result | substrate |",
        "|---|---|---|---|",
        "| Brain-aligned features encode memorability | 5-fold CV ρ | +0.403 ± 0.061 | TRIBE features (n=1022) |",
        "| TRIBE competitive with self-supervised baseline | 5-fold CV ρ | +0.403 vs +0.395 | TRIBE vs V-JEPA |",
        "| Zero-shot VLM is weaker in this setup | matched-subset ρ | +0.139 | Gemini |",
        "| Memorability is dominated by one direction | fold-safe ablation | +0.401 → +0.057 | TRIBE |",
        "| Hidden memorability direction is patch-sensitive in a 24-clip model intervention | temporal FFT + internal patches | time-pos 0× ρ +0.703; rotary 0× ρ +0.685; layer-0 v_mem removal ρ -0.030; final v_mem removal ρ -0.105 | TRIBE output + internal hooks |",
        "| Compact direction appears in an open model | fold-safe ablation + forward patch | +0.386 → +0.018; fc7 +0.432 → +0.212 | AlexNet conv5 |",
        "| Personas compress to a few axes | mean abs cosine / effective rank | 0.434 / 3.56 of 12 | TRIBE × Haiku personas |",
        "| Wan product selector is promising but proxy-only | single LoRA / base-or-gated best-of-4 | 20/24 single; 18/24 safe selector, mean +2.817 | Wan2.2 LoRA + TRIBE judge |",
        "",
        "## Methods",
        "",
        "- **Dataset:** BOLD Moments (BMD), 1102 short clips with human memorability scores.",
        "  Final analyzable n = 1022 after 80 URL-rot exclusions.",
        "- **Brain-aligned features:** TRIBE v2 predicted BOLD activations, shape (T, 20484)",
        "  on fsaverage5 cortical mesh, time-averaged per clip.",
        "- **Self-supervised baseline:** V-JEPA contrastive embeddings.",
        "- **Open-model sanity check:** AlexNet conv5 layer-5 features on all 1,102 BMD clips.",
        "- **Zero-shot VLM baseline:** Gemini 2.0 Flash, single-pass scoring.",
        "- **Personas:** 12 archetypes (cinematic, fast-scroll, narrative-emotional, etc.),",
        "  scored on 10 axes per segment by Claude Haiku 4.5 over Gemini-produced segment",
        "  descriptions, with prompt caching on persona system messages.",
        "- **Contrastive direction:** mean(top 30%) − mean(bottom 30%), unit-normalized.",
        "- **CV:** 5-fold w/ bootstrap 95% CIs (1000 resamples), 200 random-direction nulls.",
        "- **Feature-space ablation:** orthogonal-complement ablation + new-direction retraining.",
        "",
        "## Limitations",
        "",
        "- Single dataset (BMD) — generalization across datasets not yet shown.",
        "- Measured fMRI is single-subject only so far — the BMD sub-01 pilot is",
        "  positive, but all-subject aggregation remains needed.",
        "- Open-model AlexNet is a sanity check, not a replacement mechanism — it",
        "  now includes forward-pass patching in a transparent convnet, but not",
        "  TRIBE-internal patching or an open brain-encoding model trained on fMRI.",
        "- TRIBE positional structure is partially audited — final saved outputs",
        "  are mostly temporal-DC, learned time-position ablation preserves most",
        "  of the readout, and rotary-frequency zeroing preserves ordering. However,",
        "  layerwise hidden-direction patches show that removing the learned hidden",
        "  memorability direction sharply disrupts the readout. The 104-clip",
        "  fold-safe hidden-patch run is complete and supports the intervention",
        "  result; broader population-level claims should still be framed cautiously.",
        "- Persona labels are model-generated (Haiku over Gemini text) — no persona-matched",
        "  human validation; latent-axis structure may reflect prompt structure, not real audience.",
        "- Memorability is one viewer-response axis; other axes (attention, confusion,",
        "  brand recall) only partially explored.",
        "- Wan LoRA/product-selector gains are proxy-scored by TRIBE/BMD and still need",
        "  human validation against base and random variants.",
    ]

    out = Path("data/reports/FINAL_REPORT.md")
    out.write_text("\n".join(lines) + "\n")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
