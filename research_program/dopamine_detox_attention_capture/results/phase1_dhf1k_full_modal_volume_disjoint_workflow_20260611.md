# Phase 1 Modal-Volume Capture-Score Workflow

## Verdict

- Mechanical ready: True
- Claim update allowed: True
- Claim validated: False
- Scoring executed: True
- Decision reason: claim_ready_gate_failed
- Dataset: `DHF1K`
- Ground truth: `mean_fixation_density`
- Modal volume: `attention-capture-features-v1`
- Output prefix: `attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610`
- Scored rows: 350
- Claim boundary: This workflow scores ROI aggregates from full-mode TRIBE features already stored in a Modal Volume. It tests the preregistered H2 gate for this dataset only; publication-strength claims still require an additional claim-ready real dataset.

## Blocking Reasons

- none

## Feature Audit

- Existing features: 350
- Missing features: 0
- Feature errors: 0
- Valid capture denominators: 312
- Invalid capture denominators: 38

# Phase 1 Capture-Score Dry Run

## Gate

- Manifest status: real_external_attention_labels
- Claim update allowed: True
- Rule: capture_score Spearman rho >= 0.40 in at least one dataset
- Mechanical gate passed: False
- Claim validated: False
- Passed groups: none
- Samples: 350
- Invalid capture denominators: 38

## Correlations

| group | metric | n | Spearman rho | permutation p (greater) | gate |
|---|---|---:|---:|---:|---|
| DHF1K | capture_score | 312 | -0.0264 | 0.7070 |  |
| DHF1K | capture_delta | 350 | 0.0224 | 0.3460 |  |
| DHF1K | sensory_mean | 350 | -0.1010 | 0.9670 |  |
| DHF1K | frontoparietal | 350 | -0.1051 | 0.9720 |  |
| pooled | capture_score | 312 | -0.0264 | 0.6780 |  |
| pooled | capture_delta | 350 | 0.0224 | 0.3260 |  |
| pooled | sensory_mean | 350 | -0.1010 | 0.9660 |  |
| pooled | frontoparietal | 350 | -0.1051 | 0.9830 |  |

## Interpretation

This is a pipeline and metric dry run. It validates that the manifest, ROI score, correlation, and gate machinery are working; it does not validate attention capture unless the manifest contains real external ground-truth labels.
