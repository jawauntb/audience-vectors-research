# Phase 1 Capture-Score Dry Run

## Gate

- Manifest status: real_control_not_attention_capture
- Claim update allowed: False
- Rule: capture_score Spearman rho >= 0.40 in at least one dataset
- Mechanical gate passed: False
- Claim validated: False
- Passed groups: none
- Samples: 1022
- Invalid capture denominators: 362

## Correlations

| group | metric | n | Spearman rho | permutation p (greater) | gate |
|---|---|---:|---:|---:|---|
| BOLD_Moments_control | capture_score | 660 | -0.1988 | 1.0000 |  |
| BOLD_Moments_control | capture_delta | 1022 | -0.2033 | 1.0000 |  |
| BOLD_Moments_control | sensory_mean | 1022 | -0.1037 | 1.0000 |  |
| BOLD_Moments_control | frontoparietal | 1022 | 0.1935 | 0.0010 |  |
| pooled | capture_score | 660 | -0.1988 | 1.0000 |  |
| pooled | capture_delta | 1022 | -0.2033 | 1.0000 |  |
| pooled | sensory_mean | 1022 | -0.1037 | 1.0000 |  |
| pooled | frontoparietal | 1022 | 0.1935 | 0.0010 |  |

## Interpretation

This is a pipeline and metric dry run. It validates that the manifest, ROI score, correlation, and gate machinery are working; it does not validate attention capture unless the manifest contains real external ground-truth labels.

## Control Note

- Control: BOLD Moments memorability control
- Ground truth: memorability_score
- ROI source: research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_20260608.npz
- Interpretation: Control-only run. A positive result would show overlap with memorability labels, not validation of attentional capture.
