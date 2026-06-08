# Phase 1 Capture-Score Dry Run

## Gate

- Manifest status: synthetic_smoke_only
- Claim update allowed: False
- Rule: capture_score Spearman rho >= 0.40 in at least one dataset
- Mechanical gate passed: True
- Claim validated: False
- Passed groups: DHF1K_fixture, SnapUGC_fixture
- Samples: 16
- Invalid capture denominators: 0

## Correlations

| group | metric | n | Spearman rho | permutation p (greater) | gate |
|---|---|---:|---:|---:|---|
| DHF1K_fixture | capture_score | 8 | 1.0000 | 0.0010 | pass |
| DHF1K_fixture | capture_delta | 8 | 1.0000 | 0.0010 |  |
| DHF1K_fixture | sensory_mean | 8 | 1.0000 | 0.0010 |  |
| DHF1K_fixture | frontoparietal | 8 | -1.0000 | 1.0000 |  |
| SnapUGC_fixture | capture_score | 8 | 1.0000 | 0.0010 | pass |
| SnapUGC_fixture | capture_delta | 8 | 1.0000 | 0.0010 |  |
| SnapUGC_fixture | sensory_mean | 8 | 1.0000 | 0.0010 |  |
| SnapUGC_fixture | frontoparietal | 8 | -1.0000 | 1.0000 |  |
| pooled | capture_score | 16 | 0.9882 | 0.0010 | pass |
| pooled | capture_delta | 16 | 0.9912 | 0.0010 |  |
| pooled | sensory_mean | 16 | 0.9845 | 0.0010 |  |
| pooled | frontoparietal | 16 | -0.9941 | 1.0000 |  |

## Interpretation

This is a pipeline and metric dry run. It validates that the manifest, ROI score, correlation, and gate machinery are working; it does not validate attention capture unless the manifest contains real external ground-truth labels.
