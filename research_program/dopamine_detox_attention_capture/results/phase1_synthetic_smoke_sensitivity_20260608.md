# Phase 1 Capture-Score Sensitivity

## Setup

- Manifest: research_program/dopamine_detox_attention_capture/phase1_synthetic_smoke_manifest_20260608.json
- Primary: disjoint
- Gate rho: 0.40
- Claim boundary: Sensitivity compares ROI scoring policies for the same manifest. It does not validate attention capture without real external labels.

## Runs

| role | label | group | n valid | invalid denominators | capture rho | gate | claim validated |
|---|---|---|---:|---:|---:|---|---|
| primary | disjoint | pooled | 16 | 0 | 0.9882 | True | False |
| sensitivity | overlapping | pooled | 16 | 0 | 0.9882 | True | False |

## Sensitivity Delta

| group | sensitivity | primary rho | sensitivity rho | delta |
|---|---|---:|---:|---:|
| DHF1K_fixture | overlapping | 1.0000 | 1.0000 | 0.0000 |
| SnapUGC_fixture | overlapping | 1.0000 | 1.0000 | 0.0000 |
| pooled | overlapping | 0.9882 | 0.9882 | 0.0000 |
