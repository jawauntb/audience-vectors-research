# Phase 1 Data Readiness Audit

## Verdict

- Phase 1 can run now: True
- DHF1K root ready for label build: False
- DHF1K label audit ready: True
- DHF1K labels ready: True
- SnapUGC labels ready: False
- TRIBE features ready: False
- DHF1K TRIBE features ready: False
- ROI masks ready: True
- Real manifest ready: True
- Recommended next action: run scripts/run_attention_capture_phase1_workflow.py
- Claim boundary: This report audits local data availability only. It does not score TRIBE features or validate attentional capture.

## Blocking Reasons

- none

## DHF1K Candidates

| path | videos | map dirs | fixation dirs | ready |
|---|---:|---:|---:|---|
| none | 0 | 0 | 0 | False |

## DHF1K Label Audits

| path | labels CSV exists | rank column | rows | ready |
|---|---|---|---:|---|
| research_program/dopamine_detox_attention_capture/results/dhf1k_attention_label_audit_20260608.json | True | mean_map_intensity | 350 | True |
| research_program/dopamine_detox_attention_capture/results/dhf1k_attention_label_audit_fixation_density_20260609.json | True | mean_fixation_density | 350 | True |

## SnapUGC/VQualA Label Candidates

| path | rows | columns |
|---|---:|---|
| none | 0 | n/a |

## TRIBE Feature Cache Candidates

| path | npz files | sampled frames arrays | claim blocked | ready |
|---|---:|---:|---|---|
| research_program/dopamine_detox_attention_capture/fixtures/phase1_synthetic_alignment_features_20260608 | 3 | 3 | True | False |

## Phase 1 Manifests

| path | status | samples | claim blocked | provenance required | provenance ready | workflow-ready |
|---|---|---:|---|---|---|---|
| research_program/dopamine_detox_attention_capture/phase1_dhf1k_audio_only_manifest_20260609.json | real_external_attention_labels | 350 | False | True | True | True |
| research_program/dopamine_detox_attention_capture/phase1_dhf1k_fixation_density_audio_only_manifest_20260609.json | real_external_attention_labels | 350 | False | True | True | True |
| research_program/dopamine_detox_attention_capture/phase1_synthetic_alignment_manifest_20260608.json | synthetic_smoke_only | 3 | True | False | True | False |
| research_program/dopamine_detox_attention_capture/phase1_synthetic_smoke_manifest_20260608.json | synthetic_smoke_only | 16 | True | False | True | False |
