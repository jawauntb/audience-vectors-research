# Phase 1 Data Readiness Audit

## Verdict

- Phase 1 can run now: False
- DHF1K root ready for label build: True
- DHF1K label audit ready: True
- DHF1K labels ready: True
- SnapUGC labels ready: False
- TRIBE features ready: True
- DHF1K TRIBE features ready: False
- ROI masks ready: True
- Real manifest ready: False
- Recommended next action: extract DHF1K TRIBE features from the audited DHF1K labels
- Claim boundary: This report audits local data availability only. It does not score TRIBE features or validate attentional capture.

## Blocking Reasons

- DHF1K labels ready but no DHF1K TRIBE feature directory found

## DHF1K Candidates

| path | videos | map dirs | fixation dirs | ready |
|---|---:|---:|---:|---|
| data/attention_capture/DHF1K | 1000 | 700 | 700 | True |

## DHF1K Label Audits

| path | labels CSV exists | rank column | rows | ready |
|---|---|---|---:|---|
| research_program/dopamine_detox_attention_capture/results/dhf1k_attention_label_audit_20260608.json | True | mean_map_intensity | 350 | True |

## SnapUGC/VQualA Label Candidates

| path | rows | columns |
|---|---:|---|
| none | 0 | n/a |

## TRIBE Feature Cache Candidates

| path | npz files | sampled frames arrays | claim blocked | ready |
|---|---:|---:|---|---|
| /Users/jawaun/isc_mod/data/features/tribe | 1022 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_wan22_lora_proxy_train_50x4_2026-05-21 | 200 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_svd_per_persona | 108 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0 | 96 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_wan22_lora_eval_fresh_picsum_24_r16_s150_bon_24x4_s12_m1p0 | 96 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_wan22_lora_eval_fresh_picsum_24_r16_s150_s12 | 48 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_s12 | 48 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_svd_sweep | 45 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_best_of_n | 45 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_svd_n20 | 40 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_veo_bon | 40 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_wan22_replication_matrix_2026-05-20 | 32 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_wan22_lora_eval_fresh_picsum_8_r16_s150_bon_8x4_s12_m1p0 | 32 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_svd_alpha_bon | 30 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_wan22_lora_eval_fresh_picsum_8_pref_weighted_r16_s300_s12 | 16 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_wan22_lora_eval_fresh_picsum_8_r16_s150_s12_m0p5 | 16 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_wan22_lora_eval_fresh_picsum_8_r16_s150_s12 | 16 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_wan22_lora_eval_r16_s150_8x2_s12 | 16 | 8 | False | True |
| /Users/jawaun/isc_mod/data/features/tribe_cogvideox_smoke | 5 | 5 | True | False |
| /Users/jawaun/isc_mod/data/features/tribe_svd_smoke_large | 5 | 5 | True | False |

## Phase 1 Manifests

| path | status | samples | claim blocked | provenance required | provenance ready | workflow-ready |
|---|---|---:|---|---|---|---|
| research_program/dopamine_detox_attention_capture/phase1_synthetic_alignment_manifest_20260608.json | synthetic_smoke_only | 3 | True | False | True | False |
| research_program/dopamine_detox_attention_capture/phase1_synthetic_smoke_manifest_20260608.json | synthetic_smoke_only | 16 | True | False | True | False |
