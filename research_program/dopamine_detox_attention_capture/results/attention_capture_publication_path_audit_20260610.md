# Attention-Capture Publication Path Audit

## Verdict

- Publication ready: False
- Paper claim allowed: False
- Phase 2 ready: False
- Phase 1 gate passed: False
- Full multimodal path ready: True
- Claim boundary: This audit decides whether current evidence can support the attention-capture paper claim. It is stricter than data readiness: a runnable manifest is not enough when the scoring gate failed or required external or full-mode evidence is absent.

## Blocking Reasons

- current H2 capture_score failed the Phase 1 rho gate
- no SnapUGC/VQualA retention label CSV is mounted or available in audited Modal volumes
- fewer than 2 external datasets have completed claim-ready workflow reports

## Warnings

- none

## Next Actions

- Do not enter Phase 2/3 neutralization from the current H2 score; either acquire retention labels for an independent test or preregister a revised score before evaluating held-out data.
- Mount granted SnapUGC/VQualA labels and build a retention manifest with alignment-audit provenance.
- Require at least one held-out external validation dataset before claiming publication readiness.

## Workflow Evidence

| workflow | datasets | gate | best rho | p | n | invalid denominators |
|---|---|---|---:|---:|---:|---:|
| research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_audio_only_workflow_20260609.json | DHF1K, pooled | False | 0.1256 | 0.0130 | 301 | 49 |
| research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_workflow_20260609.json | DHF1K, pooled | False | -0.0348 | 0.7380 | 302 | 48 |

## Feature Cache Evidence

| audit | feature dir | ready | reproduction | npz files | expected ids | rerun cmds | aggregate sha256 |
|---|---|---|---|---:|---:|---:|---|
| research_program/dopamine_detox_attention_capture/results/dhf1k_audio_only_feature_cache_audit_20260610.json | data/features/tribe_dhf1k_attention_audio_only | True | True | 516 | 516 | 3 | 990c7605e215 |

## Modal Asset Evidence

| audit | volumes | labels | datasets | features | modal token | truncated | blockers |
|---|---:|---|---|---|---|---:|---|
| research_program/dopamine_detox_attention_capture/results/modal_asset_audit_20260610.json | 20 | False | False | True | False | 0 | no Modal-hosted SnapUGC/VQualA retention label candidate found; no Modal secret exposes a HuggingFace token env name |

## TRIBE Full-Preflight Evidence

| audit | ok | event mode | events | duration | media | error |
|---|---|---|---:|---:|---|---|
| research_program/dopamine_detox_attention_capture/results/tribe_full_preflight_audit_20260610.json | True | full | 1 | 3.5720 | /bmd-videos/generated/bo_memorability_replay/bo_replay_00_sobol_prompt_search_518_slot18_rep00.mp4 | none |
