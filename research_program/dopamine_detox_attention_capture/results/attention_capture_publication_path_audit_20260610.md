# Attention-Capture Publication Path Audit

## Verdict

- Publication ready: False
- Paper claim allowed: False
- Phase 2 ready: False
- Phase 1 gate passed: False
- Full multimodal path ready: False
- DHF1K Modal media ready: True
- Claim boundary: This audit decides whether current evidence can support the attention-capture paper claim. It is stricter than data readiness: a runnable manifest is not enough when the scoring gate failed or required external or full-mode evidence is absent.

## Blocking Reasons

- current H2 capture_score failed the Phase 1 rho gate
- no SnapUGC/VQualA retention label CSV is mounted or available in audited Modal volumes
- completed TRIBE workflows are audio-only and no successful full multimodal TRIBE prediction smoke is available
- fewer than 2 external datasets have completed claim-ready workflow reports

## Warnings

- at least one TRIBE full-mode prediction smoke audit failed

## Next Actions

- Do not enter Phase 2/3 neutralization from the current H2 score; either acquire retention labels for an independent test or preregister a revised score before evaluating held-out data.
- Mount granted SnapUGC/VQualA labels and build a retention manifest with alignment-audit provenance.
- Provide a HuggingFace token with access to the gated TRIBE text model path or pass a full-mode TRIBE prediction smoke from cached Modal weights, then rerun full multimodal feature extraction.
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
| research_program/dopamine_detox_attention_capture/results/tribe_full_preflight_dhf1k_audit_20260610.json | True | full | 50 | 14.9333 | /bmd-videos/attention_capture/DHF1K/video/003.AVI | none |

## TRIBE Full-Prediction Smoke Evidence

| audit | ok | event mode | frames | duration | media | error |
|---|---|---|---:|---:|---|---|
| research_program/dopamine_detox_attention_capture/results/tribe_full_prediction_smoke_dhf1k_audit_20260610.json | False | full | 0 x 0 | n/a | /bmd-videos/attention_capture/DHF1K/video/003.AVI | You are trying to access a gated repo. Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.2-3B. 401 Client Error. (Request ID: Root=1-6a29a2c5-6054c5445b2127c80d72e1a9;3b700567-60e6-4d7d-96ef-ca4b494c4dca)  Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-3B/resolve/main/config.json. Access to model meta-llama/Llama-3.2-3B is restricted. You must have access to it and be authenticated to access it. Please log in. |

## DHF1K Modal Media Evidence

| audit | ready | expected | found | missing | zero-byte | modal prefix | modal csv | blockers |
|---|---|---:|---:|---:|---:|---|---|---|
| research_program/dopamine_detox_attention_capture/results/dhf1k_modal_media_audit_20260610.json | True | 350 | 350 | 0 | 0 | /bmd-videos/attention_capture/DHF1K | research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_modal_20260610.csv | none |
