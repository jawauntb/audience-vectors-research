# Attention-Capture Publication Path Audit

## Verdict

- Publication ready: False
- Paper claim allowed: False
- Phase 2 ready: False
- Phase 1 gate passed: False
- Full multimodal path ready: True
- DHF1K Modal media ready: True
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
| research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_full_modal_volume_disjoint_workflow_20260611.json | DHF1K, pooled | False | -0.0264 | 0.7070 | 312 | 38 |

## Feature Cache Evidence

| audit | feature dir | ready | reproduction | npz files | expected ids | rerun cmds | aggregate sha256 |
|---|---|---|---|---:|---:|---:|---|
| none | n/a | False | False | 0 | 0 | 0 | n/a |

## Modal Asset Evidence

| audit | volumes | labels | datasets | features | modal token | truncated | blockers |
|---|---:|---|---|---|---|---:|---|
| research_program/dopamine_detox_attention_capture/results/modal_asset_audit_20260610.json | 20 | False | False | True | False | 0 | no Modal-hosted SnapUGC/VQualA retention label candidate found; no Modal secret exposes a HuggingFace token env name |

## TRIBE Full-Preflight Evidence

| audit | ok | event mode | events | duration | media | error |
|---|---|---|---:|---:|---|---|
| none | False | n/a | 0 | n/a | n/a | n/a |

## TRIBE Full-Prediction Smoke Evidence

| audit | ok | event mode | frames | duration | media | error |
|---|---|---|---:|---:|---|---|
| research_program/dopamine_detox_attention_capture/results/tribe_full_prediction_smoke_dhf1k_audit_20260610.json | True | full | 15 x 20484 | 14.9333 | /bmd-videos/attention_capture/DHF1K/video/003.AVI | none |

## DHF1K Modal Media Evidence

| audit | ready | expected | found | missing | zero-byte | modal prefix | modal csv | blockers |
|---|---|---:|---:|---:|---:|---|---|---|
| research_program/dopamine_detox_attention_capture/results/dhf1k_modal_media_audit_20260610.json | True | 350 | 350 | 0 | 0 | /bmd-videos/attention_capture/DHF1K | research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_modal_20260610.csv | none |
