# Attention-Capture Publication Path Audit

## Verdict

- Publication ready: False
- Paper claim allowed: False
- Phase 2 ready: False
- Phase 1 gate passed: False
- Full multimodal credential present: False
- Claim boundary: This audit decides whether current evidence can support the attention-capture paper claim. It is stricter than data readiness: a runnable manifest is not enough when the scoring gate failed or retention/full-multimodal evidence is absent.

## Blocking Reasons

- current H2 capture_score failed the Phase 1 rho gate
- no SnapUGC/VQualA retention label CSV is mounted
- completed TRIBE workflows are audio-only and no HuggingFace text model token is present
- fewer than 2 external datasets have completed claim-ready workflow reports

## Warnings

- TRIBE feature cache has checksum provenance, but the cache is still external to git and needs an archive location or deterministic rerun path

## Next Actions

- Do not enter Phase 2/3 neutralization from the current H2 score; either acquire retention labels for an independent test or preregister a revised score before evaluating held-out data.
- Mount granted SnapUGC/VQualA labels and build a retention manifest with alignment-audit provenance.
- Provide a HuggingFace token with access to the gated TRIBE text model path, then rerun full multimodal feature extraction.
- Require at least one held-out external validation dataset before claiming publication readiness.
- Add an object-storage/archive location or deterministic rerun instructions for the external TRIBE feature cache.

## Workflow Evidence

| workflow | datasets | gate | best rho | p | n | invalid denominators |
|---|---|---|---:|---:|---:|---:|
| research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_audio_only_workflow_20260609.json | DHF1K, pooled | False | 0.1256 | 0.0130 | 301 | 49 |
| research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_workflow_20260609.json | DHF1K, pooled | False | -0.0348 | 0.7380 | 302 | 48 |

## Feature Cache Evidence

| audit | feature dir | ready | npz files | expected ids | aggregate sha256 |
|---|---|---|---:|---:|---|
| research_program/dopamine_detox_attention_capture/results/dhf1k_audio_only_feature_cache_audit_20260610.json | data/features/tribe_dhf1k_attention_audio_only | True | 516 | 516 | 990c7605e215 |
