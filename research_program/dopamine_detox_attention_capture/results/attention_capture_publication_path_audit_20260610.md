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

- TRIBE feature cache is external to the repo and should be archived or regenerated for reproducibility

## Next Actions

- Do not enter Phase 2/3 neutralization from the current H2 score; either acquire retention labels for an independent test or preregister a revised score before evaluating held-out data.
- Mount granted SnapUGC/VQualA labels and build a retention manifest with alignment-audit provenance.
- Provide a HuggingFace token with access to the gated TRIBE text model path, then rerun full multimodal feature extraction.
- Require at least one held-out external validation dataset before claiming publication readiness.
- Create a non-git artifact plan for the external TRIBE feature cache: checksum manifest, object storage, or deterministic rerun instructions.

## Workflow Evidence

| workflow | datasets | gate | best rho | p | n | invalid denominators |
|---|---|---|---:|---:|---:|---:|
| research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_audio_only_workflow_20260609.json | DHF1K, pooled | False | 0.1256 | 0.0130 | 301 | 49 |
| research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_workflow_20260609.json | DHF1K, pooled | False | -0.0348 | 0.7380 | 302 | 48 |
