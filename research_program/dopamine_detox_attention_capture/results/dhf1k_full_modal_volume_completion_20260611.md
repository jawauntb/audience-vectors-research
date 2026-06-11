# DHF1K Full-Mode Modal Volume Completion

## Verdict

- Mechanical ready: True
- Dataset: `DHF1K`
- Event mode: `full`
- Modal app: `audience-vectors-dev`
- Modal volume: `attention-capture-features-v1`
- Output prefix: `attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610`
- Expected feature files: 350
- Observed feature files: 350

## What Changed

TRIBE full-video event construction now preserves upstream strict text alignment
first, then retries `AddSentenceToWords` with `max_unmatched_ratio=0.50` only
when the strict `0.05` unmatched-word gate fails. This recovered the three
previously blocked full-mode clips:

- `dhf1k_146`
- `dhf1k_203`
- `dhf1k_262`

The successful retry required stopping/redeploying the Modal app so the patched
class ran from a cold worker.

## Evidence

- Full resume report:
  `research_program/dopamine_detox_attention_capture/results/dhf1k_full_modal_volume_extraction_20260611.json`
- Failed strict retry report:
  `research_program/dopamine_detox_attention_capture/results/dhf1k_full_modal_volume_retry_20260611.json`
- Successful cold-deploy retry report:
  `research_program/dopamine_detox_attention_capture/results/dhf1k_full_modal_volume_retry_after_cold_deploy_20260611.json`
- Volume count command:
  `uv run --extra modal modal volume ls attention-capture-features-v1 attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610 | rg '\\.npz' | wc -l`

## Discovery-Regime Audit

Question: can DHF1K full-mode TRIBE features be completed without local CPU
extraction after Llama access was granted?

Current regime:

- Artifact types: Modal-volume TRIBE `.npz` feature files, extraction reports,
  label audits, alignment audits, Phase 1 manifests, workflow reports.
- Operations: Modal H100 TRIBE scoring, resumable feature writing, manifest
  alignment, Phase 1 capture-score scoring.
- Gates/verifiers: zero failed feature writes, 350/350 feature coverage,
  claim-gated manifest preflight, H2 Spearman gate.
- Known limitations: Modal-volume coverage is not yet a local feature-path
  manifest or a Phase 1 score.

Action class:

- Search inside the existing regime: relaxed text alignment is a preprocessing
  recovery path, not a new scientific finding.

Results:

- Accepted artifacts: 350/350 full-mode DHF1K feature files on Modal volume.
- Rejected or withheld artifacts: the earlier strict-only retry remains logged
  as a failed preprocessing attempt.
- Key metric: observed feature-file count equals expected feature-file count.

Claim boundary:

This confirms full-mode DHF1K TRIBE feature extraction coverage on Modal only.
It does not validate the H2 capture-score gate, replace SnapUGC/VQualA
retention labels, or establish a publication-ready finding.

Next move:

Build a claim-gated DHF1K full-mode manifest or Modal-native scorer over this
feature volume, then rerun Phase 1 H2 against fixation density. The project
still needs at least one additional claim-ready real dataset, preferably
SnapUGC/VQualA retention labels, before strong publication claims.
