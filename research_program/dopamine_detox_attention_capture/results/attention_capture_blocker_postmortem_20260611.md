# Attention-Capture Blocker Postmortem

## Verdict

The current blocker is no longer primarily Modal, TRIBE, or HuggingFace access.
Those paths are mechanically green enough for small full-mode runs. The blocker
is construct validity: the only completed full-mode external dataset, DHF1K,
measures ocular saliency/fixation density and did not validate the proposed H2
`capture_score`.

## What Failed

- DHF1K full-mode TRIBE feature coverage reached 350/350 videos on Modal.
- Primary disjoint ROI scoring failed: `capture_score` Spearman rho = -0.0264,
  permutation p-greater = 0.6780, n = 312.
- Overlapping ROI sensitivity also failed: rho = 0.0300, p-greater = 0.3010,
  n = 314.
- Therefore, DHF1K-only progression to Phase 2/3 neutralization remains blocked.

Primary evidence:

- `phase1_dhf1k_full_modal_volume_verdict_20260611.md`
- `attention_capture_publication_path_after_dhf1k_full_20260611.md`

## Postmortem

We were treating three blockers as one:

1. Access: gated Llama/HuggingFace and unavailable SnapUGC/VQualA labels.
2. Transport: local vs Modal storage, feature extraction, and full-mode smoke
   tests.
3. Scientific fit: whether a TRIBE ROI ratio predicts behavioral capture.

The first two produced real delays, but they are now mostly resolved for TRIBE.
The third became visible only after the full-mode DHF1K run: fixation density
is not the same construct as short-form retention.

## External Evidence

- TRIBE v2 predicts fMRI responses to naturalistic video, audio, and text, and
  maps predictions to the fsaverage5 cortical mesh:
  https://huggingface.co/facebook/tribev2
- DHF1K is a dynamic human fixation benchmark for predicting eye movements
  during free viewing:
  https://arxiv.org/abs/1801.07424
- VQualA/SnapUGC defines ECR as continuation past the first five seconds and
  presents engagement prediction as distinct from visual-quality assessment:
  https://arxiv.org/html/2509.02969v1
- The VQualA CodaLab page states that the challenge dataset was released through
  the competition flow rather than as a simple public file mirror:
  https://codalab.lisn.upsaclay.fr/competitions/23005

## Next Regime

Current regime:

- Artifact types: retention label audits, cheap baseline/control audits,
  Modal-volume TRIBE feature slices, Phase 1 workflow reports.
- Operations: label mechanics audit, CSV-metadata baseline, negative controls,
  small full-mode Modal TRIBE slice, ROI scoring.
- Gates/verifiers: ready retention labels, clean negative controls, then
  preregistered Phase 1 `capture_score` gate or explicitly exploratory metric
  search.
- Known limitation: no mounted SnapUGC/VQualA labels as of this checkpoint.

Action class:

- Retrieval first: acquire/mount already-defined SnapUGC/VQualA labels.
- Search second: test whether ECR behaves differently from DHF1K fixation.
- Discovery only if a revised score or new construct survives held-out controls.

## Shortest Honest Path

1. Mount granted SnapUGC/VQualA labels and videos.
2. Run `audit_attention_capture_retention_labels.py`.
3. Run `audit_attention_capture_retention_baselines.py`.
4. If labels pass and negative controls are clean, run a 100-300 video full-mode
   Modal TRIBE slice.
5. Score H2 unchanged first; only then run an explicitly labeled exploratory
   ROI/metric search if H2 fails.
6. If SnapUGC/ECR also fails, pivot from "detox neutralization" to either a
   negative-result paper or a revised, preregistered metric.

## Claim Boundary

No Phase 2/3 neutralization claims are allowed from the current evidence. The
only supported claim is a negative DHF1K Phase 1 result plus a prepared
SnapUGC/VQualA retention-validation path.
