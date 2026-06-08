# Soundness Audit: Dopamine Detox Attention-Capture Proposal

Date: 2026-06-08

## Verdict

Proceed with a narrow Phase 1 validation scaffold. Do not proceed yet as if
Phase 2/3 neutralization is scientifically established.

The sound version of the project is:

> Test whether a predeclared TRIBE ROI proxy for sensory-over-control dominance
> correlates with external attention labels.

The unsound version would be:

> Claim that the proxy already measures dopamine, conscious control, or a
> validated detox intervention.

## Research Considered

Local evidence:

- `START_HERE.md` and `CLAIM_LEDGER.md` preserve the evidence hierarchy:
  human/BMD labels above TRIBE proxy scores, and proxy workflows below both.
- `current_research_status.md` shows the existing TRIBE/BMD memorability
  direction has real but scoped support: BMD labels and one Prolific
  forced-choice check. It does not validate a new attentional-capture construct.
- `neural_response_guided_generation_feasibility_20260608.md` recommends a
  no-generation dry run before using neural-response scores as generation or
  intervention objectives.
- `scripts/roi_decomposition.py`, `TribeService`, and TRIBE feature NPZ caches
  are reusable infrastructure.

External checks:

- TRIBE v2 is now best cited as d'Ascoli et al., "A foundation model of vision,
  audition, and language for in-silico neuroscience," arXiv:2605.04326, 2026:
  https://arxiv.org/abs/2605.04326. The pasted proposal's 2024/TRIBE GitHub-only
  citation is stale.
- VQualA / SnapUGC appears to be a real ICCV 2025 challenge/dataset track for
  short-video engagement prediction: https://arxiv.org/abs/2509.02969. Access
  and exact ECR label availability still need to be verified before Phase 1.
- DHF1K is a real CVPR 2018 video-saliency benchmark with 1,000 videos and
  fixation data: https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Revisiting_Video_Saliency_CVPR_2018_paper.html.
- Memento10k is a real ECCV 2020 video memorability dataset:
  https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610222.pdf.
  It is better as a control/adjacent behavioral dataset than as direct evidence
  of attentional capture.
- Corbetta and Shulman support the top-down versus stimulus-driven attention
  distinction: https://www.nature.com/articles/nrn755.
- The naturalistic-stimulus review is Sonkusare, Breakspear, and Guo, not
  Bhatt: https://doi.org/10.1016/j.tics.2019.05.004.
- Finn and Bandettini support naturalistic movie viewing as behaviorally useful
  brain-state measurement context, not a direct FPN-suppression claim:
  https://www.sciencedirect.com/science/article/pii/S1053811921002408.

## What Looks Sound

Phase 1 is a good falsifiable test. The proposal has an external-label gate,
uses datasets with different attention-adjacent labels, and does not require
new human collection for the first screen.

The H1-to-H2 revision is plausible. A negative correlation between
frontoparietal activation and ECR is interpretable as evidence that FPN alone
is not a positive engagement marker. That is exactly the kind of failed
hypothesis the project should preserve.

The project can reuse existing code. TRIBE Modal scoring, NPZ feature caches,
Destrieux/fsaverage5 ROI handling, and the repo's claim-ledger discipline are
already present.

## What Needs Narrowing

The term "dopamine detox" should remain a product-facing shorthand, not a
scientific claim. Nothing in Phase 1 measures dopamine.

"FPN suppression" is too strong unless the ROI definition, TRIBE scaling, and
external labels survive controls. The safer phrase is "lower predicted
frontoparietal response under the chosen TRIBE ROI readout."

The ratio score is brittle. If frontoparietal values are signed or near zero,
the denominator can create spurious rank order. The scaffold therefore reports
invalid denominators and includes `capture_delta` as a secondary robustness
readout.

The proposed ROIs are not yet preregistered enough. V1/PPA/frontoparietal are
plausible, but language as a sensory-capture numerator can confound semantic
richness with attention capture. ROI masks should be frozen before scoring real
datasets.

Dataset labels are adjacent but not identical:

- SnapUGC ECR is platform retention, influenced by recommendation context,
  creator effects, genre, and length.
- DHF1K fixation density is overt gaze, not completion or executive control.
- Memento10k is memorability, not immediate attentional capture.

Success in one dataset should be reported as construct evidence for that label,
not as global proof that short-form-video capture has been neutralized.

## Discovery-Regime Audit

Question: does a TRIBE sensory-over-control ROI proxy discriminate externally
high-capture videos?

Current regime:

- Artifact types: video, TRIBE frame tensor, ROI mask, ROI mean, external label,
  correlation report, withheld denominator record.
- Operations: Modal TRIBE scoring, cached NPZ loading, ROI aggregation,
  Spearman/permutation gate, report rendering.
- Gates/verifiers: `rho >= 0.40` in at least one real dataset, invalid
  denominators counted, secondary metrics reported, no claim update from
  synthetic fixtures.
- Known limitations: no measured dopamine, no human validation in this lane,
  ROI definitions not yet frozen against a real atlas review.

Action class:

- Search, not discovery. The experiment tests a new proxy inside an existing
  TRIBE/ROI/correlation schema.

Experiment:

- Manifest/report paths:
  `research_program/dopamine_detox_attention_capture/phase1_synthetic_smoke_manifest_20260608.json`
  and `results/phase1_synthetic_smoke_20260608.*`.
- Positive targets: synthetic high-capture rows with higher sensory and lower
  frontoparietal ROI values.
- Negative controls: frontoparietal-only and sensory-mean correlations are
  reported separately.
- Stress tests: denominator invalidity is explicit; additive contrast is
  reported beside the primary ratio.

Gate:

- Acceptance rule for real Phase 1: primary capture-score Spearman rho >= 0.40
  in at least one real dataset, with all datasets and controls reported.
- Withheld/rejected rule: synthetic smoke runs cannot update claims; real rows
  with invalid denominators are excluded from primary ratio correlation and
  counted.

Results:

- Accepted artifacts: scaffold and synthetic smoke result only.
- Rejected or withheld artifacts: no real attention-capture claim yet.
- Key metrics: see the smoke report.
- Variance or ablation: not applicable until real datasets are loaded.

Control update:

- A BOLD Moments memorability control was run over 1,022 cached TRIBE feature
  files using exploratory Destrieux ROI masks. It is not an attention-capture
  validation dataset.
- The capture-score gate did not pass against BMD memorability:
  `rho = -0.1988` on 660 valid-ratio rows.
- The additive contrast also stayed negative: `rho = -0.2033` on 1,022 rows.
- Frontoparietal alone weakly correlated positively with memorability:
  `rho = +0.1935`.
- 362 rows had non-positive frontoparietal denominators and were withheld from
  the primary ratio. This confirms denominator handling is a live issue for
  real TRIBE tensors.
- Interpretation: useful control evidence. The proposed capture proxy does not
  appear to be a trivial re-labeling of the existing BMD memorability direction,
  but the ROI/ratio definition still needs real attention-label validation.

ROI-mask freeze update:

- The exploratory Destrieux masks are now materialized as
  `results/destrieux_roi_masks_20260608.npz` with coverage/overlap reports in
  `results/destrieux_roi_mask_audit_20260608.*`.
- Coverage is broad: V1 2,217 vertices, PPA 866, language 4,142, and
  frontoparietal 4,104.
- The audit found overlapping numerator/denominator families:
  V1/PPA overlap is 598 vertices and language/frontoparietal overlap is 742
  vertices.
- Interpretation: the current masks are now auditable and reproducible, but
  not clean enough to serve as final preregistered ROIs without refinement or a
  disjoint-mask rule.

ROI-policy refinement:

- A conservative `drop_shared` policy was added and materialized as
  `results/destrieux_roi_masks_disjoint_20260608.npz` with audit reports in
  `results/destrieux_roi_mask_audit_disjoint_20260608.*`.
- This policy removes any vertex selected by more than one ROI, avoiding
  numerator/denominator leakage rather than assigning shared vertices by an
  arbitrary priority rule.
- Disjoint coverage remains non-empty: V1 1,619 vertices, PPA 268, language
  3,400, and frontoparietal 3,362. All off-diagonal overlap counts are zero.
- The cost is that PPA becomes small. Therefore the disjoint masks should be
  the recommended default for real Phase 1, while the overlapping masks remain
  an archived sensitivity condition.
- The BOLD Moments control was rerun with the disjoint masks:
  `rho(capture_score, memorability) = -0.2444` on 739 valid-ratio rows;
  `rho(capture_delta, memorability) = -0.2492` on 1,022 rows;
  `rho(frontoparietal, memorability) = +0.2346`.
- Invalid primary-ratio denominators decreased from 362 to 283. This is a
  modest improvement in score usability without making the capture proxy a
  memorability proxy.

Manifest bridge update:

- `scripts/build_attention_capture_phase1_manifest.py` now turns a label CSV
  plus cached TRIBE NPZ feature directory into the JSON manifest consumed by
  `scripts/run_attention_capture_phase1.py`.
- This is retrieval infrastructure, not a result. It makes the next real-data
  step auditable by fixing the sample-id, label-column, feature-template, and
  missing-feature behavior in one command.

DHF1K readiness update:

- `scripts/build_dhf1k_attention_labels.py` derives external saliency/gaze
  labels from the official DHF1K directory structure documented at
  https://github.com/wenguanwang/DHF1K: `video/{id}.AVI`,
  `annotation/{id}/maps/*.png`, and optional
  `annotation/{id}/fixation/*.png`.
- The script emits multiple attention-adjacent columns:
  `mean_map_intensity`, `peak_map_intensity`, `peak_to_mean_map_ratio`,
  `mean_map_concentration`, and `mean_fixation_density`.
- This is deliberately not hidden inside TRIBE scoring. The label audit should
  be inspected first because saliency-map mean intensity may have low variance
  or encode map-generation conventions rather than attentional capture.
- `scripts/extract_attention_capture_tribe_features.py` now provides a generic
  CSV-driven TRIBE feature extractor for local or Modal-visible videos. Byte
  transport is the default so local DHF1K videos can be scored without a Modal
  volume upload pre-step.
- No local SnapUGC or DHF1K assets were found in the searched data paths during
  this pass, so no real Phase 1 claim has been attempted.

Preflight verifier update:

- `scripts/preflight_attention_capture_phase1.py` now runs a manifest preflight
  before claim-relevant Phase 1 scoring.
- It checks sample count, dataset-level label variance, missing feature files,
  feature/ROI shape compatibility, denominator validity after local ROI
  aggregation, and whether manifest status/dataset names would block claim
  updates.
- This turns the next real-data handoff into an explicit go/no-go artifact:
  a real run should create `phase1_*_preflight_*.json/md` before
  `phase1_*_disjoint_*.json/md`.
- This is still retrieval/verification infrastructure. Passing preflight only
  means the manifest is ready to score; it does not validate the capture proxy.

Sensitivity-runner update:

- `scripts/run_attention_capture_sensitivity.py` now runs the same Phase 1
  manifest through a primary ROI mask and one or more sensitivity masks, then
  writes a comparison report.
- The intended real-data use is disjoint masks as the primary run and the
  archived overlapping masks as a sensitivity condition.
- A synthetic smoke sensitivity report was generated at
  `results/phase1_synthetic_smoke_sensitivity_20260608.*`. Because the fixture
  supplies explicit ROI values, mask choice is a no-op and all deltas are zero;
  this proves command wiring only.
- On real feature-path manifests, nonzero deltas between disjoint and
  overlapping masks should be treated as ROI-definition sensitivity, not as a
  new attention-capture finding.

Guarded-workflow update:

- `scripts/run_attention_capture_phase1_workflow.py` now makes the Phase 1
  sequence atomic: run preflight, decide whether scoring is allowed, then write
  primary and optional sensitivity reports from the same manifest.
- Real claim-relevant scoring only proceeds when `claim_ready` is true. A
  smoke/control manifest can still be scored for command diagnostics with
  `--score-claim-blocked`, but the resulting report keeps
  `claim_validated=false`.
- A synthetic smoke workflow report was generated at
  `results/phase1_synthetic_smoke_workflow_20260608.*`. It shows the expected
  diagnostic-only status: mechanical preflight passes, claim updates are
  blocked, scoring executes only because it was explicitly requested, and the
  sensitivity delta is zero because explicit fixture ROI values bypass masks.
- A repeated local asset scan found only repository adapters/scripts for DHF1K,
  SnapUGC/VQualA, and Memento-style data, not real external videos/labels, so
  no real Phase 1 result has been attempted in this update.

Data-readiness audit update:

- `scripts/audit_attention_capture_data_readiness.py` now records the local
  data handoff state as a report rather than relying on ad hoc shell scans.
- The readiness audit was generated at
  `results/phase1_data_readiness_20260608.*` after scanning `.`,
  `/Users/jawaun/isc_mod/data`, `/Users/jawaun/data`, and
  `/Users/jawaun/datasets`.
- Result: `phase1_can_run_now=false`. Disjoint ROI masks are present and cached
  TRIBE feature directories exist, including `/Users/jawaun/isc_mod/data/features/tribe`
  with 1,022 NPZ files whose sampled payloads contain `frames`; however, no
  local DHF1K root, SnapUGC/VQualA ECR label CSV, or real claim-ready Phase 1
  manifest was found.
- This is a withheld/rejected handoff state, not a failed scientific result:
  the blocker is absent external attention labels/videos, not a negative
  capture-score validation.

Manifest-alignment audit update:

- `scripts/audit_attention_capture_manifest_alignment.py` now audits the join
  between an external label CSV and a cached TRIBE feature directory before the
  strict manifest builder runs.
- It checks sample-ID uniqueness, feature-file presence under the chosen
  template, finite ground-truth values, distinct label count, label variance,
  and minimum aligned sample count. This catches handoff errors before they
  become silent sample loss or a failed GPU/scoring run.
- A synthetic alignment smoke report was generated at
  `results/phase1_synthetic_alignment_20260608.*` using a tiny committed
  fixture CSV and three tiny TRIBE-shaped `frames` NPZ files. It reports
  `ready_for_manifest_build=true` for the fixture only.
- This is retrieval/verification infrastructure. It does not score TRIBE
  features and cannot validate the attention-capture proxy.

Manifest-provenance update:

- `scripts/build_attention_capture_phase1_manifest.py` can now consume an
  alignment audit via `--alignment-audit`.
- The builder fails if the audit is not `ready_for_manifest_build=true` or if
  the audit does not match the label CSV, feature directory, sample-ID column,
  ground-truth column, feature template, or dataset being used.
- A synthetic alignment manifest was generated at
  `phase1_synthetic_alignment_manifest_20260608.json`. It records the alignment
  audit path, SHA-256, aligned feature count, missing feature count, and label
  summary under `metadata.alignment_audit`.
- This closes a reproducibility gap: a real Phase 1 manifest can now carry the
  exact verifier artifact that allowed it to be built.

Residual content:

- Explained by old regime: TRIBE can produce reusable cortical features and
  proxy scores.
- New content outside old regime: a new attention-capture ROI proxy, pending
  real-data validation.
- Retractions or supersessions: update the proposal citation details and avoid
  "dopamine" or "executive restoration" as measured constructs.

Next move:

1. Mount or acquire real SnapUGC/VQualA labels or a DHF1K dataset root; rerun
   `scripts/audit_attention_capture_data_readiness.py`.
2. Before building the manifest, run
   `scripts/audit_attention_capture_manifest_alignment.py` on the chosen label
   CSV and feature directory.
3. Build a real Phase 1 manifest for SnapUGC and/or DHF1K from external labels
   and cached TRIBE NPZ files, passing `--alignment-audit` so the manifest
   records the verifier hash.
4. For DHF1K, run the label audit before GPU scoring and choose a non-degenerate
   ground-truth column.
5. Run the guarded Phase 1 workflow with
   `scripts/run_attention_capture_phase1_workflow.py`, using disjoint masks as
   primary and overlapping masks as the archived sensitivity condition.
6. If the workflow withholds scoring, treat that as the result of the handoff
   and fix the manifest/labels/features before spending more compute.
7. Run the same script with cached or Modal-generated TRIBE features.
8. Only then decide whether perturbation/neutralization is worth compute.
