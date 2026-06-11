# DHF1K Full-Mode Phase 1 Verdict

## Verdict

- Mechanical ready: True
- Feature coverage: 350/350 full-mode TRIBE files on Modal
- Dataset: `DHF1K`
- Ground truth: `mean_fixation_density`
- Primary ROI policy: disjoint Destrieux masks
- H2 gate passed: False
- Claim validated: False
- Phase 2/3 claim progression: blocked

## Primary Result

The preregistered gate was `capture_score` Spearman rho >= 0.40 in at least one
claim-ready dataset.

| metric | n | Spearman rho | permutation p (greater) | gate |
|---|---:|---:|---:|---|
| capture_score | 312 | -0.0264 | 0.6780 | fail |
| capture_delta | 350 | 0.0224 | 0.3260 | diagnostic |
| sensory_mean | 350 | -0.1010 | 0.9660 | diagnostic |
| frontoparietal | 350 | -0.1051 | 0.9830 | diagnostic |

Primary report:
`research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_full_modal_volume_disjoint_workflow_20260611.json`

## ROI Sensitivity

Overlapping Destrieux masks did not rescue the result.

| ROI policy | capture_score n | Spearman rho | permutation p (greater) | gate |
|---|---:|---:|---:|---|
| disjoint primary | 312 | -0.0264 | 0.6780 | fail |
| overlapping sensitivity | 314 | 0.0300 | 0.3010 | fail |

Sensitivity report:
`research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_full_modal_volume_overlap_sensitivity_20260611.json`

## Discovery-Regime Audit

Question: does full-mode TRIBE H2 capture_score discriminate DHF1K fixation
density after Llama access and full feature extraction are unblocked?

Current regime:

- Artifact types: Modal-volume full-mode TRIBE feature files, frozen ROI masks,
  label audits, Modal-volume Phase 1 workflow reports.
- Operations: Modal-side feature reads, ROI aggregation, Spearman/permutation
  scoring, disjoint and overlapping ROI policies.
- Gates/verifiers: 350/350 feature coverage, ready DHF1K fixation-density label
  audit, `capture_score rho >= 0.40`.
- Known limitations: DHF1K fixation density is ocular saliency, not short-form
  platform retention; this result does not test SnapUGC/VQualA ECR.

Action class:

- Search inside the existing H2 schema. The run changes infrastructure and ROI
  sensitivity, not the hypothesis definition.

Results:

- Accepted artifacts: two mechanically ready full-mode DHF1K workflow reports.
- Rejected/withheld artifacts: DHF1K H2 claim; Phase 2/3 progression from
  DHF1K alone remains blocked.
- Key metric: primary `capture_score` rho = -0.0264, far below the 0.40 gate.

Residual content:

- Explained by old regime: prior audio-only DHF1K failure was not caused by
  missing Llama text features.
- New content outside old regime: none yet. The result is a clean negative, not
  a new mechanism.
- Retraction/supersession: any DHF1K-only language claiming H2 validation should
  be superseded by this failed full-mode result.

Next move:

The shortest publication-grade path is not Phase 2 neutralization. It is adding
a second claim-ready real dataset, preferably SnapUGC/VQualA ECR labels, and
testing whether behavioral retention behaves differently from DHF1K fixation
density. If retention also fails, the proposal should pivot from
`capture_score` validation to a negative-result paper or a revised, explicitly
exploratory metric search.

## Claim Boundary

This is a negative Phase 1 result for DHF1K fixation density only. It does not
rule out behavioral-retention labels such as SnapUGC/VQualA, but it blocks
DHF1K-only progression to Phase 2/3 claims.
