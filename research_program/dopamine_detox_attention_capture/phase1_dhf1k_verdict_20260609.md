# Phase 1 DHF1K Verdict

## Discovery-Regime Audit

Question:

Does the proposed TRIBE v2 `capture_score = mean(V1, PPA, language) /
(frontoparietal + epsilon)` discriminate high- versus low-attention DHF1K
videos strongly enough to advance from Phase 1 validation into trigger
decomposition?

Current regime:

- Artifact types: DHF1K label CSVs, label audits, TRIBE NPZ feature caches,
  alignment audits, claim-gated Phase 1 manifests, primary/sensitivity ROI
  reports, ROI diagnostics.
- Operations: Modal CPU fixation-label fanout, Modal GPU TRIBE extraction,
  disjoint Destrieux ROI scoring, overlapping-mask sensitivity scoring.
- Gates/verifiers: claim-ready preflight, zero missing aligned features,
  non-degenerate ground truth, `rho >= 0.40` in at least one real dataset.
- Known limitations: completed DHF1K features are `audio-only` because the full
  TRIBE text path needs gated Llama access; language-dependent claims are
  withheld.

Action class:

- Retrieval/search/discovery: retrieval plus validation search.
- Why: Modal CPU fanout added the proposal's public fixation-density ground
  truth inside the existing Phase 1 schema, then the existing gate tested it.

Experiment:

- Mean-map proxy manifest:
  `phase1_dhf1k_audio_only_manifest_20260609.json`
- Mean-map proxy workflow:
  `results/phase1_dhf1k_audio_only_workflow_20260609.md`
- Fixation-density manifest:
  `phase1_dhf1k_fixation_density_audio_only_manifest_20260609.json`
- Fixation-density workflow:
  `results/phase1_dhf1k_fixation_density_audio_only_workflow_20260609.md`
- Fixation-density label audit:
  `results/dhf1k_attention_label_audit_fixation_density_20260609.json`
- Positive target: positive Spearman correlation between `capture_score` and
  DHF1K high/low ocular attention labels.
- Negative/control pressure: BOLD Moments memorability controls remain
  non-validating; overlapping masks are sensitivity only.
- Stress tests: disjoint primary masks, overlapping-mask sensitivity, ROI
  diagnostic decomposition after failure.

Gate:

- Acceptance rule: `capture_score` Spearman `rho >= 0.40` in at least one real
  dataset.
- Withheld/rejected rule: any lower result blocks Phase 2/3 claim progression;
  audio-only TRIBE blocks text/language claims.

Results:

- Accepted artifacts: Modal CPU fixation-density label builder; portable
  fixation-density 350-row DHF1K label CSV; complete audio-only TRIBE feature
  set for the DHF1K extreme-tail manifests; claim-ready preflight and alignment
  audits.
- Rejected or withheld artifacts: current H2 `capture_score` as a validated
  DHF1K attention proxy; Phase 2 trigger decomposition; Phase 3 neutralization.
- Key metrics:
  - Mean-map proxy: primary `rho = 0.1256`, permutation `p = 0.0130`, gate
    false; overlap sensitivity `rho = 0.2590`, gate false.
  - Mean fixation density: primary `rho = -0.0348`, permutation `p = 0.7380`,
    gate false; overlap sensitivity `rho = 0.0245`, gate false.
- Variance or ablation: fixation-density ROI diagnostics show no compensating
  single-ROI result strong enough to rescue the claim without post hoc metric
  selection.

Residual content:

- Explained by old regime: the infrastructure can run real DHF1K labels through
  TRIBE and reject unsupported attention-capture claims.
- New content outside old regime: none accepted yet; Modal CPU label fanout is
  an execution improvement, not a new scientific discovery.
- Retractions or supersessions: the Phase 1 proposal should no longer state that
  DHF1K supports H2 under the current audio-only score.

Next move:

1. Do not proceed to Phase 2/3 neutralization from the current DHF1K result.
2. Unblock full multimodal TRIBE by providing gated text-model credentials, then
   rerun only if language/text claims are required.
3. Acquire real SnapUGC/VQualA retention labels and test the current score
   against platform retention, because DHF1K fixation density may not measure
   the same construct as short-form completion.
4. If revising the score, preregister the new formula on one evidence source and
   reserve another source for held-out validation before writing a
   publication-grade paper.
