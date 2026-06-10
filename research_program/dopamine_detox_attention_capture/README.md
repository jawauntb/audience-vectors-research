# Dopamine Detox Attention-Capture Experiment

Status: Phase 1 DHF1K audio-only validation run complete; current
`capture_score` gate failed. The DHF1K audio-only feature cache now has
checksum provenance and a deterministic rerun path. DHF1K videos for the
350-sample fixation-density set are mounted in Modal `bmd-videos-v1`, and
full-mode event construction passes on an actual DHF1K AVI. Full-mode
prediction still fails without gated `meta-llama/Llama-3.2-3B` access, so the
paper remains blocked on scientific evidence gates and full-prediction access,
not DHF1K media placement or local CPU throughput.

This subfolder sets up the short-form-video attention-capture experiment from
the June 2026 proposal, with the claim boundary inherited from the existing
memorability-selector work:

```text
external human / gaze / measured-brain labels
  > TRIBE ROI proxy scores
  > perturbation and generation workflows
```

The current code path is reusable for real SnapUGC, DHF1K, or Memento10k-style
manifests. The scaffold first proved that the manifest, ROI scoring, Spearman
gate, denominator guard, and report format work; the 2026-06-09 DHF1K run then
advanced to real public gaze labels under TRIBE audio-only mode. That real run
does not validate the current attentional-capture score.

## 2026-06-09 DHF1K Verdict

The fast path is Modal CPU fanout for DHF1K label construction plus Modal GPU
TRIBE scoring. Local label scanning is no longer the preferred route for the
fixation-density ground truth: `scripts/build_dhf1k_fixation_labels_modal.py`
dispatches one annotated DHF1K video per CPU task and writes a standard label
audit/CSV pair.

Compute placement rule: use Modal CPU fanout for dataset/label scans,
manifest preflights, and checksum audits when source artifacts already live in
a Modal Volume; use Modal GPU containers for TRIBE inference; use local compute
only for small report rendering or already-local artifacts where upload would be
slower than the work.

Two DHF1K audio-only validation runs are now preserved:

```text
mean_map_intensity proxy:
  primary disjoint capture_score rho = 0.1256, permutation p = 0.0130, gate = false
  overlapping-mask sensitivity rho = 0.2590, gate = false

mean_fixation_density proposal metric:
  primary disjoint capture_score rho = -0.0348, permutation p = 0.7380, gate = false
  overlapping-mask sensitivity rho = 0.0245, gate = false
```

The preregistered Phase 1 gate remains `rho >= 0.40` in at least one real
dataset, so Phase 2 trigger decomposition and Phase 3 neutralization should not
proceed from this score. The correct public DHF1K fixation-density test is a
clear withholding result, not a near miss.

Full multimodal TRIBE status is now split into three gates. First, cached
weights and event construction work for one existing Modal-hosted BMD video.
Second, the official DHF1K `video.rar` archive has been ingested on Modal and
all 350 fixation-density videos are readable under
`/bmd-videos/attention_capture/DHF1K/video/`. Third, full prediction on an
actual DHF1K video still fails because the downstream TRIBE text path attempts
to access gated `meta-llama/Llama-3.2-3B` without an available token. The
completed DHF1K runs therefore remain `--event-mode audio-only`, and
language-dependent claims remain withheld until a successful full-prediction
smoke passes and the DHF1K or SnapUGC/VQualA features are rerun in full mode.
The shortest credible route to a publication-grade result is now real
SnapUGC/VQualA retention labels plus a working full-prediction TRIBE path, or a
preregistered revised score trained on one evidence source and tested on a
held-out source.

## 2026-06-10 Publication Path Audit

`scripts/audit_attention_capture_publication_path.py` is the current top-level
paper-readiness gate. It is intentionally stricter than the data-readiness audit:
a manifest can be runnable while the paper claim remains blocked.

Current verdict:

```text
publication_ready: false
paper_claim_allowed: false
phase2_ready: false
phase1_gate_passed: false
full_multimodal_ready: false
dhf1k_modal_media_ready: true
blocking_reasons:
  - current H2 capture_score failed the Phase 1 rho gate
  - no SnapUGC/VQualA retention label CSV is mounted or available in audited Modal volumes
  - completed TRIBE workflows are audio-only and no successful full multimodal TRIBE prediction smoke is available
  - fewer than 2 external datasets have completed claim-ready workflow reports
warnings:
  - at least one TRIBE full-mode prediction smoke audit failed
```

The shortest sound trajectory is therefore:

1. Do not run Phase 2/3 neutralization from the current H2 score.
2. Acquire or mount granted SnapUGC/VQualA retention labels; the audited Modal
   volumes do not currently contain a claim-ready retention label source.
3. Provide a HuggingFace token with access to the gated TRIBE text path, or
   otherwise make the cached Llama path usable, then rerun the full-prediction
   smoke before any full DHF1K/SnapUGC feature extraction.
4. If the score is revised, preregister the formula before evaluating held-out
   data.

The DHF1K audio-only feature-cache reproducibility warning is resolved by the
checksum audit plus recorded rerun commands. An object-storage archive is still
useful for byte-for-byte reuse, but it is no longer the shortest-path blocker.
`results/modal_asset_audit_20260610.*` records the Modal-side search over 20
volumes and 19 checked Modal secrets; it found cached TRIBE/Llama weights but no
retention-label candidates or HuggingFace token env names.
`results/dhf1k_modal_video_ingest_20260610.*` records the Modal-side download
and ingest of the official DHF1K `video.rar` into `bmd-videos-v1`.
`results/dhf1k_modal_media_audit_20260610.*` records that all 350 expected
fixation-density videos are mounted and nonzero.
`results/tribe_full_preflight_dhf1k_audit_20260610.*` records successful
full-mode event construction on an actual DHF1K AVI.
`results/tribe_full_prediction_smoke_dhf1k_audit_20260610.*` records the
stricter full-prediction failure on gated Llama access.

## Files

- `soundness_audit_20260608.md`: pre-run assessment of the approach.
- `phase1_dhf1k_verdict_20260609.md`: post-run Discovery-Regime audit and
  shortest credible trajectory after the DHF1K gate failure.
- `dhf1k_attention_labels_fixation_density_extremes_20260609.csv`: 350-video
  DHF1K high/low tail label CSV ranked by mean fixation density.
- `dhf1k_attention_labels_fixation_density_extremes_modal_20260610.csv`:
  copy of the fixation-density label CSV with `video_path` rewritten to the
  audited Modal `bmd-videos-v1` paths.
- `phase1_dhf1k_audio_only_manifest_20260609.json`: complete DHF1K
  mean-map-intensity audio-only manifest.
- `phase1_dhf1k_fixation_density_audio_only_manifest_20260609.json`: complete
  DHF1K mean-fixation-density audio-only manifest.
- `phase1_synthetic_smoke_manifest_20260608.json`: tiny fixture manifest.
- `phase1_synthetic_alignment_labels_20260608.csv`: tiny synthetic label CSV
  used only to smoke-test label-to-feature alignment.
- `phase1_synthetic_alignment_manifest_20260608.json`: synthetic manifest built
  from the alignment fixture while recording the alignment-audit hash.
- `fixtures/phase1_synthetic_alignment_features_20260608/*.npz`: tiny synthetic
  TRIBE-shaped `frames` NPZ files used only for the alignment smoke test.
- `results/phase1_synthetic_smoke_preflight_20260608.json`: machine-readable
  manifest preflight for the synthetic fixture.
- `results/phase1_synthetic_smoke_preflight_20260608.md`: readable synthetic
  fixture preflight report.
- `results/phase1_synthetic_smoke_20260608.json`: machine-readable smoke result.
- `results/phase1_synthetic_smoke_20260608.md`: readable smoke report.
- `results/phase1_synthetic_smoke_sensitivity_20260608.json`: machine-readable
  sensitivity-run smoke report. The fixture uses explicit ROI values, so mask
  choice is intentionally a no-op here.
- `results/phase1_synthetic_smoke_sensitivity_20260608.md`: readable
  sensitivity-run smoke report.
- `results/phase1_synthetic_smoke_workflow_20260608.json`: guarded workflow
  smoke report that preflights, scores diagnostically, and compares masks while
  keeping claim validation blocked.
- `results/phase1_synthetic_smoke_workflow_20260608.md`: readable guarded
  workflow smoke report.
- `results/phase1_data_readiness_20260608.json`: local data-readiness audit for
  DHF1K/SnapUGC labels, cached TRIBE feature directories, ROI masks, and Phase 1
  manifests.
- `results/phase1_data_readiness_20260608.md`: readable local data-readiness
  audit.
- `results/attention_capture_publication_path_audit_20260610.json`: current
  machine-readable paper-readiness audit.
- `results/attention_capture_publication_path_audit_20260610.md`: current
  readable paper-readiness audit.
- `results/modal_asset_audit_20260610.json`: Modal CPU audit over known volumes
  and checked secrets for remote publication unblocks.
- `results/modal_asset_audit_20260610.md`: readable Modal asset audit.
- `results/tribe_full_preflight_audit_20260610.json`: one-video Modal H100
  full-mode TRIBE event preflight audit against cached TRIBE/Llama weights.
- `results/tribe_full_preflight_audit_20260610.md`: readable full-mode TRIBE
  event preflight audit.
- `results/dhf1k_modal_video_ingest_20260610.json`: Modal-side official DHF1K
  `video.rar` ingest report for the 350 fixation-density videos.
- `results/dhf1k_modal_video_ingest_20260610.md`: readable DHF1K Modal ingest
  report.
- `results/dhf1k_modal_media_audit_20260610.json`: Modal-side media audit
  proving the 350 fixation-density DHF1K videos are mounted and nonzero.
- `results/dhf1k_modal_media_audit_20260610.md`: readable DHF1K Modal media
  audit.
- `results/tribe_full_preflight_dhf1k_audit_20260610.json`: full-mode TRIBE
  event preflight audit on the actual mounted DHF1K `003.AVI`.
- `results/tribe_full_preflight_dhf1k_audit_20260610.md`: readable DHF1K
  full-mode event preflight audit.
- `results/tribe_full_prediction_smoke_dhf1k_audit_20260610.json`: stricter
  one-video full-prediction smoke audit; currently fails on gated Llama access.
- `results/tribe_full_prediction_smoke_dhf1k_audit_20260610.md`: readable
  full-prediction smoke audit.
- `results/dhf1k_audio_only_feature_cache_audit_20260610.json`: portable
  checksum/provenance/rerun audit for the external DHF1K audio-only TRIBE
  feature cache.
- `results/dhf1k_audio_only_feature_cache_audit_20260610.md`: readable
  checksum/provenance/rerun audit for the same cache.
- `results/phase1_synthetic_alignment_20260608.json`: label-to-feature
  alignment smoke report over the tiny synthetic fixture.
- `results/phase1_synthetic_alignment_20260608.md`: readable alignment smoke
  report.
- `results/bmd_memorability_control_20260608.json`: BOLD Moments control result
  over 1,022 cached TRIBE feature files using overlapping exploratory masks.
- `results/bmd_memorability_control_20260608.md`: readable overlapping-mask
  BOLD Moments control report.
- `results/bmd_memorability_control_disjoint_20260608.json`: BOLD Moments
  control result using the disjoint `drop_shared` mask policy.
- `results/bmd_memorability_control_disjoint_20260608.md`: readable disjoint
  BOLD Moments control report.
- `results/destrieux_roi_masks_20260608.npz`: frozen exploratory Destrieux ROI
  masks with overlapping vertices allowed.
- `results/destrieux_roi_mask_audit_20260608.json`: machine-readable ROI mask
  coverage and overlap audit.
- `results/destrieux_roi_mask_audit_20260608.md`: readable ROI mask audit.
- `results/destrieux_roi_masks_disjoint_20260608.npz`: frozen exploratory
  Destrieux ROI masks after removing vertices shared by more than one ROI.
- `results/destrieux_roi_mask_audit_disjoint_20260608.json`: machine-readable
  disjoint ROI mask coverage and overlap audit.
- `results/destrieux_roi_mask_audit_disjoint_20260608.md`: readable disjoint ROI
  mask audit.
- `scripts/build_attention_capture_phase1_manifest.py`: CSV-to-manifest bridge
  for real SnapUGC, DHF1K, or similar external-label datasets once cached TRIBE
  NPZ files exist. It can consume an alignment audit and record its hash in the
  manifest metadata.
- `scripts/build_dhf1k_attention_labels.py`: DHF1K annotation-map label builder
  that emits gaze/saliency CSV rows plus a label audit with a
  `ready_for_manifest_alignment` gate and non-degenerate ground-truth column
  recommendations.
- `scripts/build_dhf1k_fixation_labels_modal.py`: Modal CPU label builder for
  DHF1K mean fixation density. This is the preferred public DHF1K label route
  for the proposal's ocular ground truth.
- `scripts/extract_attention_capture_tribe_features.py`: generic TRIBE NPZ
  extractor for local/remote videos listed in a CSV, including `--event-mode
  audio-only` for TRIBE runs where gated text weights are unavailable.
- `scripts/preflight_attention_capture_phase1.py`: manifest/feature/label
  preflight gate before claim-relevant Phase 1 scoring. Claim-updatable
  manifests must carry alignment-audit provenance in metadata.
- `scripts/run_attention_capture_sensitivity.py`: primary-vs-sensitivity ROI
  mask runner for disjoint primary and overlapping-mask sensitivity reports.
- `scripts/run_attention_capture_phase1_workflow.py`: guarded Phase 1
  orchestrator that runs preflight first, withholds scoring when the claim gate
  is not ready, and optionally emits primary plus sensitivity reports.
- `scripts/audit_attention_capture_data_readiness.py`: local readiness audit for
  external labels/videos, cached TRIBE NPZs, ROI masks, and existing manifests.
- `scripts/audit_attention_capture_manifest_alignment.py`: label-to-feature
  alignment audit to run before building a real manifest. For DHF1K, it can
  consume the upstream label audit and carry that verifier hash forward.
- `scripts/audit_attention_capture_feature_cache.py`: portable feature-cache
  checksum/provenance audit for external TRIBE NPZ directories. This verifies
  artifact integrity and manifest coverage, and can record archive URIs or
  deterministic rerun commands; it does not validate attentional capture.
- `scripts/audit_attention_capture_modal_assets.py`: Modal CPU asset audit for
  remote label, dataset, feature-cache, and token-presence checks. It keeps
  heavy discovery inside Modal and reports only secret presence, never values.
- `scripts/ingest_dhf1k_videos_modal.py`: Modal CPU ingest for the official
  DHF1K `video.rar`, copying only the requested label-set videos into
  `bmd-videos-v1`.
- `scripts/audit_attention_capture_dhf1k_modal_media.py`: Modal CPU verifier
  for DHF1K media availability under the TRIBE-readable `bmd-videos-v1` mount.
- `scripts/audit_attention_capture_tribe_full_preflight.py`: one-video
  full-mode TRIBE event preflight verifier for cached Modal weights. This
  checks runtime event construction only; it does not score Phase 1 or prove
  full prediction.
- `scripts/audit_attention_capture_tribe_full_prediction_smoke.py`: one-video
  full-mode TRIBE prediction smoke. This is the required runtime gate before
  full-mode feature extraction can be treated as runnable.
- `scripts/audit_attention_capture_publication_path.py`: stricter
  paper-readiness audit that consumes readiness, workflow, feature-cache, and
  Modal verifier reports, then blocks publication/Phase 2 when the score gate,
  retention labels, full-mode evidence, feature-cache provenance, or held-out
  evidence are missing.

## Reused Infrastructure

- `audience_vectors.services.TribeService`: Modal TRIBE predictor wrapper for
  real MP4 scoring.
- `audience_vectors.features.tribe_extractor`: existing TRIBE NPZ feature-cache
  convention.
- `scripts/roi_decomposition.py`: prior Destrieux/fsaverage5 ROI decomposition
  pattern.
- `audience_vectors.bo_replay.score_projection`: precedent for aggregating
  TRIBE frame tensors before scoring.
- Existing claim-ledger discipline around compute-proxy versus human evidence.

## Run

Freeze the exploratory Destrieux masks with overlapping vertices retained:

```bash
uv run python scripts/build_attention_capture_roi_masks.py \
  --overlap-policy allow \
  --output-npz research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_20260608.npz \
  --output-json research_program/dopamine_detox_attention_capture/results/destrieux_roi_mask_audit_20260608.json \
  --output-md research_program/dopamine_detox_attention_capture/results/destrieux_roi_mask_audit_20260608.md
```

Freeze the recommended real-Phase-1 masks with shared vertices removed:

```bash
uv run python scripts/build_attention_capture_roi_masks.py \
  --overlap-policy drop_shared \
  --output-npz research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_disjoint_20260608.npz \
  --output-json research_program/dopamine_detox_attention_capture/results/destrieux_roi_mask_audit_disjoint_20260608.json \
  --output-md research_program/dopamine_detox_attention_capture/results/destrieux_roi_mask_audit_disjoint_20260608.md
```

Run the synthetic smoke test:

```bash
uv run python scripts/run_attention_capture_phase1.py \
  --manifest research_program/dopamine_detox_attention_capture/phase1_synthetic_smoke_manifest_20260608.json \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_synthetic_smoke_20260608.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_synthetic_smoke_20260608.md \
  --permutations 999 \
  --seed 20260608
```

Audit local data readiness before trying a real Phase 1 handoff:

```bash
uv run python scripts/audit_attention_capture_data_readiness.py \
  --search-root . \
  --search-root data/attention_capture \
  --search-root /Users/jawaun/isc_mod/data \
  --search-root /Users/jawaun/data \
  --search-root /Users/jawaun/datasets \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_data_readiness_20260608.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_data_readiness_20260608.md
```

Audit Modal-hosted assets without local dataset crawling:

```bash
uv run --extra modal modal run scripts/audit_attention_capture_modal_assets.py \
  --output-json research_program/dopamine_detox_attention_capture/results/modal_asset_audit_20260610.json \
  --output-md research_program/dopamine_detox_attention_capture/results/modal_asset_audit_20260610.md \
  --max-entries-per-volume 20000 \
  --max-depth 5 \
  --preview-limit 120
```

Audit full-mode TRIBE event construction from cached Modal weights:

```bash
uv run --extra modal python scripts/audit_attention_capture_tribe_full_preflight.py \
  --media-path /bmd-videos/generated/bo_memorability_replay/bo_replay_00_sobol_prompt_search_518_slot18_rep00.mp4 \
  --output-json research_program/dopamine_detox_attention_capture/results/tribe_full_preflight_audit_20260610.json \
  --output-md research_program/dopamine_detox_attention_capture/results/tribe_full_preflight_audit_20260610.md \
  --event-mode full
```

Ingest the official DHF1K videos into the Modal volume used by TRIBE, copying
only the 350 fixation-density videos needed for the current public rerun:

```bash
uv run --extra modal modal run scripts/ingest_dhf1k_videos_modal.py \
  --labels-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_20260609.csv \
  --output-json research_program/dopamine_detox_attention_capture/results/dhf1k_modal_video_ingest_20260610.json \
  --output-md research_program/dopamine_detox_attention_capture/results/dhf1k_modal_video_ingest_20260610.md \
  --expected-min-videos 350
```

Audit that the DHF1K videos are visible from the same Modal mount used by
TRIBE, and emit a Modal-path label CSV for later extraction:

```bash
uv run --extra modal modal run scripts/audit_attention_capture_dhf1k_modal_media.py \
  --labels-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_20260609.csv \
  --output-json research_program/dopamine_detox_attention_capture/results/dhf1k_modal_media_audit_20260610.json \
  --output-md research_program/dopamine_detox_attention_capture/results/dhf1k_modal_media_audit_20260610.md \
  --output-modal-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_modal_20260610.csv \
  --preview-limit 25
```

Then run the two one-video DHF1K runtime gates. Event construction currently
passes; full prediction currently fails on gated Llama access:

```bash
uv run --extra modal python scripts/audit_attention_capture_tribe_full_preflight.py \
  --media-path /bmd-videos/attention_capture/DHF1K/video/003.AVI \
  --output-json research_program/dopamine_detox_attention_capture/results/tribe_full_preflight_dhf1k_audit_20260610.json \
  --output-md research_program/dopamine_detox_attention_capture/results/tribe_full_preflight_dhf1k_audit_20260610.md \
  --event-mode full

uv run --extra modal python scripts/audit_attention_capture_tribe_full_prediction_smoke.py \
  --media-path /bmd-videos/attention_capture/DHF1K/video/003.AVI \
  --output-json research_program/dopamine_detox_attention_capture/results/tribe_full_prediction_smoke_dhf1k_audit_20260610.json \
  --output-md research_program/dopamine_detox_attention_capture/results/tribe_full_prediction_smoke_dhf1k_audit_20260610.md \
  --event-mode full
```

Audit publication readiness after any Phase 1 run:

```bash
uv run python scripts/audit_attention_capture_publication_path.py \
  --readiness-json research_program/dopamine_detox_attention_capture/results/phase1_data_readiness_20260608.json \
  --workflow-json research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_audio_only_workflow_20260609.json \
  --workflow-json research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_workflow_20260609.json \
  --feature-cache-audit research_program/dopamine_detox_attention_capture/results/dhf1k_audio_only_feature_cache_audit_20260610.json \
  --modal-asset-audit research_program/dopamine_detox_attention_capture/results/modal_asset_audit_20260610.json \
  --tribe-full-preflight-audit research_program/dopamine_detox_attention_capture/results/tribe_full_preflight_audit_20260610.json \
  --tribe-full-preflight-audit research_program/dopamine_detox_attention_capture/results/tribe_full_preflight_dhf1k_audit_20260610.json \
  --tribe-full-prediction-smoke-audit research_program/dopamine_detox_attention_capture/results/tribe_full_prediction_smoke_dhf1k_audit_20260610.json \
  --dhf1k-modal-media-audit research_program/dopamine_detox_attention_capture/results/dhf1k_modal_media_audit_20260610.json \
  --output-json research_program/dopamine_detox_attention_capture/results/attention_capture_publication_path_audit_20260610.json \
  --output-md research_program/dopamine_detox_attention_capture/results/attention_capture_publication_path_audit_20260610.md
```

Current DHF1K audio-only readiness/verdict:

```text
phase1_can_run_now: true for the preserved DHF1K audio-only manifests
dhf1k_root_ready_for_label_build: true
dhf1k_label_audit_ready: true
dhf1k_labels_ready: true
snapugc_labels_ready: false
tribe_features_ready: true
dhf1k_tribe_features_ready: true for audio-only DHF1K features
roi_masks_ready: true
real_manifest_ready: true for DHF1K audio-only mean-map and fixation-density runs
full_multimodal_event_preflight_ready: true from cached Modal weights
dhf1k_modal_media_ready: true for the 350 fixation-density videos
full_multimodal_prediction_ready: false; gated Llama access blocks full prediction
full_multimodal_dhf1k_features_ready: false; preserved DHF1K features are audio-only
primary_h2_gate_passed: false
```

The scan found a mounted DHF1K root at `data/attention_capture/DHF1K/` with
1,000 videos and 700 annotation-map directories. DHF1K-specific TRIBE audio-only
features were extracted for the completed mean-map and fixation-density extreme
tail manifests. Existing generic TRIBE feature caches remain useful
infrastructure, but claim-updatable DHF1K manifests must still align to the
audited DHF1K sample IDs.

The preferred local intake point is `data/attention_capture/`, which is ignored
for datasets but tracked with a README. Mount DHF1K at
`data/attention_capture/DHF1K/` or place granted SnapUGC/VQualA labels under
`data/attention_capture/` before rerunning the readiness audit.

DHF1K readiness is intentionally split: a dataset root can be ready for label
building, but `dhf1k_labels_ready` stays false until a
`dhf1k_attention_label_audit` artifact reports
`ready_for_manifest_alignment=true` and its label CSV still exists. Synthetic,
fixture, smoke, and control CSV/feature paths may appear in diagnostics, but do
not count as real external labels or real feature caches.

`real_manifest_ready` is also provenance-gated. A claim-updatable manifest must
carry `metadata.alignment_audit` with `ready_for_manifest_build=true`, a valid
SHA-256, enough aligned features for its sample count, and zero missing
features. DHF1K manifests additionally require a ready nested label audit.
If a discovered manifest fails this provenance gate, the top-level readiness
blockers point to the manifest fix before recommending more data acquisition.

Smoke-test label-to-feature alignment:

```bash
uv run python scripts/audit_attention_capture_manifest_alignment.py \
  --labels-csv research_program/dopamine_detox_attention_capture/phase1_synthetic_alignment_labels_20260608.csv \
  --feature-dir research_program/dopamine_detox_attention_capture/fixtures/phase1_synthetic_alignment_features_20260608 \
  --dataset synthetic_alignment_fixture \
  --ground-truth-column ecr \
  --min-samples 3 \
  --min-distinct-ground-truth 3 \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_synthetic_alignment_20260608.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_synthetic_alignment_20260608.md
```

The synthetic alignment fixture is not claim evidence; it only proves that a
label CSV, feature directory, and manifest-builder inputs can be audited before
the strict manifest builder runs.

Build the synthetic alignment manifest with audit provenance:

```bash
uv run python scripts/build_attention_capture_phase1_manifest.py \
  --labels-csv research_program/dopamine_detox_attention_capture/phase1_synthetic_alignment_labels_20260608.csv \
  --feature-dir research_program/dopamine_detox_attention_capture/fixtures/phase1_synthetic_alignment_features_20260608 \
  --output research_program/dopamine_detox_attention_capture/phase1_synthetic_alignment_manifest_20260608.json \
  --dataset synthetic_alignment_fixture \
  --ground-truth-name ecr \
  --ground-truth-column ecr \
  --status synthetic_smoke_only \
  --alignment-audit research_program/dopamine_detox_attention_capture/results/phase1_synthetic_alignment_20260608.json
```

Build a real Phase 1 manifest from external labels and cached TRIBE NPZ files:

```bash
uv run python scripts/build_attention_capture_phase1_manifest.py \
  --labels-csv /absolute/path/to/labels.csv \
  --feature-dir /absolute/path/to/tribe_npz_features \
  --output research_program/dopamine_detox_attention_capture/phase1_real_manifest.json \
  --dataset SnapUGC \
  --ground-truth-name ECR \
  --sample-id-column sample_id \
  --ground-truth-column ecr \
  --alignment-audit research_program/dopamine_detox_attention_capture/results/phase1_real_alignment.json
```

For DHF1K specifically, first derive external saliency labels from the official
dataset layout (`video/001.AVI`, `annotation/001/maps/*.png`, and optional
`annotation/001/fixation/*.png`). The official repository describes 1,000
videos, with released annotations for the first 700 train/validation videos:
https://github.com/wenguanwang/DHF1K.

```bash
uv run python scripts/build_dhf1k_attention_labels.py \
  --dhf1k-root data/attention_capture/DHF1K \
  --split annotated \
  --rank-column mean_map_intensity \
  --metric-scope rank \
  --extreme-count-per-tail 175 \
  --min-rows 350 \
  --min-distinct-rank-values 3 \
  --output-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_extremes_20260608.csv \
  --output-json research_program/dopamine_detox_attention_capture/results/dhf1k_attention_label_audit_20260608.json
```

The DHF1K label audit must report `ready_for_manifest_alignment=true` before
feature extraction and manifest alignment. If the chosen `--rank-column` is
degenerate, use one of `candidate_ground_truth_columns` from the audit and rerun
the label builder before spending TRIBE compute.

For the proposal's fixation-density ground truth, prefer the Modal CPU builder:

```bash
uv run modal run scripts/build_dhf1k_fixation_labels_modal.py \
  --dhf1k-root data/attention_capture/DHF1K \
  --output-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_20260609.csv \
  --output-json research_program/dopamine_detox_attention_capture/results/dhf1k_attention_label_audit_fixation_density_20260609.json \
  --extreme-count-per-tail 175
```

Then extract TRIBE features for those videos:

```bash
uv run python scripts/extract_attention_capture_tribe_features.py \
  --source-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_20260609.csv \
  --output-dir data/features/tribe_dhf1k_attention_audio_only \
  --sample-id-column sample_id \
  --media-path-column video_path \
  --transport bytes \
  --event-mode audio-only \
  --concurrency 8
```

Audit the external feature cache before manifest construction or paper-readiness
claims. This audit checks cache integrity and manifest coverage only; it does
not validate attentional capture. If the cache is already in a Modal Volume,
run equivalent CPU fanout next to the volume rather than downloading it locally.

```bash
uv run python scripts/audit_attention_capture_feature_cache.py \
  --feature-dir data/features/tribe_dhf1k_attention_audio_only \
  --display-feature-dir data/features/tribe_dhf1k_attention_audio_only \
  --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_audio_only_manifest_20260609.json \
  --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_fixation_density_audio_only_manifest_20260609.json \
  --rerun-command "uv run python scripts/extract_attention_capture_tribe_features.py --source-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_extremes_20260608.csv --output-dir data/features/tribe_dhf1k_attention_audio_only --sample-id-column sample_id --media-path-column video_path --transport bytes --event-mode audio-only --concurrency 8" \
  --rerun-command "uv run python scripts/extract_attention_capture_tribe_features.py --source-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_20260609.csv --output-dir data/features/tribe_dhf1k_attention_audio_only --sample-id-column sample_id --media-path-column video_path --transport bytes --event-mode audio-only --concurrency 8" \
  --rerun-command "uv run python scripts/audit_attention_capture_feature_cache.py --feature-dir data/features/tribe_dhf1k_attention_audio_only --display-feature-dir data/features/tribe_dhf1k_attention_audio_only --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_audio_only_manifest_20260609.json --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_fixation_density_audio_only_manifest_20260609.json --output-json research_program/dopamine_detox_attention_capture/results/dhf1k_audio_only_feature_cache_audit_20260610.json --output-md research_program/dopamine_detox_attention_capture/results/dhf1k_audio_only_feature_cache_audit_20260610.md" \
  --output-json research_program/dopamine_detox_attention_capture/results/dhf1k_audio_only_feature_cache_audit_20260610.json \
  --output-md research_program/dopamine_detox_attention_capture/results/dhf1k_audio_only_feature_cache_audit_20260610.md
```

Then audit DHF1K label-to-feature alignment, including the upstream label audit:

```bash
uv run python scripts/audit_attention_capture_manifest_alignment.py \
  --labels-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_20260609.csv \
  --feature-dir data/features/tribe_dhf1k_attention_audio_only \
  --label-audit research_program/dopamine_detox_attention_capture/results/dhf1k_attention_label_audit_fixation_density_20260609.json \
  --dataset DHF1K \
  --ground-truth-column mean_fixation_density \
  --min-samples 350 \
  --min-distinct-ground-truth 3 \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_alignment_20260609.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_alignment_20260609.md
```

Then build the DHF1K Phase 1 manifest:

```bash
uv run python scripts/build_attention_capture_phase1_manifest.py \
  --labels-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_20260609.csv \
  --feature-dir data/features/tribe_dhf1k_attention_audio_only \
  --output research_program/dopamine_detox_attention_capture/phase1_dhf1k_fixation_density_audio_only_manifest_20260609.json \
  --dataset DHF1K \
  --ground-truth-name mean_fixation_density \
  --ground-truth-column mean_fixation_density \
  --alignment-audit research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_alignment_20260609.json
```

Preflight the manifest before claim-relevant scoring:

```bash
uv run python scripts/preflight_attention_capture_phase1.py \
  --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_fixation_density_audio_only_manifest_20260609.json \
  --roi-masks research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_disjoint_20260608.npz \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_preflight_20260609.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_preflight_20260609.md \
  --min-samples 30 \
  --min-distinct-ground-truth 3
```

For real claim-updatable manifests, preflight requires
`metadata.alignment_audit.ready_for_manifest_build=true`, a recorded alignment
audit path and SHA-256, zero missing features in that audit, and enough aligned
features for the requested sample threshold. DHF1K manifests additionally
require a ready nested `metadata.alignment_audit.label_audit`.

Preferred guarded workflow for the real DHF1K run:

```bash
uv run python scripts/run_attention_capture_phase1_workflow.py \
  --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_fixation_density_audio_only_manifest_20260609.json \
  --primary-label disjoint \
  --roi-masks research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_disjoint_20260608.npz \
  --sensitivity-roi-masks overlapping=research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_20260608.npz \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_workflow_20260609.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_workflow_20260609.md \
  --min-samples 30 \
  --min-distinct-ground-truth 3 \
  --permutations 999 \
  --seed 20260609 \
  --omit-rows
```

The workflow exits non-zero after writing its report if preflight fails or the
manifest is claim-blocked. Use `--score-claim-blocked` only for smoke/control
diagnostics, never to turn a fixture into evidence.

```bash
uv run python scripts/run_attention_capture_phase1.py \
  --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_fixation_density_audio_only_manifest_20260609.json \
  --roi-masks research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_disjoint_20260608.npz \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_disjoint_20260609.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_disjoint_20260609.md \
  --permutations 999 \
  --seed 20260609
```

Run the archived overlapping-mask sensitivity check on the same DHF1K manifest:

```bash
uv run python scripts/run_attention_capture_sensitivity.py \
  --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_fixation_density_audio_only_manifest_20260609.json \
  --primary-label disjoint \
  --primary-roi-masks research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_disjoint_20260608.npz \
  --sensitivity-roi-masks overlapping=research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_20260608.npz \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_sensitivity_20260609.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_sensitivity_20260609.md \
  --permutations 999 \
  --seed 20260609
```

The DHF1K label audit should be inspected before GPU scoring. If
`mean_map_intensity` has weak variance, treat it as a proxy diagnostic rather
than the proposal's ocular ground truth. The 2026-06-09 fixation-density run is
the current public DHF1K test of the proposal metric.

For a generic real manifest, run the same preflight before scoring with the
disjoint ROI masks. Generic real manifests must be built with
`--alignment-audit`; otherwise preflight withholds claim-ready scoring:

```bash
uv run python scripts/preflight_attention_capture_phase1.py \
  --manifest research_program/dopamine_detox_attention_capture/phase1_real_manifest.json \
  --roi-masks research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_disjoint_20260608.npz \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_real_preflight.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_real_preflight.md
```

```bash
uv run python scripts/run_attention_capture_phase1.py \
  --manifest research_program/dopamine_detox_attention_capture/phase1_real_manifest.json \
  --roi-masks research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_disjoint_20260608.npz \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_real_disjoint.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_real_disjoint.md \
  --permutations 999 \
  --seed 20260608
```

Run the local BOLD Moments control if `/Users/jawaun/isc_mod/data` is present:

```bash
uv run python scripts/run_attention_capture_bmd_control.py \
  --roi-masks research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_disjoint_20260608.npz \
  --output-json research_program/dopamine_detox_attention_capture/results/bmd_memorability_control_disjoint_20260608.json \
  --output-md research_program/dopamine_detox_attention_capture/results/bmd_memorability_control_disjoint_20260608.md \
  --permutations 999 \
  --seed 20260608
```

The BMD control is deliberately marked `real_control_not_attention_capture`.
It can show whether the capture proxy overlaps with memorability, but it cannot
validate the attention-capture claim.

## Real Manifest Shape

Each sample should include either explicit ROI values:

```json
{
  "sample_id": "snapugc_000001",
  "dataset": "SnapUGC",
  "ground_truth": 0.83,
  "ground_truth_name": "ECR",
  "roi_values": {
    "V1": 0.71,
    "PPA": 0.55,
    "language": 0.42,
    "frontoparietal": 0.20
  }
}
```

or a cached TRIBE feature path, plus a separate `--roi-masks` NPZ:

```json
{
  "sample_id": "dhf1k_000001",
  "dataset": "DHF1K",
  "ground_truth": 0.68,
  "ground_truth_name": "mean_fixation_density",
  "tribe_feature_path": "/absolute/path/to/dhf1k_000001.npz"
}
```

The primary gate is `Spearman rho(capture_score, ground_truth) >= 0.40` in at
least one real dataset, where:

```text
capture_score = mean(V1, PPA, language) / (frontoparietal + epsilon)
```

Ratios with non-positive frontoparietal denominators are withheld from the
primary correlation and counted in the report. `capture_delta =
mean(V1, PPA, language) - frontoparietal` is reported as a secondary robustness
readout.

## Current Control Result

The BOLD Moments control used 1,022 cached TRIBE feature files and
memorability labels. With the recommended disjoint masks, it did not pass the
capture-score gate:

```text
capture_score vs memorability: rho = -0.2444, n = 739
capture_delta vs memorability: rho = -0.2492, n = 1022
frontoparietal vs memorability: rho = +0.2346, n = 1022
invalid ratio denominators: 283
```

This is useful negative/control evidence. Under the broad exploratory
Destrieux ROI masks, the new capture proxy is not simply the existing BMD
memorability direction. The disjoint policy improved denominator validity
relative to the overlapping masks, reducing withheld rows from 362 to 283, but
did not turn the proxy into a memorability predictor.

The overlapping mask audit also shows that the broad string-matched ROI
defaults are anatomically entangled:

```text
V1/PPA overlap: 598 vertices
language/frontoparietal overlap: 742 vertices
```

The recommended `drop_shared` mask removes all off-diagonal overlap and keeps
non-empty ROI coverage:

```text
V1: 1,619 vertices
PPA: 268 vertices
language: 3,400 vertices
frontoparietal: 3,362 vertices
```

The PPA mask is now small, so Phase 1 should report both the disjoint default
and the archived overlapping-mask sensitivity check rather than claiming the
ROI definition is final.
