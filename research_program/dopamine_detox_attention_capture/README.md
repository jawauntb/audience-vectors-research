# Dopamine Detox Attention-Capture Experiment

Status: Phase 1 scaffold and synthetic smoke run.

This subfolder sets up the short-form-video attention-capture experiment from
the June 2026 proposal, with the claim boundary inherited from the existing
memorability-selector work:

```text
external human / gaze / measured-brain labels
  > TRIBE ROI proxy scores
  > perturbation and generation workflows
```

The current code path is reusable for real SnapUGC, DHF1K, or Memento10k-style
manifests, but the committed result is only a synthetic smoke test. It proves
that the manifest, ROI scoring, Spearman gate, denominator guard, and report
format work. It does not validate attentional capture.

## Files

- `soundness_audit_20260608.md`: pre-run assessment of the approach.
- `phase1_synthetic_smoke_manifest_20260608.json`: tiny fixture manifest.
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
  NPZ files exist.
- `scripts/build_dhf1k_attention_labels.py`: DHF1K annotation-map label builder
  that emits gaze/saliency CSV rows plus a label audit.
- `scripts/extract_attention_capture_tribe_features.py`: generic TRIBE NPZ
  extractor for local/remote videos listed in a CSV.
- `scripts/preflight_attention_capture_phase1.py`: manifest/feature/label
  preflight gate before claim-relevant Phase 1 scoring.
- `scripts/run_attention_capture_sensitivity.py`: primary-vs-sensitivity ROI
  mask runner for disjoint primary and overlapping-mask sensitivity reports.
- `scripts/run_attention_capture_phase1_workflow.py`: guarded Phase 1
  orchestrator that runs preflight first, withholds scoring when the claim gate
  is not ready, and optionally emits primary plus sensitivity reports.

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

Build a real Phase 1 manifest from external labels and cached TRIBE NPZ files:

```bash
uv run python scripts/build_attention_capture_phase1_manifest.py \
  --labels-csv /absolute/path/to/labels.csv \
  --feature-dir /absolute/path/to/tribe_npz_features \
  --output research_program/dopamine_detox_attention_capture/phase1_real_manifest.json \
  --dataset SnapUGC \
  --ground-truth-name ECR \
  --sample-id-column sample_id \
  --ground-truth-column ecr
```

For DHF1K specifically, first derive external saliency labels from the official
dataset layout (`video/001.AVI`, `annotation/001/maps/*.png`, and optional
`annotation/001/fixation/*.png`). The official repository describes 1,000
videos, with released annotations for the first 700 train/validation videos:
https://github.com/wenguanwang/DHF1K.

```bash
uv run python scripts/build_dhf1k_attention_labels.py \
  --dhf1k-root /absolute/path/to/DHF1K \
  --split annotated \
  --rank-column mean_map_intensity \
  --extreme-count-per-tail 175 \
  --output-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_extremes_20260608.csv \
  --output-json research_program/dopamine_detox_attention_capture/results/dhf1k_attention_label_audit_20260608.json
```

Then extract TRIBE features for those videos:

```bash
uv run python scripts/extract_attention_capture_tribe_features.py \
  --source-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_extremes_20260608.csv \
  --output-dir data/features/tribe_dhf1k_attention \
  --sample-id-column sample_id \
  --media-path-column video_path \
  --transport bytes \
  --concurrency 4
```

Then build and score the DHF1K Phase 1 manifest:

```bash
uv run python scripts/build_attention_capture_phase1_manifest.py \
  --labels-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_extremes_20260608.csv \
  --feature-dir data/features/tribe_dhf1k_attention \
  --output research_program/dopamine_detox_attention_capture/phase1_dhf1k_manifest_20260608.json \
  --dataset DHF1K \
  --ground-truth-name mean_map_intensity \
  --ground-truth-column mean_map_intensity
```

Preflight the manifest before claim-relevant scoring:

```bash
uv run python scripts/preflight_attention_capture_phase1.py \
  --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_manifest_20260608.json \
  --roi-masks research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_disjoint_20260608.npz \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_preflight_20260608.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_preflight_20260608.md \
  --min-samples 30 \
  --min-distinct-ground-truth 3
```

Preferred guarded workflow for the real DHF1K run:

```bash
uv run python scripts/run_attention_capture_phase1_workflow.py \
  --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_manifest_20260608.json \
  --primary-label disjoint \
  --roi-masks research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_disjoint_20260608.npz \
  --sensitivity-roi-masks overlapping=research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_20260608.npz \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_workflow_20260608.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_workflow_20260608.md \
  --min-samples 30 \
  --min-distinct-ground-truth 3 \
  --permutations 999 \
  --seed 20260608 \
  --omit-rows
```

The workflow exits non-zero after writing its report if preflight fails or the
manifest is claim-blocked. Use `--score-claim-blocked` only for smoke/control
diagnostics, never to turn a fixture into evidence.

```bash
uv run python scripts/run_attention_capture_phase1.py \
  --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_manifest_20260608.json \
  --roi-masks research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_disjoint_20260608.npz \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_disjoint_20260608.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_disjoint_20260608.md \
  --permutations 999 \
  --seed 20260608
```

Run the archived overlapping-mask sensitivity check on the same DHF1K manifest:

```bash
uv run python scripts/run_attention_capture_sensitivity.py \
  --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_manifest_20260608.json \
  --primary-label disjoint \
  --primary-roi-masks research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_disjoint_20260608.npz \
  --sensitivity-roi-masks overlapping=research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_20260608.npz \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_sensitivity_20260608.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_sensitivity_20260608.md \
  --permutations 999 \
  --seed 20260608
```

The DHF1K label audit should be inspected before GPU scoring. If
`mean_map_intensity` has weak variance, use one of the emitted concentration
columns as a stress test rather than treating map intensity as a validated
capture metric.

For a generic real manifest, run the same preflight before scoring with the
disjoint ROI masks:

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
