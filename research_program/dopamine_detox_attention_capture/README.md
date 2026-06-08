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
- `results/phase1_synthetic_smoke_20260608.json`: machine-readable smoke result.
- `results/phase1_synthetic_smoke_20260608.md`: readable smoke report.
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

Then score that manifest with the disjoint ROI masks:

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
