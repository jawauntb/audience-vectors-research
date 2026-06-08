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

```bash
uv run python scripts/run_attention_capture_phase1.py \
  --manifest research_program/dopamine_detox_attention_capture/phase1_synthetic_smoke_manifest_20260608.json \
  --output-json research_program/dopamine_detox_attention_capture/results/phase1_synthetic_smoke_20260608.json \
  --output-md research_program/dopamine_detox_attention_capture/results/phase1_synthetic_smoke_20260608.md \
  --permutations 999 \
  --seed 20260608
```

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
