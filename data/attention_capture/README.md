# Attention-Capture External Data Mount

This directory is the preferred local mount point for claim-relevant Phase 1
attention-capture data. Datasets, clips, labels, features, and model outputs are
ignored by git; only this README and `.gitkeep` are tracked.

## DHF1K

Mount or unpack DHF1K here as:

```text
data/attention_capture/DHF1K/
  video/
    001.AVI
    ...
  annotation/
    001/
      maps/
        0001.png
        ...
      fixation/
        0001.png
        ...
```

The Phase 1 readiness audit scans `data/attention_capture` by default. Once the
root is mounted, the verdict should move from `no external attention-label
source found` to `DHF1K root found but no ready DHF1K label audit found`.

Then derive the external DHF1K labels and audit provenance:

```bash
uv run python scripts/build_dhf1k_attention_labels.py \
  --dhf1k-root data/attention_capture/DHF1K \
  --split annotated \
  --rank-column mean_map_intensity \
  --extreme-count-per-tail 175 \
  --min-rows 350 \
  --min-distinct-rank-values 3 \
  --output-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_extremes_20260608.csv \
  --output-json research_program/dopamine_detox_attention_capture/results/dhf1k_attention_label_audit_20260608.json
```

Only a label audit with `ready_for_manifest_alignment=true` can unblock the
DHF1K handoff. This folder does not make synthetic, fixture, smoke, or control
artifacts claim-ready.

## SnapUGC / VQualA

Place real platform-retention labels here only if access has been granted, for
example:

```text
data/attention_capture/snapugc_vquala_labels.csv
data/attention_capture/snapugc_videos/
```

The CSV must include a stable video/sample id and an external attention or
retention metric such as ECR, completion rate, retention, or engagement.

Before spending Modal GPU time, audit the granted labels:

```bash
uv run python scripts/audit_attention_capture_retention_labels.py \
  --labels-csv data/attention_capture/snapugc_vquala_labels.csv \
  --dataset SnapUGC \
  --sample-id-column sample_id \
  --ground-truth-column ecr \
  --media-path-column video_path \
  --ground-truth-name ecr \
  --output-json research_program/dopamine_detox_attention_capture/results/snapugc_retention_label_audit.json \
  --output-md research_program/dopamine_detox_attention_capture/results/snapugc_retention_label_audit.md
```

Only a report with `ready_for_manifest_alignment=true` should be used for a
claim-relevant Phase 1 run. `ready_for_modal_feature_extraction=true` additionally
means the CSV has a usable media path column for the Modal feature extractor.

Then run the cheap CSV-metadata baseline/control audit before launching TRIBE:

```bash
uv run python scripts/audit_attention_capture_retention_baselines.py \
  --labels-csv data/attention_capture/snapugc_vquala_labels.csv \
  --label-audit research_program/dopamine_detox_attention_capture/results/snapugc_retention_label_audit.json \
  --dataset SnapUGC \
  --sample-id-column sample_id \
  --ground-truth-column ecr \
  --media-path-column video_path \
  --ground-truth-name ecr \
  --output-json research_program/dopamine_detox_attention_capture/results/snapugc_retention_baseline_audit.json \
  --output-md research_program/dopamine_detox_attention_capture/results/snapugc_retention_baseline_audit.md
```

This baseline is diagnostic, not a replacement for TRIBE. It checks whether
simple metadata has signal and whether deterministic negative controls such as
row order or sample-id hashes look suspicious before any Modal GPU spend.
