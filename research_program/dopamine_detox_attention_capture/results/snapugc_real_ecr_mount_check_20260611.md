# SnapUGC Real ECR Mount Check

## Verdict

No granted behavioral SnapUGC/VQualA ECR label CSV is mounted locally as of this
checkpoint.

- Real SnapUGC ECR labels mounted: false
- Phase 1 SnapUGC can run now: false
- Modal TRIBE slice launched: false
- Claim-ready: false

This is a deliberate stop. The public `submission_baseline.csv` style file is
usable for format smoke tests only; it is not external behavioral ground truth.

## What Was Checked

The local search covered the active worktrees plus likely user data locations:

- `/Users/jawaun/.codex/worktrees`
- `/Users/jawaun/Downloads`
- `/Users/jawaun/Desktop`
- `/Users/jawaun/Documents`
- `/Users/jawaun/isc_mod`
- `data/attention_capture`

The search used SnapUGC/VQualA/ECR/retention/completion/val-data/submission
hints. It found only the previous public-format smoke artifacts:

- `snapugc_public_format_smoke_20260611.json`
- `snapugc_public_format_smoke_20260611.md`

No claim-relevant granted ECR label file was found.

## Required Inputs

Mount the real files here:

```text
data/attention_capture/snapugc_val_data.csv
data/attention_capture/snapugc_ecr_labels.csv
```

The metadata CSV should contain:

```text
Id,Title,Description,Download_link
```

The granted behavioral label CSV should contain:

```text
Id,ECR
```

Optional media can be mounted under:

```text
data/attention_capture/snapugc_videos/
```

If the clips are mirrored only in Modal, use
`--media-path-template '/bmd-videos/attention_capture/SnapUGC/{sample_id}.mp4'`
when building the canonical labels.

## Next Commands Once Mounted

```bash
uv run python scripts/build_snapugc_retention_labels.py \
  --metadata-csv data/attention_capture/snapugc_val_data.csv \
  --scores-csv data/attention_capture/snapugc_ecr_labels.csv \
  --output-csv data/attention_capture/snapugc_vquala_labels.csv \
  --output-json research_program/dopamine_detox_attention_capture/results/snapugc_retention_label_builder.json \
  --output-md research_program/dopamine_detox_attention_capture/results/snapugc_retention_label_builder.md
```

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

Only after those audits pass should we launch a Modal TRIBE slice.

## Claim Boundary

This artifact records a blocker, not an experimental result. It does not score
TRIBE, does not validate H2, and does not support Phase 2/3 neutralization
claims.
