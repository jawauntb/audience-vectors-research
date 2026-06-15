# Seedance 2.0 And Modal Generation Runbook

Date: 2026-06-15

Status: `runbook_no_api_calls_yet`

## What Modal Should Handle

Modal is appropriate for everything except the Prolific researcher UI:

- parallel Seedance API generation jobs;
- candidate and lure file inventory;
- sampled-frame contact sheets;
- visual artifact screening;
- TRIBE/V-JEPA/CLIP scoring jobs;
- upload to durable HTTPS storage;
- manifest hashing and release-bundle preparation.

The human study itself should use Prolific plus a durable collector
(Supabase/Postgres, Modal-backed API with persistent DB, or another stable
database). Do not use webhook.site for the confirmatory run.

## Credential Boundary

Seedance credentials should be read at runtime from Doppler in the
`/Users/jawaun/superoptimizers` project, not committed here.

Expected local wrapper pattern:

```bash
cd /Users/jawaun/superoptimizers
doppler run --project cofounder --config dev -- bash -lc '
  cd /Users/jawaun/isc_mod/.worktrees/codex/confirmatory-recognition-study-20260615
  uv run --extra modal python scripts/generate_confirmatory_seedance_videos.py \
    --config research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/confirmatory_study_config_20260615.json \
    --phase candidate_old_videos \
    --dry-run
'
```

The first implementation pass should run `--dry-run` and print the resolved
model ID, job count, estimated cost, and output paths. Do not spend generation
budget until the dry-run manifest is reviewed.

Committed dry-run entry point:

```bash
uv run python scripts/generate_confirmatory_seedance_videos.py \
  --phase candidate_old_videos \
  --dry-run
```

To clear the model/cost preflight blockers without logging credentials:

```bash
uv run python scripts/generate_confirmatory_seedance_videos.py \
  --phase candidate_old_videos \
  --model-id <provider_seedance_2_model_id> \
  --estimated-cost-per-video-usd <reviewed_cost> \
  --dry-run
```

The script may also read `SEEDANCE_MODEL_ID` and
`SEEDANCE_ESTIMATED_COST_PER_VIDEO_USD` from the environment. It records only
whether recognized credential variables are present; it never records secret
values.

## Generation Phases

Phase 0: freeze prompt and candidate manifest.

- 12 content families.
- 8 candidate old-video prompts per family.
- no API calls until manifest review.

Phase 1: generate candidate old videos.

- 96 analysis candidate old videos.
- no lures yet.
- retain failed and visually rejected candidates.

Phase 2: score and select.

- score all 96 candidates with the required baseline plan;
- select one `selector_top` and one `quality_matched_control` per family;
- freeze selected old-video manifest.

Phase 3: generate lures.

- 2 lure attempts per selected old video;
- same broad category, different object layout and camera geometry;
- reject near-duplicates and artifact failures;
- freeze one accepted lure per selected old video plus backups where possible.

Phase 4: generate fillers.

- 24 unrelated old/lure filler pairs;
- avoid content overlap with analysis families;
- screen exactly like analysis videos.

Phase 5: upload and hash.

- upload accepted videos to stable HTTPS storage;
- write URL map and SHA256 hashes;
- verify every URL with range/video load checks;
- preserve local and remote path inventory.

## Recommended Durable Storage

For Prolific and public review, use Supabase Storage, Cloudflare R2, S3, or
another stable object store. Modal can orchestrate upload, but Modal web
endpoints should not be the only long-term source of truth for stimuli.

For publication scrutiny, export the final accepted old/lure videos to OSF or
Zenodo with a DOI.

## Pre-Launch Stop Rules

Stop before Prolific if any of these hold:

- no stable HTTPS URL for a launch video;
- missing SHA256 hash;
- selected old and lure are too visually similar;
- any selected video has text/watermark/personally identifying content;
- compute scores are missing for any analysis old video;
- visual quality differs dramatically between selector-top and control items;
- form plan cannot be regenerated from committed config.
