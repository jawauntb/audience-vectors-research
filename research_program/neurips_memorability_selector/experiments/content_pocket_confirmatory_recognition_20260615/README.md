# Content-Pocket Confirmatory Recognition Study

Date: 2026-06-15

Status: `setup_no_generation_no_human_data`

This folder sets up the next, larger content-pocket recognition-memory study.
The goal is not to make another retrospective pocket note. The goal is a
prospective, preregistered, durable, multi-pocket study that asks:

> Can compute selectors prospectively choose generated videos that humans
> recognize better after a delay?

## Why This Exists

The first content-pocket human result supports a narrow claim: the pooled
orange-flowers/hanging-clothes packet was recognized better than hard controls
in a delayed old-vs-lure task. The next study upgrades the regime:

- from two primary pockets to a multi-pocket benchmark;
- from old SVD replay artifacts to fresh Seedance 2.0 generations;
- from retrospective pocket explanation to prospective selector assignment;
- from webhook-limited collection to durable response capture;
- from aggregate-only paper figures to a releasable stimulus artifact bundle.

## Files

- `protocol_20260615.md`: study design, gates, and claim boundary.
- `confirmatory_study_config_20260615.json`: machine-readable study config.
- `preregistration_analysis_plan_20260615.md`: primary endpoint and model.
- `model_baseline_scoring_plan_20260615.md`: required compute baselines.
- `seedance_modal_generation_runbook_20260615.md`: generation/scoring/upload
  runbook for Modal plus Seedance credentials from Doppler.
- `artifact_release_plan_20260615.md`: public stimulus bundle and privacy
  boundary.
- `confirmatory_form_plan_20260615.json`: generated counterbalanced form plan.
- `confirmatory_form_plan_20260615.md`: readable form-plan summary.

## Immediate Next Operations

1. Review and freeze the protocol/config.
2. Run the Seedance generation adapter in dry-run/cost-estimate mode.
3. Implement the live provider call only after the dry-run report is reviewed.
4. Generate candidate old videos only after the manifest is frozen.
5. Score candidates with TRIBE/BMD, V-JEPA, CLIP, quality, saliency, and simple
   visual descriptors.
6. Select one `selector_top` and one `quality_matched_control` item per content
   family.
7. Generate same-category lures for selected old videos.
8. Screen videos, upload to durable HTTPS storage, and freeze launch assets.
9. Launch Session 1/Session 2 with a durable collector and preregistered
   mixed-effects analysis.

The dry-run entry point is:

```bash
uv run python scripts/generate_confirmatory_seedance_videos.py \
  --phase candidate_old_videos \
  --dry-run
```

It writes `seedance_candidate_generation_dry_run_20260615.json` and
`seedance_candidate_generation_dry_run_20260615.md`. The report resolves the
OpenRouter model as `bytedance/seedance-2.0` and estimates cost from the
committed 5-second 1280x720 settings. Live generation still requires a separate
provider-call implementation and final spend review.

## Allowed Claim Before Human Data

Only a protocol/setup claim is allowed now. No new human-memory, broad
memorability, or generator-control claim is created by this folder.
