# Prolific Launch Assets - V-JEPA Selector Pilot

Date prepared: 2026-06-01

This folder collects the human-evaluation launch materials for the current
V-JEPA-augmented selector pilot. It is a review packet, not evidence that the
study has launched. Use it after faculty/PI review and IRB exemption, approval,
or written determination.

## Core Question

When several generated videos are available for the same prompt, does the
TRIBE/BMD selector choose clips that people judge as more memorable than clips
chosen by non-brain baselines such as CLIP preservation and V-JEPA memorability?

## Current Local Assets

- `../current_selector_manifest_with_vjepa.json`
- `../current_selector_pairwise_tasks_with_vjepa.json`
- `../current_selector_prolific_survey_with_vjepa.html`
- `../launch_preflight/build_hosted_survey.py`
- `hosted_video_url_map.template.json`
- `task_randomization_freeze.json`
- `../selector_human_eval_protocol.md`
- `../pilot_runbook.md`
- `../../irb/irb_protocol_draft.md`
- `../../irb/prolific_launch_checklist.md`

## Current Task Set

- Seeds: 24
- Pairwise tasks: 185
- Per-participant survey sample: 24 trials, balanced across comparison families
  when possible

Comparison families:

| Comparison | Tasks |
|---|---:|
| gated_vs_base | 24 |
| product_vs_single_lora | 24 |
| product_vs_clip_prompt | 20 |
| product_vs_clip_preservation | 19 |
| product_vs_clip_seed_image | 19 |
| product_vs_base | 18 |
| product_vs_vjepa_memorability | 17 |
| gated_vs_clip_preservation | 17 |
| gated_vs_vjepa_memorability | 14 |
| product_vs_raw_best | 13 |

## What Must Happen Before Launch

- Faculty/PI decides whether this is exempt human-subjects research or requires
  formal IRB submission.
- Consent language is finalized with institutional contact information.
- Stimulus videos are hosted at stable public URLs and screened for sensitive
  content.
- The hosted URL map is filled, screened, and used to render the hosted survey
  HTML from `../launch_preflight/build_hosted_survey.py`.
- Completion code, compensation, inclusion criteria, exclusion rules, and data
  retention are frozen.
- A local dry run is completed in Chrome and Safari.

## What This Pilot Can Support

This pilot can calibrate whether the selector produces human-visible preference
signal against baselines and whether the task mechanics work. It cannot support
submission-grade claims without enough independent human responses and
prompt-clustered analysis.
