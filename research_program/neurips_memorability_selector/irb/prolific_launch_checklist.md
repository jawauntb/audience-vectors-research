# Prolific Launch Checklist

Use this only after faculty/IRB review or a determination that the pilot can run.

## Before Launch

- [ ] Faculty sponsor / PI confirmed.
- [ ] IRB exemption, approval, or written determination received.
- [ ] Consent form finalized with institutional contact information.
- [ ] Stimulus videos screened for sensitive, graphic, sexual, medical,
      political, or distressing content.
- [ ] Final participant inclusion/exclusion criteria documented.
- [ ] Compensation set to meet Prolific fair-pay guidelines and institutional
      requirements.
- [ ] Data-retention location and access list documented.
- [ ] Prolific IDs handling plan finalized.
- [ ] Attention checks finalized.
- [ ] Exclusion rules predeclared.

## Study Assets

- [ ] Hosted video URLs work outside the local filesystem.
- [ ] `../experiments/prolific_launch_assets_2026-06-01/hosted_video_url_map.template.json`
      filled with HTTPS URLs for every unique MP4.
- [ ] Every hosted URL map entry marked screened only after stimulus screening.
- [ ] Hosted survey rendered with `../experiments/launch_preflight/build_hosted_survey.py`.
- [ ] Pairwise task file and randomization metadata frozen in
      `../experiments/prolific_launch_assets_2026-06-01/task_randomization_freeze.json`.
- [ ] Pilot run tested locally in Chrome and Safari.
- [ ] Completion code configured.
- [ ] Webhook or response export path tested.

## Current Local Assets

- `../experiments/current_selector_manifest_with_vjepa.json`
- `../experiments/current_selector_pairwise_tasks_with_vjepa.json`
- `../experiments/current_selector_prolific_survey_with_vjepa.html`
- `../experiments/selector_human_eval_protocol.md`

## Recommended First Pilot

- 24-50 prompts if using the current candidate pool.
- 20 raters per comparison family for calibration.
- Pairwise predicted-memorability task first.
- Delayed-recognition task only after the pairwise pilot shows a useful effect.

## Primary Analysis After Data Collection

- Prompt-clustered bootstrap CI.
- Mixed-effects logistic regression.
- Per-prompt win-rate table.
- Attention-check exclusion report.
- Failure-case review for prompts where TRIBE+gate loses.
