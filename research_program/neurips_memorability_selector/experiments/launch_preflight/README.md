# Launch Preflight - V-JEPA Prolific Survey

This lane prepares the V-JEPA-augmented pairwise survey for hosting. It does not
launch the study, contact participants, or send faculty/IRB materials.

## Inputs

- `../current_selector_manifest_with_vjepa.json`
- `../current_selector_pairwise_tasks_with_vjepa.json`
- `../current_selector_prolific_survey_with_vjepa.html`

## Outputs

Run from `research_program/neurips_memorability_selector/experiments`:

```bash
python launch_preflight/build_hosted_survey.py
```

This validates the current task pool and writes:

- `prolific_launch_assets_2026-06-01/hosted_video_url_map.template.json`
- `prolific_launch_assets_2026-06-01/task_randomization_freeze.json`

The URL map template contains every unique MP4 path used by the 185-task pool.
Fill each `hosted_url` with a stable public HTTPS URL and set `screened` to
`true` only after content screening is complete.

After the URL map is filled, render a hostable copy of the survey:

```bash
python launch_preflight/build_hosted_survey.py \
  --url-map prolific_launch_assets_2026-06-01/hosted_video_url_map.template.json \
  --hosted-survey prolific_launch_assets_2026-06-01/current_selector_prolific_survey_with_vjepa.hosted.html
```

The script refuses to render if any required hosted URL is blank, non-HTTPS, or
not marked screened.

## Frozen Randomization Facts

- Task pool: 185 pairwise tasks over 24 seeds.
- Per-participant sample: 24 trials.
- Participant seed: Prolific ID/query parameter when available.
- Randomization: `hashString` plus `mulberry32`, grouped by comparison family,
  then shuffled before display.

The freeze JSON records SHA-256 hashes of the task file, manifest, survey HTML,
task ID order, task payload, and unique video path set.

## Remaining Launch Blocks

- Faculty/PI and IRB exemption, approval, or written determination.
- Hosted video URLs for every unique MP4.
- Stimulus content screening.
- Completion code, compensation, Prolific settings, and response export path.
- Browser dry run of the rendered hosted survey.
