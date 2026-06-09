# Content-Pocket Recognition Response Analysis Plan

Date: 2026-06-08

## Status

- Status: `analysis_plan_ready_no_response_data`
- Analysis script:
  `scripts/analyze_content_pocket_recognition_responses.py`
- This is an analysis-prep artifact only. It does not contain human-memory
  evidence.

## Discovery-Regime Audit

Question: do primary SVD content pockets produce actual delayed old-vs-lure
recognition-memory gains over hard-negative controls?

Current regime:

- Artifact types: Session 1 exposure payloads, Session 2 recognition payloads,
  Prolific IDs, form IDs, trial-level media-error flags, old-vs-lure
  correctness, exclusion reasons, and gate summaries.
- Operations: load JSON/JSONL/CSV-wrapped response payloads, select latest
  payload per participant/session, join sessions by participant and target,
  reason-code exclusions, enforce complete-case analysis arms, and compute
  accuracy contrasts.
- Gates/verifiers: minimum usable delayed participants, complete-case coverage,
  primary-positive versus hard-negative recognition accuracy, pocket-level
  positive directions, above-chance primary accuracy, and a two-proportion
  z-test for the pooled primary contrast.
- Known limitation: no response data have been collected or analyzed yet.

Action class: validation plumbing inside the recognition-memory regime.

## Run Command

Use this after the response endpoint export exists:

```bash
uv run python scripts/analyze_content_pocket_recognition_responses.py \
  --responses data/prolific/content_pocket_recognition_20260608 \
  --out-json research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_response_analysis_result_20260608.json \
  --out-md research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_response_analysis_result_20260608.md
```

Use `--exclude-participant-id`, `--exclude-participant-prefix`, or
`--exclude-study-id` for dry-run/shakedown rows. Do not delete those rows from
the source export.

## Primary Gate

- Minimum usable delayed participants: 200.
- Complete-case requirement: each usable participant must have usable Session 1
  exposure and Session 2 recognition rows for all five analysis arms:
  `orange_flowers`, `hanging_clothes`, `aerial_beach`, `city_street`, and
  `storm_beach`.
- Pooled `primary_positive` recognition accuracy must exceed pooled
  `hard_negative_control` recognition accuracy.
- `orange_flowers` and `hanging_clothes` must each exceed the pooled hard
  negative control accuracy.
- Pooled `primary_positive`, `orange_flowers`, and `hanging_clothes`
  accuracies must be above chance.
- The pooled primary-positive versus hard-negative-control two-proportion
  z-test must have two-sided `p < 0.05`.

## Exclusion Policy

- Manual dry-run or shakedown participants are excluded only through explicit
  CLI flags.
- Participants missing Session 1 or Session 2 payloads are not usable for the
  gate.
- Form mismatches between sessions are excluded with `form_mismatch`.
- Session 1 `media_error` or incomplete exposure excludes the matched analysis
  trial.
- Session 2 `left_media_error`, `right_media_error`, or `any_media_error`
  excludes the matched recognition trial.
- Participants without all five usable analysis arms are excluded from the
  complete-case primary gate.

Excluded rows remain in the analysis JSON with `exclusion_reasons`.

## Claim Boundary

- This plan and script do not prove human memorability.
- Underpowered, dry-run, media-error, missing-session, and incomplete-case
  outputs are not human memorability evidence.
- Only a passed delayed Session 2 analysis can support a human
  recognition-memory claim for orange flowers and hanging clothes.
