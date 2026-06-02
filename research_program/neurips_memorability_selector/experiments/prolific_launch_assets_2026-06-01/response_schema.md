# Response Schema And Analysis Contract

This schema is the expected response shape for the V-JEPA selector pilot.

## Trial-Level Response

| Field | Type | Meaning |
|---|---|---|
| participant_id | string | Prolific ID or anonymous platform ID |
| session_id | string | Browser/session UUID |
| task_id | string | Frozen task identifier from `current_selector_pairwise_tasks_with_vjepa.json` |
| seed | string | Prompt/seed identifier |
| comparison | string | Selector comparison family |
| left_policy | string | Policy represented by the left video |
| right_policy | string | Policy represented by the right video |
| left_asset | string | Hosted URL or asset ID for the left video |
| right_asset | string | Hosted URL or asset ID for the right video |
| choice | string | `left` or `right` |
| chosen_policy | string | Policy selected by participant |
| response_ms | number | Time from trial render to answer |
| trial_index | number | Trial order shown to participant |

## Participant-Level Response

| Field | Type | Meaning |
|---|---|---|
| participant_id | string | Prolific ID or anonymous platform ID |
| session_id | string | Browser/session UUID |
| started_at | ISO timestamp | Survey start time |
| completed_at | ISO timestamp | Survey completion time |
| user_agent | string | Browser metadata, if collected |
| attention_check_passed | boolean | Final attention-check status |
| excluded | boolean | Whether participant is excluded by predeclared rules |
| exclusion_reason | string | Empty when not excluded |

## Primary Derived Table

The analysis script should derive one row per trial with:

- `target_policy`
- `baseline_policy`
- `target_chosen`
- `comparison`
- `seed`
- `participant_id`

This supports prompt-clustered bootstrap confidence intervals and mixed-effects
logistic regression.

## Anti-Circularity Rule

Do not use TRIBE/BMD scores as the outcome variable for the human-validation
claim. TRIBE/BMD can define a selector policy, but the endpoint must be
participant choice or delayed recognition.
