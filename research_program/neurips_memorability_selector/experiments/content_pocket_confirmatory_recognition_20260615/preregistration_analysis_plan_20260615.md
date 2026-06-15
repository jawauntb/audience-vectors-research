# Preregistration Analysis Plan

Date: 2026-06-15

Status: `draft_before_generation`

## Primary Endpoint

The primary endpoint is Session 2 old-vs-lure recognition correctness on
analysis trials:

```text
correct = 1 if participant selects the Session 1 old video
correct = 0 otherwise
```

Media-error trials remain in raw exports and are excluded from the primary
analysis with explicit reason codes.

## Primary Contrast

The primary contrast is:

```text
selector_top > quality_matched_control
```

The contrast is within the same content-family set but between participants
through counterbalanced forms. Participants should not see both selected items
from the same family.

## Primary Model

Fit a mixed-effects logistic regression:

```text
correct ~ selection_condition
          + visual_quality
          + motion_magnitude
          + colorfulness
          + saliency_score
          + old_lure_similarity
          + (1 | participant_id)
          + (1 | item_id)
          + (1 | family_id)
```

Primary success requires the `selection_condition = selector_top` coefficient
to be positive with two-sided `p < 0.05` or a 95% interval excluding zero in the
positive direction.

## Sensitivity Analyses

- all-trial analysis with media-error covariate;
- no-free-text subset;
- high-confidence-only subset;
- leave-one-family-out;
- leave-one-form-out;
- item-level bootstrap;
- participant-level bootstrap;
- exact binomial summary by condition;
- family-level sign test over selector-top advantage;
- repeat model with only low-level visual covariates;
- repeat model with TRIBE, V-JEPA, CLIP, and ensemble scalar scores as
  continuous predictors.

## Exclusion Rules

- Exclude dry-run and staff/test participant IDs by explicit flag or study ID.
- Exclude participants missing either session.
- Exclude participants with form mismatch between sessions.
- Exclude participants with fewer than 10 usable analysis recognition trials.
- Exclude trial-level media failures from the primary model.
- Keep every excluded row with reason codes in private raw exports and
  participant-safe aggregate summaries.

## Secondary Endpoints

- Confidence-rated recognition.
- Optional post-choice free-text recall, coded after recognition to avoid
  contaminating the primary endpoint.
- Filler-trial accuracy as attention and task-compliance diagnostic.
- Response time as a quality-control diagnostic, not a primary memory endpoint.

## Power Reading

The target delayed sample is 400 usable participants. With 12 analysis trials
per participant, this yields roughly 4,800 analysis observations before
exclusions. Each of the 24 selected analysis items should receive about 200
participant observations under balanced 8-form assignment.

The minimum delayed sample of 300 is still interpretable. The stretch sample of
500 improves item/family-level robustness and publication credibility.

## Claim Boundary

Passing this plan supports delayed recognition selection in this benchmark. It
does not prove broad memorability, preference, virality, commercial conversion,
generator control, or measured neural validation.
