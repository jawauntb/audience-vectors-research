# Model Baseline Scoring Plan

Date: 2026-06-15

## Purpose

The next study should not merely show that one hand-picked packet works. It
should discover which compute signals prospectively predict delayed human
recognition.

## Required Pre-Human Scores

Every candidate old video must receive these scores before any human launch:

- TRIBE/BMD memorability projection.
- Exact V-JEPA video embedding margin.
- CLIP image/video semantic margin where available.
- Video quality/artifact score.
- Motion magnitude.
- Colorfulness.
- Saliency or gaze-predicted concentration score.
- Old-lure similarity score after lure generation.
- Random baseline rank.

## Selector Definitions

Primary selector:

```text
ensemble_score = z(TRIBE/BMD) + z(V-JEPA margin)
```

Only candidates passing visual quality, prompt fidelity, and artifact gates are
eligible.

Matched control:

```text
quality_matched_control = lower ensemble candidate from the same family
                          matched on quality, motion, colorfulness,
                          and prompt fidelity
```

The matched control is not necessarily the worst video. It should be a credible
generated video whose main difference is lower preregistered memory-selector
score.

## Baseline Questions

The analysis should answer:

- Does the ensemble selector predict human recognition?
- Does TRIBE add signal beyond V-JEPA?
- Does V-JEPA add signal beyond CLIP and visual descriptors?
- Are effects mostly content-family/category effects?
- Do low-level descriptors explain away the selector effect?
- Which pockets fail or reverse?

## Rejected Shortcut

Do not select only the visually most distinctive videos. Distinctiveness,
quality, and color are covariates and matching constraints, not the primary
scientific claim.
