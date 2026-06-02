# IRB Protocol Draft

Date: 2026-06-01

This is a draft for faculty/IRB review. Institutional language, investigator
names, data-retention periods, and compensation details should be edited before
submission.

## Study Title

Brain-Aligned Memorability Signals for Generated Video Selection

## Principal Investigator / Faculty Sponsor

TBD.

## Research Team

- Jawaun Brown, independent researcher / study developer.
- Faculty sponsor: TBD.
- Additional research assistants: TBD.

## Purpose

The purpose of this study is to evaluate whether a computational
brain-aligned memorability selector chooses generated videos that human raters
judge as more memorable than videos selected by non-brain baselines.

## Background

The project uses a supervised memorability readout learned from BOLD Moments and
TRIBE-predicted cortical-response features. In computational validation, TRIBE
predicts held-out BOLD Moments memorability at Spearman rho about +0.403, while
V-JEPA reaches about +0.395. A generated-video selector built from the TRIBE/BMD
readout improves candidate scores under the proxy metric, but this does not
establish human-visible improvement. Independent human evaluation is required.

## Study Design

This is an online behavioral study. Participants will view pairs of short videos
generated for the same prompt or image seed. For each pair, participants will
answer which video they believe would be more memorable.

### Primary Endpoint

The primary endpoint is the prompt-clustered win rate of the TRIBE+gate selected
video against the strongest non-brain baseline, expected to be V-JEPA or a
CLIP/preservation-based selector.

### Secondary Endpoints

- Win rate against base generation.
- Win rate against CLIP/preservation selectors.
- Win rate against V-JEPA memorability selector.
- Per-prompt disagreement and failure-case analysis.
- Participant-level response consistency and attention-check pass rate.

## Participants

### Inclusion Criteria

- Age 18 or older.
- Able to read English.
- Prolific participant in approved countries/regions selected by the final study
  team.
- Normal or corrected-to-normal vision sufficient for online video viewing.

### Exclusion Criteria

- Under age 18.
- Failure to provide consent.
- Failure of attention checks.
- Duplicate submissions.
- Optional: self-reported sensitivity to flashing or rapidly moving visual
  content, if the final stimulus set includes rapid motion.

## Recruitment

Participants will be recruited through Prolific. The study listing will describe
the task as a short video-rating study about memorability. Recruitment language
will not mention brain prediction in a way that implies participants are being
personally diagnosed or neurologically assessed.

## Procedures

1. Participant opens the Prolific study and reads the consent form.
2. Participant confirms eligibility and consent.
3. Participant completes practice or instruction screens.
4. Participant views a randomized sequence of video pairs.
5. For each pair, participant chooses which clip seems more memorable.
6. Participant completes attention checks embedded in the task.
7. Participant submits responses and receives the completion code.

Each video pair will contain videos generated for the same prompt or image seed.
Pair order and left/right presentation will be randomized. The study will avoid
sensitive, graphic, sexual, political persuasion, medical, or otherwise
distressing stimuli unless separately reviewed.

## Duration

The first pilot is expected to take approximately 8-15 minutes per participant,
depending on the number of trials.

## Compensation

Compensation will be set before launch to meet or exceed Prolific's fair-pay
guidelines and any institutional requirements. The planned rate should be
documented in the final IRB submission.

## Data Collected

The study may collect:

- trial-level choices;
- response times;
- randomized trial order;
- attention-check responses;
- optional demographic fields only if approved and necessary;
- Prolific participant ID for payment, duplicate prevention, and data quality.

The study should not collect names, emails, phone numbers, IP addresses, clinical
data, biometric data, or sensitive personal histories.

## Privacy And Confidentiality

Response data will be stored under pseudonymous participant identifiers. Prolific
IDs, if retained, should be stored separately from response data or converted to
hashed/pseudonymous IDs before analysis. Only authorized study personnel should
have access to raw data. Public release should include only aggregate results or
de-identified trial-level data if permitted by the IRB and consent form.

## Risks

Risks are minimal and may include:

- boredom or fatigue;
- mild visual discomfort from viewing videos;
- ordinary privacy risks associated with online survey data.

Participants may stop at any time without penalty. Stimuli should be screened to
avoid graphic or sensitive content.

## Benefits

Participants may not directly benefit. The study may benefit research by
improving evaluation methods for generated media and by clarifying whether
brain-aligned computational signals predict human judgments of memorability.

## Data Analysis

The primary analysis will estimate whether humans choose TRIBE+gate selected
videos above chance against the strongest non-brain baseline. Analyses will
include:

- prompt-clustered bootstrap confidence intervals;
- mixed-effects logistic regression with selector policy as a fixed effect and
  random intercepts for prompt and participant when possible;
- per-prompt win-rate distributions;
- attention-check exclusion reporting;
- correction for multiple selector comparisons.

## Stopping Rules And Data Quality

Responses may be excluded for:

- failed attention checks;
- duplicate Prolific IDs;
- unrealistically fast completion;
- incomplete submissions;
- technical failures that prevent video playback.

Exclusion rules should be finalized before launching the confirmatory study.

## Dissemination

Results may be reported in papers, preprints, talks, demos, and internal product
research materials. Only aggregate or de-identified data will be reported.

## Open Questions For IRB / Faculty Sponsor

1. Should the first predicted-memorability study be submitted as exempt or
   expedited/minimal risk?
2. Is collecting Prolific ID acceptable for duplicate prevention and payment?
3. Should the protocol explicitly exclude people with photosensitive epilepsy or
   visual sensitivity?
4. Does the institution require a specific data-retention period?
5. Is a delayed-recognition follow-up covered by this protocol, or should it be
   submitted as an amendment after the first pilot?
