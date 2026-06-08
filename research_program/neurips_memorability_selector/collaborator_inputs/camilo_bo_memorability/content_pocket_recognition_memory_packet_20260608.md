# Content-Pocket Recognition-Memory Validation Packet

Date: 2026-06-08T22:19:35+00:00

## Purpose

Build the direct human-memory validation study for the accepted
content-pocket candidates. Unlike the current forced-choice survey, this
design asks whether participants can later recognize the exact clip they
saw, against a newly generated same-category lure.

This packet is not launchable yet. It defines the old targets, the lure
generation requirements, the sparse-exposure form structure, and the
pre-registered success gate.

## Discovery-Regime Audit

Question: do primary SVD content pockets produce actual human recognition
memory gains, not just perceived memorability preferences?

Current regime:

- Artifact types: frozen old MP4s, pocket labels, TRIBE/V-JEPA verifier
  status, generated lure requests, recognition form templates, hosted
  survey tasks, recognition responses.
- Operations: sparse exposure, same-category lure generation, visual
  screening, delayed old-vs-lure recognition, mixed-effects analysis.
- Gates/verifiers: old-target freeze integrity, lure distinctiveness, visual
  screening, 2AFC recognition accuracy, primary-pocket positive direction.
- Known limitation: no human/BMD recognition result has run yet.

Action class: discovery-transition design. The validation endpoint changes
from a preference readout to an actual memory-behavior readout.

## Design Summary

- Session 1: each participant sees one old target from each analysis arm,
  plus unrelated fillers, and performs a light cover task.
- Session 2: 24-48 hours later, each analysis trial shows the old target
  against a newly generated same-category lure.
- Sparse exposure rule: no participant sees more than one old clip from the
  same analysis arm.
- Primary endpoint: old-vs-lure 2AFC recognition accuracy.
- Participant assignment: hash the Prolific participant ID to one
  of the six sparse forms and persist that form into Session 2.

## Analysis Arms

| arm | pocket | group | source |
|---|---|---|---|
| `orange_flowers` | `fresh24_orange_flowers` | `primary_positive` | primary/candidate |
| `hanging_clothes` | `fresh24_hanging_clothes` | `primary_positive` | primary/candidate |
| `aerial_beach` | `fresh24_aerial_beach` | `hard_negative_control` | primary/control |
| `city_street` | `fresh24_city_street` | `hard_negative_control` | primary/control |
| `storm_beach` | `fresh24_storm_beach` | `hard_negative_control` | primary/control |

## Old Target Variants

| arm | variants | selected labels |
|---|---:|---|
| `orange_flowers` | 3 | `bo_replay_05_sobol_prompt_search_519_slot10_rep01`, `bo_replay_10_sobol_prompt_search_520_slot10_rep01`, `bo_replay_05_sobol_prompt_search_519_slot10_rep02` |
| `hanging_clothes` | 3 | `bo_replay_11_sobol_prompt_search_520_slot12_rep00`, `bo_replay_16_sobol_prompt_search_521_slot12_rep02`, `bo_replay_16_sobol_prompt_search_521_slot12_rep01` |
| `aerial_beach` | 3 | `bo_replay_12_sobol_prompt_search_520_slot03_rep00`, `bo_replay_17_sobol_prompt_search_521_slot03_rep00`, `bo_replay_07_sobol_prompt_search_519_slot03_rep00` |
| `city_street` | 3 | `bo_replay_18_sobol_prompt_search_521_slot08_rep01`, `bo_replay_13_sobol_prompt_search_520_slot08_rep01`, `bo_replay_08_sobol_prompt_search_519_slot08_rep01` |
| `storm_beach` | 3 | `bo_replay_14_sobol_prompt_search_520_slot14_rep02`, `bo_replay_09_sobol_prompt_search_519_slot14_rep02`, `bo_replay_19_sobol_prompt_search_521_slot14_rep02` |

## Lure Generation

Required same-category lures: 15

Lure seed images must be visually distinct from the frozen old target
clip while preserving the broad category. Prompt rewrites alone are not
sufficient in the current image-conditioned SVD path.

## Sample Size

- Recommended Session 1 slots: 350
- Target Session 2 usable participants: 300
- Minimum Session 2 usable participants: 200
- Small dry runs are only for plumbing and must be excluded from
  the evidence gate.

## Response Capture

Session 1 rows must retain participant ID, form ID, target ID, arm,
analysis group, video URL, cover-task rating, exposure-completion
status, and timestamps.

Session 2 rows must retain participant ID, form ID, target ID, arm,
analysis group, old/lure URLs, old side, choice side, correctness,
response time, and timestamps.

Excluded participants and failed-load trials must remain in the
exported dataset with explicit exclusion reasons.

## Primary Gate

- pooled primary_positive recognition accuracy exceeds hard_negative_control accuracy
- fresh24_orange_flowers effect direction is positive
- fresh24_hanging_clothes effect direction is positive
- same-category lure false familiarity does not collapse accuracy to chance

## Launch Blockers

- Need distinct same-category lure seed images for every old target variant.
- Need generated lure MP4s from those seed images.
- Need unrelated filler targets and filler lures.
- Need MP4 screening/contact sheets for generated lures and fillers.
- Need hosted HTTPS URLs and Prolific two-session setup.

## Next Action

Acquire or generate the required distinct lure seed images, generate lure
and filler MP4s under matched SVD settings, screen them, then freeze the
complete recognition stimulus set before creating the two-session
Prolific study.
