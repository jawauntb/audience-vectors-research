# Content-Pocket Recognition Stimulus Production Manifest

Date: 2026-06-08T23:17:30+00:00

## Discovery-Regime Audit

Question: can the recognition-memory validation set be materialized
without using near-duplicate same-category lures?

Current regime:

- Artifact types: frozen old MP4s, seed-image requests, SVD generation
  jobs, generated lure/filler MP4s, visual screening records, hosted URLs,
  two-session recognition responses.
- Operations: acquire distinct seed images, generate SVD MP4s, screen
  images and videos, freeze launchable old-vs-lure recognition forms.
- Gates/verifiers: seed-image distinctiveness, MP4 visual validity,
  complete-candidate retention, old-vs-lure human recognition accuracy.
- Known limitation: seed images and generated recognition lures are
  not accepted for launch until their corresponding screening records
  exist.

Action class: production search inside the accepted recognition-memory
validation regime.

## Status

- Status: `generated_videos_present_screening_required`
- Seed image requests: 60
- Seed images present: 60
- Seed images missing: 0
- SVD generation jobs: 60
- Output MP4s present: 60
- Output MP4s missing: 0

## Required Production Blocks

| block | count | purpose |
|---|---:|---|
| analysis lures | 15 | same-category old-vs-lure trials for primary and hard-negative arms |
| filler old targets | 25 | Session 1 unrelated filler exposures |
| filler lures | 20 | Session 2 unrelated filler recognition trials |

## Launch Blockers

- Generated MP4 visual screening has unresolved flags.
- Final human/IRB-facing content review is not complete.
- Hosted HTTPS URLs and two-session Prolific wiring are not complete.

## Claim Boundary

- This artifact is a production manifest, not human evidence.
- Do not claim actual memorability until the recognition gate clears.
- Do not use near-duplicate lures to rescue an underpowered recognition result.

## Next Action

Review or replace the generated MP4s flagged by video screening,
preserve rejected artifacts with reasons, then rerun screening before
hosting accepted videos.
