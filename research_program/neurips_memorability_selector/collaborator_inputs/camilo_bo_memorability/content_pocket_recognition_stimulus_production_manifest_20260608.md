# Content-Pocket Recognition Stimulus Production Manifest

Date: 2026-06-08T22:28:41+00:00

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
- Known limitation: no seed images or generated recognition lures are
  accepted until screening records exist.

Action class: production search inside the accepted recognition-memory
validation regime.

## Status

- Status: `missing_seed_images_not_ready_for_generation`
- Seed image requests: 60
- Seed images present: 0
- Seed images missing: 60
- SVD generation jobs: 60
- Output MP4s present: 0
- Output MP4s missing: 60

## Required Production Blocks

| block | count | purpose |
|---|---:|---|
| analysis lures | 15 | same-category old-vs-lure trials for primary and hard-negative arms |
| filler old targets | 25 | Session 1 unrelated filler exposures |
| filler lures | 20 | Session 2 unrelated filler recognition trials |

## Launch Blockers

- 60 seed images are missing or not materialized.
- 60 SVD output MP4s are missing.
- Manual image distinctiveness screening has not been recorded.
- Generated MP4 visual screening/contact sheets have not been recorded.
- Hosted HTTPS URLs and two-session Prolific wiring are not complete.

## Claim Boundary

- This artifact is a production manifest, not human evidence.
- Do not claim actual memorability until the recognition gate clears.
- Do not use near-duplicate lures to rescue an underpowered recognition result.

## Next Action

Materialize the listed seed images under the manifest seed root, review
their contact sheet for category match and distinctiveness, generate SVD
MP4s from the generation jobs, then screen/contact-sheet those MP4s before
freezing the launchable recognition set.
