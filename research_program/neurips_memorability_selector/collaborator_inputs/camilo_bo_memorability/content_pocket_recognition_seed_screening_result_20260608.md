# Content-Pocket Recognition Seed Screening Result

Date: 2026-06-08T23:35:17Z

## Discovery-Regime Audit

Question: are the materialized recognition-memory seed images acceptable for
SVD MP4 generation without accepting obvious near-duplicate lures?

Current regime:

- Artifact types: seed PNGs, seed hashes, contact sheets, old-vs-lure pair
  sheets, screening status, superseded/rejected local artifacts.
- Operations: OpenAI Image API seed materialization, contact-sheet review,
  replacement of a brittle unrelated filler pair, and production-manifest
  gate refresh.
- Gates/verifiers: all 60 seed images present; no visible text/watermark/severe
  artifacts; analysis lures remain same-category but compositionally distinct
  from matched old targets; fillers remain acceptable as unrelated balancing
  stimuli.
- Known limitation: this is only a seed-image screening gate. It does not
  generate videos and does not validate human recognition memory or
  memorability.

Action class: production search inside the accepted recognition-memory
validation regime.

## Result

Accepted for SVD generation: yes.

- Analysis lures accepted: 15/15.
- Filler old seeds accepted: 25/25.
- Filler lure seeds accepted: 20/20.
- Rejected or withheld: 0.
- Superseded filler seed pairs: 1.

The concert-stage filler pair `filler_old_v06` / `filler_lure_v06` was
superseded by a ceramic-teacup filler pair after the concert-stage lure
twice triggered sampled-frame `tail_sharpness_collapse` in video screening.
The superseded seeds and rejected concert-stage video attempts are preserved
as ignored local data-lake artifacts and are not silently discarded.

Refreshed review accepts the 15 analysis lures for SVD generation. They
preserve the required broad categories while changing composition enough to
avoid obvious old-vs-lure near duplicates in the screening sheets. No obvious
text, watermark, poster layout, or severe generation artifact is visible.

Refreshed review also accepts the 45 filler seed images for unrelated filler
use. Some filler pairs are naturally same-category similar, but they are not
analysis claims and remain acceptable for old/new task balancing before
video-level screening.

## Contact Sheets

- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_screening_sheets_20260608/analysis_lures.jpg`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_screening_sheets_20260608/filler_old.jpg`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_screening_sheets_20260608/filler_lures.jpg`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_screening_sheets_20260608/analysis_old_vs_lure_pairs.jpg`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_screening_sheets_20260608/filler_old_vs_lure_pairs.jpg`

## Remaining Caveats

- This screening result is not human recognition-memory or memorability
  evidence.
- SVD MP4 generation and video-level visual screening are still required
  before launch.
- Hosted HTTPS URLs, two-session Prolific wiring, and response-collection
  validation are still required.
- Human/BMD claim language remains blocked until the behavioral validation
  gate clears.

## Next Action

Use the refreshed SVD MP4 screening report, then host accepted videos at
stable HTTPS URLs and wire the two-session Prolific recognition study.
