# Content-Pocket Recognition Seed Materialization Result

Date: 2026-06-08T23:31:00+00:00

## Discovery-Regime Audit

Question: can the recognition-memory production manifest's seed-image
requests be materialized without accepting near-duplicate lures?

Current regime:

- Artifact types: seed PNGs, seed hashes, generation metadata, contact
  sheets, screening status.
- Operations: OpenAI Image API generation from production-manifest prompts
  and requirements; contact-sheet construction.
- Gates/verifiers: all 60 seed images present, manual contact-sheet review
  before SVD generation, no human-memory claim until recognition data.
- Known limitation: this result materializes seed images only. It does not
  generate SVD videos or validate memorability.

Action class: production search inside the accepted recognition-memory
validation regime.

## Counts

- Requested: 2
- Generated: 2
- Already present: 0
- Failed: 0
- Seed images present after run: 2

## Contact Sheets

- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_replacement_screening_sheets_20260608/analysis_lures.jpg` (missing, items=0)
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_replacement_screening_sheets_20260608/filler_old.jpg` (present, items=1)
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_replacement_screening_sheets_20260608/filler_lures.jpg` (present, items=1)
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_replacement_screening_sheets_20260608/analysis_old_vs_lure_pairs.jpg` (present, items=15)
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_replacement_screening_sheets_20260608/filler_old_vs_lure_pairs.jpg` (present, items=1)

## Next Action

Review the contact sheets for same-category match, non-duplication, no
text/watermarks, and no obvious artifacts. Only after image screening
passes should the SVD generation jobs in the production manifest run.
