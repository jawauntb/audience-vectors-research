# Content-Pocket Validation MP4 Screening

Date: 2026-06-08T21:59:38+00:00

## Status

Agent prelaunch sampled-frame screening only. This artifact checks frozen
MP4 availability, byte hashes, sampled-frame visual stability, and contact
sheet review readiness. It does not launch a study and does not validate
human memorability or measured-BMD grounding.

Result: No automated screening failures were found.

## Summary

- Frozen task payload SHA-256: `b151326f1e120d7d6c6440c97e9341784bb3b25a1393b8ca7a8481fbcb3cef6c`
- Stimuli screened: 45
- Sampled frames per video: 3
- Contact sheets: 4
- Automated screening failures: 0
- Visual-gate failures: 0
- Agent contact-sheet review: Codex sampled-frame contact-sheet review found retained candidate subjects and visually distinct hard-negative controls, with no obvious sampled-frame text/watermark, frame collapse, or attention-check leakage. Final human/IRB-facing screening still required.

## Task Counts

| comparison | tasks |
|---|---:|
| `exploratory_boundary_content_pocket_vs_hard_negative` | 12 |
| `primary_content_pocket_vs_hard_negative` | 12 |

## Contact Sheets

| tier | role | stimuli | sheet |
|---|---|---:|---|
| exploratory_boundary | candidate | 12 | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_validation_screening_sheets_20260608/exploratory_boundary_candidate.jpg` |
| exploratory_boundary | control | 12 | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_validation_screening_sheets_20260608/exploratory_boundary_control.jpg` |
| primary | candidate | 12 | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_validation_screening_sheets_20260608/primary_candidate.jpg` |
| primary | control | 9 | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_validation_screening_sheets_20260608/primary_control.jpg` |

## Screening Flags

None from byte/hash/frame-gate screening.

## Launch Blockers

- This is an agent sampled-frame pre-screen, not final IRB/faculty sign-off.
- Stable HTTPS hosted video URLs are still required before launch.
- Participant-facing consent, compensation, and response collection remain open.
- Human/BMD validation has not run; all content-pocket claims remain proxy-selected.

## Next Action

Review the contact sheets and selected MP4s, fill the hosted-video URL
map template for the frozen task JSON, and mark hosted videos screened
only after final human/IRB-facing content review.
