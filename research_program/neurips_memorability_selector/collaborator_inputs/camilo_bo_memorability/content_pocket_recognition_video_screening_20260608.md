# Content-Pocket Recognition Video Screening

Date: 2026-06-08T23:13:44+00:00

## Status

Agent sampled-frame screening only. This artifact checks generated
recognition MP4 availability and sampled-frame visual stability. It does
not launch a study and does not validate human memorability.

Result: 1 video screening failures require review.

## Summary

- Videos screened: 60
- Sampled frames per video: 3
- Contact sheets: 3
- Automated screening failures: 1
- Visual-gate failures: 1
- Accepted for hosting prep: false
- Agent contact-sheet review: Contact sheets generated for review; final human/IRB-facing screening still required.

## Role Counts

| role | videos |
|---|---:|
| `analysis_lure_video` | 15 |
| `filler_lure_video` | 20 |
| `filler_old_video` | 25 |

## Contact Sheets

| role | videos | sheet |
|---|---:|---|
| `analysis_lure_video` | 15 | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_video_screening_sheets_20260608/analysis_lure_video.jpg` |
| `filler_lure_video` | 20 | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_video_screening_sheets_20260608/filler_lure_video.jpg` |
| `filler_old_video` | 25 | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_video_screening_sheets_20260608/filler_old_video.jpg` |

## Screening Flags

| role | job | flags |
|---|---|---|
| `filler_lure_video` | `filler_lure_v06` | `tail_sharpness_collapse` |

## Launch Blockers

- This is an agent sampled-frame pre-screen, not final IRB/faculty sign-off.
- Stable HTTPS hosted video URLs are still required before launch.
- Two-session Prolific wiring and response collection remain open.
- Human recognition-memory validation has not run.

## Next Action

Review the contact sheets and generated MP4s, host the accepted videos
at stable HTTPS URLs, then wire those URLs into the two-session
recognition study.
