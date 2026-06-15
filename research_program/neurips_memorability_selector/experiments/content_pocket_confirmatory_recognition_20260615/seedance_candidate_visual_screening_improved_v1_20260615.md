# Improved-v1 Seedance Candidate Visual Screening

Date: `2026-06-15T20:46:39+00:00`
Status: `visual_gate_passed_manual_review_needed_proxy_selection_blocked`

## Discovery-Regime Audit

Question: are the 96 improved-v1 Seedance candidate old videos visually and technically eligible for proxy scoring and later selector/control assignment?

Current regime:

- Artifact types: Seedance MP4 candidates, prompt manifest rows, sampled-frame gates, contact sheets, low-level descriptors, later proxy-score tables.
- Operations: local MP4 byte/hash/metadata checks, sampled-frame artifact gate, visual descriptor extraction, manual review queueing.
- Gates/verifiers: all 12 families have 8 candidates; MP4s are playable 1280x720 around 5 seconds; sampled frames avoid collapse/low contrast; text/signage risk is queued for manual review.
- Known limitation: this gate cannot read OCR reliably and cannot choose memory selector winners without TRIBE/V-JEPA/CLIP scores.

Action class: production search inside the confirmatory recognition-study regime.

## Summary

- Candidates screened: `96`
- Samples per video: `5`
- Hard visual failures: `0`
- Manual review required: `59`
- Source manifest: `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_generation_manifest_improved_v1_20260615.json`
- Video root prefix: `/Users/jawaun/isc_mod`

## Family Counts

| family | candidates |
|---|---:|
| `aerial_beach` | 8 |
| `blue_jellyfish` | 8 |
| `butterflies_on_flowers` | 8 |
| `candle_flame_table` | 8 |
| `city_street` | 8 |
| `hands_pottery_wheel` | 8 |
| `hanging_clothes` | 8 |
| `old_car` | 8 |
| `orange_flowers` | 8 |
| `rain_on_window` | 8 |
| `storm_beach` | 8 |
| `street_food_grill` | 8 |

## Screening Flags

| flag | count |
|---|---:|
| `low_sampled_sharpness` | 33 |
| `manual_text_review_recommended` | 24 |
| `tail_sharpness_collapse` | 5 |

## Hard Blockers

| flag | count |
|---|---:|
| none | 0 |

## Contact Sheets

| family | candidates | sheet |
|---|---:|---|
| `overview` | 12 | `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_sheets_improved_v1_20260615/family_v00_overview.jpg` |
| `aerial_beach` | 8 | `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_sheets_improved_v1_20260615/aerial_beach_sampled_frames.jpg` |
| `blue_jellyfish` | 8 | `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_sheets_improved_v1_20260615/blue_jellyfish_sampled_frames.jpg` |
| `butterflies_on_flowers` | 8 | `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_sheets_improved_v1_20260615/butterflies_on_flowers_sampled_frames.jpg` |
| `candle_flame_table` | 8 | `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_sheets_improved_v1_20260615/candle_flame_table_sampled_frames.jpg` |
| `city_street` | 8 | `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_sheets_improved_v1_20260615/city_street_sampled_frames.jpg` |
| `hands_pottery_wheel` | 8 | `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_sheets_improved_v1_20260615/hands_pottery_wheel_sampled_frames.jpg` |
| `hanging_clothes` | 8 | `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_sheets_improved_v1_20260615/hanging_clothes_sampled_frames.jpg` |
| `old_car` | 8 | `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_sheets_improved_v1_20260615/old_car_sampled_frames.jpg` |
| `orange_flowers` | 8 | `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_sheets_improved_v1_20260615/orange_flowers_sampled_frames.jpg` |
| `rain_on_window` | 8 | `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_sheets_improved_v1_20260615/rain_on_window_sampled_frames.jpg` |
| `storm_beach` | 8 | `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_sheets_improved_v1_20260615/storm_beach_sampled_frames.jpg` |
| `street_food_grill` | 8 | `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_sheets_improved_v1_20260615/street_food_grill_sampled_frames.jpg` |

## Selection Readiness

| family | visually eligible | manual review | status | visual review head |
|---|---:|---:|---|---|
| `aerial_beach` | 8 | 4 | `blocked_pending_tribe_vjepa_clip_scores` | aerial_beach_candidate_old_v01, aerial_beach_candidate_old_v07, aerial_beach_candidate_old_v04 |
| `blue_jellyfish` | 8 | 8 | `blocked_pending_tribe_vjepa_clip_scores` | blue_jellyfish_candidate_old_v01, blue_jellyfish_candidate_old_v04, blue_jellyfish_candidate_old_v03 |
| `butterflies_on_flowers` | 8 | 0 | `blocked_pending_tribe_vjepa_clip_scores` | butterflies_on_flowers_candidate_old_v05, butterflies_on_flowers_candidate_old_v02, butterflies_on_flowers_candidate_old_v01 |
| `candle_flame_table` | 8 | 7 | `blocked_pending_tribe_vjepa_clip_scores` | candle_flame_table_candidate_old_v07, candle_flame_table_candidate_old_v00, candle_flame_table_candidate_old_v02 |
| `city_street` | 8 | 8 | `blocked_pending_tribe_vjepa_clip_scores` | city_street_candidate_old_v07, city_street_candidate_old_v03, city_street_candidate_old_v02 |
| `hands_pottery_wheel` | 8 | 4 | `blocked_pending_tribe_vjepa_clip_scores` | hands_pottery_wheel_candidate_old_v01, hands_pottery_wheel_candidate_old_v02, hands_pottery_wheel_candidate_old_v03 |
| `hanging_clothes` | 8 | 2 | `blocked_pending_tribe_vjepa_clip_scores` | hanging_clothes_candidate_old_v00, hanging_clothes_candidate_old_v06, hanging_clothes_candidate_old_v02 |
| `old_car` | 8 | 8 | `blocked_pending_tribe_vjepa_clip_scores` | old_car_candidate_old_v02, old_car_candidate_old_v03, old_car_candidate_old_v01 |
| `orange_flowers` | 8 | 4 | `blocked_pending_tribe_vjepa_clip_scores` | orange_flowers_candidate_old_v03, orange_flowers_candidate_old_v06, orange_flowers_candidate_old_v00 |
| `rain_on_window` | 8 | 1 | `blocked_pending_tribe_vjepa_clip_scores` | rain_on_window_candidate_old_v01, rain_on_window_candidate_old_v02, rain_on_window_candidate_old_v03 |
| `storm_beach` | 8 | 5 | `blocked_pending_tribe_vjepa_clip_scores` | storm_beach_candidate_old_v02, storm_beach_candidate_old_v07, storm_beach_candidate_old_v06 |
| `street_food_grill` | 8 | 8 | `blocked_pending_tribe_vjepa_clip_scores` | street_food_grill_candidate_old_v03, street_food_grill_candidate_old_v01, street_food_grill_candidate_old_v04 |

## Claim Boundary

- This is visual/technical screening, not human memorability evidence.
- Do not freeze selector_top or quality_matched_control until TRIBE/BMD, exact V-JEPA, and CLIP/proxy scores are attached.
- low_sampled_sharpness, tail_sharpness_collapse, and manual_text_review_recommended are review cues, not automated proxy-scoring blockers.
- Preserve all generated candidates and rejected/withheld reasons.

## Next Actions

- Review contact sheets and MP4s flagged for manual text/content review.
- Attach TRIBE/BMD, exact V-JEPA, CLIP, and optional saliency scores to the 96 exact MP4s.
- Then select one selector_top and one quality_matched_control per family from visually eligible candidates.
- Generate lures only after selected old videos are frozen.
