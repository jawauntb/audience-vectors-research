# Improved-v1 Seedance Proxy-Scoring Manifest

Date: `2026-06-15T20:56:25+00:00`
Status: `ready_for_proxy_scoring`

## Purpose

This freezes the exact visually eligible MP4 byte targets for TRIBE/BMD, V-JEPA, and CLIP scoring. It is not a selection result and not human memorability evidence.

## Summary

- Candidate MP4s queued: `96`
- Families queued: `12`
- Manual review cues retained: `59`
- Source visual report: `research_program/neurips_memorability_selector/experiments/content_pocket_confirmatory_recognition_20260615/seedance_candidate_visual_screening_improved_v1_20260615.json`

## Required Score Contract

- `tribe_bmd_projection`
- `vjepa_centroid_margin`
- `clip_seed_video_preservation`

## Family Queue

| family | queued | manual review | status | head |
|---|---:|---:|---|---|
| `aerial_beach` | 8 | 4 | `ready_for_proxy_scoring` | aerial_beach_candidate_old_v04, aerial_beach_candidate_old_v02, aerial_beach_candidate_old_v00 |
| `blue_jellyfish` | 8 | 8 | `ready_for_proxy_scoring` | blue_jellyfish_candidate_old_v01, blue_jellyfish_candidate_old_v04, blue_jellyfish_candidate_old_v03 |
| `butterflies_on_flowers` | 8 | 0 | `ready_for_proxy_scoring` | butterflies_on_flowers_candidate_old_v05, butterflies_on_flowers_candidate_old_v02, butterflies_on_flowers_candidate_old_v01 |
| `candle_flame_table` | 8 | 7 | `ready_for_proxy_scoring` | candle_flame_table_candidate_old_v07, candle_flame_table_candidate_old_v00, candle_flame_table_candidate_old_v02 |
| `city_street` | 8 | 8 | `ready_for_proxy_scoring` | city_street_candidate_old_v07, city_street_candidate_old_v03, city_street_candidate_old_v02 |
| `hands_pottery_wheel` | 8 | 4 | `ready_for_proxy_scoring` | hands_pottery_wheel_candidate_old_v01, hands_pottery_wheel_candidate_old_v02, hands_pottery_wheel_candidate_old_v06 |
| `hanging_clothes` | 8 | 2 | `ready_for_proxy_scoring` | hanging_clothes_candidate_old_v00, hanging_clothes_candidate_old_v06, hanging_clothes_candidate_old_v02 |
| `old_car` | 8 | 8 | `ready_for_proxy_scoring` | old_car_candidate_old_v02, old_car_candidate_old_v03, old_car_candidate_old_v01 |
| `orange_flowers` | 8 | 4 | `ready_for_proxy_scoring` | orange_flowers_candidate_old_v03, orange_flowers_candidate_old_v06, orange_flowers_candidate_old_v00 |
| `rain_on_window` | 8 | 1 | `ready_for_proxy_scoring` | rain_on_window_candidate_old_v01, rain_on_window_candidate_old_v02, rain_on_window_candidate_old_v03 |
| `storm_beach` | 8 | 5 | `ready_for_proxy_scoring` | storm_beach_candidate_old_v06, storm_beach_candidate_old_v04, storm_beach_candidate_old_v05 |
| `street_food_grill` | 8 | 8 | `ready_for_proxy_scoring` | street_food_grill_candidate_old_v03, street_food_grill_candidate_old_v01, street_food_grill_candidate_old_v04 |

## Screening Flags Retained

| flag | count |
|---|---:|
| `low_sampled_sharpness` | 33 |
| `manual_text_review_recommended` | 24 |
| `tail_sharpness_collapse` | 5 |

## Next Scoring Steps

- Run TRIBE/BMD projection on each exact source_absolute_path MP4.
- Extract exact V-JEPA embeddings for each exact MP4 and compute prospective centroid margins.
- Extract CLIP video/seed-image preservation features for each exact MP4.
- Attach scores back to this manifest without changing candidate hashes.
- Freeze selector/control roles per family only after manual review and proxy scores are complete.

## Claim Boundary

Proxy scoring can rank generated candidates, but it is not human memorability evidence until the delayed-recognition study clears its preregistered gate.
