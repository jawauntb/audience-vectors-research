# Improved-v1 Seedance Proxy Scores

Date: `2026-06-15T21:27:55+00:00`
Status: `proxy_scores_complete_roles_frozen_proxy_only`

## Summary

- Candidates: `96`
- Complete proxy scores: `96`
- Feature cache dir: `data/features/content_pocket_confirmatory_recognition_20260615/proxy_scores_improved_v1`

## Model Status

| model | status counts |
|---|---|
| `tribe` | `{"cached": 96}` |
| `vjepa` | `{"cached": 96}` |
| `clip` | `{"cached": 94, "written": 2}` |

## Family Selection Audit

| family | complete | status | selector_top_proxy | quality_matched_control_proxy |
|---|---:|---|---|---|
| `aerial_beach` | 8 | `roles_frozen_proxy_only` | `aerial_beach_candidate_old_v01` | `aerial_beach_candidate_old_v04` |
| `blue_jellyfish` | 8 | `roles_frozen_proxy_only` | `blue_jellyfish_candidate_old_v03` | `blue_jellyfish_candidate_old_v06` |
| `butterflies_on_flowers` | 8 | `roles_frozen_proxy_only` | `butterflies_on_flowers_candidate_old_v06` | `butterflies_on_flowers_candidate_old_v04` |
| `candle_flame_table` | 8 | `roles_frozen_proxy_only` | `candle_flame_table_candidate_old_v06` | `candle_flame_table_candidate_old_v00` |
| `city_street` | 8 | `roles_frozen_proxy_only` | `city_street_candidate_old_v04` | `city_street_candidate_old_v03` |
| `hands_pottery_wheel` | 8 | `roles_frozen_proxy_only` | `hands_pottery_wheel_candidate_old_v07` | `hands_pottery_wheel_candidate_old_v03` |
| `hanging_clothes` | 8 | `roles_frozen_proxy_only` | `hanging_clothes_candidate_old_v07` | `hanging_clothes_candidate_old_v00` |
| `old_car` | 8 | `roles_frozen_proxy_only` | `old_car_candidate_old_v05` | `old_car_candidate_old_v06` |
| `orange_flowers` | 8 | `roles_frozen_proxy_only` | `orange_flowers_candidate_old_v06` | `orange_flowers_candidate_old_v02` |
| `rain_on_window` | 8 | `roles_frozen_proxy_only` | `rain_on_window_candidate_old_v04` | `rain_on_window_candidate_old_v01` |
| `storm_beach` | 8 | `roles_frozen_proxy_only` | `storm_beach_candidate_old_v00` | `storm_beach_candidate_old_v07` |
| `street_food_grill` | 8 | `roles_frozen_proxy_only` | `street_food_grill_candidate_old_v06` | `street_food_grill_candidate_old_v02` |

## Composite Rule

`z_family(TRIBE/BMD) + 0.5*z_family(V-JEPA centroid margin) + 0.25*z_family(CLIP prompt-video alignment) + 0.10*z_family(visual quality)`

## Claim Boundary

These are compute-proxy selection scores for exact generated MP4 bytes. They are not human memorability evidence and do not replace the preregistered delayed-recognition study.
