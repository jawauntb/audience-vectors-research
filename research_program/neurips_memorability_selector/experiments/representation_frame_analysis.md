# Representation-Frame Analysis

This analysis compares TRIBE, V-JEPA, CLIP-preservation, and available human
response frames on the current Wan2.2 candidate pool.

## Coverage

- Candidate rows: **144**
- Complete TRIBE+V-JEPA+CLIP rows: **144**
- Missing TRIBE features: **0**
- Missing V-JEPA features: **0**
- Human participants loaded: **0**
- Human responses loaded: **0**

## Score Correlations

| scores | Spearman rho |
|---|---:|
| v_mem_projection vs vjepa_memorability_score | +0.632 |
| v_mem_projection vs clip_preservation_score | +0.142 |
| vjepa_memorability_score vs clip_preservation_score | -0.066 |
| v_mem_projection vs clip_preservation_z2_score | +0.110 |
| vjepa_memorability_score vs clip_preservation_z2_score | -0.058 |

## Representation Geometry

| frames | RSA Spearman | linear CKA |
|---|---:|---:|
| TRIBE pooled cortical vs V-JEPA video | +0.345 | +0.392 |
| TRIBE pooled cortical vs CLIP preservation scalars | +0.060 | +0.076 |
| V-JEPA video vs CLIP preservation scalars | +0.068 | +0.166 |

## Within-Seed Rank Agreement

| scores | seed groups | mean rho | median rho | top-1 agreement |
|---|---:|---:|---:|---:|
| v_mem_projection vs vjepa_memorability_score | 24 | +0.443 | +0.543 | 0.458 |
| v_mem_projection vs clip_preservation_score | 24 | -0.136 | -0.200 | 0.125 |
| vjepa_memorability_score vs clip_preservation_score | 24 | -0.229 | -0.286 | 0.000 |

## TRIBE vs V-JEPA Top Disagreements

| seed | TRIBE top | V-JEPA top | TRIBE top V-JEPA score | V-JEPA top TRIBE score |
|---|---|---|---:|---:|
| `fresh24_aerial_beach` | `fresh24_aerial_beach_base` | `fresh24_aerial_beach_m1p0_n01` | -0.0637 | -6.6142 |
| `fresh24_blue_jellyfish` | `fresh24_blue_jellyfish_m1p0_n01` | `fresh24_blue_jellyfish_m1p0_n02` | +0.0159 | +7.0617 |
| `fresh24_coastal_tracks` | `fresh24_coastal_tracks_m1p0_n03` | `fresh24_coastal_tracks_lora` | +0.0163 | -0.7121 |
| `fresh24_forest_canopy` | `fresh24_forest_canopy_m1p0_n00` | `fresh24_forest_canopy_lora` | +0.0336 | -5.3752 |
| `fresh24_golden_grass` | `fresh24_golden_grass_lora` | `fresh24_golden_grass_m1p0_n03` | -0.0469 | +0.3897 |
| `fresh24_hanging_clothes` | `fresh24_hanging_clothes_m1p0_n03` | `fresh24_hanging_clothes_lora` | +0.0193 | +5.1289 |
| `fresh24_old_car` | `fresh24_old_car_base` | `fresh24_old_car_lora` | +0.1462 | +5.9058 |
| `fresh24_orange_flowers` | `fresh24_orange_flowers_m1p0_n02` | `fresh24_orange_flowers_m1p0_n01` | +0.0784 | +8.2033 |
| `fresh24_red_mailbox` | `fresh24_red_mailbox_m1p0_n01` | `fresh24_red_mailbox_m1p0_n03` | -0.0181 | -4.2485 |
| `fresh24_storm_beach` | `fresh24_storm_beach_m1p0_n00` | `fresh24_storm_beach_lora` | -0.0223 | +2.0457 |
| `fresh24_suspension_bridge` | `fresh24_suspension_bridge_m1p0_n00` | `fresh24_suspension_bridge_m1p0_n02` | +0.0307 | -3.8555 |
| `fresh24_tall_building` | `fresh24_tall_building_m1p0_n02` | `fresh24_tall_building_m1p0_n00` | -0.0243 | -7.1130 |
| `fresh24_wheat_closeup` | `fresh24_wheat_closeup_base` | `fresh24_wheat_closeup_m1p0_n03` | -0.0702 | +1.3582 |

## Interpretation

This is a representation-frame audit, not a human validation. It tells us how much TRIBE, V-JEPA, and CLIP-preservation agree before collecting the augmented survey responses. Large top-1 disagreement between TRIBE and V-JEPA is useful: it means the human study can actually adjudicate between brain-aligned and self-supervised video frames rather than comparing two selectors that choose the same clips.
