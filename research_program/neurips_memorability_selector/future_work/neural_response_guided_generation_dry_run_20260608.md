# Neural-Response-Guided Generation Dry-Run Spike

Date: 2026-06-08

## Scope

This is a no-generation, proxy-only feasibility dry run. It does not change the frozen content-pocket validation set and does not claim that TRIBE, V-JEPA, CLIP, saliency, or any composite score validates human memorability.

## Local Signal Availability

- Frozen pairwise tasks: 24 tasks, 48 side observations, 45 unique MP4 paths.
- MP4s resolved locally: 45 / 45 unique paths.
- V-JEPA features resolved: 48 / 48 side observations.
- CLIP seed-video preservation scores resolved: 48 / 48 side observations.
- Visual first-frame statuses: {"retained": 48}.

## Pairwise Proxy Outcomes

| Proxy | Decisions | Selects content-pocket target | Rate | Primary rate | Exploratory rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| tribe_bmd_projection | 24 | 24 | 100.0% | 100.0% | 100.0% |
| vjepa_centroid_margin | 24 | 22 | 91.7% | 100.0% | 83.3% |
| clip_seed_video_preservation | 24 | 9 | 37.5% | 75.0% | 0.0% |
| composite_proxy_score | 24 | 24 | 100.0% | 100.0% | 100.0% |

Interpretation: these are proxy agreement counts over existing clips, not validation. CLIP seed-video preservation is intentionally listed as a guardrail because it can select hard negatives when those clips preserve the seed image more strongly.

## Disagreement Examples

- content_pocket_exploratory_boundary_blue_jellyfish_sobol_prompt_search_519_slot18_rep00_vs_aerial_beach (exploratory_boundary): fresh24_blue_jellyfish vs fresh24_aerial_beach; winner sides {"clip_seed_video_preservation": "right", "composite_proxy_score": "left", "tribe_bmd_projection": "left", "vjepa_centroid_margin": "left"}.
- content_pocket_exploratory_boundary_blue_jellyfish_sobol_prompt_search_519_slot18_rep01_vs_city_street (exploratory_boundary): fresh24_blue_jellyfish vs fresh24_city_street; winner sides {"clip_seed_video_preservation": "left", "composite_proxy_score": "right", "tribe_bmd_projection": "right", "vjepa_centroid_margin": "left"}.
- content_pocket_exploratory_boundary_blue_jellyfish_sobol_prompt_search_519_slot18_rep02_vs_storm_beach (exploratory_boundary): fresh24_blue_jellyfish vs fresh24_storm_beach; winner sides {"clip_seed_video_preservation": "right", "composite_proxy_score": "left", "tribe_bmd_projection": "left", "vjepa_centroid_margin": "left"}.
- content_pocket_exploratory_boundary_blue_jellyfish_sobol_prompt_search_521_slot18_rep00_vs_aerial_beach (exploratory_boundary): fresh24_blue_jellyfish vs fresh24_aerial_beach; winner sides {"clip_seed_video_preservation": "left", "composite_proxy_score": "right", "tribe_bmd_projection": "right", "vjepa_centroid_margin": "right"}.
- content_pocket_exploratory_boundary_blue_jellyfish_sobol_prompt_search_521_slot18_rep01_vs_city_street (exploratory_boundary): fresh24_blue_jellyfish vs fresh24_city_street; winner sides {"clip_seed_video_preservation": "right", "composite_proxy_score": "left", "tribe_bmd_projection": "left", "vjepa_centroid_margin": "right"}.
- content_pocket_exploratory_boundary_blue_jellyfish_sobol_prompt_search_521_slot18_rep02_vs_storm_beach (exploratory_boundary): fresh24_blue_jellyfish vs fresh24_storm_beach; winner sides {"clip_seed_video_preservation": "left", "composite_proxy_score": "right", "tribe_bmd_projection": "right", "vjepa_centroid_margin": "right"}.
- content_pocket_exploratory_boundary_old_car_sobol_prompt_search_518_slot00_rep00_vs_aerial_beach (exploratory_boundary): fresh24_old_car vs fresh24_aerial_beach; winner sides {"clip_seed_video_preservation": "left", "composite_proxy_score": "right", "tribe_bmd_projection": "right", "vjepa_centroid_margin": "right"}.
- content_pocket_exploratory_boundary_old_car_sobol_prompt_search_518_slot00_rep01_vs_city_street (exploratory_boundary): fresh24_old_car vs fresh24_city_street; winner sides {"clip_seed_video_preservation": "right", "composite_proxy_score": "left", "tribe_bmd_projection": "left", "vjepa_centroid_margin": "left"}.

## Feasible Loops

- candidate_reranking: feasible_now. existing MP4s can be scored by TRIBE, V-JEPA, and committed CLIP summaries without generating new pixels
- evolutionary_selection_over_generated_candidates: feasible_now_with_existing_or_new_generated_batches. selection can operate on candidate pools after generation, while keeping proxy-only language
- prompt_search: blocked_for_current_svd_runner. do not treat prompt rewrites as pixel-affecting unless the generator is prompt-conditioned in that runner
- latent_or_guidance_optimization: blocked_until_generator_exposes_pixel_affecting_controls. requires a generator path where latent, noise, guidance, or conditioning changes actually affect MP4 pixels

## Smallest Safe Next Spike

- Run this proxy-only scorer on a larger generated candidate batch with held-out naming and no human-validation language.
- Review disagreement cases where CLIP preservation picks controls but TRIBE/V-JEPA pick content-pocket candidates.
- Only after proxy behavior is documented, decide whether a new candidate-generation batch is worth human or measured-BMD evaluation.

## Stop Rules

- Stop if the task is framed as proving human memorability or replacing human/BMD validation.
- Stop if the frozen 24-task content-pocket stimulus set would be edited by a proxy experiment.
- Stop if prompt text is varied in a runner where prompt text does not affect generated pixels.
- Stop if V-JEPA, CLIP, saliency, or TRIBE outputs are described as validated human reward models.
- Stop before launch if MP4s, V-JEPA features, or visual-gate statuses are missing for the intended candidate pool.

## Launch Blockers

- No local artifact blocker for this dry run; scientific overclaiming gates still apply.

## Relation To Content-Pocket Validation

The frozen 24-task set is used here only as an existing target for a no-generation proxy-agreement dry run. It remains separate from the manual MP4 screening, hosted-video, forced-choice, and measured-BMD validation lane.
