# Seed-Content Fixed-Recipe Probe Result

Date: 2026-06-08

## Question

The per-prompt Sobol search showed that prompt/seed identity explains retained
TRIBE score variance far better than alpha/guidance recipe choice. The content
axis audit then showed why prompt rewriting is not a valid intervention under
the current SVD replay path: prompt text is metadata only, while the generator
actually consumes the seed image.

This follow-up therefore asks a stricter content question:

> If we hold two alpha/guidance recipes fixed and replay them across the five
> currently available seed images, does the seed-content slot still dominate
> score and visual retention?

## Protocol

Manifest builder:
`scripts/build_bo_prompt_search_manifest.py`

Local trial table:
`data/reports/bo_seed_content_recipe_probe_trial_table_sobol516_517_x5_reps2_20260608.json`

Local generated/scored reports:

- initial generation/visual report:
  `data/reports/bo_seed_content_recipe_probe_sobol516_517_x5_reps2_steps50_motion5_noise0_20260608.json`
- rescored retained-row report:
  `data/reports/bo_seed_content_recipe_probe_sobol516_517_x5_reps2_steps50_motion5_noise0_rescored_20260608.json`

The manifest used two Sobol alpha/guidance recipes, indices `516` and `517`,
across the five locally available seed images:

- slot 00: fireworks
- slot 01: ocean cliffs
- slot 02: concert stage
- slot 03: blue jellyfish
- slot 04: forest canopy

Each recipe/slot pair received two stochastic SVD replays. Generation used the
same tuned visual-first replay settings as the preceding runs: 50 SVD inference
steps, motion bucket 5, noise augmentation 0, full TRIBE scoring from video
bytes, and the current TRIBE/BMD cortical direction.

## Gate Result

- 20/20 requested clips generated.
- 2/20 rows failed the visual artifact gate, both fireworks rows:
  - `sobol_prompt_search_516_slot00` replicate 0:
    `tail_sharpness_collapse`
  - `sobol_prompt_search_517_slot00` replicate 1:
    `tail_sharpness_collapse`
- Because the run used complete-candidate visual-first retention, both
  fireworks candidates were withheld entirely: 4/20 rows withheld.
- 16/16 retained rows completed full TRIBE scoring after the TRIBE Modal image
  was repinned to `transformers==4.56.1` and stale Modal tasks were stopped.

## Retained Scores

| seed-content slot | scored / requested | mean TRIBE | std | retained scores |
|---|---:|---:|---:|---|
| fireworks | 0 / 4 | n/a | n/a | withheld by visual-first retention |
| ocean cliffs | 4 / 4 | -7.9930 | 0.4766 | -8.4165, -8.4717, -7.3260, -7.7580 |
| concert stage | 4 / 4 | -0.8903 | 1.2008 | 0.2682, 0.2584, -1.5720, -2.5157 |
| blue jellyfish | 4 / 4 | 2.2202 | 0.8731 | 1.2728, 2.2457, 3.6103, 1.7522 |
| forest canopy | 4 / 4 | -3.7754 | 0.7675 | -3.9357, -2.5129, -4.0722, -4.5806 |

The best retained candidate remained the blue-jellyfish slot:
`sobol_prompt_search_517_slot03`, with mean `2.6812` over two stochastic
replicates.

Recipe-level means were close and both negative:

| Sobol recipe | scored rows | mean TRIBE | std |
|---:|---:|---:|---:|
| 516 | 8 | -2.4115 | 3.9480 |
| 517 | 8 | -2.8078 | 3.7626 |

## Variance Audit

On the 16 retained rows, a one-hot least-squares diagnostic again found that
the seed-content slot explains the score structure, while recipe identity
explains essentially none of it:

| model | retained-score R2 | RMSE |
|---|---:|---:|
| recipe only | 0.0026 | 3.8564 |
| seed-content slot only | 0.9494 | 0.8690 |
| recipe + seed-content slot | 0.9520 | 0.8461 |

## Interpretation

This is a stronger content-axis result than the previous alpha/guidance search
because it fixes the recipe set and still recovers the same hierarchy: blue
jellyfish is the only positive retained content slot; ocean cliffs and forest
canopy are strongly negative; fireworks remains visually brittle enough that
complete-candidate retention withholds it before scoring.

The result supports a regime-level conclusion for the current SVD replay path:
the actionable broadening variable is seed-image/content selection, not further
alpha/guidance tuning around the same limited seed bank. It also reinforces
that the current positive compute-proxy signal is a jellyfish/content pocket,
not a global recipe effect.

Do not claim:

- these SVD clips improve human memorability;
- alpha/guidance recipes 516 or 517 are globally good;
- prompt rewriting has been tested under SVD replay.

Next valid experiments:

1. expand or restore the seed-image bank and run a seed-selection tournament
   with equal generated/scored budget per seed;
2. switch to a prompt-conditioned video generator before testing prompt-rewrite
   tournaments;
3. use this fixed-recipe panel as a small reproducible content-axis sanity
   check when changing generation settings.
