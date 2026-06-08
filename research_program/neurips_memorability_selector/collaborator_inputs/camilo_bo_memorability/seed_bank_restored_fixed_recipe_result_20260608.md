# Restored Seed-Bank Fixed-Recipe Result

Date: 2026-06-08

## Question

The five-seed fixed-recipe probe showed that current SVD replay scores are
mostly content driven, but the local seed bank was incomplete: only 5 of 24
catalog seed images were present. This run asks whether restoring the full
catalog exposes additional positive content pockets beyond blue jellyfish.

## Seed-bank restoration

Restoration used `scripts/restore_bo_seed_bank.py`, which reads
`original/seeds/prompts.json`, downloads missing `source_image` rows, and
normalizes them to the existing seed format: RGB PNG, 640 x 352.

Local restore report:
`data/reports/bo_seed_bank_restore_20260608.json`

Audit after restoration:

| quantity | value |
|---|---:|
| catalog rows | 24 |
| available seed images | 24 |
| missing seed images | 0 |
| prompt axis in current SVD replay | metadata only |
| seed-bank expansion status | optional |

Restored PNG files are raw seed data and are kept local under
`original/seeds/`; they are reproducible from the catalog with the restore
script rather than committed as research text.

## Replay design

Trial table:
`data/reports/bo_seed_bank_restored_trial_table_sobol516_517_x24_reps2_20260608.json`

Replay report:
`data/reports/bo_seed_bank_restored_sobol516_517_x24_reps2_steps50_motion5_noise0_20260608.json`

Command shape:

```bash
python scripts/build_bo_prompt_search_manifest.py \
  --replay-seed-pool-size 24 \
  --sobol-samples-per-seed 2 \
  --sobol-start-index 516 \
  --sobol-scramble-seed 42 \
  --report-path data/reports/bo_seed_bank_restored_trial_table_sobol516_517_x24_reps2_20260608.json

python scripts/modal_bo_memorability_replay.py \
  --trial-table data/reports/bo_seed_bank_restored_trial_table_sobol516_517_x24_reps2_20260608.json \
  --selection first \
  --max-evals 48 \
  --replicates 2 \
  --replay-seed-pool-size 24 \
  --num-inference-steps 50 \
  --svd-motion-bucket-id 5 \
  --svd-noise-aug-strength 0 \
  --visual-first-retention complete-candidates \
  --report-path data/reports/bo_seed_bank_restored_sobol516_517_x24_reps2_steps50_motion5_noise0_20260608.json \
  --output-dir data/generated/bo_seed_bank_restored_sobol516_517_x24_reps2_steps50_motion5_noise0_20260608
```

The table replayed Sobol recipes 516 and 517 across all 24 image-backed seed
slots, with two stochastic noise-seed replicates per recipe/slot candidate.

## Visual gate

The run generated 96/96 requested clips. The visual artifact gate failed 2/96
videos, both `tail_sharpness_collapse` failures in the fireworks slot:

| candidate | failed replicate |
|---|---|
| `sobol_prompt_search_516_slot01` | replicate 0 |
| `sobol_prompt_search_517_slot01` | replicate 1 |

Complete-candidate visual-first retention therefore kept 46/48 candidates and
92/96 rows for full TRIBE scoring. The fireworks slot was withheld for both
recipes because each recipe had one failed replicate.

## Pooled seed-content ranking

Scores below pool over both recipes and both stochastic replicates for each
retained seed slot.

| seed-content slot | scored / requested | mean TRIBE | interpretation |
|---|---:|---:|---|
| fresh24_orange_flowers | 4 / 4 | 4.2013 | strongest restored positive pocket |
| fresh24_hanging_clothes | 4 / 4 | 3.6167 | stable restored positive pocket |
| fresh24_blue_jellyfish | 4 / 4 | 2.1849 | known positive pocket remains positive |
| fresh24_old_car | 4 / 4 | 1.0488 | newly positive, lower than top restored pockets |
| fresh24_sidewalk_steps | 4 / 4 | 0.0402 | near neutral |
| fresh24_concert_stage | 4 / 4 | -0.9507 | weak or recipe-sensitive |
| fresh24_dewy_grass | 4 / 4 | -1.2453 | negative |
| fresh24_mountain_fog | 4 / 4 | -1.9546 | negative |
| fresh24_red_mailbox | 4 / 4 | -2.4673 | negative |
| fresh24_forest_canopy | 4 / 4 | -3.7593 | negative |
| fresh24_golden_grass | 4 / 4 | -4.5581 | negative |
| fresh24_wheat_closeup | 4 / 4 | -4.5827 | negative |
| fresh24_suspension_bridge | 4 / 4 | -5.9046 | strongly negative |
| fresh24_coastal_tracks | 4 / 4 | -6.0013 | strongly negative |
| fresh24_lighthouse | 4 / 4 | -7.4457 | strongly negative |
| fresh24_ocean_cliffs | 4 / 4 | -7.9619 | strongly negative |
| fresh24_sparse_forest | 4 / 4 | -8.0197 | strongly negative |
| fresh24_aerial_beach | 4 / 4 | -8.7943 | strongly negative |
| fresh24_misty_woods | 4 / 4 | -8.8783 | strongly negative |
| fresh24_cloud_mountain | 4 / 4 | -9.0798 | strongly negative |
| fresh24_city_street | 4 / 4 | -9.4384 | strongly negative |
| fresh24_tall_building | 4 / 4 | -9.5164 | strongly negative |
| fresh24_storm_beach | 4 / 4 | -10.3075 | strongest negative pocket |
| fresh24_fireworks | 0 / 4 | withheld | visual brittle under complete-candidate retention |

## Positive candidate means

| seed-content slot | recipe | reps | mean TRIBE |
|---|---:|---:|---:|
| fresh24_orange_flowers | 516 | 2 | 4.2075 |
| fresh24_orange_flowers | 517 | 2 | 4.1951 |
| fresh24_hanging_clothes | 516 | 2 | 4.0186 |
| fresh24_hanging_clothes | 517 | 2 | 3.2148 |
| fresh24_blue_jellyfish | 517 | 2 | 2.6671 |
| fresh24_blue_jellyfish | 516 | 2 | 1.7027 |
| fresh24_old_car | 516 | 2 | 1.2473 |
| fresh24_old_car | 517 | 2 | 0.8502 |
| fresh24_sidewalk_steps | 516 | 2 | 0.2916 |
| fresh24_concert_stage | 516 | 2 | 0.1864 |

## Variance decomposition

On the 92 retained/scored rows:

| model | row-level R2 |
|---|---:|
| seed-content only | 0.9804 |
| recipe only | 0.0008 |
| seed-content + recipe | 0.9941 |

Mean score by recipe:

| recipe | retained rows | mean TRIBE |
|---|---:|---:|
| 516 | 46 | -4.4628 |
| 517 | 46 | -4.2132 |

## Finding

The restored seed-bank screen breaks the earlier jellyfish-only picture. Under
the same two fixed alpha/guidance recipes, the strongest retained content
pockets are `fresh24_orange_flowers` and `fresh24_hanging_clothes`, both stable
across two recipes and two stochastic replicates. Blue jellyfish remains
positive, but it is no longer the best seed-content slot once the catalog is
complete.

The main scientific interpretation is unchanged in kind but stronger in scope:
current SVD replay is content-slot dominated, not alpha/guidance dominated.
However, the content optimum is not a single idiosyncratic jellyfish artifact.
The restored catalog exposes at least two new positive non-jellyfish pockets
and a broad negative tail of landscape, street, shore, and fog scenes.

## Next step

Run a small local-neighborhood search around the top restored pockets:

1. `fresh24_orange_flowers`
2. `fresh24_hanging_clothes`
3. `fresh24_blue_jellyfish`
4. `fresh24_old_car`

The next design should spend budget on recipe variation inside these content
pockets, with held-out negative pockets retained as controls. A prompt-only
rewrite tournament is still invalid for current SVD replay because prompt text
is metadata-only in the generation path.
