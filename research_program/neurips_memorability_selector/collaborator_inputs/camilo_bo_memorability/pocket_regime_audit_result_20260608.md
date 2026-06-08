# Pocket Regime-Audit Result

Date: 2026-06-08

## Question

The restored seed-bank fixed-recipe screen found new non-jellyfish positive
content pockets (`fresh24_orange_flowers` and `fresh24_hanging_clothes`) under
Sobol recipes 516 and 517. This run asks whether those pockets survive local
alpha/guidance recipe variation and hard negative controls.

## Regime-Audit Setup

This was run as a discovery-regime audit rather than a broad optimizer sweep.
The preregistered manifest is
`pocket_regime_audit_manifest_20260608.md`.

Current regime:

- Artifact types: restored seed image, Sobol alpha/guidance recipe, SVD replay
  clip, visual artifact-gate record, TRIBE score, replay report.
- Operations: seed-bank restore, target-slot Sobol manifest construction,
  Modal SVD-XT generation, complete-candidate visual-first retention, TRIBE
  scoring.
- Gate: all generated clips must be explicit; visually failed candidates are
  withheld as complete two-replicate families; a pocket is provisionally
  accepted only if it remains positive while hard negative controls remain
  negative.

Action class:

- The recipe sweep itself is fixed-regime search.
- The scientific content under test is a possible regime-level artifact class:
  stable positive content pockets in the SVD replay schema, beyond the original
  blue-jellyfish pocket.

## Replay Design

Manifest:

`data/reports/bo_pocket_regime_audit_trial_table_sobol518_523_x7_reps2_20260608.json`

Dry-run replay report:

`data/reports/bo_pocket_regime_audit_trial_table_sobol518_523_x7_reps2_dry_run_20260608.json`

Full replay report:

`data/reports/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608.json`

Generated videos:

`data/generated/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608`

Parameters:

| parameter | value |
|---|---:|
| target seed slots | 7 |
| Sobol recipes | 518-523 |
| alpha range | -3.0 to 4.0 |
| guidance range | 2.5 to 10.0 |
| candidates | 42 |
| stochastic replicates | 2 |
| replay jobs | 84 |
| SVD steps | 50 |
| motion bucket | 5 |
| noise augmentation | 0 |

Target and control slots:

| slot | bmd_name | role |
|---:|---|---|
| 0 | `fresh24_old_car` | positive target, lower-scoring restored pocket |
| 10 | `fresh24_orange_flowers` | strongest restored positive target |
| 12 | `fresh24_hanging_clothes` | stable restored positive target |
| 18 | `fresh24_blue_jellyfish` | known positive target |
| 3 | `fresh24_aerial_beach` | hard negative control |
| 8 | `fresh24_city_street` | hard negative control |
| 14 | `fresh24_storm_beach` | strongest negative control |

Command shape:

```bash
python scripts/build_bo_prompt_search_manifest.py \
  --replay-seed-pool-size 24 \
  --target-seed-slot 0 \
  --target-seed-slot 10 \
  --target-seed-slot 12 \
  --target-seed-slot 18 \
  --target-seed-slot 3 \
  --target-seed-slot 8 \
  --target-seed-slot 14 \
  --sobol-samples-per-seed 6 \
  --sobol-start-index 518 \
  --sobol-scramble-seed 42 \
  --alpha-range=-3.0,4.0 \
  --guidance-range=2.5,10.0 \
  --report-path data/reports/bo_pocket_regime_audit_trial_table_sobol518_523_x7_reps2_20260608.json

BO_MEM_STEERING_ARTIFACT=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --trial-table data/reports/bo_pocket_regime_audit_trial_table_sobol518_523_x7_reps2_20260608.json \
  --selection first \
  --max-evals 42 \
  --replicates 2 \
  --replay-seed-pool-size 24 \
  --num-inference-steps 50 \
  --svd-motion-bucket-id 5 \
  --svd-noise-aug-strength 0 \
  --visual-first-retention complete-candidates \
  --report-path data/reports/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608.json \
  --output-dir data/generated/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608
```

## Visual Gate

The run generated and scored the full planned panel:

| quantity | value |
|---|---:|
| requested clips | 84 |
| generated clips | 84 |
| visual-gate failures | 0 |
| retained rows | 84 / 84 |
| retained candidates | 42 / 42 |
| TRIBE-scored rows | 84 / 84 |

This is cleaner than the restored fixed-recipe screen, where fireworks caused
two visual-gate failures. None of the target/control pockets in this local
regime-audit panel triggered complete-candidate withholding.

## Seed-Content Results

Scores below pool six Sobol recipes and two stochastic replicates per target
slot.

| seed-content slot | scored / requested | mean TRIBE | min | max | positive rows |
|---|---:|---:|---:|---:|---:|
| `fresh24_orange_flowers` | 12 / 12 | 4.1043 | 3.6386 | 4.7864 | 12 / 12 |
| `fresh24_hanging_clothes` | 12 / 12 | 2.8991 | 1.8805 | 3.5741 | 12 / 12 |
| `fresh24_blue_jellyfish` | 12 / 12 | 2.0901 | 0.7267 | 3.3580 | 12 / 12 |
| `fresh24_old_car` | 12 / 12 | 1.1695 | 0.4140 | 1.6482 | 12 / 12 |
| `fresh24_aerial_beach` | 12 / 12 | -8.8447 | -10.0352 | -7.1844 | 0 / 12 |
| `fresh24_city_street` | 12 / 12 | -9.2525 | -9.6196 | -8.4299 | 0 / 12 |
| `fresh24_storm_beach` | 12 / 12 | -10.4170 | -11.3646 | -9.4116 | 0 / 12 |

All four positive targets stayed positive for every row. All three hard
negative controls stayed negative for every row. The separation is therefore
not a single lucky recipe or a visual-gate artifact.

## Top Candidate Families

| candidate | seed-content slot | alpha | guidance | mean TRIBE | replicate scores |
|---|---|---:|---:|---:|---|
| `sobol_prompt_search_519_slot10` | `fresh24_orange_flowers` | 2.813 | 7.645 | 4.2626 | 4.7864, 3.7389 |
| `sobol_prompt_search_523_slot10` | `fresh24_orange_flowers` | 0.613 | 6.763 | 4.1972 | 4.1930, 4.2015 |
| `sobol_prompt_search_520_slot10` | `fresh24_orange_flowers` | 2.392 | 2.567 | 4.1972 | 4.2388, 4.1556 |
| `sobol_prompt_search_521_slot10` | `fresh24_orange_flowers` | -1.961 | 8.631 | 4.1915 | 4.6070, 3.7760 |
| `sobol_prompt_search_522_slot10` | `fresh24_orange_flowers` | -0.182 | 4.450 | 4.0061 | 3.9886, 4.0236 |
| `sobol_prompt_search_518_slot10` | `fresh24_orange_flowers` | -1.476 | 6.149 | 3.7709 | 3.9031, 3.6386 |
| `sobol_prompt_search_518_slot12` | `fresh24_hanging_clothes` | -1.476 | 6.149 | 3.3653 | 3.4288, 3.3018 |
| `sobol_prompt_search_521_slot12` | `fresh24_hanging_clothes` | -1.961 | 8.631 | 3.1367 | 3.4448, 2.8286 |
| `sobol_prompt_search_521_slot18` | `fresh24_blue_jellyfish` | -1.961 | 8.631 | 3.1135 | 2.8689, 3.3580 |
| `sobol_prompt_search_519_slot12` | `fresh24_hanging_clothes` | 2.813 | 7.645 | 3.0912 | 3.2494, 2.9329 |

The top six candidate families are all orange flowers. This is stronger than
the restored fixed-recipe screen: orange flowers is not merely positive under
two recipes; it is the dominant pocket across a local recipe neighborhood.

## Variance Decomposition

On all 84 scored rows:

| model | row-level R2 |
|---|---:|
| seed-content only | 0.9912 |
| recipe only | 0.0021 |
| seed-content + recipe | 0.9983 |

Recipe choice has some local interaction effects, especially for blue jellyfish
and hanging clothes, but recipe identity alone explains essentially none of the
score structure. Content slot remains the operative variable in the current
SVD replay regime.

## Finding

This run turns the restored seed-bank result from a fixed-recipe observation
into a stronger content-pocket finding. Under the current SVD replay regime,
stable positive content pockets exist outside blue jellyfish. The strongest is
`fresh24_orange_flowers`, which averaged 4.1043 across 12/12 scored rows, had
no negative replicates, and occupied all top six candidate-family slots.

The important residual content is therefore not a portable alpha/guidance
recipe. It is a content-pocket artifact class: seed-image identity creates
stable score basins that survive local recipe variation while hard negative
controls remain negative. This is a genuine regime clarification for the
research process. Further alpha/guidance-only broadening is low leverage unless
conditioned on content-pocket selection.

## Next Step

Consolidate the new pocket rather than widen blindly:

1. Build a feature/embedding audit for orange flowers versus hanging clothes,
   blue jellyfish, old car, and the three negative controls.
2. Run a narrow orange-flowers replication with more stochastic seeds around
   the best local recipe families (`519`, `520`, `521`, `523`) to estimate
   within-pocket variance.
3. If prompt-conditioned generation becomes available, use orange flowers and
   hanging clothes as source content pockets for a true prompt/editing regime
   transition rather than continuing metadata-only prompt rewrites.
