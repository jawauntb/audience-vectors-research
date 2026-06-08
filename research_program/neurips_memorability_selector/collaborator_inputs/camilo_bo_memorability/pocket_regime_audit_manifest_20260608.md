# Pocket Regime-Audit Manifest

Date: 2026-06-08

## Question

The restored seed-bank screen found that the current SVD replay regime is
dominated by seed-content pockets rather than alpha/guidance recipes. This
run asks whether the newly positive pockets survive local recipe variation and
negative controls, or whether the prior fixed-recipe screen was a narrow
recipe artifact.

## Discovery-Regime Audit

Current regime:

| component | value |
|---|---|
| Artifact types | restored seed image, prompt metadata, Sobol alpha/guidance recipe, SVD replay clip, visual gate record, TRIBE score, replay report |
| Operations | restore seed bank, build target-slot Sobol manifest, generate SVD-XT clips on Modal, apply visual artifact gate, retain complete candidates, score retained clips with TRIBE |
| Gates/verifiers | 2-replicate complete-candidate visual-first retention; TRIBE score summary by content slot and recipe; positive-vs-negative control separation |
| Known limitation | prompt text is metadata-only in the current SVD replay path, so prompt rewriting is not an actionable intervention yet |

Action class:

| item | classification |
|---|---|
| Manifest construction | search inside the current replay schema |
| Claim under test | possible discovery only if a non-jellyfish content pocket remains positive under local recipe variation and negative controls |

Transported evidence:

- The old regime already explains that recipes 516 and 517 do not dominate the
  restored seed bank globally.
- The restored fixed-recipe screen identified positive pockets at
  `fresh24_orange_flowers`, `fresh24_hanging_clothes`,
  `fresh24_blue_jellyfish`, and `fresh24_old_car`.
- The same screen identified hard negative controls at
  `fresh24_storm_beach`, `fresh24_city_street`, and
  `fresh24_aerial_beach`.

Residual content sought:

- A stable positive content pocket outside the original jellyfish seed.
- A local recipe neighborhood that improves or preserves those positive pockets
  while leaving negative controls negative.
- Evidence that content-pocket structure is an accepted artifact class for this
  SVD replay regime, not just a one-off fixed-recipe observation.

## Experiment

Manifest:

`data/reports/bo_pocket_regime_audit_trial_table_sobol518_523_x7_reps2_20260608.json`

Dry-run replay report:

`data/reports/bo_pocket_regime_audit_trial_table_sobol518_523_x7_reps2_dry_run_20260608.json`

Planned full replay report:

`data/reports/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608.json`

Planned output directory:

`data/generated/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608`

Target seed slots:

| slot | bmd_name | role |
|---:|---|---|
| 0 | `fresh24_old_car` | positive target, lower-scoring restored pocket |
| 10 | `fresh24_orange_flowers` | strongest restored positive target |
| 12 | `fresh24_hanging_clothes` | stable restored positive target |
| 18 | `fresh24_blue_jellyfish` | known positive target |
| 3 | `fresh24_aerial_beach` | hard negative control |
| 8 | `fresh24_city_street` | hard negative control |
| 14 | `fresh24_storm_beach` | strongest negative control |

Recipe neighborhood:

| parameter | value |
|---|---:|
| Sobol indices | 518-523 |
| Sobol scramble seed | 42 |
| alpha range | -3.0 to 4.0 |
| guidance range | 2.5 to 10.0 |
| candidates | 42 |
| replicates | 2 |
| replay jobs | 84 |

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

python scripts/modal_bo_memorability_replay.py \
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

## Gate

Acceptance rule:

- Generate all 84 requested replay jobs, or record missing jobs explicitly.
- Withhold any candidate whose two-replicate set fails complete-candidate visual
  retention.
- Call a content pocket provisionally accepted only if it has at least one fully
  retained recipe with positive mean TRIBE and remains separated from the hard
  negative controls.

Rejected/withheld rule:

- Preserve visual failures and negative-control scores as first-class evidence.
- Do not claim prompt effects from this run because prompt text is not yet an
  intervention in the current generation path.

## Next Move

If the positive pockets survive, consolidate by probing the strongest pocket
with a narrower recipe replication and an embedding/visual feature audit. If
they collapse, keep the rejection and move the regime transition upstream to a
new generator path where prompt text or image-content editing is actually an
operation.
