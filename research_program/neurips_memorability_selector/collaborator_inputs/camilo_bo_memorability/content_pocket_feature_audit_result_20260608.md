# Content-Pocket Feature Audit Result - 2026-06-08

## Discovery-Regime Audit

Question: are the stable positive pockets explainable by lightweight visual descriptors, or are they only TRIBE score islands?

Current regime:

- Artifact types: restored seed images, generated SVD replay videos, TRIBE replay scores, visual-gate status, seed/video descriptors, and positive/control pocket labels.
- Operations: join the pocket-regime replay report to restored seed images, compute visual descriptors on seed images and sampled generated video frames, then compare positive pockets with hard negative controls.
- Gates/verifiers: descriptor separation must not use TRIBE score as an input feature; acceptance requires the pre-registered AUC and effect-size threshold in the gate below.

Action class: search inside the current compute-proxy regime. It becomes discovery-relevant only if the descriptor becomes an accepted verifier or artifact class for the content-pocket regime.

## Inputs

- Replay report: `data/reports/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608.json`
- Seed root: `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/original`
- Candidates: 42 task-level candidates from 84 scored replicate rows.
- Positive targets: orange flowers, hanging clothes, blue jellyfish, old car.
- Negative controls: aerial beach, city street, storm beach.

## Score By Pocket

| pocket | label | candidates | mean | min | max | positive candidates |
|---|---|---:|---:|---:|---:|---:|
| fresh24_orange_flowers | positive | 6 | 4.1043 | 3.7709 | 4.2626 | 6 |
| fresh24_hanging_clothes | positive | 6 | 2.8991 | 1.9789 | 3.3653 | 6 |
| fresh24_blue_jellyfish | positive | 6 | 2.0901 | 0.9065 | 3.1135 | 6 |
| fresh24_old_car | positive | 6 | 1.1695 | 0.7951 | 1.4398 | 6 |
| fresh24_aerial_beach | negative_control | 6 | -8.8447 | -9.8623 | -7.5131 | 0 |
| fresh24_city_street | negative_control | 6 | -9.2525 | -9.4150 | -8.6676 | 0 |
| fresh24_storm_beach | negative_control | 6 | -10.4170 | -10.9942 | -9.6029 | 0 |

## Best Descriptor Separators

| family | feature | direction | positive mean | negative mean | AUC | abs d | r(score) |
|---|---|---|---:|---:|---:|---:|---:|
| seed | colorfulness | higher_for_positive | 0.2381 | 0.1126 | 0.8333 | 1.8471 | 0.7469 |
| seed | bright_fraction | higher_for_positive | 0.5143 | 0.1568 | 0.8333 | 1.3371 | 0.5908 |
| video | bright_fraction | higher_for_positive | 0.5221 | 0.1869 | 0.8102 | 1.2287 | 0.5626 |
| video | colorfulness | higher_for_positive | 0.2797 | 0.1366 | 0.7963 | 1.6930 | 0.7225 |
| seed | green_excess | lower_for_positive | -0.1592 | -0.0235 | 0.7500 | 1.2174 | -0.6073 |
| video | green_excess | lower_for_positive | -0.1954 | -0.0335 | 0.7500 | 1.1151 | -0.5788 |
| seed | neutral_fraction | lower_for_positive | 0.1770 | 0.3586 | 0.7500 | 1.0174 | -0.4708 |
| video | red_fraction | higher_for_positive | 0.1960 | 0.0036 | 0.7500 | 0.8260 | 0.4814 |
| seed | red_fraction | higher_for_positive | 0.1653 | 0.0035 | 0.7500 | 0.8064 | 0.4734 |
| video | rgb_r_mean | higher_for_positive | 0.5381 | 0.4343 | 0.7477 | 0.6042 | 0.3938 |

## Gate

Acceptance rule: separation_auc >= 0.85 and abs_cohen_d >= 1.00.

Gate result: **not accepted**.

Best feature:

- family: `seed`
- feature: `colorfulness`
- direction: higher_for_positive
- separation AUC: 0.8333
- absolute Cohen d: 1.8471
- correlation with mean TRIBE score: 0.7469

## Interpretation

The audit does not find a lightweight descriptor that clears the gate. C-017 should remain a black-box compute-proxy pocket finding until a stronger embedding or human/BMD verifier explains it.

## Next Move

Try a stronger embedding audit, such as CLIP/V-JEPA video embeddings, before spending more replication budget.
