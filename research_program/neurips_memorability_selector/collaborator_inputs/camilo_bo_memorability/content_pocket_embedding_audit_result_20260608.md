# Content-Pocket Embedding Audit Result - 2026-06-08

## Discovery-Regime Audit

Question: do CLIP/V-JEPA-style embeddings explain the stable positive content pockets after lightweight visual descriptors failed?

Current regime:

- Artifact types: restored seed images, generated SVD replay videos, TRIBE replay scores, CLIP seed/video/text embeddings, optional V-JEPA embeddings, pocket labels, and leakage-aware verifier outputs.
- Operations: encode seed images and generated-video frame samples with CLIP, aggregate stochastic replicates to task-level embeddings, compute pocket-held-out centroid margins, and train leave-one-pocket-out classifiers.
- Gates/verifiers: embedding descriptors cannot use TRIBE score as an input. Acceptance requires either the descriptor or classifier rule below.

Action class: search inside the current compute-proxy regime. It becomes discovery-relevant only if an embedding descriptor becomes an accepted verifier for content-pocket consolidation.

## Inputs

- Replay report: `data/reports/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608.json`
- Seed root: `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/original`
- CLIP model: `openai/clip-vit-base-patch32`
- V-JEPA status: not run; exact pocket-replay feature dir not provided
- Candidates: 42 task-level candidates from 84 scored replicate rows.

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

## Embedding Descriptor Separators

| family | feature | direction | positive mean | negative mean | AUC | abs d | r(score) |
|---|---|---|---:|---:|---:|---:|---:|
| clip_seed_image | pocket_heldout_centroid_margin | higher_for_positive | 0.0605 | -0.0409 | 1.0000 | 2.8573 | 0.8541 |
| clip_video | pocket_heldout_centroid_margin | higher_for_positive | 0.0604 | -0.0359 | 0.8796 | 2.0280 | 0.7620 |
| clip_seed_video | seed_video_clip_cosine | lower_for_positive | 0.9585 | 0.9701 | 0.5556 | 0.5203 | -0.2108 |

## Leakage-Aware Classifiers

| family | embedding | validation | predictions | AUC | balanced accuracy | pos prob mean | neg prob mean |
|---|---|---|---:|---:|---:|---:|---:|
| clip_seed_image | seed_embedding | leave_one_pocket_out | 42 | 1.0000 | 0.8333 | 0.5109 | 0.4901 |
| clip_video | video_embedding | leave_one_pocket_out | 42 | 0.9514 | 0.8333 | 0.5110 | 0.4913 |

## Gate

Descriptor rule: separation_auc >= 0.85 and abs_cohen_d >= 1.00.

Classifier rule: leave-one-pocket-out roc_auc >= 0.85 and balanced_accuracy >= 0.75.

Gate result: **accepted**.

Best descriptor:

- family: `clip_seed_image`
- feature: `pocket_heldout_centroid_margin`
- separation AUC: 1.0000
- absolute Cohen d: 2.8573
- correlation with mean TRIBE score: 0.8541

Best classifier:

- family: `clip_seed_image`
- validation: `leave_one_pocket_out`
- ROC AUC: 1.0000
- balanced accuracy: 0.8333

## Interpretation

The CLIP embedding audit clears the verifier gate. That means the stable positive pockets are not merely opaque score islands: their seed/video embeddings contain enough structure to distinguish them from hard negative controls under leakage-aware evaluation. This is still compute-proxy evidence, not human memorability, but it gives the next replication a real descriptor to track.

## Next Move

Run the orange-flowers and hanging-clothes stochastic replication with the accepted CLIP descriptor as a covariate and stopping rule. If new variants preserve both positive TRIBE score and descriptor margin, the content-pocket verifier becomes stronger.
