# Content-Pocket Embedding Audit Result - 2026-06-08

## Discovery-Regime Audit

Question: do CLIP/V-JEPA-style embeddings explain the stable positive content pockets after lightweight visual descriptors failed?

Current regime:

- Artifact types: restored seed images, generated SVD replay videos, TRIBE replay scores, CLIP seed/video/text embeddings, optional V-JEPA embeddings, pocket labels, and leakage-aware verifier outputs.
- Operations: encode seed images and generated-video frame samples with CLIP, aggregate stochastic replicates to task-level embeddings, compute pocket-held-out centroid margins, and train leave-one-pocket-out classifiers.
- Gates/verifiers: embedding descriptors cannot use TRIBE score as an input. Acceptance requires either the descriptor or classifier rule below.

Action class: search inside the current compute-proxy regime. It becomes discovery-relevant only if an embedding descriptor becomes an accepted verifier for content-pocket consolidation.

## Inputs

- Replay report: `data/reports/bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_noise0_20260608.json`
- Seed root: `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/original`
- CLIP model: `openai/clip-vit-base-patch32`
- V-JEPA status: integrated for all 30/30 candidates from `data/features/vjepa_boundary_pocket_audit_20260608`
- Candidates: 30 task-level candidates from 90 scored replicate rows.

## Score By Pocket

| pocket | label | candidates | mean | min | max | positive candidates |
|---|---|---:|---:|---:|---:|---:|
| fresh24_blue_jellyfish | positive | 6 | 1.8844 | 1.0461 | 2.6652 | 6 |
| fresh24_old_car | positive | 6 | 1.3110 | 0.9776 | 1.5023 | 6 |
| fresh24_aerial_beach | negative_control | 6 | -8.8361 | -9.6327 | -7.9565 | 0 |
| fresh24_city_street | negative_control | 6 | -9.3657 | -9.5246 | -9.1931 | 0 |
| fresh24_storm_beach | negative_control | 6 | -10.3363 | -11.0822 | -9.5990 | 0 |

## Embedding Descriptor Separators

| family | feature | direction | positive mean | negative mean | AUC | abs d | r(score) |
|---|---|---|---:|---:|---:|---:|---:|
| clip_seed_video | seed_video_clip_cosine | lower_for_positive | 0.9404 | 0.9728 | 0.9722 | 2.2252 | -0.7860 |
| clip_prompt_seed | prompt_seed_clip_cosine | higher_for_positive | 0.3117 | 0.2939 | 0.8333 | 1.9506 | 0.7094 |
| vjepa_video | pocket_heldout_centroid_margin | higher_for_positive | -0.0390 | -0.0960 | 0.8194 | 1.0704 | 0.4838 |
| clip_prompt_video | prompt_video_clip_cosine | higher_for_positive | 0.3211 | 0.3103 | 0.6667 | 0.7997 | 0.4265 |
| clip_seed_image | pocket_heldout_centroid_margin | higher_for_positive | -0.0442 | -0.0502 | 0.6667 | 0.3429 | 0.2256 |
| clip_video | pocket_heldout_centroid_margin | higher_for_positive | -0.0180 | -0.0464 | 0.6620 | 0.7490 | 0.4113 |

## Leakage-Aware Classifiers

| family | embedding | validation | predictions | AUC | balanced accuracy | pos prob mean | neg prob mean |
|---|---|---|---:|---:|---:|---:|---:|
| clip_seed_image | seed_embedding | leave_one_pocket_out | 30 | 1.0000 | 0.5000 | 0.4977 | 0.4933 |
| clip_video | video_embedding | leave_one_pocket_out | 30 | 0.9676 | 0.7917 | 0.5013 | 0.4942 |
| vjepa_video | vjepa_embedding | leave_one_pocket_out | 30 | 0.8333 | 0.7500 | 0.4977 | 0.4869 |

## Gate

Descriptor rule: separation_auc >= 0.85 and abs_cohen_d >= 1.00.

Classifier rule: leave-one-pocket-out roc_auc >= 0.85 and balanced_accuracy >= 0.75.

Gate result: **accepted**.

Best descriptor:

- family: `clip_seed_video`
- feature: `seed_video_clip_cosine`
- separation AUC: 0.9722
- absolute Cohen d: 2.2252
- correlation with mean TRIBE score: -0.7860

Best classifier:

- family: `clip_video`
- validation: `leave_one_pocket_out`
- ROC AUC: 0.9676
- balanced accuracy: 0.7917

## Interpretation

The embedding audit clears the verifier gate. The best accepted family is clip_seed_video, and the stable positive pockets are not merely opaque score islands: their seed/video embeddings contain enough structure to distinguish them from hard negative controls under leakage-aware evaluation. This is still compute-proxy evidence, not human memorability, but it gives the next replication or validation packet a real descriptor to track.

## Next Move

Use the accepted embedding descriptor as a covariate and stopping rule in the next replication or human/BMD validation packet. If a specific family fails here, keep that caveat explicit rather than promoting it through the broader accepted-gate result.
