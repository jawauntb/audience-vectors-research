# Descriptor-Conditioned Replication Embedding Result - 2026-06-08

## Discovery-Regime Audit

Question: do CLIP/V-JEPA-style embeddings transport prospectively on the fresh orange-flowers and hanging-clothes replication?

Current regime:

- Artifact types: restored seed images, generated SVD replay videos, TRIBE replay scores, CLIP seed/video/text embeddings, optional V-JEPA embeddings, pocket labels, and leakage-aware verifier outputs.
- Operations: encode seed images and generated-video frame samples with CLIP, aggregate stochastic replicates to task-level embeddings, compute pocket-held-out centroid margins, and train leave-one-pocket-out classifiers.
- Gates/verifiers: embedding descriptors cannot use TRIBE score as an input. Acceptance requires either the descriptor or classifier rule below.

Action class: search inside the current compute-proxy regime. It becomes discovery-relevant only if an embedding descriptor becomes an accepted verifier for content-pocket consolidation.

## Inputs

- Replay report: `data/reports/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608.json`
- Seed root: `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/original`
- CLIP model: `openai/clip-vit-base-patch32`
- V-JEPA status: integrated for all 30/30 candidates from `data/features/vjepa_descriptor_conditioned_replication_20260608`
- Candidates: 30 task-level candidates from 90 scored replicate rows.

## Score By Pocket

| pocket | label | candidates | mean | min | max | positive candidates |
|---|---|---:|---:|---:|---:|---:|
| fresh24_orange_flowers | positive | 6 | 3.8569 | 3.6750 | 4.1035 | 6 |
| fresh24_hanging_clothes | positive | 6 | 3.1519 | 2.7363 | 3.6030 | 6 |
| fresh24_aerial_beach | negative_control | 6 | -8.7489 | -9.9707 | -7.4204 | 0 |
| fresh24_city_street | negative_control | 6 | -9.3232 | -9.4662 | -9.2046 | 0 |
| fresh24_storm_beach | negative_control | 6 | -10.4614 | -10.9369 | -9.6384 | 0 |

## Embedding Descriptor Separators

| family | feature | direction | positive mean | negative mean | AUC | abs d | r(score) |
|---|---|---|---:|---:|---:|---:|---:|
| vjepa_video | pocket_heldout_centroid_margin | higher_for_positive | 0.0750 | -0.1371 | 1.0000 | 2.8636 | 0.8380 |
| clip_seed_image | pocket_heldout_centroid_margin | higher_for_positive | -0.0104 | -0.0996 | 0.8333 | 1.4599 | 0.6038 |
| clip_seed_video | seed_video_clip_cosine | higher_for_positive | 0.9814 | 0.9708 | 0.8194 | 1.0973 | 0.3868 |
| clip_video | pocket_heldout_centroid_margin | higher_for_positive | -0.0210 | -0.0971 | 0.6667 | 1.0056 | 0.4671 |

## Leakage-Aware Classifiers

| family | embedding | validation | predictions | AUC | balanced accuracy | pos prob mean | neg prob mean |
|---|---|---|---:|---:|---:|---:|---:|
| clip_seed_image | seed_embedding | leave_one_pocket_out | 30 | 0.8333 | 0.7500 | 0.5018 | 0.4869 |
| clip_video | video_embedding | leave_one_pocket_out | 30 | 0.8333 | 0.5833 | 0.5004 | 0.4873 |
| vjepa_video | vjepa_embedding | leave_one_pocket_out | 30 | 1.0000 | 1.0000 | 0.5128 | 0.4825 |

## Gate

Descriptor rule: separation_auc >= 0.85 and abs_cohen_d >= 1.00.

Classifier rule: leave-one-pocket-out roc_auc >= 0.85 and balanced_accuracy >= 0.75.

Gate result: **accepted via exact V-JEPA only**.

Best descriptor:

- family: `vjepa_video`
- feature: `pocket_heldout_centroid_margin`
- separation AUC: 1.0000
- absolute Cohen d: 2.8636
- correlation with mean TRIBE score: 0.8380

Best classifier:

- family: `vjepa_video`
- validation: `leave_one_pocket_out`
- ROC AUC: 1.0000
- balanced accuracy: 1.0000

## Interpretation

The embedding gate clears only through exact V-JEPA on the fresh descriptor-conditioned replication. V-JEPA video centroid margin separates the replicated positives from hard controls with AUC 1.0000 and abs d 2.8636, and the V-JEPA leave-one-pocket-out classifier reaches balanced accuracy 1.0000. Generated-video CLIP does not transport prospectively in this run: its centroid-margin AUC is 0.6667 and its classifier balanced accuracy is 0.5833.

This is still compute-proxy evidence, not human memorability, but it strengthens orange flowers and hanging clothes as TRIBE/V-JEPA-verified candidate pockets while narrowing the prospective CLIP claim.

## Next Move

Use `descriptor_conditioned_replication_result_20260608.md` as the controlling result note. Do not describe this replication as a two-descriptor V-JEPA+CLIP pass. Decide whether to build a human/BMD packet with an explicit V-JEPA-only compute-screen caveat or run a targeted CLIP prospective diagnostic first.
