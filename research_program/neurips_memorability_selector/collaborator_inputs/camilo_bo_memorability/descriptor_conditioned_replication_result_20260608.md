# Descriptor-Conditioned Content-Pocket Replication Result

Date: 2026-06-08

## Discovery-Regime Audit

Question: do `fresh24_orange_flowers` and `fresh24_hanging_clothes`
replicate under fresh stochastic SVD seeds while preserving positive TRIBE
score and the accepted V-JEPA/CLIP content-pocket verifiers?

Current regime:

- Artifact types: restored seed images, Sobol alpha/guidance recipes, fresh SVD
  replay clips, visual-gate records, TRIBE scores, exact V-JEPA features, CLIP
  seed/video embeddings, pocket-held-out centroid margins, and claim-ledger
  entries.
- Operations: restore the seed bank, build the selected-slot Sobol trial table,
  generate 90 SVD clips on Modal, apply complete-candidate visual retention,
  score retained rows with TRIBE, extract exact V-JEPA features by MP4 bytes, and
  run the embedding verifier.
- Gates/verifiers: complete generation and visual retention; positive TRIBE
  means for target pockets; negative hard controls; exact V-JEPA generated-video
  centroid margin and leave-one-pocket-out classifier; generated-video CLIP
  centroid margin as the prospective CLIP verifier.
- Known limitation: this is compute-proxy SVD replay evidence. It is not human
  memorability, measured-BMD validation, delayed recognition, or
  prompt-conditioned generation.

Action class: search inside the accepted SVD content-pocket regime. The result
is discovery-relevant only as a verifier stress test: exact V-JEPA transported
prospectively, while generated-video CLIP did not.

## Inputs

- Manifest:
  `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/descriptor_conditioned_replication_manifest_20260608.md`
- Local seed-bank restore report:
  `data/reports/bo_seed_bank_restore_descriptor_conditioned_replication_20260608.json`
- Trial table:
  `data/reports/bo_descriptor_conditioned_replication_trial_table_sobol518_523_x5_noise250k_reps3_20260608.json`
- Dry-run expansion:
  `data/reports/bo_descriptor_conditioned_replication_trial_table_sobol518_523_x5_noise250k_reps3_dry_run_20260608.json`
- Replay report:
  `data/reports/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608.json`
- Generated videos:
  `data/generated/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608`
- Exact V-JEPA feature directory:
  `data/features/vjepa_descriptor_conditioned_replication_20260608`
- V-JEPA extraction summary:
  `descriptor_conditioned_replication_vjepa_extraction_summary_20260608.json`
- Embedding verifier summary:
  `descriptor_conditioned_replication_embedding_summary_20260608.json`

## Generation And Retention

| item | result |
|---|---:|
| task-level candidates | 30 |
| requested generated clips | 90 |
| generated clips scored by TRIBE | 90 |
| visual-gate failures | 0 / 90 |
| retained rows under complete-candidate retention | 90 / 90 |
| retained task-level candidates | 30 / 30 |
| withheld candidates | 0 |

No failed, withheld, visually rejected, or unscored rows were produced in this
run. The local generated videos and full replay reports remain in the ignored
`data/` lake.

## TRIBE Scores

Rows are individual fresh stochastic-seed clips. Task candidates are
recipe-by-pocket candidates aggregated over three stochastic replicates.

| pocket | role | task candidates | rows | row mean | row min | row max | positive rows | positive task means |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fresh24_orange_flowers` | primary positive | 6 | 18 | 3.8569 | 3.4717 | 4.4512 | 18 / 18 | 6 / 6 |
| `fresh24_hanging_clothes` | primary positive | 6 | 18 | 3.1519 | 1.5230 | 4.1540 | 18 / 18 | 6 / 6 |
| `fresh24_aerial_beach` | hard negative | 6 | 18 | -8.7489 | -10.0803 | -7.0079 | 0 / 18 | 0 / 6 |
| `fresh24_city_street` | hard negative | 6 | 18 | -9.3232 | -9.8582 | -8.9727 | 0 / 18 | 0 / 6 |
| `fresh24_storm_beach` | hard negative | 6 | 18 | -10.4614 | -11.0996 | -9.4910 | 0 / 18 | 0 / 6 |

The TRIBE replication gate passed cleanly: both target pockets stayed positive
across every retained row and every task-level candidate, while all hard
negative controls stayed negative under the matched recipe and stochastic seed
schedule.

## Embedding Verifiers

Exact V-JEPA feature extraction was complete: 90/90 scored replay MP4s produced
new `.npz` feature files. The embedding verifier then aggregated 90 scored rows
to 30 task-level candidates.

| family | feature | positive mean | negative mean | AUC | abs d | r(score) | result |
|---|---|---:|---:|---:|---:|---:|---|
| `vjepa_video` | pocket-held-out centroid margin | 0.0750 | -0.1371 | 1.0000 | 2.8636 | 0.8380 | accepted |
| `clip_seed_image` | pocket-held-out centroid margin | -0.0104 | -0.0996 | 0.8333 | 1.4599 | 0.6038 | not accepted |
| `clip_seed_video` | seed-video CLIP cosine | 0.9814 | 0.9708 | 0.8194 | 1.0973 | 0.3868 | not accepted |
| `clip_video` | pocket-held-out centroid margin | -0.0210 | -0.0971 | 0.6667 | 1.0056 | 0.4671 | not accepted |

| family | validation | predictions | AUC | balanced accuracy | result |
|---|---|---:|---:|---:|---|
| `vjepa_video` | leave-one-pocket-out | 30 | 1.0000 | 1.0000 | accepted |
| `clip_seed_image` | leave-one-pocket-out | 30 | 0.8333 | 0.7500 | not accepted |
| `clip_video` | leave-one-pocket-out | 30 | 0.8333 | 0.5833 | not accepted |

The embedding gate was accepted only through exact V-JEPA. Generated-video CLIP
did not transport prospectively: its pocket-held-out centroid-margin AUC fell to
0.6667, its positive-pocket mean margin was still negative, and its
leave-one-pocket-out classifier balanced accuracy was 0.5833.

## Gate Result

| gate item | result |
|---|---|
| complete generation | passed: 90 / 90 clips generated |
| complete-candidate visual retention | passed: 0 / 90 visual failures, 30 / 30 candidates retained |
| target TRIBE score | passed: orange flowers mean 3.8569, hanging clothes mean 3.1519 |
| hard negative controls | passed: aerial beach mean -8.7489, city street mean -9.3232, storm beach mean -10.4614 |
| exact V-JEPA prospective verifier | passed: centroid-margin AUC 1.0000, abs d 2.8636; LOPO balanced accuracy 1.0000 |
| generated-video CLIP prospective verifier | not accepted: centroid-margin AUC 0.6667 and LOPO balanced accuracy 0.5833 |

Overall gate: partial pass. The run strengthens orange flowers and hanging
clothes as fresh-seed TRIBE/V-JEPA-verified compute-proxy pockets. It does not
support the stronger claim that both V-JEPA and CLIP generated-video margins
agree prospectively on the fresh replication.

## Claim Impact

- Strengthens C-017 for the two strongest non-jellyfish pockets: orange flowers
  and hanging clothes replicated under fresh stochastic seeds with complete
  visual retention and matched hard negative controls.
- Narrows the prospective scope of C-018: exact V-JEPA transported cleanly as a
  prospective verifier, but generated-video CLIP did not clear the fresh
  replication gate.
- Does not prove human memorability, measured-BMD grounding, delayed
  recognition, prompt-conditioned generation, or broad BO/control superiority.

## Next Move

Do not spend human/BMD validation budget under a "V-JEPA plus CLIP both agree"
claim. The next control decision is whether to:

1. build a human/BMD packet with orange flowers and hanging clothes explicitly
   labeled as TRIBE/V-JEPA-verified compute-proxy candidates, or
2. run a targeted CLIP-prospective diagnostic or boundary audit before the
   human/BMD packet.

Either path should preserve the CLIP non-replication as evidence rather than
averaging it away.
