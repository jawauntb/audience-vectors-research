# Boundary Pocket Audit Result

Date: 2026-06-08

## Discovery-Regime Audit

Question: do the weaker positive pockets, `fresh24_blue_jellyfish` and
`fresh24_old_car`, replicate under fresh stochastic SVD seeds strongly enough to
enter the next human/BMD validation packet, or should they remain outside the
primary validation set behind orange flowers and hanging clothes?

Current regime:

- Artifact types: restored seed images, prompt metadata, Sobol alpha/guidance
  recipes, fresh SVD replay clips, visual-gate records, TRIBE scores, exact
  V-JEPA features from generated MP4 bytes, CLIP seed/video/text embeddings,
  pocket labels, and claim-ledger entries.
- Operations: restore seed bank, build selected-slot Sobol manifest, generate
  SVD-XT clips on Modal, apply complete-candidate visual retention, score
  retained clips with TRIBE, extract exact V-JEPA features, and run embedding
  verifier diagnostics.
- Gates/verifiers: complete generation and visual retention; positive TRIBE
  means for boundary pockets; negative hard controls; exact V-JEPA and CLIP
  embedding gates tracked separately.
- Known limitation: this is compute-proxy SVD replay evidence. It is not human
  memorability, measured-BMD validation, delayed recognition, or
  prompt-conditioned generation.

Action class: search inside the accepted SVD content-pocket regime. The result
is discovery-relevant only because it changes validation-packet membership:
blue jellyfish and old car are no longer just old-run supporting positives, but
their forward verifier status is weaker than orange flowers and hanging clothes.

## Inputs

- Manifest:
  `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/boundary_pocket_audit_manifest_20260608.md`
- Local seed-bank restore report:
  `data/reports/bo_seed_bank_restore_boundary_audit_20260608.json`
- Trial table:
  `data/reports/bo_boundary_pocket_audit_trial_table_sobol518_523_x5_noise350k_reps3_20260608.json`
- Dry-run expansion:
  `data/reports/bo_boundary_pocket_audit_trial_table_sobol518_523_x5_noise350k_reps3_dry_run_20260608.json`
- Replay report:
  `data/reports/bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_noise0_20260608.json`
- Generated videos:
  `data/generated/bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_noise0_20260608`
- Exact V-JEPA feature directory:
  `data/features/vjepa_boundary_pocket_audit_20260608`
- V-JEPA extraction summary:
  `boundary_pocket_audit_vjepa_extraction_summary_20260608.json`
- Embedding verifier summary:
  `boundary_pocket_audit_embedding_summary_20260608.json`

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

No failed, withheld, visually rejected, or unscored rows were produced. The
local generated videos and full replay reports remain in the ignored `data/`
lake.

## TRIBE Scores

Task candidates are recipe-by-pocket candidates aggregated over three fresh
stochastic replicates.

| pocket | role | task candidates | row-level clips | candidate mean | candidate min | candidate max | positive task means |
|---|---|---:|---:|---:|---:|---:|---:|
| `fresh24_blue_jellyfish` | boundary positive | 6 | 18 | 1.8844 | 1.0461 | 2.6652 | 6 / 6 |
| `fresh24_old_car` | boundary positive | 6 | 18 | 1.3110 | 0.9776 | 1.5023 | 6 / 6 |
| `fresh24_aerial_beach` | hard negative | 6 | 18 | -8.8361 | -9.6327 | -7.9565 | 0 / 6 |
| `fresh24_city_street` | hard negative | 6 | 18 | -9.3657 | -9.5246 | -9.1931 | 0 / 6 |
| `fresh24_storm_beach` | hard negative | 6 | 18 | -10.3363 | -11.0822 | -9.5990 | 0 / 6 |

The TRIBE boundary gate passed: both boundary pockets stayed positive across
all retained task-level candidates, while all hard negative controls stayed
negative under the matched recipe and stochastic seed schedule.

## Embedding Verifiers

Exact V-JEPA extraction was complete: 90/90 scored replay MP4s produced new
`.npz` feature files. The embedding verifier then aggregated 90 scored rows to
30 task-level candidates.

| family | feature | positive mean | negative mean | AUC | abs d | r(score) | result |
|---|---|---:|---:|---:|---:|---:|---|
| `clip_seed_video` | seed-video CLIP cosine | 0.9404 | 0.9728 | 0.9722 | 2.2252 | -0.7860 | accepted; lower for positive |
| `clip_prompt_seed` | prompt-seed CLIP cosine | 0.3117 | 0.2939 | 0.8333 | 1.9506 | 0.7094 | not accepted by AUC gate |
| `vjepa_video` | pocket-held-out centroid margin | -0.0390 | -0.0960 | 0.8194 | 1.0704 | 0.4838 | not accepted by AUC gate |
| `clip_prompt_video` | prompt-video CLIP cosine | 0.3211 | 0.3103 | 0.6667 | 0.7997 | 0.4265 | not accepted |
| `clip_seed_image` | pocket-held-out centroid margin | -0.0442 | -0.0502 | 0.6667 | 0.3429 | 0.2256 | not accepted |
| `clip_video` | pocket-held-out centroid margin | -0.0180 | -0.0464 | 0.6620 | 0.7490 | 0.4113 | not accepted |

| family | validation | AUC | balanced accuracy | result |
|---|---|---:|---:|---|
| `clip_video` | leave-one-pocket-out classifier | 0.9676 | 0.7917 | accepted |
| `vjepa_video` | leave-one-pocket-out classifier | 0.8333 | 0.7500 | not accepted by AUC gate |
| `clip_seed_image` | leave-one-pocket-out classifier | 1.0000 | 0.5000 | not accepted by balanced-accuracy gate |

The embedding gate was accepted, but not through V-JEPA. Boundary pockets are
TRIBE-stable and CLIP-supported in this fresh audit; they are not
V-JEPA-verified boundary pockets under the pre-registered AUC >= 0.85 rule.

## Gate Result

| gate item | result |
|---|---|
| complete generation | passed: 90 / 90 clips generated |
| complete-candidate visual retention | passed: 0 / 90 visual failures, 30 / 30 candidates retained |
| target TRIBE score | passed: blue jellyfish mean 1.8844, old car mean 1.3110 |
| hard negative controls | passed: aerial beach mean -8.8361, city street mean -9.3657, storm beach mean -10.3363 |
| exact V-JEPA boundary verifier | not accepted: centroid-margin AUC 0.8194; classifier AUC 0.8333 |
| CLIP boundary verifier | accepted: seed-video descriptor AUC 0.9722 / abs d 2.2252; video classifier AUC 0.9676 / balanced accuracy 0.7917 |

Overall gate: partial pass. Blue jellyfish and old car graduate from
old-run supporting positives to fresh-seed TRIBE-stable boundary pockets with
CLIP-side support. They should not be merged into the primary V-JEPA-verified
candidate claim.

## Claim Impact

- Strengthens C-017 for blue jellyfish and old car as stable positive content
  pockets under fresh stochastic SVD seeds and matched hard controls.
- Narrows the boundary-pocket part of C-018: exact V-JEPA remains accepted for
  the original pocket-regime residual and orange/hanging replication, but did
  not pass this boundary audit.
- Adds a separate CLIP-side boundary descriptor: seed-video CLIP cosine and a
  generated-video CLIP classifier separate boundary positives from hard
  negatives in this run.
- Does not prove human memorability, measured-BMD grounding, delayed
  recognition, prompt-conditioned generation, or broad BO/control superiority.

## Validation-Packet Decision

Use a tiered human/BMD packet:

- Primary candidate pockets: `fresh24_orange_flowers` and
  `fresh24_hanging_clothes`, labeled as TRIBE/V-JEPA-verified compute-proxy
  pockets with generated-video CLIP non-replication preserved as a caveat.
- Secondary boundary arms: `fresh24_blue_jellyfish` and `fresh24_old_car`,
  labeled as TRIBE-stable and CLIP-supported but not V-JEPA-verified in the
  fresh boundary audit.
- Hard negative controls: `fresh24_aerial_beach`, `fresh24_city_street`, and
  `fresh24_storm_beach`.

Do not pool secondary boundary arms into the primary human/BMD success claim.
They can test generality and failure modes, but the primary validation gate
should remain orange flowers and hanging clothes.
