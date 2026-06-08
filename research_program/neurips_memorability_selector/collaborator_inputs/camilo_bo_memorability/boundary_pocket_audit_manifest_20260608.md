# Boundary Pocket Audit Manifest

Date: 2026-06-08

## Question

Do the weaker positive pockets, `fresh24_blue_jellyfish` and
`fresh24_old_car`, replicate under fresh stochastic SVD seeds strongly enough
to enter the next human/BMD validation packet, or should they remain
supporting/exploratory pockets behind orange flowers and hanging clothes?

## Discovery-Regime Audit

Current regime:

| component | value |
|---|---|
| Artifact types | restored seed image, prompt metadata, Sobol alpha/guidance recipe, fresh SVD replay clip, visual gate record, TRIBE score, exact V-JEPA generated-video feature, CLIP seed/video/text embedding, centroid-margin verifier, pocket label, result note |
| Operations | restore seed bank, build selected-slot Sobol manifest, generate SVD-XT clips on Modal, apply visual artifact gate, retain complete candidates, score retained clips with TRIBE, extract exact V-JEPA features from MP4 bytes, run embedding verifier |
| Gates/verifiers | complete-candidate visual retention; target-vs-control TRIBE separation; exact V-JEPA generated-video centroid margin; generated-video CLIP and prompt/seed CLIP diagnostics as non-primary supporting readouts |
| Known limitation | this is compute-proxy SVD replay evidence. Passing this gate does not prove human memorability, measured-BMD grounding, delayed recognition, or prompt-conditioned generation. Prompt text remains metadata-only in the current SVD path. |

Action class:

| item | classification |
|---|---|
| Manifest construction and replay | search inside the accepted SVD content-pocket regime |
| Discovery relevance | possible only if the boundary pockets change validation-packet membership: promote to validation packet, keep supporting, or demote |

Transported evidence:

- C-017 accepted that restored content pockets survive local SVD recipe stress
  tests, with orange flowers and hanging clothes as the strongest positives and
  blue jellyfish/old car as supporting positives.
- C-019 accepted orange flowers and hanging clothes as fresh-seed
  TRIBE/V-JEPA-verified compute-proxy candidate pockets.
- C-020 resolved the fresh replication CLIP gap: exact V-JEPA remains the
  primary prospective generated-video verifier; generated-video CLIP is not
  accepted prospectively, while prompt-seed CLIP is only ancillary.

Residual content sought:

- Determine whether blue jellyfish and old car are stable enough to join the
  human/BMD validation packet or should remain outside the first packet.
- Preserve hard negative separation against `fresh24_aerial_beach`,
  `fresh24_city_street`, and `fresh24_storm_beach`.

## Experiment

Local seed-bank restore report:

`data/reports/bo_seed_bank_restore_boundary_audit_20260608.json`

Trial-table manifest:

`data/reports/bo_boundary_pocket_audit_trial_table_sobol518_523_x5_noise350k_reps3_20260608.json`

Dry-run replay report:

`data/reports/bo_boundary_pocket_audit_trial_table_sobol518_523_x5_noise350k_reps3_dry_run_20260608.json`

Planned full replay report:

`data/reports/bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_noise0_20260608.json`

Planned generated-video directory:

`data/generated/bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_noise0_20260608`

Planned exact V-JEPA feature directory:

`data/features/vjepa_boundary_pocket_audit_20260608`

Planned V-JEPA extraction summaries:

- `boundary_pocket_audit_vjepa_extraction_summary_20260608.json`
- `boundary_pocket_audit_vjepa_extraction_result_20260608.md`

Planned embedding verifier summaries:

- `boundary_pocket_audit_embedding_summary_20260608.json`
- `boundary_pocket_audit_embedding_result_20260608.md`

Target seed slots:

| slot | bmd_name | role |
|---:|---|---|
| 18 | `fresh24_blue_jellyfish` | boundary positive target |
| 0 | `fresh24_old_car` | boundary positive target |
| 3 | `fresh24_aerial_beach` | hard negative control |
| 8 | `fresh24_city_street` | hard negative control |
| 14 | `fresh24_storm_beach` | hard negative control |

Recipe neighborhood:

| parameter | value |
|---|---:|
| Sobol indices | 518-523 |
| Sobol scramble seed | 42 |
| alpha range | -3.0 to 4.0 |
| guidance range | 2.5 to 10.0 |
| candidate pockets | 5 |
| recipes per pocket | 6 |
| task-level candidates | 30 |

Fresh stochastic seed schedule:

| parameter | value |
|---|---:|
| source noise seed offset | 350000 |
| source noise seeds | 350518-350523 |
| replicate seed stride | 10000 |
| replicates per task-level candidate | 3 |
| replicate seeds for recipe 518 | 350518, 360518, 370518 |
| total replay jobs | 90 |

This keeps the accepted Sobol recipe neighborhood fixed while moving the SVD
generation noise seeds away from both the old pocket-regime replay and the
orange/hanging descriptor-conditioned replication seed pools.

## Commands

Restore the local seed bank:

```bash
uv run python scripts/restore_bo_seed_bank.py \
  --report-path data/reports/bo_seed_bank_restore_boundary_audit_20260608.json
```

Build the replay-compatible trial table:

```bash
uv run python scripts/build_bo_prompt_search_manifest.py \
  --replay-seed-pool-size 24 \
  --target-seed-slot 18 \
  --target-seed-slot 0 \
  --target-seed-slot 3 \
  --target-seed-slot 8 \
  --target-seed-slot 14 \
  --sobol-samples-per-seed 6 \
  --sobol-start-index 518 \
  --sobol-scramble-seed 42 \
  --noise-seed-offset 350000 \
  --alpha-range=-3.0,4.0 \
  --guidance-range=2.5,10.0 \
  --report-path data/reports/bo_boundary_pocket_audit_trial_table_sobol518_523_x5_noise350k_reps3_20260608.json
```

Dry-run the replay expansion:

```bash
uv run python scripts/modal_bo_memorability_replay.py \
  --trial-table data/reports/bo_boundary_pocket_audit_trial_table_sobol518_523_x5_noise350k_reps3_20260608.json \
  --selection first \
  --max-evals 30 \
  --replicates 3 \
  --replay-seed-pool-size 24 \
  --num-inference-steps 50 \
  --svd-motion-bucket-id 5 \
  --svd-noise-aug-strength 0 \
  --visual-first-retention complete-candidates \
  --dry-run \
  --report-path data/reports/bo_boundary_pocket_audit_trial_table_sobol518_523_x5_noise350k_reps3_dry_run_20260608.json \
  --output-dir data/generated/bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_noise0_20260608
```

Run generation, visual retention, and TRIBE scoring:

```bash
uv run python scripts/modal_bo_memorability_replay.py \
  --trial-table data/reports/bo_boundary_pocket_audit_trial_table_sobol518_523_x5_noise350k_reps3_20260608.json \
  --selection first \
  --max-evals 30 \
  --replicates 3 \
  --replay-seed-pool-size 24 \
  --num-inference-steps 50 \
  --svd-motion-bucket-id 5 \
  --svd-noise-aug-strength 0 \
  --visual-first-retention complete-candidates \
  --report-path data/reports/bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_noise0_20260608.json \
  --output-dir data/generated/bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_noise0_20260608
```

Extract exact V-JEPA features:

```bash
uv run python scripts/extract_pocket_replay_vjepa.py \
  --replay-report data/reports/bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_noise0_20260608.json \
  --out-dir data/features/vjepa_boundary_pocket_audit_20260608 \
  --summary-json research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/boundary_pocket_audit_vjepa_extraction_summary_20260608.json \
  --summary-md research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/boundary_pocket_audit_vjepa_extraction_result_20260608.md
```

Run the embedding verifier and CLIP diagnostic readouts:

```bash
uv run python scripts/audit_content_pocket_embeddings.py \
  --replay-report data/reports/bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_noise0_20260608.json \
  --vjepa-features-dir data/features/vjepa_boundary_pocket_audit_20260608 \
  --include-text \
  --max-video-frames 8 \
  --out-json research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/boundary_pocket_audit_embedding_summary_20260608.json \
  --out-md research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/boundary_pocket_audit_embedding_result_20260608.md
```

## Gate

Generation and retention:

- Generate all 90 requested replay jobs, or record missing jobs explicitly.
- Apply the visual artifact gate before TRIBE scoring.
- Use complete-candidate retention: if any replicate for a task-level candidate
  fails the visual gate, withhold the full candidate and preserve the failed
  videos/reasons.
- Target-pocket interpretation fails if visual retention leaves no retained
  task-level candidates for either boundary positive pocket.

TRIBE acceptance:

- `fresh24_blue_jellyfish` has positive mean TRIBE score across retained
  fresh-seed candidates.
- `fresh24_old_car` has positive mean TRIBE score across retained fresh-seed
  candidates.
- `fresh24_aerial_beach`, `fresh24_city_street`, and `fresh24_storm_beach`
  remain negative under matched recipes and stochastic seed schedule.

Boundary-pocket membership:

- Promote a boundary pocket into the validation packet only if it has positive
  retained mean TRIBE score and every retained task-level candidate mean is
  positive.
- Keep a boundary pocket as supporting/exploratory if its retained mean stays
  positive but some task-level candidate means fail or variance is materially
  weaker than orange flowers/hanging clothes.
- Demote a boundary pocket if its retained mean is non-positive, if hard
  controls fail, or if visual retention removes too much evidence to interpret.

Exact V-JEPA acceptance:

- Exact V-JEPA feature coverage is complete for every scored retained MP4.
- V-JEPA generated-video pocket-held-out centroid margin remains positive for
  replicated positives and does not collapse toward the hard-negative centroid.
- Descriptor gate remains `separation_auc >= 0.85` and `abs_cohen_d >= 1.00`;
  classifier support remains leave-one-pocket-out `roc_auc >= 0.85` and
  `balanced_accuracy >= 0.75`.

CLIP interpretation:

- Generated-video CLIP is reported diagnostically only unless it clears the
  full generated-video descriptor/classifier gate.
- Prompt-seed CLIP may be used as an ancillary seed/prompt sanity check, not as
  a generated-video verifier.

Rejected/withheld rule:

- Preserve failed, withheld, visually rejected, unscored, and negative-control
  artifacts as part of the result.
- If a boundary pocket fails, narrow C-017/C-019 rather than averaging it into a
  broad success claim.
- Do not claim human memorability, measured-BMD grounding, delayed recognition,
  or prompt-conditioned generation from this compute-proxy boundary audit.

## Next Move

After the audit, build the human/BMD validation packet around:

- the already replicated primary pockets: orange flowers and hanging clothes;
- any boundary pocket that passes the membership gate above;
- hard negative controls retained as contrast examples.

If neither boundary pocket passes, keep the first validation packet narrow:
orange flowers and hanging clothes only, with V-JEPA as the primary
compute-proxy verifier and generated-video CLIP explicitly listed as a failed
prospective verifier.
