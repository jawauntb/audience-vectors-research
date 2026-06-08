# Descriptor-Conditioned Content-Pocket Replication Manifest

Date: 2026-06-08

## Question

Do `fresh24_orange_flowers` and `fresh24_hanging_clothes` replicate under
fresh stochastic SVD seeds while preserving positive TRIBE replay score and
the accepted exact V-JEPA/CLIP content-pocket margins?

## Discovery-Regime Audit

Current regime:

| component | value |
|---|---|
| Artifact types | restored seed image, prompt metadata, Sobol alpha/guidance recipe, fresh SVD replay clip, visual gate record, TRIBE score, exact V-JEPA generated-video feature, CLIP seed/video embedding, centroid-margin verifier, pocket label, result note |
| Operations | restore seed bank, build selected-slot Sobol manifest, generate SVD-XT clips on Modal, apply visual artifact gate, retain complete candidates, score retained clips with TRIBE, extract exact V-JEPA features from new MP4 bytes, run CLIP/V-JEPA embedding verifier |
| Gates/verifiers | complete-candidate visual retention; target-vs-control TRIBE separation; exact V-JEPA generated-video centroid margin; CLIP generated-video centroid margin; leakage-aware classifier summaries as supporting stress tests |
| Known limitation | this is still compute-proxy SVD replay evidence. Passing this gate does not prove human memorability, measured-BMD grounding, delayed recognition, or prompt-conditioned generation. Prompt text remains metadata-only in the current SVD path. |

Action class:

| item | classification |
|---|---|
| Manifest construction and replay | search inside the accepted SVD content-pocket regime |
| Discovery relevance | possible only if V-JEPA/CLIP margins work prospectively as candidate-selection constraints rather than merely explaining the old seed pool |

Transported evidence:

- C-017 accepted that restored non-jellyfish content pockets survive local SVD
  recipe stress tests.
- C-018 accepted exact V-JEPA and CLIP as compute-proxy verifiers for the
  current pocket-regime replay residual.
- The old pocket-regime audit used Sobol recipes 518-523, two stochastic reps,
  `steps=50`, `motion_bucket_id=5`, and `noise_aug_strength=0`.
- The strongest pockets to consolidate are `fresh24_orange_flowers` and
  `fresh24_hanging_clothes`; `fresh24_aerial_beach`,
  `fresh24_city_street`, and `fresh24_storm_beach` are hard negative controls.

Residual content sought:

- Positive TRIBE replay means for orange flowers and hanging clothes under
  fresh stochastic noise seeds.
- Hard controls remain negative under the same recipes and matched stochastic
  seed schedule.
- Exact V-JEPA and CLIP generated-video centroid margins remain positive for
  replicated positives and do not collapse toward the hard-negative centroid.

## Experiment

Local seed-bank restore report:

`data/reports/bo_seed_bank_restore_descriptor_conditioned_replication_20260608.json`

Trial-table manifest:

`data/reports/bo_descriptor_conditioned_replication_trial_table_sobol518_523_x5_noise250k_reps3_20260608.json`

Dry-run replay report:

`data/reports/bo_descriptor_conditioned_replication_trial_table_sobol518_523_x5_noise250k_reps3_dry_run_20260608.json`

Planned full replay report:

`data/reports/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608.json`

Planned generated-video directory:

`data/generated/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608`

Planned exact V-JEPA feature directory:

`data/features/vjepa_descriptor_conditioned_replication_20260608`

Planned V-JEPA extraction summaries:

- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/descriptor_conditioned_replication_vjepa_extraction_summary_20260608.json`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/descriptor_conditioned_replication_vjepa_extraction_result_20260608.md`

Planned embedding verifier summaries:

- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/descriptor_conditioned_replication_embedding_summary_20260608.json`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/descriptor_conditioned_replication_embedding_result_20260608.md`

Target seed slots:

| slot | bmd_name | role |
|---:|---|---|
| 10 | `fresh24_orange_flowers` | primary positive target |
| 12 | `fresh24_hanging_clothes` | primary positive target |
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
| source noise seed offset | 250000 |
| source noise seeds | 250518-250523 |
| replicate seed stride | 10000 |
| replicates per task-level candidate | 3 |
| replicate seeds for recipe 518 | 250518, 260518, 270518 |
| total replay jobs | 90 |

This keeps the accepted Sobol recipe neighborhood fixed while moving the SVD
generation noise seeds away from the old pocket-regime replay seed pool.

## Commands

Restore the local seed bank in any fresh worktree before building the trial
table:

```bash
uv run python scripts/restore_bo_seed_bank.py \
  --report-path data/reports/bo_seed_bank_restore_descriptor_conditioned_replication_20260608.json
```

Build the replay-compatible trial table:

```bash
uv run python scripts/build_bo_prompt_search_manifest.py \
  --replay-seed-pool-size 24 \
  --target-seed-slot 10 \
  --target-seed-slot 12 \
  --target-seed-slot 3 \
  --target-seed-slot 8 \
  --target-seed-slot 14 \
  --sobol-samples-per-seed 6 \
  --sobol-start-index 518 \
  --sobol-scramble-seed 42 \
  --noise-seed-offset 250000 \
  --alpha-range=-3.0,4.0 \
  --guidance-range=2.5,10.0 \
  --report-path data/reports/bo_descriptor_conditioned_replication_trial_table_sobol518_523_x5_noise250k_reps3_20260608.json
```

Dry-run the replay expansion:

```bash
uv run python scripts/modal_bo_memorability_replay.py \
  --trial-table data/reports/bo_descriptor_conditioned_replication_trial_table_sobol518_523_x5_noise250k_reps3_20260608.json \
  --selection first \
  --max-evals 30 \
  --replicates 3 \
  --replay-seed-pool-size 24 \
  --num-inference-steps 50 \
  --svd-motion-bucket-id 5 \
  --svd-noise-aug-strength 0 \
  --visual-first-retention complete-candidates \
  --dry-run \
  --report-path data/reports/bo_descriptor_conditioned_replication_trial_table_sobol518_523_x5_noise250k_reps3_dry_run_20260608.json \
  --output-dir data/generated/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608
```

Run generation, visual retention, and TRIBE scoring:

```bash
uv run python scripts/modal_bo_memorability_replay.py \
  --trial-table data/reports/bo_descriptor_conditioned_replication_trial_table_sobol518_523_x5_noise250k_reps3_20260608.json \
  --selection first \
  --max-evals 30 \
  --replicates 3 \
  --replay-seed-pool-size 24 \
  --num-inference-steps 50 \
  --svd-motion-bucket-id 5 \
  --svd-noise-aug-strength 0 \
  --visual-first-retention complete-candidates \
  --report-path data/reports/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608.json \
  --output-dir data/generated/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608
```

Extract exact V-JEPA features for the new generated MP4 bytes:

```bash
uv run python scripts/extract_pocket_replay_vjepa.py \
  --replay-report data/reports/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608.json \
  --out-dir data/features/vjepa_descriptor_conditioned_replication_20260608 \
  --summary-json research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/descriptor_conditioned_replication_vjepa_extraction_summary_20260608.json \
  --summary-md research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/descriptor_conditioned_replication_vjepa_extraction_result_20260608.md
```

Run the accepted CLIP/V-JEPA embedding verifier:

```bash
uv run python scripts/audit_content_pocket_embeddings.py \
  --replay-report data/reports/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608.json \
  --vjepa-features-dir data/features/vjepa_descriptor_conditioned_replication_20260608 \
  --out-json research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/descriptor_conditioned_replication_embedding_summary_20260608.json \
  --out-md research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/descriptor_conditioned_replication_embedding_result_20260608.md
```

## Gate

Generation and retention:

- Generate all 90 requested replay jobs, or record missing jobs explicitly.
- Apply the visual artifact gate before TRIBE scoring.
- Use complete-candidate retention: if any replicate for a task-level candidate
  fails the visual gate, withhold the full candidate and preserve the failed
  videos/reasons.
- Target-pocket interpretation fails if visual retention leaves no retained
  task-level candidates for either primary positive pocket.

TRIBE acceptance:

- `fresh24_orange_flowers` has positive mean TRIBE score across retained
  fresh-seed candidates.
- `fresh24_hanging_clothes` has positive mean TRIBE score across retained
  fresh-seed candidates.
- `fresh24_aerial_beach`, `fresh24_city_street`, and `fresh24_storm_beach`
  remain negative under matched recipes and stochastic seed schedule.

Exact V-JEPA acceptance:

- Exact V-JEPA feature coverage is complete for every scored retained MP4.
- V-JEPA generated-video pocket-held-out centroid margin remains positive for
  replicated positives and does not collapse toward the hard-negative centroid.
- Descriptor gate remains `separation_auc >= 0.85` and `abs_cohen_d >= 1.00`;
  classifier support remains leave-one-pocket-out `roc_auc >= 0.85` and
  `balanced_accuracy >= 0.75`.

CLIP acceptance:

- Generated-video CLIP pocket-held-out centroid margin remains positive for
  replicated positives and does not collapse toward the hard-negative centroid.
- The same descriptor/classifier thresholds apply. Seed-image CLIP geometry may
  be reported as a reference, but generated-video CLIP is the prospective
  replication verifier.

Rejected/withheld rule:

- Preserve failed, withheld, visually rejected, unscored, and negative-control
  artifacts as part of the result.
- If the target pockets fail TRIBE or embedding margins, narrow C-017/C-018
  rather than blending failed examples into a broad success claim.
- Do not claim human memorability, measured-BMD grounding, delayed recognition,
  or prompt-conditioned generation from this compute-proxy replication.

## Next Move

If the gate passes, promote orange flowers and hanging clothes only to
descriptor-verified compute-proxy candidate pockets for human/BMD validation,
then update `CLAIM_LEDGER.md`, `NEXT_STEPS.md`, and
`research_program/neurips_memorability_selector/experiments/current_research_status.md`.

If the gate partially passes or fails, preserve the failed replication, demote
unstable pockets, and decide whether to run the blue jellyfish/old car boundary
audit or switch to a prompt-conditioned generator regime.
