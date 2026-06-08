# Prompt-Transfer Stress Test Result - 2026-06-08

Last updated: 2026-06-08.

This note records the first prompt-broadened stress test after the max-3
regenerated-control result. The question is narrower and more mechanistic than
"does BO win?": do the top saved BO parameter recipes transfer across prompt
strata, or are they prompt-pocket artifacts?

## Protocol

The table was built with
`scripts/build_bo_prompt_transfer_manifest.py` and then replayed through
`scripts/modal_bo_memorability_replay.py`.

Prompt-transfer design:

- BO-transfer anchors: the top 3 saved BO recipes by original table TRIBE score:
  `bo07_cand01`, `bo04_cand01`, and `bo02_cand01`
- target prompt slots: the 5 locally image-backed replay seed slots:
  fireworks, ocean cliffs, concert stage, blue jellyfish, and forest canopy
- BO-transfer rows: each BO recipe retargeted to each prompt slot, 15 rows total
- Sobol-transfer controls: 3 deterministic Sobol alpha/guidance recipes
  retargeted to each prompt slot, 15 rows total
- visual policy: `--visual-first-retention complete-candidates`
- replicates: 1 per candidate
- SVD settings: 50 inference steps, motion bucket 5, noise augmentation 0
- TRIBE mode: full, direct bytes input, 300 second timeout, concurrency 3

Local ignored outputs:

- trial table:
  `data/reports/bo_prompt_transfer_trial_table_top3x5_sobol3_20260608.json`
- replay report:
  `data/reports/bo_prompt_transfer_top3x5_sobol3_reps1_steps50_motion5_noise0_20260608.json`
- videos:
  `data/generated/bo_prompt_transfer_top3x5_sobol3_reps1_steps50_motion5_noise0_20260608/`

## Commands

Build the transfer table:

```bash
uv run python scripts/build_bo_prompt_transfer_manifest.py \
  --top-bo-anchors 3 \
  --sobol-controls-per-seed 3 \
  --sobol-start-index 256 \
  --report-path data/reports/bo_prompt_transfer_trial_table_top3x5_sobol3_20260608.json
```

Dry-run the replay expansion:

```bash
BO_MEM_STEERING_ARTIFACT=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --dry-run \
  --require-artifacts \
  --trial-table data/reports/bo_prompt_transfer_trial_table_top3x5_sobol3_20260608.json \
  --selection first \
  --max-evals 30 \
  --stratify-by prompt \
  --replicates 1 \
  --report-path /tmp/bo_prompt_transfer_top3x5_sobol3_dry_run_20260608.json
```

Full run:

```bash
BO_MEM_STEERING_ARTIFACT=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --trial-table data/reports/bo_prompt_transfer_trial_table_top3x5_sobol3_20260608.json \
  --selection first \
  --max-evals 30 \
  --stratify-by prompt \
  --replicates 1 \
  --num-inference-steps 50 \
  --svd-motion-bucket-id 5 \
  --svd-noise-aug-strength 0 \
  --generation-timeout 1800 \
  --tribe-mode full \
  --tribe-input bytes \
  --tribe-timeout 300 \
  --tribe-concurrency 3 \
  --visual-first-retention complete-candidates \
  --report-path data/reports/bo_prompt_transfer_top3x5_sobol3_reps1_steps50_motion5_noise0_20260608.json \
  --output-dir data/generated/bo_prompt_transfer_top3x5_sobol3_reps1_steps50_motion5_noise0_20260608
```

## Generation And Visual Gate

The run generated all 30 requested MP4s. The automated visual artifact gate
failed 2/30 videos:

| failed row | task | prompt slot | flags |
|---|---|---|---|
| `bo_replay_00_bo_transfer_bo07_cand01_slot00` | `bo_transfer_bo07_cand01_slot00` | fireworks | `tail_sharpness_collapse`, `tail_contrast_collapse` |
| `bo_replay_25_sobol_transfer_258_slot00` | `sobol_transfer_258_slot00` | fireworks | `tail_sharpness_collapse` |

Complete-candidate retention kept 28/30 rows and 28/30 candidates for TRIBE
scoring. Because this was a one-replicate stress test, complete-candidate
retention is equivalent to withholding the two visually failed rows.

## TRIBE Results

All retained rows completed full TRIBE scoring: 28/28.

Policy summary after retention:

| policy | requested candidates | scored candidates | mean TRIBE | std | best retained candidate |
|---|---:|---:|---:|---:|---|
| BO-transfer | 15 | 14 | -3.5444 | 3.1705 | `bo_transfer_bo04_cand01_slot03` mean 1.5428 |
| Sobol-transfer | 15 | 14 | -3.0223 | 3.5794 | `sobol_transfer_257_slot03` mean 2.0134 |

Stratum summary:

| prompt slot | BO-transfer mean | Sobol-transfer mean | interpretation |
|---|---:|---:|---|
| fireworks | -4.0593 | -3.9195 | visually brittle and negative for both policies |
| ocean cliffs | -7.5721 | -8.0602 | very negative for both policies |
| concert stage | -2.4617 | -0.3998 | Sobol-transfer is less bad; BO recipes do not help |
| blue jellyfish | 1.0982 | 1.2100 | only positive prompt slot; Sobol-transfer slightly higher |
| forest canopy | -4.8986 | -4.2410 | negative for both policies |

Top retained candidates:

| rank | task | policy | prompt slot | score |
|---:|---|---|---|---:|
| 1 | `sobol_transfer_257_slot03` | Sobol-transfer | blue jellyfish | 2.0134 |
| 2 | `bo_transfer_bo04_cand01_slot03` | BO-transfer | blue jellyfish | 1.5428 |
| 3 | `bo_transfer_bo02_cand01_slot03` | BO-transfer | blue jellyfish | 1.1928 |
| 4 | `sobol_transfer_258_slot03` | Sobol-transfer | blue jellyfish | 1.1034 |
| 5 | `bo_transfer_bo07_cand01_slot03` | BO-transfer | blue jellyfish | 0.5589 |
| 6 | `sobol_transfer_256_slot03` | Sobol-transfer | blue jellyfish | 0.5133 |

## Claim Impact

This run produces the strongest compute-side evidence so far that the saved BO
recipes are not portable strategy recipes.

Reviewer-safe statements:

- The prompt-transfer stress test broadened the visual-gated replay panel from
  two saved-table prompt strata to five locally image-backed prompt slots.
- The top three saved BO recipes did not generalize across prompt slots. Their
  transferred scores were positive only on blue jellyfish and negative on
  fireworks, ocean cliffs, concert stage, and forest canopy.
- The same positive-pocket pattern also appears in Sobol-transfer controls:
  blue jellyfish is the only positive prompt slot, and the best retained
  candidate is a Sobol-transfer control rather than a BO-transfer row.
- Overall, Sobol-transfer slightly beat BO-transfer in this one-replicate panel
  (-3.0223 vs -3.5444), so this run is evidence against broad BO recipe
  transfer.

Do not claim:

- BO recipes generalize across prompts.
- BO broadly beats Sobol controls.
- The prompt-transfer test is a full new BO/search panel.
- The one-replicate transfer panel is human-memorability evidence.

New working hypothesis:

```text
Under the current SVD/TRIBE replay settings, prompt identity dominates parameter
recipe transfer: the saved high-scoring BO recipes are jellyfish-pocket recipes,
not reusable global steering/guidance policies.
```

Next scientific step: run a true per-prompt search or cheap proxy prefilter over
more prompt strata, then reserve full TRIBE scoring for visually retained
candidate families. A pure transfer of old BO recipes is no longer a promising
path to a broad BO/control claim.
