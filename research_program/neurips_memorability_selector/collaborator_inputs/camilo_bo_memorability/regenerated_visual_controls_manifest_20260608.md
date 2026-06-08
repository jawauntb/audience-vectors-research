# Regenerated Visual Controls Manifest

Last updated: 2026-06-08.

This manifest freezes the next compute-side gate after the saved-table
visual-first retention run. The saved table cannot supply a complete matched
visual-gated BO/Sobol panel because both fireworks Sobol candidates were
withheld by complete-candidate visual-first retention. The next defensible
move is therefore to regenerate deterministic Sobol controls for the same BO
prompt strata, apply the visual gate before scoring, and preserve every
withheld clip as provenance.

This is a foundation/protocol artifact, not a claim upgrade.

## What This Adds

The replay script now supports:

- `--selection top-bo-per-stratum`: select the top saved BO candidates inside
  each BO-covered prompt or seed stratum.
- `--regenerated-sobol-controls-per-stratum`: append unscored deterministic
  Sobol controls for each selected BO stratum.
- `--regenerated-sobol-pool-size`, `--regenerated-sobol-start-index`, and
  `--regenerated-sobol-scramble-seed`: freeze the Sobol control search space.

The regenerated controls are not selected by TRIBE score. They are selected by
deterministic Sobol sequence order after matching the selected BO strata. The
report records a `regenerated_sobol_controls` block with the Sobol index,
alpha, guidance, seed slot, prompt stratum, and any missing strata.

## Dry-Run Preview

Command:

```bash
uv run python scripts/modal_bo_memorability_replay.py \
  --dry-run \
  --selection top-bo-per-stratum \
  --stratify-by prompt \
  --max-evals 2 \
  --regenerated-sobol-controls-per-stratum 2 \
  --regenerated-sobol-pool-size 128 \
  --replicates 3 \
  --report-path /tmp/bo_regenerated_controls_preview.json
```

Result:

- loaded 32 saved-table trials;
- selected 4 saved BO anchors;
- appended 4 regenerated Sobol controls;
- expanded to 24 replay jobs (`8` candidates x `3` replicates);
- found no missing regenerated-control strata.

Candidate preview:

| stratum | policy | task id | alpha | guidance | seed idx | original TRIBE |
|---|---|---|---:|---:|---:|---:|
| fireworks | BO | `bo06_cand01` | -4.1262 | 7.8464 | 10 | -0.3899 |
| fireworks | BO | `bo09_cand01` | 7.0962 | 2.4844 | 10 | -0.9172 |
| fireworks | Sobol regen | `sobol_regen_016` | 7.9397 | 4.7943 | 10 | n/a |
| fireworks | Sobol regen | `sobol_regen_017` | -7.1886 | 8.2405 | 0 | n/a |
| jellyfish | BO | `bo07_cand01` | 7.0735 | 3.2311 | 13 | 6.1509 |
| jellyfish | BO | `bo04_cand01` | -3.9674 | 7.7753 | 13 | 5.3993 |
| jellyfish | Sobol regen | `sobol_regen_013` | -1.8405 | 5.5206 | 3 | n/a |
| jellyfish | Sobol regen | `sobol_regen_020` | 1.3305 | 1.2554 | 3 | n/a |

## Modal Run Command

Use the tuned SVD settings from the accepted one-replicate visual-gated smoke
and complete-candidate visual-first retention:

```bash
BO_MEM_STEERING_ARTIFACT=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --selection top-bo-per-stratum \
  --stratify-by prompt \
  --max-evals 2 \
  --regenerated-sobol-controls-per-stratum 2 \
  --regenerated-sobol-pool-size 128 \
  --replicates 3 \
  --num-inference-steps 50 \
  --svd-motion-bucket-id 5 \
  --svd-noise-aug-strength 0 \
  --generation-timeout 1800 \
  --tribe-mode full \
  --tribe-input bytes \
  --tribe-timeout 300 \
  --tribe-concurrency 3 \
  --visual-first-retention complete-candidates \
  --report-path data/reports/bo_regenerated_visual_controls_topbo2_regensobol2_reps3_steps50_motion5_noise0_20260608.json \
  --output-dir data/generated/bo_regenerated_visual_controls_topbo2_regensobol2_reps3_steps50_motion5_noise0_20260608
```

Local ignored artifacts expected from the full run:

- `data/reports/bo_regenerated_visual_controls_topbo2_regensobol2_reps3_steps50_motion5_noise0_20260608.json`
- `data/generated/bo_regenerated_visual_controls_topbo2_regensobol2_reps3_steps50_motion5_noise0_20260608/*.mp4`

## Acceptance Readout

Primary report fields:

- `regenerated_sobol_controls`: verifies regenerated control provenance and
  missing-stratum status.
- `visual_artifact_gate`: records all visual failures before scoring.
- `visual_first_retention`: records which complete candidates were retained or
  withheld.
- `stratum_policy_summary`: primary BO/Sobol comparison after retention.
- `replicate_summary`: candidate-level mean/std/SEM over retained TRIBE scores.
- `policy_group_summary`: pooled summary, secondary to per-stratum results.

The run only unblocks a candidate set for human-panel preparation if at least
one BO candidate and one regenerated Sobol candidate are retained in the same
prompt stratum after complete-candidate visual-first retention. A broader
BO-over-control claim still requires more prompt strata and human validation.

Reviewer-safe wording after this PR, before the full run:

```text
We added a regenerated-control protocol that selects top saved BO candidates
per prompt stratum, appends deterministic unscored Sobol controls for those
same strata, applies visual-first complete-candidate retention before TRIBE
scoring, and records regenerated-control provenance in the replay report. A
dry run selected four BO anchors and four regenerated Sobol controls across
the fireworks and jellyfish strata, expanding to 24 planned replay jobs.
```

