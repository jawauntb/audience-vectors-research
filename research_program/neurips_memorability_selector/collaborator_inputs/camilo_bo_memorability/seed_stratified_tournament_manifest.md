# Seed-Stratified BO/Sobol Tournament Manifest

Last updated: 2026-06-05.

This manifest is the next compute-side gate after the equal-budget BO/Sobol
panel. It runs a small matched-stratum replay before spending on a broader
regenerated baseline.

## Purpose

The prior equal-budget panel showed BO beating the saved Sobol top-5 under
replicated replay, but the BO top-5 were all from the
`fresh24_blue_jellyfish` seed pocket. This manifest asks a narrower and fairer
question:

```text
When BO and Sobol both have saved candidates for the same prompt/seed-image
content, does BO still replay better under the same stochastic replicate budget?
```

## Required Local Artifacts

These files are intentionally not committed:

- `tribe_clip_adapter.pt`, passed via `BO_MEM_STEERING_ARTIFACT` or
  `--steering-artifact`;
- `v_mem.npz`, passed via `BO_MEM_CORTICAL_VMEM` or `--cortical-vmem`.

Use a preflight dry-run before queuing Modal work:

```bash
BO_MEM_STEERING_ARTIFACT=/path/to/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/path/to/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --dry-run \
  --require-artifacts \
  --selection seed-stratified-bo-vs-sobol \
  --stratify-by prompt \
  --max-evals 1 \
  --replicates 3 \
  --report-path /tmp/bo_seed_stratified_preflight.json
```

## Selected Saved-Table Strata

Dry-run selection on the saved 32-trial 3-objective table selects four
candidates across two matched prompt strata:

| stratum | policy | task id | seed image | seed_idx | original TRIBE |
|---|---|---|---|---:|---:|
| fireworks | BO | `bo06_cand01` | `fresh24_fireworks` | 10 | -0.3899 |
| fireworks | Sobol | `sobol_007` | `fresh24_fireworks` | 10 | -0.7994 |
| jellyfish | BO | `bo07_cand01` | `fresh24_blue_jellyfish` | 13 | 6.1509 |
| jellyfish | Sobol | `sobol_005` | `fresh24_blue_jellyfish` | 13 | 3.5215 |

With `--replicates 3`, this expands to 12 generated/scored videos.

## Modal Run Command

```bash
BO_MEM_STEERING_ARTIFACT=/path/to/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/path/to/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --selection seed-stratified-bo-vs-sobol \
  --stratify-by prompt \
  --max-evals 1 \
  --replicates 3 \
  --num-inference-steps 4 \
  --tribe-mode full \
  --tribe-input bytes \
  --tribe-timeout 300 \
  --tribe-concurrency 3 \
  --report-path data/reports/bo_modal_replay_seed_stratified_20260605.json \
  --output-dir data/generated/bo_modal_replay_seed_stratified_20260605
```

## Acceptance Readout

Use the report fields already emitted by the replay script:

- `stratum_policy_summary`: primary matched-stratum comparison;
- `replicate_summary`: candidate-level mean/std/SEM and score ranges;
- `policy_group_summary`: pooled BO/Sobol comparison, secondary only because
  there are only two strata.

The result should not upgrade the BO claim to broad prompt coverage unless BO
wins outside jellyfish and the visual artifact gate also passes. If this saved
table panel is inconclusive, regenerate a matched baseline across all available
seed images/prompts rather than selecting only from the historical table.
