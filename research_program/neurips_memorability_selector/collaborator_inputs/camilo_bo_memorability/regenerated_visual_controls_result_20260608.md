# Regenerated Visual Controls Result - 2026-06-08

Last updated: 2026-06-08.

This note records the full regenerated-control follow-up for the BO
memorability replay. It completes the run proposed in
`regenerated_visual_controls_manifest_20260608.md`.

## Protocol

Selection:

- saved BO anchors: `--selection top-bo-per-stratum`
- strata: `--stratify-by prompt`
- saved BO budget: `--max-evals 2` per prompt stratum
- regenerated controls: `--regenerated-sobol-controls-per-stratum 2`
- regenerated control pool: `--regenerated-sobol-pool-size 128`
- replicates: `3` per selected candidate
- visual policy: `--visual-first-retention complete-candidates`
- SVD settings: 50 inference steps, motion bucket 5, noise augmentation 0
- TRIBE mode: full, direct bytes input, 300 second timeout, concurrency 3

Local ignored outputs:

- report:
  `data/reports/bo_regenerated_visual_controls_topbo2_regensobol2_reps3_steps50_motion5_noise0_20260608.json`
- videos:
  `data/generated/bo_regenerated_visual_controls_topbo2_regensobol2_reps3_steps50_motion5_noise0_20260608/`

## Commands

Preflight:

```bash
BO_MEM_STEERING_ARTIFACT=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --dry-run \
  --require-artifacts \
  --selection top-bo-per-stratum \
  --stratify-by prompt \
  --max-evals 2 \
  --regenerated-sobol-controls-per-stratum 2 \
  --regenerated-sobol-pool-size 128 \
  --replicates 3 \
  --report-path /tmp/bo_regenerated_controls_preflight.json
```

Full run:

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

## Selection

The run selected 4 saved BO anchors and appended 4 deterministic, unscored
Sobol controls. No target prompt stratum was missing regenerated controls.

| task | policy | prompt stratum | alpha | guidance | seed_idx | noise_seed |
|---|---|---|---:|---:|---:|---:|
| `bo06_cand01` | BO | fireworks | -4.1262 | 7.8464 | 10 | 601 |
| `bo09_cand01` | BO | fireworks | 7.0962 | 2.4844 | 10 | 901 |
| `sobol_regen_016` | regenerated Sobol | fireworks | 7.9397 | 4.7943 | 10 | 16 |
| `sobol_regen_017` | regenerated Sobol | fireworks | -7.1886 | 8.2405 | 0 | 17 |
| `bo07_cand01` | BO | jellyfish | 7.0735 | 3.2311 | 13 | 701 |
| `bo04_cand01` | BO | jellyfish | -3.9674 | 7.7753 | 13 | 401 |
| `sobol_regen_013` | regenerated Sobol | jellyfish | -1.8405 | 5.5206 | 3 | 13 |
| `sobol_regen_020` | regenerated Sobol | jellyfish | 1.3305 | 1.2554 | 3 | 20 |

## Visual-First Retention

The full run generated all 24 requested MP4s. The automated visual artifact
gate failed 1/24 videos: `bo_replay_01_bo09_cand01_rep01`, with
`tail_sharpness_collapse`.

Because the run used complete-candidate visual-first retention, all 3 rows for
`bo09_cand01` were withheld before upload/scoring. The retained panel contains
21/24 rows and 7/8 candidates.

| task | policy | stratum | retained | scored rows | visual note |
|---|---|---|---|---:|---|
| `bo06_cand01` | BO | fireworks | yes | 3/3 | all replicates passed |
| `bo09_cand01` | BO | fireworks | no | 0/3 | replicate 1 failed tail sharpness |
| `sobol_regen_016` | regenerated Sobol | fireworks | yes | 3/3 | all replicates passed |
| `sobol_regen_017` | regenerated Sobol | fireworks | yes | 3/3 | all replicates passed |
| `bo07_cand01` | BO | jellyfish | yes | 3/3 | all replicates passed |
| `bo04_cand01` | BO | jellyfish | yes | 3/3 | all replicates passed |
| `sobol_regen_013` | regenerated Sobol | jellyfish | yes | 3/3 | all replicates passed |
| `sobol_regen_020` | regenerated Sobol | jellyfish | yes | 3/3 | all replicates passed |

The regenerated-control gate's structural pass condition is satisfied: after
complete-candidate retention, at least one BO candidate and one regenerated
Sobol control remain in each selected prompt stratum.

## TRIBE Results

All retained rows completed full TRIBE scoring: 21/21.

Candidate ranking by mean replay TRIBE score:

| rank | task | policy | scored | mean | std | sem | original score |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `bo04_cand01` | BO | 3/3 | 2.1171 | 0.3615 | 0.2087 | 5.3993 |
| 2 | `bo07_cand01` | BO | 3/3 | 1.2318 | 0.1038 | 0.0599 | 6.1509 |
| 3 | `sobol_regen_013` | regenerated Sobol | 3/3 | 1.1270 | 0.1379 | 0.0796 | n/a |
| 4 | `sobol_regen_020` | regenerated Sobol | 3/3 | 0.9846 | 0.9671 | 0.5583 | n/a |
| 5 | `sobol_regen_017` | regenerated Sobol | 3/3 | -3.0944 | 1.1475 | 0.6625 | n/a |
| 6 | `sobol_regen_016` | regenerated Sobol | 3/3 | -3.5538 | 0.5455 | 0.3150 | n/a |
| 7 | `bo06_cand01` | BO | 3/3 | -3.9426 | 0.4228 | 0.2441 | -0.3899 |
| 8 | `bo09_cand01` | BO | 0/3 | n/a | n/a | n/a | -0.9172 |

Policy summary after retention:

| policy | candidates | requested rows | scored rows | pooled mean | pooled std | best retained candidate |
|---|---:|---:|---:|---:|---:|---|
| BO | 4 | 12 | 9 | -0.1979 | 2.8487 | `bo04_cand01` mean 2.1171 |
| regenerated Sobol | 4 | 12 | 12 | -1.1342 | 2.3938 | `sobol_regen_013` mean 1.1270 |

Stratum summary:

| stratum | BO scored rows | BO mean | Sobol scored rows | Sobol mean | local winner |
|---|---:|---:|---:|---:|---|
| fireworks | 3 | -3.9426 | 6 | -3.3241 | regenerated Sobol |
| jellyfish | 6 | 1.6745 | 6 | 1.0558 | BO |

## Claim Impact

This run advances the foundation, but it should not be used as a broad BO win.

Reviewer-safe statements:

- The regenerated-control protocol is now executable end to end under a
  complete-candidate visual-first filter.
- The small two-stratum panel preserved matched BO/control coverage after visual
  screening.
- The result is mixed by stratum: BO wins jellyfish; regenerated Sobol wins
  fireworks.
- Pooled BO is higher than pooled regenerated Sobol in this retained panel, but
  that pooled contrast is dominated by the two prompt strata and by the withheld
  BO fireworks candidate.

Do not claim:

- BO-generated videos are more memorable to humans.
- BO broadly beats random/Sobol controls across prompts.
- The saved collaborator scores are stable point estimates.

The defensible next step is to either broaden the regenerated-control panel
across more prompt/seed strata and strategy baselines, or freeze only the
visually retained matched candidates for a small human panel with explicit
compute-proxy caveats.
