# Max-3 Regenerated Visual Controls Result - 2026-06-08

Last updated: 2026-06-08.

This note records the balanced max-3 regenerated-control stress test proposed
in `next_research_foundation_manifest_20260608.md`.

## Protocol

Selection:

- saved BO anchors: `--selection top-bo-per-stratum`
- strata: `--stratify-by prompt`
- saved BO budget: `--max-evals 3` per prompt stratum
- regenerated controls: `--regenerated-sobol-controls-per-stratum 3`
- regenerated control pool: `--regenerated-sobol-pool-size 256`
- regenerated control start index: `--regenerated-sobol-start-index 128`
- replicates: `3` per selected candidate
- visual policy: `--visual-first-retention complete-candidates`
- SVD settings: 50 inference steps, motion bucket 5, noise augmentation 0
- TRIBE mode: full, direct bytes input, 300 second timeout, concurrency 3

Local ignored outputs:

- report:
  `data/reports/bo_regenerated_visual_controls_max3_regensobol3_reps3_steps50_motion5_noise0_start128_20260608.json`
- videos:
  `data/generated/bo_regenerated_visual_controls_max3_regensobol3_reps3_steps50_motion5_noise0_start128_20260608/`

## Commands

Preflight with artifact requirement:

```bash
BO_MEM_STEERING_ARTIFACT=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --dry-run \
  --require-artifacts \
  --selection top-bo-per-stratum \
  --stratify-by prompt \
  --max-evals 3 \
  --regenerated-sobol-controls-per-stratum 3 \
  --regenerated-sobol-pool-size 256 \
  --regenerated-sobol-start-index 128 \
  --replicates 3 \
  --report-path /tmp/regenerated_controls_max3_start128_preflight_required.json
```

Full run:

```bash
BO_MEM_STEERING_ARTIFACT=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --selection top-bo-per-stratum \
  --stratify-by prompt \
  --max-evals 3 \
  --regenerated-sobol-controls-per-stratum 3 \
  --regenerated-sobol-pool-size 256 \
  --regenerated-sobol-start-index 128 \
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
  --report-path data/reports/bo_regenerated_visual_controls_max3_regensobol3_reps3_steps50_motion5_noise0_start128_20260608.json \
  --output-dir data/generated/bo_regenerated_visual_controls_max3_regensobol3_reps3_steps50_motion5_noise0_start128_20260608
```

## Selection

The run selected 6 saved BO anchors and appended 6 deterministic, unscored
Sobol controls from sequence index 128 onward. No target prompt stratum was
missing regenerated controls.

| task | policy | prompt stratum | alpha | guidance | seed_idx | noise_seed |
|---|---|---|---:|---:|---:|---:|
| `bo06_cand01` | BO | fireworks | -4.1262 | 7.8464 | 10 | 601 |
| `bo09_cand01` | BO | fireworks | 7.0962 | 2.4844 | 10 | 901 |
| `bo03_cand01` | BO | fireworks | 7.0735 | 3.6069 | 10 | 301 |
| `sobol_regen_133` | regenerated Sobol | fireworks | -3.7734 | 5.6754 | 10 | 133 |
| `sobol_regen_136` | regenerated Sobol | fireworks | 6.9236 | 2.0850 | 5 | 136 |
| `sobol_regen_138` | regenerated Sobol | fireworks | -2.7452 | 4.3513 | 0 | 138 |
| `bo07_cand01` | BO | jellyfish | 7.0735 | 3.2311 | 13 | 701 |
| `bo04_cand01` | BO | jellyfish | -3.9674 | 7.7753 | 13 | 401 |
| `bo02_cand01` | BO | jellyfish | -3.9785 | 7.9710 | 13 | 201 |
| `sobol_regen_128` | regenerated Sobol | jellyfish | 9.7878 | 3.7529 | 8 | 128 |
| `sobol_regen_135` | regenerated Sobol | jellyfish | 5.9328 | 7.9415 | 13 | 135 |
| `sobol_regen_142` | regenerated Sobol | jellyfish | -6.8373 | 2.8278 | 8 | 142 |

## Visual-First Retention

The full run generated all 36 requested MP4s. Generation time ranged from 64.1
to 168.8 seconds, with mean 113.8 seconds per clip.

The automated visual artifact gate failed 2/36 videos:

| failed row | task | flags | tail sharpness ratio | tail contrast ratio | min tail contrast |
|---|---|---|---:|---:|---:|
| `bo_replay_01_bo09_cand01_rep01` | `bo09_cand01` | `tail_sharpness_collapse` | 0.3066 | 0.6966 | 0.1477 |
| `bo_replay_02_bo03_cand01_rep02` | `bo03_cand01` | `tail_sharpness_collapse` | 0.1178 | 0.7041 | 0.1452 |

Because the run used complete-candidate visual-first retention, all rows for
`bo09_cand01` and `bo03_cand01` were withheld before upload/scoring. The
retained panel contains 30/36 rows and 10/12 candidates.

| task | policy | stratum | retained | scored rows | visual note |
|---|---|---|---|---:|---|
| `bo06_cand01` | BO | fireworks | yes | 3/3 | all replicates passed |
| `bo09_cand01` | BO | fireworks | no | 0/3 | replicate 1 failed tail sharpness |
| `bo03_cand01` | BO | fireworks | no | 0/3 | replicate 2 failed tail sharpness |
| `sobol_regen_133` | regenerated Sobol | fireworks | yes | 3/3 | all replicates passed |
| `sobol_regen_136` | regenerated Sobol | fireworks | yes | 3/3 | all replicates passed |
| `sobol_regen_138` | regenerated Sobol | fireworks | yes | 3/3 | all replicates passed |
| `bo07_cand01` | BO | jellyfish | yes | 3/3 | all replicates passed |
| `bo04_cand01` | BO | jellyfish | yes | 3/3 | all replicates passed |
| `bo02_cand01` | BO | jellyfish | yes | 3/3 | all replicates passed |
| `sobol_regen_128` | regenerated Sobol | jellyfish | yes | 3/3 | all replicates passed |
| `sobol_regen_135` | regenerated Sobol | jellyfish | yes | 3/3 | all replicates passed |
| `sobol_regen_142` | regenerated Sobol | jellyfish | yes | 3/3 | all replicates passed |

The structural gate passed: after complete-candidate retention, at least one BO
candidate and one regenerated Sobol control remained in each prompt stratum.

## TRIBE Results

All retained rows completed full TRIBE scoring: 30/30. TRIBE wall time ranged
from 18.1 to 190.8 seconds, with mean 38.0 seconds per retained row.

Candidate ranking by mean replay TRIBE score:

| rank | task | policy | scored | mean | std | sem | original score |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `bo04_cand01` | BO | 3/3 | 2.1171 | 0.3615 | 0.2087 | 5.3993 |
| 2 | `sobol_regen_135` | regenerated Sobol | 3/3 | 1.9839 | 0.1316 | 0.0760 | n/a |
| 3 | `bo02_cand01` | BO | 3/3 | 1.8839 | 0.3579 | 0.2066 | 4.8678 |
| 4 | `bo07_cand01` | BO | 3/3 | 1.2318 | 0.1038 | 0.0599 | 6.1509 |
| 5 | `sobol_regen_128` | regenerated Sobol | 3/3 | 0.8472 | 0.6000 | 0.3464 | n/a |
| 6 | `sobol_regen_142` | regenerated Sobol | 3/3 | 0.3413 | 0.1946 | 0.1124 | n/a |
| 7 | `bo06_cand01` | BO | 3/3 | -3.9426 | 0.4228 | 0.2441 | -0.3899 |
| 8 | `sobol_regen_138` | regenerated Sobol | 3/3 | -4.3249 | 0.0811 | 0.0468 | n/a |
| 9 | `sobol_regen_133` | regenerated Sobol | 3/3 | -4.6413 | 0.6316 | 0.3646 | n/a |
| 10 | `sobol_regen_136` | regenerated Sobol | 3/3 | -6.3424 | 0.8138 | 0.4698 | n/a |
| 11 | `bo09_cand01` | BO | 0/3 | n/a | n/a | n/a | -0.9172 |
| 12 | `bo03_cand01` | BO | 0/3 | n/a | n/a | n/a | -2.9498 |

Policy summary after retention:

| policy | candidates | requested rows | scored rows | pooled mean | pooled std | best retained candidate |
|---|---:|---:|---:|---:|---:|---|
| BO | 6 | 18 | 12 | 0.3226 | 2.6099 | `bo04_cand01` mean 2.1171 |
| regenerated Sobol | 6 | 18 | 18 | -2.0227 | 3.2993 | `sobol_regen_135` mean 1.9839 |

Stratum summary:

| stratum | BO retained candidates | BO mean | Sobol retained candidates | Sobol mean | local winner |
|---|---:|---:|---:|---:|---|
| fireworks | 1/3 | -3.9426 | 3/3 | -5.1029 | BO, but both policies are poor |
| jellyfish | 3/3 | 1.7443 | 3/3 | 1.0575 | BO |

## Claim Impact

This run produces a real compute-side finding, but it is not a broad BO win.

Reviewer-safe statements:

- The max-3 regenerated-control stress test passed the structural gate: all
  videos generated, visual-first retention preserved matched BO/control
  coverage in both selected prompt strata, and all retained rows completed full
  TRIBE scoring.
- The jellyfish stratum is a stable positive pocket for saved BO replay:
  three retained BO candidates scored positive with mean 1.7443, while three
  regenerated Sobol controls averaged 1.0575.
- The fireworks stratum is visually brittle and low scoring under the current
  tuned SVD/TRIBE replay: two of three BO fireworks candidates were withheld by
  the visual gate, and the only retained BO fireworks candidate scored -3.9426.
- BO numerically beats regenerated Sobol in both retained strata, but the
  fireworks contrast is a weak/negative-pocket contrast rather than evidence
  that BO found good fireworks videos.
- The saved collaborator table still cannot support broad prompt-level
  BO/control claims because BO coverage remains limited to fireworks and
  jellyfish prompt strata.

Do not claim:

- BO-generated videos are more memorable to humans.
- BO broadly beats random/Sobol controls across prompts.
- The fireworks prompt is solved by BO.
- Pooled BO > Sobol in this run establishes a general strategy advantage.

New working hypothesis:

```text
The current BO replay evidence is best explained by prompt-pocket behavior:
the saved BO run reliably exploits a jellyfish replay pocket, while fireworks
remains a visually brittle low-scoring stratum for both BO and regenerated
controls.
```

Next scientific step: run a new prompt-broadened BO/search panel with at least
four prompt strata and the same visual-first complete-candidate retention gate.
