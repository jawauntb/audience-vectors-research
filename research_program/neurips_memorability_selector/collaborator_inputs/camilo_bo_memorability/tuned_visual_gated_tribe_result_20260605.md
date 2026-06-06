# Tuned Visual-Gated BO/Sobol TRIBE Result

Last updated: 2026-06-05.

This note records the first seed-stratified BO/Sobol replay panel that both
passed the automated visual artifact gate and completed full TRIBE scoring.
Raw reports and generated MP4s are local ignored artifacts and are intentionally
not committed.

## Tuning Path

The original visual-gated smoke used 4 SVD inference steps and failed 4/4
generated videos. Increasing SVD inference steps improved the gate result but
did not fully solve it:

| setting | visual failures | observation |
|---|---:|---|
| 4 steps, default motion/noise | 4/4 | all clips failed visual gate |
| 25 steps, default motion/noise | 2/4 | improved, but `sobol_007` and `bo07_cand01` failed tail sharpness |
| 50 steps, default motion/noise | 1/4 | only `sobol_007` failed tail sharpness |
| 50 steps, motion bucket 1, noise 0 | 1/4 | fixed `sobol_007`, but `bo06_cand01` failed |
| 50 steps, motion bucket 5, noise 0 | 0/4 | all four clips passed visual gate |

The accepted visual-gate setting is therefore:

- `--num-inference-steps 50`
- `--svd-motion-bucket-id 5`
- `--svd-noise-aug-strength 0`

## Scored Run

```bash
BO_MEM_STEERING_ARTIFACT=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --selection seed-stratified-bo-vs-sobol \
  --stratify-by prompt \
  --max-evals 1 \
  --replicates 1 \
  --num-inference-steps 50 \
  --svd-motion-bucket-id 5 \
  --svd-noise-aug-strength 0 \
  --generation-timeout 1800 \
  --tribe-mode full \
  --tribe-input bytes \
  --tribe-timeout 300 \
  --tribe-concurrency 3 \
  --fail-on-visual-artifacts \
  --report-path data/reports/bo_visual_gated_tribe_steps50_motion5_noise0_20260605.json \
  --output-dir data/generated/bo_visual_gated_tribe_steps50_motion5_noise0_20260605
```

Local ignored artifacts:

- `data/reports/bo_visual_gated_tribe_steps50_motion5_noise0_20260605.json`
- `data/generated/bo_visual_gated_tribe_steps50_motion5_noise0_20260605/*.mp4`

The run generated 4/4 videos, passed 4/4 through the visual artifact gate,
uploaded the clips, and completed 4/4 full TRIBE scores.

## Visual Gate

| task | policy | stratum | tail sharpness ratio | tail contrast ratio | gate |
|---|---|---|---:|---:|---|
| `bo06_cand01` | BO | fireworks | 0.5383 | 0.8076 | pass |
| `sobol_007` | Sobol | fireworks | 0.9690 | 0.9608 | pass |
| `bo07_cand01` | BO | jellyfish | 0.8394 | 0.9724 | pass |
| `sobol_005` | Sobol | jellyfish | 0.7006 | 0.9683 | pass |

## TRIBE Result

| stratum | policy | task | replay TRIBE | original TRIBE | delta |
|---|---|---|---:|---:|---:|
| fireworks | BO | `bo06_cand01` | -4.3017 | -0.3899 | -3.9118 |
| fireworks | Sobol | `sobol_007` | -3.8594 | -0.7994 | -3.0600 |
| jellyfish | BO | `bo07_cand01` | 1.3402 | 6.1509 | -4.8106 |
| jellyfish | Sobol | `sobol_005` | 2.4330 | 3.5215 | -1.0885 |

Pooled one-replicate means:

| policy | scored | replay mean | best task |
|---|---:|---:|---|
| BO | 2/2 | -1.4807 | `bo07_cand01` |
| Sobol | 2/2 | -0.7132 | `sobol_005` |

Sobol wins both matched strata in this visually accepted one-replicate panel.
This does not establish a robust Sobol advantage because there is only one
replicate per candidate, but it does further block any broad claim that the
saved-table BO candidates robustly outperform Sobol controls.

## Claim Impact

This run validates the tuned visual-gated replay path and gives the first
human-ready-enough compute panel by the current automated gate. The scientific
claim remains conservative: under a visually accepted one-replicate replay,
the saved BO candidates do not beat the matched saved Sobol candidates.

Reviewer-safe wording:

```text
After tuning SVD generation to 50 inference steps with low motion and no seed
noise, a one-replicate seed-stratified BO/Sobol panel passed the automated
visual artifact gate for 4/4 generated clips and completed 4/4 TRIBE scores.
Sobol beat BO in both matched strata in this small panel, so the saved-table BO
result remains insufficient for a broad BO-over-control claim.
```
