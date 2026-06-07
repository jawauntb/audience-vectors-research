# Replicated Visual-Gated Replay Blocked Result

Last updated: 2026-06-07.

This note records the follow-up to the tuned one-replicate visual-gated
BO/Sobol panel. The goal was to replay the same matched candidates with three
stochastic noise seeds each, keep the automated visual artifact gate as a
blocking verifier, and run full TRIBE scoring only if all generated clips passed.

Raw reports and generated MP4s are local ignored artifacts and are intentionally
not committed.

## Primary Run

```bash
BO_MEM_STEERING_ARTIFACT=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --selection seed-stratified-bo-vs-sobol \
  --stratify-by prompt \
  --max-evals 1 \
  --replicates 3 \
  --num-inference-steps 50 \
  --svd-motion-bucket-id 5 \
  --svd-noise-aug-strength 0 \
  --generation-timeout 1800 \
  --tribe-mode full \
  --tribe-input bytes \
  --tribe-timeout 300 \
  --tribe-concurrency 3 \
  --fail-on-visual-artifacts \
  --report-path data/reports/bo_visual_gated_tribe_replicates3_steps50_motion5_noise0_20260607.json \
  --output-dir data/generated/bo_visual_gated_tribe_replicates3_steps50_motion5_noise0_20260607
```

Local ignored artifacts:

- `data/reports/bo_visual_gated_tribe_replicates3_steps50_motion5_noise0_20260607.json`
- `data/generated/bo_visual_gated_tribe_replicates3_steps50_motion5_noise0_20260607/*.mp4`

The run generated 12/12 videos. The visual gate failed 1/12 and, because
`--fail-on-visual-artifacts` was enabled, upload and TRIBE scoring were skipped.

## Primary Visual Gate Result

| task | policy | replicate | seed | tail sharpness ratio | tail contrast ratio | gate |
|---|---|---:|---:|---:|---:|---|
| `bo06_cand01` | BO | 0 | 601 | 0.5383 | 0.8076 | pass |
| `bo06_cand01` | BO | 1 | 10601 | 0.7222 | 0.8562 | pass |
| `bo06_cand01` | BO | 2 | 20601 | 0.8492 | 0.8917 | pass |
| `sobol_007` | Sobol | 0 | 7 | 0.9690 | 0.9608 | pass |
| `sobol_007` | Sobol | 1 | 10007 | 0.8363 | 0.9819 | pass |
| `sobol_007` | Sobol | 2 | 20007 | 0.1200 | 0.5942 | fail: tail sharpness collapse |
| `bo07_cand01` | BO | 0 | 701 | 0.8394 | 0.9724 | pass |
| `bo07_cand01` | BO | 1 | 10701 | 0.9190 | 0.9849 | pass |
| `bo07_cand01` | BO | 2 | 20701 | 0.9663 | 0.9831 | pass |
| `sobol_005` | Sobol | 0 | 5 | 0.7006 | 0.9683 | pass |
| `sobol_005` | Sobol | 1 | 10005 | 0.6973 | 0.9696 | pass |
| `sobol_005` | Sobol | 2 | 20005 | 0.8778 | 0.9701 | pass |

## Tuning Probes

The failure was not removed by nearby global SVD settings:

| probe | visual failures | failed clip(s) |
|---|---:|---|
| Full panel, 50 steps, motion bucket 5, noise 0 | 1/12 | `sobol_007` replicate 2, seed `20007`, sharpness 0.1200 |
| Targeted `sobol_007`, 75 steps, motion bucket 5, noise 0 | 1/3 | `sobol_007` replicate 2, seed `20007`, sharpness 0.1192 |
| Full panel, 50 steps, motion bucket 3, noise 0 | 1/12 | `sobol_007` replicate 2, seed `20007`, sharpness 0.1292 |
| Full panel, 50 steps, motion bucket 2, noise 0 | 2/12 | `bo06_cand01` replicate 0, seed `601`, sharpness 0.3264; `sobol_007` replicate 2, seed `20007`, sharpness 0.1386 |
| Targeted `sobol_007`, 50 steps, motion bucket 5, noise 0.005 | 1/3 | `sobol_007` replicate 2, seed `20007`, sharpness 0.1047 |

The repeated failure is therefore best treated as a stochastic generation
instability for this candidate/seed combination, not as a solved tuning issue.

## Claim Impact

The one-replicate visual-gated result remains useful as a smoke test, but it is
not enough to proceed to a human panel. The replicated gate blocks the current
matched panel because one Sobol replicate repeatedly collapses before scoring.

Reviewer-safe wording:

```text
The tuned SVD settings that passed a one-replicate visual gate did not survive
a three-replicate visual-gated panel. All 12 videos generated, but one Sobol
replicate repeatedly failed the blocking visual artifact gate with tail
sharpness collapse, so upload and TRIBE scoring were skipped. The next step is
visual-first candidate replacement or resampling under a matched budget, not a
human panel.
```

## Next Step

Do not tune per replicate or silently change failed stochastic seeds after
observing a collapse. The next defensible protocol should make visual screening
part of selection up front, for example:

- generate a small matched replacement budget per policy/stratum,
- apply the visual artifact gate before TRIBE scoring,
- retain only candidates or replicate sets that pass the gate under the declared
  budget, and
- record rejected visual failures as first-class provenance artifacts.
