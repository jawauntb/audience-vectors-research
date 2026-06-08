# Visual-First Retention Replay Result

Last updated: 2026-06-07.

This note records the first BO/Sobol replay run where the visual artifact gate
was part of the scoring protocol rather than only a blocker. The run generated a
small replacement pool, applied the visual gate, retained only candidates whose
complete three-replicate set passed, and scored only those retained rows.

Raw reports and generated MP4s are local ignored artifacts and are intentionally
not committed.

## Protocol

The replay script now supports:

- `--visual-first-retention passing-videos`: upload/score only visual-passing
  rows.
- `--visual-first-retention complete-candidates`: upload/score only candidates
  whose full replicate set passed the visual gate.

For this run we used complete-candidate retention:

```bash
BO_MEM_STEERING_ARTIFACT=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --selection seed-stratified-bo-vs-sobol \
  --stratify-by prompt \
  --max-evals 2 \
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
  --report-path data/reports/bo_visual_first_complete_candidates_max2_reps3_steps50_motion5_noise0_20260607.json \
  --output-dir data/generated/bo_visual_first_complete_candidates_max2_reps3_steps50_motion5_noise0_20260607
```

Local ignored artifacts:

- `data/reports/bo_visual_first_complete_candidates_max2_reps3_steps50_motion5_noise0_20260607.json`
- `data/generated/bo_visual_first_complete_candidates_max2_reps3_steps50_motion5_noise0_20260607/*.mp4`

## Visual-First Retention

The run selected 7 candidates and generated 21 videos:

| retained? | task | policy | stratum | scored rows | visual result |
|---|---|---|---|---:|---|
| yes | `bo06_cand01` | BO | fireworks | 3/3 | all passed |
| no | `bo09_cand01` | BO | fireworks | 0/3 | replicate 1 failed tail sharpness |
| no | `sobol_007` | Sobol | fireworks | 0/3 | replicate 2 failed tail sharpness |
| no | `sobol_008` | Sobol | fireworks | 0/3 | replicate 2 failed tail sharpness |
| yes | `bo07_cand01` | BO | jellyfish | 3/3 | all passed |
| yes | `bo04_cand01` | BO | jellyfish | 3/3 | all passed |
| yes | `sobol_005` | Sobol | jellyfish | 3/3 | all passed |

Complete-candidate retention kept 4/7 candidates and 12/21 generated rows for
TRIBE scoring. The max-3 dry-run preview selected 9 candidates / 27 videos but
did not add any additional fireworks Sobol candidate, so the saved table does
not contain a third fireworks Sobol replacement to rescue that stratum.

## TRIBE Scores After Retention

| rank | task | policy | stratum | scored | replay mean | replay std | original TRIBE |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `bo04_cand01` | BO | jellyfish | 3/3 | 2.1171 | 0.3615 | 5.3993 |
| 2 | `sobol_005` | Sobol | jellyfish | 3/3 | 2.0256 | 0.6227 | 3.5215 |
| 3 | `bo07_cand01` | BO | jellyfish | 3/3 | 1.2318 | 0.1038 | 6.1509 |
| 4 | `bo06_cand01` | BO | fireworks | 3/3 | -3.9426 | 0.4228 | -0.3899 |

Policy summaries over retained scored rows:

| policy | retained candidates | scored rows | mean candidate replay score | best retained candidate |
|---|---:|---:|---:|---|
| BO | 3 | 9 | -0.1979 | `bo04_cand01` |
| Sobol | 1 | 3 | 2.0256 | `sobol_005` |

Stratum-level interpretation:

- Fireworks: BO retained one candidate (`bo06_cand01`), but both saved-table
  Sobol candidates (`sobol_007`, `sobol_008`) were withheld. There is no
  retained matched BO/Sobol comparison for fireworks.
- Jellyfish: BO retained two candidates and Sobol retained one. In this retained
  subset, Sobol `sobol_005` mean 2.0256 beats BO `bo07_cand01` mean 1.2318 but
  is narrowly below BO `bo04_cand01` mean 2.1171.

## Claim Impact

The protocol now records visual failures as first-class provenance and prevents
failed generated videos from entering TRIBE scoring. That is a real improvement
over both "score everything" and "abort everything" modes.

The scientific result remains conservative. The saved-table candidate pool still
does not support a complete matched visual-gated BO/Sobol claim because no
fireworks Sobol candidate survived complete-candidate retention. The next
defensible step is not a human panel from this saved pool; it is regenerated
matched controls with visual screening before TRIBE scoring.

Reviewer-safe wording:

```text
We added a visual-first retention protocol that generates a matched candidate
pool, applies the automated visual artifact gate, and scores only candidates
whose full replicate set passes. In a saved-table max-2 BO/Sobol pool, 21/21
videos generated, 3/21 failed the visual gate, and complete-candidate retention
kept 4/7 candidates for 12 TRIBE scores. Because both fireworks Sobol candidates
were withheld, the saved-table pool still cannot support a complete matched
visual-gated BO/Sobol claim.
```
