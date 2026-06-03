# Collaborator Intake: BO Memorability

**Status:** code/provenance intake, not main-paper evidence yet.

This folder imports the collaborator BO memorability prototype from
`https://github.com/cerqarth/bo_memorability` for reproducibility review and
Modal adaptation.

## What Is Included

- `original/src/bo_mem/`: original BO package code.
- `original/scripts/`: original local GPU runner/report scripts.
- `original/seeds/`: small seed-image/prompt assets available in the source repo.
- `raw_results/gpu_run_all_results.json`: original 2-objective run table.
- `raw_results/gpu_run_3obj_all_results.json`: original 3-objective run table.
- `figures/`: original run figures from the source repo.

## What Is Deliberately Excluded

- Generated MP4s.
- Model weights and adapter checkpoints.
- `bo_state.pt` checkpoint binaries.
- Report PDFs.
- Secrets or Hugging Face tokens.

The source repo includes `artifacts/v_mem.npz` and
`artifacts/tribe_clip_adapter.pt`; those are intentionally not committed here.
Pass local copies to the replay script when running reproduction.

## Reviewer-Safe Framing

This is compute/control evidence, not human memorability evidence. The current
claim should be:

```text
Multi-objective BO can be a sample-efficient search policy over generated video
candidates under TRIBE/CLIP/R3D proxy objectives.
```

Do not claim:

```text
BO-steered videos are proven more memorable to humans.
```

## Modal Replay

Dry-run the imported table and seed mapping:

```bash
uv run python scripts/modal_bo_memorability_replay.py --dry-run
```

Run a tiny Modal smoke replay, assuming local artifact copies:

```bash
BO_MEM_STEERING_ARTIFACT=/path/to/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/path/to/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --selection top-tribe \
  --max-evals 2 \
  --num-inference-steps 8
```

The script:

1. queues Modal SVD generations in parallel;
2. passes the collaborator `alpha` and `guidance` values;
3. uploads generated MP4s to `bmd-videos-v1`;
4. scores or preflights with the deployed `TribeV2Predictor`;
5. writes `data/reports/bo_modal_replay.json`.

TRIBE modes:

- `--tribe-mode full`: run full TRIBE prediction and project onto cortical
  `v_mem`.
- `--tribe-mode preflight`: validate TRIBE path/bytes handling, ffprobe
  duration, and `get_events_dataframe` without running the expensive model
  prediction path.
- `--tribe-input bytes`: send local generated MP4 bytes directly to TRIBE,
  avoiding Modal volume staleness as a confound.
- `--tribe-input volume`: score the uploaded `/bmd-videos/...` path.

### 2026-06-03 Modal Smoke Result

Dry-run table/seed mapping passed for the top two TRIBE trials. A bounded
single-trial Modal smoke generated and uploaded the top collaborator trial
(`bo07_cand01`) successfully, but full TRIBE scoring timed out:

- source objective row: `alpha=7.0735`, `guidance=3.2311`, `seed_idx=13`,
  `noise_seed=701`;
- generated `data/generated/bo_modal_replay_smoke/bo_replay_00_bo07_cand01.mp4`
  in 40.0 seconds at 1,474,551 bytes;
- uploaded to
  `/bmd-videos/generated/bo_memorability_replay/bo_replay_00_bo07_cand01.mp4`;
- bounded TRIBE replay scoring reached the video extractor but timed out after
  60 seconds, so `replay_tribe_score` is not validated yet.

Cold SVD setup was the expensive part before caching: unauthenticated Hugging
Face downloads took about five minutes for the first successful SVD cache fill.
SVD generation itself returned quickly once the cache was warm. The remaining
compute-side blocker is TRIBE video feature extraction on generated MP4s.

Follow-up replay diagnostics on the same day narrowed the blocker:

- 2-eval top-TRIBE smoke passed SVD generation, upload, and direct-bytes TRIBE
  preflight for both selected trials.
- 1-eval full-score probe with `--tribe-input bytes` still timed out after 90
  seconds, while automatic timeout preflight passed on the same MP4. This rules
  out Modal volume staleness, bad MP4 duration probing, and TRIBE event
  construction as the primary issue.
- 32-eval fixed-budget replay with `--selection first --max-evals 32
  --tribe-mode preflight --tribe-input bytes --num-inference-steps 4` generated
  and uploaded all 32 videos. All 32 passed TRIBE preflight.
- 32-eval generation timing: min 28.2s, max 44.5s, mean 37.8s per generated
  MP4. Total generated payload was 58,782,919 bytes.

The post-outage retry showed the TRIBE infrastructure is healthy again when
given a longer scoring timeout:

- 1-eval full-score probe with `--tribe-input bytes --tribe-timeout 300`
  completed with replay TRIBE score `0.6754`.
- 2-eval full-score probe completed with replay TRIBE scores `0.6754` and
  `1.3557`.
- 32/32 pre-generated replay MP4s completed full TRIBE scoring with direct
  bytes input. The replay distribution ranged from `-4.5179` to `2.4868`, with
  mean `-0.0345`.

Current interpretation: Modal can make SVD replay and fixed-budget generation
practical once the SVD cache is warm, and TRIBE scoring is usable again. The
remaining scientific issue is score stability: replay scores differ materially
from the collaborator table, so candidate ranking should be treated as
stochastic until we summarize multiple generated replicates per BO point.

## Replicated Stochastic Replay

For probabilistic video generation, a single replay score can be a lucky or
unlucky draw. Use `--replicates` to replay each selected BO point across
deterministic noise-seed offsets and report per-candidate mean/std/min/max:

```bash
BO_MEM_STEERING_ARTIFACT=/path/to/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/path/to/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --selection top-tribe \
  --max-evals 2 \
  --replicates 3 \
  --num-inference-steps 4 \
  --tribe-mode full \
  --tribe-input bytes \
  --tribe-timeout 300 \
  --tribe-concurrency 3 \
  --report-path data/reports/bo_modal_replay_replicates_top2.json \
  --output-dir data/generated/bo_modal_replay_replicates_top2
```

Replicate `0` preserves the collaborator's original `noise_seed`. Later
replicates use deterministic offsets controlled by `--replicate-seed-stride`
and `--replicate-seed-offset`. Reports include `replicate_summary`, sorted by
mean replay TRIBE score, so follow-up figures can plot score distributions or
error bars instead of relying on one point estimate.

### 2026-06-03 Top-2 Replicate Smoke Result

The top two original TRIBE candidates were replayed with 3 noise seeds each,
4 SVD inference steps, direct-bytes TRIBE input, and a 300 second TRIBE timeout.
All 6/6 full TRIBE scores completed.

Replicated replay changed the top-2 order:

- `bo04_cand01`: mean `1.5123`, std `0.3245`, sem `0.1873`, range
  `1.2857..1.8840`, original score `5.3993`.
- `bo07_cand01`: mean `1.3184`, std `0.7007`, sem `0.4046`, range
  `0.5102..1.7557`, original score `6.1509`.

Interpretation: the original top score should not be treated as a stable point
estimate yet. Replicated replay is the right next compute-side validation
because it separates robust candidates from stochastic high draws.

### 2026-06-03 Top-5 Replicate Panel Result

After merging the replicate tooling and redeploying the TRIBE Modal image, the
top five original TRIBE candidates were replayed with 3 noise seeds each,
4 SVD inference steps, direct-bytes TRIBE input, and a 300 second TRIBE timeout.
All 15/15 full TRIBE scores completed.

SVD generation remained practical: 15 generated MP4s completed with min 33.6s,
max 38.1s, and mean 35.5s per clip. TRIBE full scoring was more variable:
min 19.1s, max 242.4s, and mean 76.3s per clip.

Replicated replay substantially changed the original top-5 order:

| replay rank | task | replay mean | replay std | replay range | original score |
|---:|---|---:|---:|---:|---:|
| 1 | `bo10_cand01` | 1.8826 | 0.3552 | 1.6549..2.2920 | 4.1715 |
| 2 | `bo02_cand01` | 1.7181 | 0.3977 | 1.3791..2.1559 | 4.8678 |
| 3 | `bo04_cand01` | 1.5397 | 0.3086 | 1.3557..1.8960 | 5.3993 |
| 4 | `bo07_cand01` | 1.5224 | 0.5935 | 0.8371..1.8739 | 6.1509 |
| 5 | `bo02_cand00` | 0.6132 | 1.5041 | -1.1183..1.5963 | 4.6728 |

Important interpretation: the top original score is not the top replicated
mean, and one candidate has a large negative replicate. All five selected
points also use the same `fresh24_blue_jellyfish` seed image, so this panel is
good evidence of stochastic replay variance but not evidence of broad prompt
coverage. Next compute validation should compare replicated BO points against
random/Sobol or best-of-N under equal budget across multiple seed images.

## Validation Checklist

- Confirm exact `v_mem_CLIP` derivation from cortical `v_mem`.
- Reproduce a 2-4 eval Modal smoke with completed TRIBE replay scores. Current
  status: passed after the post-outage Modal retry.
- Reproduce the 32-eval table under a fixed generation/scoring budget. Current
  status: generation/upload/preflight pass for all 32; full TRIBE direct-bytes
  scoring passed for all 32 with longer timeout.
- Run replicated stochastic replay for the top candidates and report mean,
  standard deviation, standard error, and rank stability. Current status:
  top-2 and top-5 replicated panels passed; top-rank stability is weak.
- Compare against random/Sobol/best-of-N under equal evaluation count.
- Inspect top videos for prompt drift and artifacts.
- Report wall-clock and average minutes per evaluation.
- Treat runtime as a limitation: BO is sample-efficient, not yet wall-clock
  efficient.
