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
3. attaches start/mid/end-frame visual artifact-gate metrics;
4. uploads generated MP4s to `bmd-videos-v1`;
5. scores or preflights with the deployed `TribeV2Predictor`;
6. writes `data/reports/bo_modal_replay.json`.

Visual gate options:

- `--skip-visual-gate`: omit generated-video artifact metrics.
- `--fail-on-visual-artifacts`: write the report and exit nonzero if any
  generated video fails the visual gate; upload and TRIBE scoring are skipped
  when the gate fails.
- `--visual-gate-samples`: number of evenly spaced frames sampled per video.

SVD quality controls:

- `--svd-num-frames`: number of SVD-XT frames to generate.
- `--svd-motion-bucket-id`: SVD-XT motion bucket; lower values preserve the
  seed image more strongly.
- `--svd-noise-aug-strength`: seed-image noise augmentation strength.
- `--svd-fps`: MP4 encoding FPS.

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

For an equal-budget BO/Sobol comparison, use `--selection top-bo-vs-top-sobol`.
In that mode, `--max-evals` applies per policy group, so `--max-evals 5
--replicates 3` expands to 5 BO candidates, 5 Sobol candidates, and 30 total
replay jobs. Reports include `policy_group_summary` for the group-level
comparison.

For the stricter seed-coverage check, use
`--selection seed-stratified-bo-vs-sobol`. In that mode, the script only selects
strata where both BO and Sobol have candidates, applies `--max-evals` per policy
inside each matched stratum, and writes `stratum_policy_summary` so the
policy-by-seed comparison is explicit. The default `--stratify-by prompt`
groups by repeated seed-image content; use `--stratify-by seed_idx` when the raw
optimizer slot is the desired control.

The runnable manifest is
`seed_stratified_tournament_manifest.md`. It records the selected saved-table
strata, required local artifacts, preflight command, Modal run command, and
acceptance readout.

The completed result note is
`seed_stratified_tournament_result_20260605.md`. It records a 12/12 generated
and TRIBE-scored replay, a mixed BO/Sobol stratum outcome, and a failed visual
artifact gate for the panel.

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
  --report-path data/reports/bo_modal_replay_seed_stratified.json \
  --output-dir data/generated/bo_modal_replay_seed_stratified
```

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

### 2026-06-05 Equal-Budget BO vs Sobol Panel Result

The next panel compared the top five BO candidates against the top five Sobol
candidates from the same saved 32-trial table, with 3 stochastic replay seeds
per candidate, 4 SVD inference steps, direct-bytes TRIBE input, and a 300 second
TRIBE timeout. All 30/30 full TRIBE scores completed.

SVD generation completed with min 44.8s, max 69.2s, and mean 60.6s per clip.
TRIBE full scoring completed with min 19.0s, max 114.1s, and mean 42.8s per
clip.

Group-level result:

| policy | candidates | scored replays | original mean | replay mean | replay std | best replay candidate |
|---|---:|---:|---:|---:|---:|---|
| BO | 5 | 15/15 | 5.0525 | 1.3281 | 0.8399 | `bo10_cand01` mean 1.8148 |
| Sobol | 5 | 15/15 | 0.6798 | -1.5901 | 2.6747 | `sobol_005` mean 1.4614 |

Candidate-level replay ranking:

| replay rank | policy | task | seed image | replay mean | replay std | replay range | original score |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | BO | `bo10_cand01` | `fresh24_blue_jellyfish` | 1.8148 | 0.3930 | 1.5706..2.2681 | 4.1715 |
| 2 | BO | `bo02_cand01` | `fresh24_blue_jellyfish` | 1.6081 | 0.3863 | 1.2632..2.0254 | 4.8678 |
| 3 | Sobol | `sobol_005` | `fresh24_blue_jellyfish` | 1.4614 | 0.3173 | 1.2154..1.8194 | 3.5215 |
| 4 | BO | `bo04_cand01` | `fresh24_blue_jellyfish` | 1.4415 | 0.2722 | 1.2830..1.7558 | 5.3993 |
| 5 | BO | `bo07_cand01` | `fresh24_blue_jellyfish` | 1.3257 | 0.6996 | 0.5191..1.7686 | 6.1509 |
| 6 | BO | `bo02_cand00` | `fresh24_blue_jellyfish` | 0.4505 | 1.5569 | -1.3372..1.5089 | 4.6728 |
| 7 | Sobol | `sobol_007` | `fresh24_fireworks` | 0.0621 | 0.4795 | -0.4728..0.4536 | -0.7994 |
| 8 | Sobol | `sobol_000` | `fresh24_concert_stage` | -1.4724 | 0.2648 | -1.7751..-1.2838 | 2.2894 |
| 9 | Sobol | `sobol_003` | `fresh24_concert_stage` | -2.6244 | 1.2964 | -3.9476..-1.3565 | -0.9350 |
| 10 | Sobol | `sobol_002` | `fresh24_concert_stage` | -5.3771 | 2.6070 | -7.6397..-2.5262 | -0.6776 |

Interpretation: under this equal-count replay budget, BO retains a higher
replicated mean than the top Sobol/random initialization points. The strong
caveat is seed coverage. The BO top five are all `fresh24_blue_jellyfish`, while
the Sobol top five include one jellyfish point plus concert/fireworks points.
The fair claim is therefore not "BO robustly beats random across prompts"; it
is "BO beats the saved Sobol top-5 under replicated replay, but the result is
still entangled with a jellyfish seed pocket." The next baseline should be
seed-stratified or regenerated so BO, random/Sobol, and best-of-N compare under
matched seed-image coverage.

### 2026-06-05 Seed-Stratified BO vs Sobol Panel Result

The stricter saved-table panel compared BO and Sobol inside matched prompt
strata, with 3 stochastic replay seeds per candidate, 4 SVD inference steps,
direct-bytes TRIBE input, and a 300 second TRIBE timeout. After SVD cache
warmup, all 12/12 generated videos completed full TRIBE scoring.

Matched-stratum result:

| stratum | BO task | BO replay mean | Sobol task | Sobol replay mean | winner |
|---|---|---:|---|---:|---|
| fireworks | `bo06_cand01` | 0.3778 | `sobol_007` | 0.1514 | BO |
| jellyfish | `bo07_cand01` | 1.4744 | `sobol_005` | 1.5559 | Sobol |

Pooled means are close: BO `0.9261`, Sobol `0.8536`. The panel is therefore
mixed, not a broad BO win. Visual inspection also fails the artifact gate:
fireworks clips collapse into dark blur after the first frame, and jellyfish
clips collapse into smooth blue/white gradients with weak subject persistence.
This run should be used to narrow the BO claim, not strengthen it.

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
- Compare against random/Sobol/best-of-N under equal evaluation count. Current
  status: BO vs saved Sobol top-5 panel passed, with seed-coverage caveat.
- Run a seed-stratified BO/Sobol tournament panel. Current status: tooling
  and run manifest added; 2026-06-05 saved-table BO/Sobol replay completed
  12/12 scores but produced a mixed stratum result.
- Inspect top videos for prompt drift and artifacts. Current status: failed for
  the 2026-06-05 seed-stratified panel because mid/end frames collapse into
  low-content blur or gradients. The automated visual-gated smoke also failed
  4/4 generated clips and correctly skipped upload/TRIBE scoring. Tuned SVD
  settings (`50` steps, motion bucket `5`, noise `0`) passed 4/4 clips and
  completed 4/4 TRIBE scores in a one-replicate panel.
- Report wall-clock and average minutes per evaluation.
- Treat runtime as a limitation: BO is sample-efficient, not yet wall-clock
  efficient.
