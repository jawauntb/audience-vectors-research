# Seed-Stratified BO/Sobol Tournament Result

Last updated: 2026-06-05.

This note records the completed compute-side replay from
`seed_stratified_tournament_manifest.md`. Raw reports and generated MP4s are
local artifacts and are intentionally not committed.

## Run

Local artifacts were fetched from the upstream collaborator repository and kept
under ignored `data/artifacts/camilo_bo_memorability/`:

- `tribe_clip_adapter.pt`
- `v_mem.npz`

The artifact-required preflight passed and selected four saved-table candidates:
BO/Sobol for the fireworks stratum and BO/Sobol for the jellyfish stratum. The
full warm run generated and scored 12/12 videos:

```bash
BO_MEM_STEERING_ARTIFACT=data/artifacts/camilo_bo_memorability/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=data/artifacts/camilo_bo_memorability/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --selection seed-stratified-bo-vs-sobol \
  --stratify-by prompt \
  --max-evals 1 \
  --replicates 3 \
  --num-inference-steps 4 \
  --generation-timeout 1800 \
  --tribe-mode full \
  --tribe-input bytes \
  --tribe-timeout 300 \
  --tribe-concurrency 3 \
  --report-path data/reports/bo_modal_replay_seed_stratified_20260605.json \
  --output-dir data/generated/bo_modal_replay_seed_stratified_20260605
```

An earlier one-replicate smoke populated the SVD cache and produced 3/4 videos;
one first-call SVD job hit `ConnectionError('Deadline exceeded')`. The warm
full run completed without generation or TRIBE scoring failures.

## TRIBE Replay Result

The matched result is mixed rather than a BO win across strata.

| stratum | policy | task | scored | replay mean | replay std | replay sem | replay range | original TRIBE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| fireworks | BO | `bo06_cand01` | 3/3 | 0.3778 | 0.7869 | 0.4543 | -0.5121..0.9817 | -0.3899 |
| fireworks | Sobol | `sobol_007` | 3/3 | 0.1514 | 0.3662 | 0.2114 | -0.2507..0.4656 | -0.7994 |
| jellyfish | BO | `bo07_cand01` | 3/3 | 1.4744 | 0.6920 | 0.3995 | 0.6754..1.8740 | 6.1509 |
| jellyfish | Sobol | `sobol_005` | 3/3 | 1.5559 | 0.3229 | 0.1865 | 1.2731..1.9078 | 3.5215 |

Pooled across the two matched strata, BO is only slightly higher than Sobol:

| policy | candidates | scored replays | replay mean | replay std | best candidate |
|---|---:|---:|---:|---:|---|
| BO | 2 | 6/6 | 0.9261 | 0.8944 | `bo07_cand01` mean 1.4744 |
| Sobol | 2 | 6/6 | 0.8536 | 0.8290 | `sobol_005` mean 1.5559 |

Candidate replay rank is `sobol_005`, `bo07_cand01`, `bo06_cand01`,
`sobol_007`. The panel therefore does not support stronger language that BO
robustly beats Sobol across matched seed-image/prompt strata.

## Visual Artifact Gate

A contact-sheet inspection of start/mid/end frames from all 12 generated clips
fails the visual artifact gate for this panel.

- Fireworks clips start with a recognizable firework but most mid/end frames
  collapse into dark blue, low-content blur with horizon streaking.
- Jellyfish clips start with recognizable translucent forms but mid/end frames
  collapse into smooth blue/white gradients with weak subject persistence.
- The clips are useful for proxy stress testing, but they are not suitable as a
  human memorability panel without improving generation settings and artifact
  filtering.

## Claim Impact

This run resolves the immediate seed-stratified BO/Sobol gate as mixed and
visually weak. It should be cited as evidence that the saved-table BO result is
not ready to become a broad control claim.

Reviewer-safe wording:

```text
A seed-stratified replay across two matched BO/Sobol prompt strata completed
12/12 generated and TRIBE-scored videos. BO won the fireworks stratum, Sobol won
the jellyfish stratum, and pooled means were close. Visual inspection found
substantial mid/end-frame collapse, so the result remains a proxy stress test
rather than human-ready generated-video evidence.
```

Next compute move: either regenerate a matched baseline across all seed images
with improved generation settings, or add an automated visual-quality/prompt
preservation gate before spending on more TRIBE replay.
