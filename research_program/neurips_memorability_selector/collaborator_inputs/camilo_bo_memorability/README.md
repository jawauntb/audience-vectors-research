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
- `--visual-first-retention`: visual-first scoring mode. Use `passing-videos`
  to upload/score only visual-passing rows, or `complete-candidates` to
  upload/score only candidates whose full replicate set passes the visual gate.
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

For a regenerated-control follow-up when the saved Sobol table is exhausted, use
`--selection top-bo-per-stratum` together with
`--regenerated-sobol-controls-per-stratum`. This selects saved BO anchors per
prompt/seed stratum and appends deterministic, unscored Sobol controls for the
same strata before generation. Reports include `regenerated_sobol_controls` so
the control sequence, prompt strata, and missing strata are auditable. The
runnable manifest is `regenerated_visual_controls_manifest_20260608.md`.
The completed regenerated-control run is
`regenerated_visual_controls_result_20260608.md`.
The next foundation manifest is
`next_research_foundation_manifest_20260608.md`.

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

### 2026-06-08 Regenerated Visual Controls Result

The regenerated-control follow-up selected 2 saved BO anchors and 2
deterministic, unscored regenerated Sobol controls in each BO-covered prompt
stratum, then replayed 3 stochastic replicates per candidate under the tuned
visual-first settings: 50 SVD steps, motion bucket 5, noise augmentation 0, and
complete-candidate visual-first retention.

The run generated 24/24 requested clips. One clip failed the automated visual
gate: `bo09_cand01` replicate 1 with `tail_sharpness_collapse`. Because the run
used complete-candidate retention, all 3 rows for `bo09_cand01` were withheld
before upload/scoring. The retained set kept 7/8 candidates and scored 21/21
retained rows with full TRIBE.

Matched BO/control coverage survived in both selected prompt strata:

| stratum | BO retained candidates | regenerated Sobol candidates | BO mean | Sobol mean | local winner |
|---|---:|---:|---:|---:|---|
| fireworks | 1 | 2 | -3.9426 | -3.3241 | regenerated Sobol |
| jellyfish | 2 | 2 | 1.6745 | 1.0558 | BO |

Pooled retained means were BO `-0.1979` and regenerated Sobol `-1.1342`, but
this is still a small two-stratum proxy panel with a mixed per-stratum result.
Use it as evidence that the regenerated-control and visual-first protocol can
run end to end, not as evidence that BO broadly beats controls or improves
human memorability.

### 2026-06-08 Next Foundation Audit

The saved 3-objective table cannot broaden the BO/control claim beyond two
prompt strata by replay alone. A `top-bo-per-stratum --max-evals 99` dry-run
finds 20 saved BO rows: 3 fireworks and 17 jellyfish. A saved-table
`seed-stratified-bo-vs-sobol --max-evals 99` dry-run also remains limited to
the same two prompt strata.

The next low-friction compute step is therefore a balanced within-table stress
test, not a broad prompt claim: 3 saved BO anchors and 3 fresh regenerated
Sobol controls per available prompt stratum, with 3 stochastic replicates each.
The preflight in `next_research_foundation_manifest_20260608.md` selects 12
candidates and expands to 36 replay jobs.

True broad prompt evidence requires a new BO/search panel over additional seed
prompts.

### 2026-06-08 Prompt and Content-Axis Follow-Ups

Three follow-up runs converted the foundation audit into a tighter regime
diagnosis.

First, the prompt-transfer stress test retargeted the top saved BO
alpha/guidance recipes across all five locally image-backed prompt slots and
compared them with matched Sobol-transfer controls. It generated 30/30 clips,
withheld two fireworks rows under the visual gate, scored 28/28 retained rows,
and found that saved high-scoring BO recipes were not portable global recipes:
BO-transfer averaged `-3.5444`, Sobol-transfer averaged `-3.0223`, and blue
jellyfish was the only positive prompt slot.

Second, the per-prompt Sobol search ran eight shared Sobol alpha/guidance points
across each of the five image-backed prompt slots. It generated 40/40 clips,
withheld two fireworks rows, scored 38/38 retained rows, and found that prompt
identity explained the retained score structure far better than alpha/guidance
recipe choice: prompt-only R2 = `0.9196`, Sobol recipe-index-only R2 = `0.0062`,
and alpha/guidance/interaction-only R2 = `0.0042`.

Third, the SVD content-axis audit showed that prompt text is metadata-only in
the current SVD replay path: `SVDGenerator.generate` does not accept prompt text
and the replay runner does not pass it. The only currently valid SVD content
variables are seed-image selection and seed-bank expansion.

The fixed-recipe seed-content probe then tested that actual intervention: Sobol
recipes 516 and 517 were replayed across the five available seed images with
two stochastic reps each. It generated 20/20 clips, withheld both fireworks
candidates by complete-candidate visual-first retention, scored 16/16 retained
rows after the Modal/TRIBE dependency fix, and found that seed-content slot
explained retained TRIBE score variance almost entirely:

| model | retained-score R2 |
|---|---:|
| recipe only | 0.0026 |
| seed-content slot only | 0.9494 |
| recipe + seed-content slot | 0.9520 |

The practical conclusion is now narrower and cleaner: under current SVD replay,
do not spend more broadening budget on alpha/guidance-only search or prompt
rewriting. The next valid content-broadening move is to restore/expand the
seed-image bank or switch to a prompt-conditioned video generator before running
prompt-rewrite tournaments.

### 2026-06-08 Restored Seed-Bank and Pocket Regime Audit

The full 24-row seed catalog was restored locally from `source_image` URLs
using `scripts/restore_bo_seed_bank.py`. Restored PNGs are raw data and remain
local; the script makes them reproducible.

The restored fixed-recipe screen replayed Sobol recipes 516 and 517 across all
24 seed-image slots with two stochastic reps per candidate. It generated 96/96
clips, withheld the visually brittle fireworks candidates, and scored 92/96
rows. Seed-content-only R2 was 0.9804 while recipe-only R2 was 0.0008. The top
retained pockets were orange flowers (mean TRIBE 4.2013), hanging clothes
(3.6167), blue jellyfish (2.1849), and old car (1.0488). The committed run note
is `seed_bank_restored_fixed_recipe_result_20260608.md`.

The pocket regime-audit then stress-tested those four positive targets against
three hard negative controls (aerial beach, city street, storm beach) across
six nearby Sobol recipes (`518`-`523`) and two stochastic reps per candidate.
It generated 84/84 clips, had 0/84 visual-gate failures, retained 42/42
complete candidates, and scored 84/84 rows.

| seed-content slot | scored / requested | mean TRIBE | min | max | positive rows |
|---|---:|---:|---:|---:|---:|
| orange flowers | 12 / 12 | 4.1043 | 3.6386 | 4.7864 | 12 / 12 |
| hanging clothes | 12 / 12 | 2.8991 | 1.8805 | 3.5741 | 12 / 12 |
| blue jellyfish | 12 / 12 | 2.0901 | 0.7267 | 3.3580 | 12 / 12 |
| old car | 12 / 12 | 1.1695 | 0.4140 | 1.6482 | 12 / 12 |
| aerial beach | 12 / 12 | -8.8447 | -10.0352 | -7.1844 | 0 / 12 |
| city street | 12 / 12 | -9.2525 | -9.6196 | -8.4299 | 0 / 12 |
| storm beach | 12 / 12 | -10.4170 | -11.3646 | -9.4116 | 0 / 12 |

On all scored rows, seed-content-only R2 was 0.9912, recipe-only R2 was 0.0021,
and seed-content + recipe R2 was 0.9983. This upgrades the restored-bank result
from a two-recipe observation into a stable content-pocket finding. The
accepted artifact class is not a portable alpha/guidance recipe; it is
seed-image content-pocket structure under the current SVD replay regime. The
committed preregistration and result notes are
`pocket_regime_audit_manifest_20260608.md` and
`pocket_regime_audit_result_20260608.md`.

### 2026-06-08 Content-Pocket Feature Audit

The next audit asked whether the positive pocket residual has a lightweight
visual explanation, or whether it should remain a black-box TRIBE compute-proxy
finding until a stronger verifier is added. The script
`scripts/audit_content_pocket_features.py` joined the pocket-regime replay
report to the restored seed images and generated replay videos, then computed
color, brightness, hue-region, edge/texture, entropy, and center/border
descriptors on both seed images and sampled generated-video frames.

The pre-registered descriptor gate was strict but simple: accept a descriptor
only if it separates the four positive pockets from the three hard negative
controls with `separation_auc >= 0.85` and `abs_cohen_d >= 1.00`, without using
TRIBE score as an input feature.

No lightweight descriptor cleared that gate.

| family | feature | direction | positive mean | negative mean | AUC | abs d | r(score) |
|---|---|---|---:|---:|---:|---:|---:|
| seed | colorfulness | higher for positive | 0.2381 | 0.1126 | 0.8333 | 1.8471 | 0.7469 |
| seed | bright fraction | higher for positive | 0.5143 | 0.1568 | 0.8333 | 1.3371 | 0.5908 |
| video | bright fraction | higher for positive | 0.5221 | 0.1869 | 0.8102 | 1.2287 | 0.5626 |
| video | colorfulness | higher for positive | 0.2797 | 0.1366 | 0.7963 | 1.6930 | 0.7225 |

This is useful negative structure. The stable pockets are not explained well
enough by simple color/edge/frame descriptors to promote one of those
descriptors into a verifier. C-017 therefore stays scoped as a compute-proxy
content-pocket finding. The next mechanistic move should be a stronger
CLIP/V-JEPA embedding audit, or a human/BMD-grounded validation gate, before
spending much more budget on blind stochastic replication. The committed audit
artifacts are `content_pocket_feature_audit_manifest_20260608.md`,
`content_pocket_feature_audit_result_20260608.md`, and
`content_pocket_feature_audit_summary_20260608.json`.

### 2026-06-08 Content-Pocket Embedding Audit

The stronger embedding audit then asked whether CLIP/V-JEPA-style representations
explain the pocket residual that simple visual descriptors missed. The script
`scripts/audit_content_pocket_embeddings.py` encoded the exact pocket-regime
seed images and generated SVD replay clips with `openai/clip-vit-base-patch32`,
aggregated replicates to 42 task-level candidates, and tested pocket-held-out
centroid margins plus leave-one-pocket-out classifiers. Exact V-JEPA feature
files for these replay-video stems were not available, so V-JEPA was recorded
as missing rather than mixed with mismatched Wan/BMD features.

The CLIP gate passed.

| family | feature | direction | positive mean | negative mean | AUC | abs d | r(score) |
|---|---|---|---:|---:|---:|---:|---:|
| clip_seed_image | pocket-held-out centroid margin | higher for positive | 0.0605 | -0.0409 | 1.0000 | 2.8573 | 0.8541 |
| clip_video | pocket-held-out centroid margin | higher for positive | 0.0604 | -0.0359 | 0.8796 | 2.0280 | 0.7620 |

The leakage-aware classifiers also passed: seed-image CLIP reached AUC 1.0000
and balanced accuracy 0.8333, while generated-video CLIP reached AUC 0.9514 and
balanced accuracy 0.8333. This promotes CLIP embedding geometry to an accepted
compute-proxy verifier for the content-pocket residual. It still does not prove
human memorability, delayed recognition, or exact V-JEPA agreement. The next
SVD experiment should replicate orange flowers and hanging clothes under fresh
stochastic seeds while preserving both positive TRIBE score and the CLIP
centroid-margin verifier. The committed audit artifacts are
`content_pocket_embedding_audit_manifest_20260608.md`,
`content_pocket_embedding_audit_result_20260608.md`, and
`content_pocket_embedding_audit_summary_20260608.json`.

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
  completed 4/4 TRIBE scores in a one-replicate panel, but the three-replicate
  panel failed 1/12 clips because `sobol_007` replicate 2 repeatedly collapsed.
  A visual-first `max_evals=2` replacement pool generated 21/21 clips, withheld
  3/7 complete candidates, and scored 12/21 retained rows. The saved-table pool
  still lacked a retained fireworks Sobol candidate. The 2026-06-08
  regenerated-control run generated 24/24 clips, withheld one BO fireworks
  candidate under complete-candidate retention, scored 21/21 retained rows, and
  kept matched BO/control coverage in both selected strata. The balanced max-3
  regenerated-control stress test then generated 36/36 clips, withheld two
  visual-failed rows, scored 30/30 retained rows, and confirmed prompt-pocket
  behavior. The prompt-transfer, per-prompt Sobol, content-axis audit, and
  fixed-recipe seed-content probe now show that current SVD broadening should
  target seed-image/content expansion rather than alpha/guidance-only search.
  The restored seed-bank and pocket regime-audit runs extend this: restored
  non-jellyfish pockets survive local recipe stress tests while hard negative
  controls remain negative.
- Report wall-clock and average minutes per evaluation.
- Treat runtime as a limitation: BO is sample-efficient, not yet wall-clock
  efficient.
