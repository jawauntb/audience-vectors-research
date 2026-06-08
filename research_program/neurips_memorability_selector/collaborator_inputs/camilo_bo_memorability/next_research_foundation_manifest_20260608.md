# Next Research Foundation Manifest - 2026-06-08

Last updated: 2026-06-08.

This manifest turns the post-regenerated-control fork into concrete next steps.
It is deliberately conservative: it records what the saved BO table can still
support, what it cannot support, and which gate should run next.

## Current State

The completed regenerated-control run in
`regenerated_visual_controls_result_20260608.md` established a working protocol:

- deterministic regenerated Sobol controls can be appended per selected BO
  prompt stratum;
- complete-candidate visual-first retention can withhold visually unstable
  candidate families before TRIBE scoring;
- matched BO/control coverage survived in the two selected prompt strata;
- the result remained mixed by stratum and proxy-only.

The conceptual guardrail from the categorical/autopoietic/ruliology notes is:
keep claims typed, gated, and slack-preserving. The system should not collapse
onto a lucky candidate pocket before the evidence supports that narrowing.

## Saved-Table Coverage Audit

The replay script defaults to the 3-objective collaborator table:

```text
research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/raw_results/gpu_run_3obj_all_results.json
```

Dry-run audit commands:

```bash
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --dry-run \
  --selection top-bo-per-stratum \
  --stratify-by prompt \
  --max-evals 99 \
  --regenerated-sobol-controls-per-stratum 0 \
  --report-path /tmp/top_bo_per_prompt_all.json

uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --dry-run \
  --selection seed-stratified-bo-vs-sobol \
  --stratify-by prompt \
  --max-evals 99 \
  --report-path /tmp/saved_matched_prompt_all.json
```

Audit result:

| selector | selected rows | prompt strata | interpretation |
|---|---:|---:|---|
| `top-bo-per-stratum`, `max_evals=99` | 20 saved BO rows | 2 | The saved BO table cannot broaden beyond fireworks and jellyfish prompt strata. |
| `seed-stratified-bo-vs-sobol`, `max_evals=99` | 23 saved BO/Sobol rows | 2 | Saved Sobol coverage is also limited and previously visually brittle. |

Saved BO prompt coverage:

| prompt stratum | saved BO candidates available |
|---|---:|
| fireworks | 3 |
| jellyfish | 17 |

This means "broader regenerated controls" has two different meanings:

1. within-table stress test: use more saved candidates inside the same two
   prompt strata;
2. true prompt-broadened claim: run a new BO/search panel over more seed prompts.

Only the second can support a broad prompt-level BO/control claim.

## Next Operational Gate: Balanced Max-3 Stress Test

The next low-friction foundation run is a balanced within-table stress test:
3 saved BO anchors and 3 fresh regenerated Sobol controls per available prompt
stratum, 3 stochastic replicates each.

Preflight command:

```bash
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --dry-run \
  --selection top-bo-per-stratum \
  --stratify-by prompt \
  --max-evals 3 \
  --regenerated-sobol-controls-per-stratum 3 \
  --regenerated-sobol-pool-size 256 \
  --regenerated-sobol-start-index 128 \
  --replicates 3 \
  --report-path /tmp/regenerated_controls_max3_start128_preflight.json
```

Preflight result:

- loaded 32 trials;
- selected 6 saved BO anchors;
- appended 6 regenerated Sobol controls;
- expanded to 36 replay jobs;
- no target stratum was missing regenerated controls.

Selected saved BO anchors:

| task | stratum | alpha | guidance | seed_idx | original TRIBE |
|---|---|---:|---:|---:|---:|
| `bo06_cand01` | fireworks | -4.1262 | 7.8464 | 10 | -0.3899 |
| `bo09_cand01` | fireworks | 7.0962 | 2.4844 | 10 | -0.9172 |
| `bo03_cand01` | fireworks | 7.0735 | 3.6069 | 10 | -2.9498 |
| `bo07_cand01` | jellyfish | 7.0735 | 3.2311 | 13 | 6.1509 |
| `bo04_cand01` | jellyfish | -3.9674 | 7.7753 | 13 | 5.3993 |
| `bo02_cand01` | jellyfish | -3.9785 | 7.9710 | 13 | 4.8678 |

Fresh regenerated controls:

| task | stratum | alpha | guidance | seed_idx | noise_seed |
|---|---|---:|---:|---:|---:|
| `sobol_regen_133` | fireworks | -3.7734 | 5.6754 | 10 | 133 |
| `sobol_regen_136` | fireworks | 6.9236 | 2.0850 | 5 | 136 |
| `sobol_regen_138` | fireworks | -2.7452 | 4.3513 | 0 | 138 |
| `sobol_regen_128` | jellyfish | 9.7878 | 3.7529 | 8 | 128 |
| `sobol_regen_135` | jellyfish | 5.9328 | 7.9415 | 13 | 135 |
| `sobol_regen_142` | jellyfish | -6.8373 | 2.8278 | 8 | 142 |

Full-run command:

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

Pass condition:

- 36/36 requested clips are generated or every generation failure is recorded;
- complete-candidate visual-first retention is applied before upload/scoring;
- after retention, at least one BO candidate and one regenerated Sobol control
  remain in each prompt stratum;
- all retained rows complete full TRIBE scoring;
- the result note reports candidate means, per-stratum means, visual failures,
  withheld candidate families, and claim impact.

Interpretation rule:

- If the result remains mixed by stratum, keep BO/control claims narrow and move
  to a new prompt-broadened BO run before any paper-facing upgrade.
- If BO wins both strata, treat it as a stronger within-table result only; it is
  still not broad prompt evidence because the saved BO table covers only two
  prompt strata.
- If visual-first retention removes a whole policy from either stratum, the
  gate is blocked rather than negative evidence.

## Parallel Foundation Path: Exploratory Human Panel Freeze

If the goal is a small human pilot instead of more compute foundation, freeze
only the visually retained matched candidates from the completed regenerated
run:

| stratum | BO candidates | regenerated Sobol controls | excluded |
|---|---|---|---|
| fireworks | `bo06_cand01` | `sobol_regen_016`, `sobol_regen_017` | `bo09_cand01` withheld for tail sharpness collapse |
| jellyfish | `bo04_cand01`, `bo07_cand01` | `sobol_regen_013`, `sobol_regen_020` | none |

Human-panel wording must be exploratory:

- ask whether viewers prefer or better remember visually retained generated
  clips, not whether BO is generally superior;
- balance prompt strata and policy labels;
- keep replicate provenance;
- report exclusions before results;
- do not include the withheld `bo09_cand01` family.

## True Broad Claim Requirement

A broad BO/control claim requires a new search panel, not just a replay of the
saved collaborator table.

Minimum requirements:

- at least 4 prompt/seed strata with BO and control coverage;
- equal generated clips and full TRIBE scores per strategy per stratum;
- visual-first complete-candidate retention before scoring;
- strategy set: BO, regenerated Sobol/random, saved Sobol where available, and
  if feasible best-of-N or a cheap CLIP/R3D/VBench-style prefilter;
- candidate-level mean/std/SEM over stochastic replicates;
- stratum-level summaries before pooled summaries;
- explicit human/BMD gate before any human-memorability claim.

The saved-table max-3 stress test is still worth running because it tests
whether the current two-stratum result is stable under a slightly wider
within-table panel. It cannot, by itself, solve prompt coverage.
