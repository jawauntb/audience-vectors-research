# Per-Prompt Sobol Search Result

Date: 2026-06-08

## Question

The prompt-transfer stress test showed that the best saved BO alpha/guidance
recipes do not transfer across prompt strata. This run asks the next narrower
question: if we search alpha/guidance locally inside each available prompt slot,
do any non-jellyfish prompts produce positive TRIBE/BMD proxy candidates?

## Protocol

Manifest builder:
`scripts/build_bo_prompt_search_manifest.py`

Local trial table:
`data/reports/bo_prompt_search_trial_table_sobol8x5_20260608.json`

Local replay report:
`data/reports/bo_prompt_search_sobol8x5_reps1_steps50_motion5_noise0_20260608.json`

The manifest used the five locally image-backed prompt slots and eight shared
Sobol alpha/guidance points per slot, starting at Sobol index 512. Generation
and scoring reused the tuned replay settings from the prior visual-first runs:
50 SVD steps, motion bucket 5, noise augmentation 0, visual-first
complete-candidate retention, full TRIBE scoring from bytes, and the current
TRIBE/BMD cortical direction.

## Gate Result

- 40/40 requested clips generated.
- 2/40 rows failed the visual artifact gate, both in the fireworks prompt slot:
  - `sobol_prompt_search_516_slot00`: `tail_sharpness_collapse`
  - `sobol_prompt_search_519_slot00`: `tail_sharpness_collapse`
- Complete-candidate visual-first retention kept 38/40 candidates.
- 38/38 retained rows completed full TRIBE scoring.

## Prompt-Level Scores

| prompt slot | scored / requested | mean TRIBE | std | best retained candidate | best score |
|---|---:|---:|---:|---|---:|
| fireworks | 6 / 8 | -3.9068 | 1.3723 | `sobol_prompt_search_515_slot00` | -1.9761 |
| ocean cliffs | 8 / 8 | -8.1526 | 0.6898 | `sobol_prompt_search_515_slot01` | -6.8832 |
| concert stage | 8 / 8 | -1.3808 | 1.0446 | `sobol_prompt_search_516_slot02` | -0.1553 |
| blue jellyfish | 8 / 8 | 1.6597 | 0.6887 | `sobol_prompt_search_517_slot03` | 2.9734 |
| forest canopy | 8 / 8 | -3.7246 | 1.3450 | `sobol_prompt_search_514_slot04` | -1.7318 |

The top eight retained candidates were all blue jellyfish rows. The best row was
`sobol_prompt_search_517_slot03`, with alpha = -0.7250, guidance = 9.4145, and
TRIBE score = 2.9734.

## Variance Audit

An additive least-squares diagnostic on the 38 retained rows quantified how much
of the score structure is explained by prompt identity versus the shared
alpha/guidance recipe index:

| model | retained-score R2 |
|---|---:|
| prompt only | 0.9196 |
| Sobol recipe index only | 0.0062 |
| alpha + guidance + alpha*guidance only | 0.0042 |
| prompt + Sobol recipe index | 0.9254 |
| prompt + alpha + guidance + alpha*guidance | 0.9234 |

## Interpretation

This is stronger than the prompt-transfer result. Under the current
SVD/TRIBE replay regime, prompt/seed identity dominates alpha/guidance choice.
The per-prompt Sobol search found a higher-scoring jellyfish candidate than the
prompt-transfer panel, but it did not uncover a positive non-jellyfish prompt
slot. Alpha/guidance-only search is therefore the wrong next broadening axis.

The next representational move should add content variables to the search
regime: prompt rewriting, seed-image selection, or a seed-bank expansion before
spending more budget on BO over alpha/guidance alone.

