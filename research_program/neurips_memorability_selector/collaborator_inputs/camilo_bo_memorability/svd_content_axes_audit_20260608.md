# SVD Content-Axis Audit

Date: 2026-06-08

## Question

After the per-prompt Sobol search showed that prompt/seed identity dominates
alpha/guidance choice, the obvious next idea was prompt or content search. This
audit asks which content axes the current SVD replay runner can actually
intervene on.

## Protocol

Audit script:
`scripts/audit_bo_content_axes.py`

Local audit report:
`data/reports/bo_svd_content_axes_audit_20260608.json`

The audit checks two things:

1. seed-bank availability in
   `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/original/seeds`;
2. whether prompt text is passed from `scripts/modal_bo_memorability_replay.py`
   into `SVDGenerator.generate`.

## Result

| audit item | result |
|---|---:|
| prompt catalog rows | 24 |
| locally available seed images | 5 |
| missing seed images | 19 |
| SVD generator accepts prompt text | false |
| replay runner passes prompt text | false |
| current prompt axis | metadata only |

The five locally available seed images are:

- `fresh24_fireworks`
- `fresh24_ocean_cliffs`
- `fresh24_concert_stage`
- `fresh24_blue_jellyfish`
- `fresh24_forest_canopy`

The prompt field remains useful for provenance and stratification, but it is not
currently a generation-conditioning variable for SVD replay. The actionable
content variables under the current runner are seed-image selection and
seed-bank expansion.

## Attempted Follow-Up Probe

I attempted the next small fixed-recipe seed-content replicate panel using Sobol
indices 516 and 517 across the five available seed images with two stochastic
replicates each. The dry-run expanded correctly to 20 jobs, but the full Modal
run was blocked before the first generation completed:

```text
workspace billing cycle spend limit reached
```

No new scored replay result was produced by that attempted probe.

## Interpretation

This is a regime audit rather than a memorability result. It prevents a
misleading next experiment: prompt rewriting alone cannot test content
broadening in the current SVD replay path because prompt text is not passed into
the image-to-video generator.

The next valid content-broadening step is one of:

1. restore or create a larger seed-image bank, then run matched seed-selection
   panels under the existing SVD replay;
2. change the generator path to a prompt-conditioned model such as CogVideoX,
   Wan2.2, or Veo before running prompt-rewrite tournaments;
3. modify SVD replay plumbing only if a prompt-conditioned SVD variant is
   actually available.

