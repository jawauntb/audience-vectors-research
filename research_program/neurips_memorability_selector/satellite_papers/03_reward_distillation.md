# From Brain-Aligned Selectors To Video-Generator Distillation

**Draft status:** satellite paper draft, regenerated 2026-06-02.
**Core purpose:** define the LoRA/DPO project without pretending the proxy result
is already behavioral success.

## Abstract

Brain-aligned reward signals can be used in two ways: selecting among generated
candidates at test time, or distilling the selector into the generator through
preference optimization. The current project supports the first path more
strongly than the second. On a 24-seed Wan2.2 proxy run, a preference-weighted
single LoRA improves 20/24 prompts under the TRIBE/BMD projection, and
base-or-gated best-of-4 improves 18/24 with mean lift about +2.817 while avoiding
negative seed-level regressions by falling back to the base clip. This is useful
engineering evidence for a generation-ranking workflow, but it is not proof that
the generator learned human memorability. A distillation paper should begin only
after the selector itself is validated by humans.

## Current Product Workflow

```text
generate base video
generate LoRA / best-of-N variants
score each candidate with TRIBE/BMD v_mem
score preservation with CLIP-style gates
choose the best gated candidate
fallback to base when the gated candidate is worse
```

## Current Proxy Result

| Selection rule | Improved seeds | Mean lift | Median lift |
|---|---:|---:|---:|
| Single LoRA | 20/24 | +1.274 | +1.218 |
| Raw best-of-N | 20/24 | +3.561 | +3.451 |
| Base or raw best-of-N | 20/24 | +3.674 | +3.451 |
| Base or gated best-of-N | 18/24 | +2.817 | +2.432 |

## Why DPO Is Harder Than Selection

Selection is cheap because the reward model only chooses among candidates.
Distillation is harder because the generator can learn reward-model loopholes:
artifact-heavy clips, semantic drift, or high-scoring visual tropes that humans
do not actually remember. Video DPO also needs temporally meaningful preference
pairs, consistent seeds, enough diversity, and held-out human evaluation. The
training objective must not be evaluated only by the reward model that created
the labels.

## Minimal DPO Study Design

1. Validate the selector against humans on 50-100 prompts.
2. Generate 4-8 candidates per prompt for 500-2,000 prompts.
3. Label preference pairs with the validated selector plus preservation gates.
4. Train a small LoRA or DPO adapter.
5. Evaluate on fresh prompts with:
   - human pairwise memorability;
   - V-JEPA and CLIP baselines;
   - video-quality gates;
   - delayed recognition if budget allows.

## Product Direction

The most useful near-term product is not a fully trained memorability generator.
It is a queueable selector that lets users upload or generate multiple variants,
then returns:

- raw TRIBE dimensions;
- BMD memorability projection;
- V-JEPA and CLIP baseline scores;
- threshold checks;
- natural-language failure analysis;
- recommended edits or next generations.

Distillation becomes worthwhile only after the selector reliably predicts human
judgment.
