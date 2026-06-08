# Current Research Status

Last updated: 2026-06-08

## Confirmed Enough To Treat As Real

- A supervised TRIBE/BMD memorability direction predicts held-out BMD
  memorability at roughly `rho ~= +0.40`.
- BOLD Moments human memorability labels are the ground-truth behavioral target
  for the core `v_mem` direction.
- The completed Prolific best-of-N study supports the broader TRIBE/BMD selector
  signal: 41 raters, 41 passed attention checks, and Study A found humans chose
  the TRIBE-ranked best-of-N winner over a within-seed median variant 290/451
  times, or 64.3% (Wilson 95% CI [0.598, 0.686], binomial p = 1.3e-9).
- Persona vectors are structured, but the Spencer critique was right: they are
  not independent orthogonal axes. Signed cosine structure implies a smaller set
  of shared latent directions.
- Best-of-N and base-or-gated selection are the most practical current workflow.
  Direct continuous steering is not solved.
- The current Wan proxy run is promising but still proxy-only:
  preference-weighted single LoRA improves 20/24 prompts, and base-or-gated
  best-of-4 improves 18/24 prompts under the TRIBE/BMD proxy.
- V-JEPA is now an active baseline for the same current-pilot candidate pool:
  103 unique V-JEPA embeddings, 24/24 seeds covered, 0 missing features.
- BO/SVD generated-video evidence is currently a compute-proxy regime result,
  not a human-memorability result. The latest regenerated-control,
  prompt-transfer, per-prompt Sobol, content-axis audit, and fixed-recipe
  seed-content panels show prompt-pocket behavior: blue jellyfish is the stable
  positive pocket, fireworks is visually brittle, and seed-image/content slot
  explains retained SVD replay score variance far better than alpha/guidance
  recipe identity.

## Newly Built V-JEPA Pilot State

- Augmented manifest:
  `research_program/neurips_memorability_selector/experiments/current_selector_manifest_with_vjepa.json`
- V-JEPA score report:
  `research_program/neurips_memorability_selector/experiments/vjepa_selector_report.json`
- Augmented pairwise tasks:
  `research_program/neurips_memorability_selector/experiments/current_selector_pairwise_tasks_with_vjepa.json`
- Augmented survey:
  `research_program/neurips_memorability_selector/experiments/current_selector_prolific_survey_with_vjepa.html`

Coverage:

```text
unique V-JEPA embeddings: 103
seeds with V-JEPA-selected video path: 24/24
augmented pairwise tasks: 185
missing local video paths in task file: 0
```

Selector overlap:

```text
V-JEPA equals product selector: 7/24 seeds
V-JEPA equals gated selector: 10/24 seeds
```

This is exactly what we want for a useful baseline: V-JEPA is neither identical
to TRIBE/product nor obviously irrelevant.

## Not Yet Proven

- We have not shown that the newer V-JEPA-adjudicated TRIBE-selected generated
  videos are more memorable to humans.
- We have not shown that TRIBE beats V-JEPA, CLIP, or quality baselines in the
  current independent generated-video selector pilot.
- We have not shown that Arthur/Camilo's BO-generated videos improve human
  memorability. Their BO work is compute/control evidence until human-tested,
  and current SVD broadening should target seed-image/content expansion or a
  prompt-conditioned generator path rather than more alpha/guidance-only search.
- We have not shown delayed-recognition memory gains.
- We have not shown that a LoRA or DPO model has actually learned
  memorability; the present model-side results are selector/proxy evidence.
- We have a fold-safe TRIBE-internal hidden-direction intervention on 104
  balanced high/low clips. We have not yet shown a full population-level causal
  mechanism with content stratification and matched-control patches.

## Submission-Critical Next Step

Run the V-JEPA-augmented blinded human pilot. The key tests are:

- `product_vs_vjepa_memorability`
- `gated_vs_vjepa_memorability`
- `product_vs_clip_preservation`
- `gated_vs_clip_preservation`
- `product_vs_base`
- `gated_vs_base`

If TRIBE/product or gated selection loses to V-JEPA, the paper should become a
more honest "brain-aligned and self-supervised video features both expose
memorability-like signals" paper. If it wins, the paper has a strong main
claim: brain-aligned features improve generated-video selection for a cognitive
property beyond standard video-feature baselines.

## After The Pilot

1. Scale from 24 prompts to 50-100 fresh prompts with frozen selector policies.
2. Add a standard video-quality metric or VBench-style score so reviewers cannot
   say the selector only finds artifact-heavy memorable clips.
3. Add a representation-frame analysis: compare TRIBE, V-JEPA, CLIP, and human
   pairwise similarity/order matrices with RSA or CKA, especially on prompts
   where the selectors disagree.
4. Run prompt-clustered bootstrap and mixed-effects logistic analysis.
5. Only then decide whether LoRA/DPO distillation is worth the budget.
