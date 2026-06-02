# Claim Contract

This document is the guardrail for a submission-grade paper. Every claim in the
paper should be either already supported, pending a named experiment, or cut.

## Allowed Core Claims Now

- A supervised contrastive direction in TRIBE/BMD features predicts held-out BMD
  memorability around `rho ~= +0.40`.
- V-JEPA is competitive for global memorability prediction, so the contribution
  is not "TRIBE beats all visual baselines."
- Brain alignment gives TRIBE extra scientific affordances: fMRI comparability,
  ROI hypotheses, and a way to connect generated-video selection to neural
  response models.
- Persona directions are structured but not orthogonal; they compress to a few
  signed latent axes.
- The 104-clip fold-safe hidden-direction patch shows the learned TRIBE hidden
  direction is load-bearing across disjoint train/eval folds on balanced
  high/low memorability tail clips.
- Current Wan LoRA/product-selector results are proxy-scored and need human
  validation.

## Claims That Require The Next Human Eval

- "TRIBE-selected generated videos are more memorable to humans."
- "TRIBE improves generation ranking beyond V-JEPA/CLIP/aesthetic baselines."
- "The base-or-gated selector is a practical product workflow."
- "Brain-aligned metrics add value over generic video quality metrics."

## Claims To Avoid

- TRIBE is a human preference oracle.
- The persona axes are independent audience segments.
- A learned vector is the ontological essence of memorability, attention, or
  audience identity.
- The active-inference / emptiness frame validates any empirical result.
- The hidden-direction patch proves population-level TRIBE-internal causality
  or a fully isolated mechanism.
- The Wan LoRA is behaviorally improved just because the TRIBE/BMD proxy rises.
- Direct steering is solved.
- Memorability is the same thing as quality, engagement, virality, emotion, or
  commercial effectiveness.

## Reviewer-Expected Baselines

The main paper should include at least:

- Random candidate selection.
- CLIP or text-video alignment selection.
- V-JEPA memorability or feature-linear baseline.
- Generic video quality/temporal consistency baseline where feasible.
- TRIBE memorability selection.
- TRIBE memorability plus preservation/quality gate.
- Optional: LoRA/DPO selector as a separate treatment, clearly marked proxy
  trained.

## Required Statistical Shape

- Frozen train/validation/test split.
- Predeclared primary endpoint.
- Clustered bootstrap by prompt, not just pooled votes.
- Per-prompt win rates and confidence intervals.
- Multiple-comparison correction if many selectors are compared.
- Failure-case analysis: when TRIBE selects worse videos, why?

## Optional Theory Frame

The broader program can be described as generated media acting on human
generative models. In that frame, TRIBE, V-JEPA, CLIP, and human behavior are
different representation/measurement frames over the same stimulus space. The
empirical question is whether a readout learned in one frame preserves the human
memorability ordering better than competing frames. This is useful discussion
language, but it should not replace the predeclared human endpoint.
