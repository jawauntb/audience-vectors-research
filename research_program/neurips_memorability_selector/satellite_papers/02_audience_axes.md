# Synthetic Audience Axes Are Structured But Not Orthogonal

**Draft status:** satellite paper draft, regenerated 2026-06-01.
**Core purpose:** preserve the useful audience-vector work while removing the
original overclaim.

## Abstract

Synthetic persona-conditioned scoring can reveal structured differences in how
generated or natural videos are ranked, but signed cosine averages can make
shared axes look falsely orthogonal. We revisit the audience-vector
decomposition after a reviewer critique. Twelve persona-conditioned TRIBE
contrastive directions have signed off-diagonal cosine mean
+0.023, but this statistic
hides large positive and negative alignments. The corrected unsigned overlap is
mean abs cosine 0.434,
and the effective rank is 3.56
out of 12, compared with nearly 12 for random directions in the same dimension.
Thus the audience directions are structured, stable, and potentially useful for
ranking, but they do not span twelve independent audience axes. The right next
step is persona-matched human validation, not stronger claims about cognitive
modules.

## Corrected Claim

Original unsafe claim:

```text
Per-persona directions are near-orthogonal and reveal independent audience axes.
```

Current safe claim:

```text
Synthetic persona directions compress to a small set of signed latent axes.
Opposite signs can reflect the same underlying axis, and ROI localization is
exploratory.
```

## Why The Correction Matters

A cosine of -0.99 is not evidence for a different memorability axis. It is the
same axis with opposite sign. Since squared projection removes sign, any ROI or
energy localization analysis must be interpreted with this sign ambiguity in
mind.

## Current Numbers

| Quantity | Value |
|---|---:|
| Personas | 12 |
| Signed off-diagonal cosine mean | +0.023 |
| Signed off-diagonal cosine range | -0.987 to +0.934 |
| Mean absolute off-diagonal cosine | 0.434 |
| Effective rank | 3.56 / 12 |
| Components for 90 percent variance | 4 |

## Interpretation

The persona system is still interesting. It suggests that model-generated
audience archetypes induce repeatable orderings over stimuli. But the structure
looks low-dimensional and signed, not cleanly modular. This may be useful for
product interfaces, where users want sliders such as "fast-scroll salience" or
"narrative emotionality." It is not yet evidence that real audiences divide
into those exact archetypes.

## Validation Needed

1. Recruit participants who self-identify with or are behaviorally matched to
   the persona profiles.
2. Ask them to rank or choose videos under identical prompt-conditioned sets.
3. Compare persona-derived selector wins with matched and mismatched raters.
4. Report whether synthetic persona axes predict real subgroup preferences
   beyond the global memorability direction.

## Keep Out Of The Main Selector Paper

This work is useful, but it can distract from the primary selector claim. The
main paper should mention persona axes only as exploratory context or appendix
material unless persona-matched human validation is complete.
