# Satellite Paper Tracks

The exploratory project contains too much for one paper. These tracks should be
kept separate unless a result directly supports the main selector claim.

## Paper 1: Main Selector Paper

**Brain-Aligned Memorability Signals Improve Video Generation Selection**

Core:

- BMD/TRIBE memorability direction.
- Selector policies.
- Human validation against baselines.

Venue target:

- NeurIPS main track if human result is strong.
- NeurIPS E&D if framed as evaluation methodology/tooling.
- CVPR/ICCV workshop if first pass is smaller.

## Paper 2: Audience Axes / Persona Decomposition

Core:

- Synthetic persona scoring reveals low-dimensional signed axes.
- Reviewer-corrected analysis: not 12 orthogonal personas, closer to four
  effective signed axes.
- Needs persona-matched human validation before strong claims.

Venue target:

- CHI/CSCW workshop, recommender systems workshop, or ML interpretability
  workshop.

## Paper 3: Mechanistic Audit Of Brain-Encoder Memorability

Core:

- Temporal Fourier and hidden-direction patching inside TRIBE.
- Positional-artifact critique is weakened, but hidden sequence dependence
  remains.
- Needs larger fold-safe hidden-patch run.

Venue target:

- NeuroAI / mechanistic interpretability workshop.

## Paper 4: Distilling Brain-Aligned Rewards Into Video Generators

Core:

- Use validated memorability selector to create preference pairs.
- Train LoRA/DPO-style adapter.
- Evaluate against humans, not only the proxy reward.

Venue target:

- Generative model alignment workshop first.
- Main-track only if human gains are strong and compute scale is credible.

## Rule

Do not let a satellite claim become a burden for the main paper unless it
directly improves the selector's human-validated result.
