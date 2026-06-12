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
- 104-clip fold-safe hidden-patch run is complete; remaining caution is
  population-level generalization and matched-control analysis.

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

## Paper 6: Proxy-Guided BO Video Control

Core:

- Multi-objective BO over SVD-XT steering/search parameters.
- Proxy objectives: TRIBE/BMD memorability, CLIP fidelity, R3D quality.
- Current status is collaborator intake plus Modal replay tooling.
- Needs exact `v_mem_CLIP` provenance, fixed-budget reproduction, and visual
  inspection before being treated as evidence.

Venue target:

- Generative media control / black-box optimization workshop.
- Main-track only if paired with human validation and strong equal-budget
  baselines.

## Paper 7: Neural-Response-Guided Generated Video

Core:

- Treat TRIBE-like brain-predictive models, V-JEPA/CLIP audit frames, and
  memorability predictors as candidate-selection or guidance signals.
- Separate media-to-predicted-brain-response scoring from brain-to-media
  reconstruction and from reward/guidance optimization.
- Validate generated-video memorability with human or BMD-grounded endpoints
  before making memorability claims.

Venue target:

- NeuroAI / generative media workshop first.
- Main-track only if the selector or guidance gain survives human/BMD
  validation against strong V-JEPA, CLIP, and quality baselines.

## Paper 8: Content-Pocket Human Recognition Memory

**Generated-Video Content Pockets Predict Delayed Human Recognition Memory**

Core:

- SVD content-pocket regime produced primary candidates: orange flowers and
  hanging clothes.
- TRIBE/V-JEPA compute-proxy verification preceded a frozen old-vs-lure human
  recognition-memory task.
- Wave 2 Prolific result provides narrow delayed human recognition evidence for
  the pooled primary-positive packet against hard negative controls.
- Scope remains exact recognition in this packet, not broad generated-video
  memorability, measured-BMD grounding, or prompt-conditioned generator control.

Venue target:

- NeuroAI / generative media evaluation workshop.
- Main-track only if framed carefully as a content-pocket validation study or
  paired with a larger confirmatory endpoint.

## Rule

Do not let a satellite claim become a burden for the main paper unless it
directly improves the selector's human-validated result.
