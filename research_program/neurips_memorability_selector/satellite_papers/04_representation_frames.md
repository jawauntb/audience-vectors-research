# Representation Frames For Cognitive Media Selection

**Draft status:** theory and methods note, regenerated 2026-06-01.
**Core purpose:** keep the broader intellectual frame without letting it
overclaim the empirical paper.

## Abstract

A learned vector in a model should not be treated as the essence of a cognitive
property. It is a readout in a representation frame. The current memorability
project can be formalized as a frame-transfer problem: TRIBE, V-JEPA, CLIP, and
human behavior each map the same stimulus into different coordinates, and the
scientific question is whether an ordering learned in one frame survives in
another. On the current Wan candidate pool, TRIBE and V-JEPA show moderate
geometric relationship, with RSA Spearman about
+0.345 and linear CKA
about +0.392, while CLIP-preservation
signals are much less aligned with the memorability scores. This motivates a
broader program: generated media as controlled intervention on human generative
models, with memorability as the first validated axis.

## Mathematical Frame

Let `X` be the space of prompt-conditioned video candidates. We observe several
representation maps:

```text
f_T : X -> V_T      TRIBE / brain-aligned frame
f_J : X -> V_J      V-JEPA / self-supervised video frame
f_C : X -> V_C      CLIP / prompt-preservation frame
h   : X -> Y        human memory behavior
```

The current readout is:

```text
l_T : V_T -> R
```

The empirical question is whether `l_T(f_T(x))` predicts `h(x)` better than
comparable readouts in `V_J` and `V_C`.

## Anti-Reification Rule

The vector is not the property. The vector is a coordinate that becomes useful
when its induced ordering transfers across frames. This matters for:

- memorability vectors;
- attention vectors;
- persona vectors;
- social-cohesion or volition vectors in future projects.

## Current Frame Audit

TRIBE and V-JEPA are partially aligned, not interchangeable. CLIP preservation
is closer to prompt/seed fidelity than memorability. That separation is useful:
it lets a product enforce prompt fidelity while still optimizing memory signal.

The existing artifact is:

- `experiments/representation_frame_analysis.md`

## Future Work

1. Compare TRIBE, V-JEPA, CLIP, and human pairwise similarity matrices with RSA
   and CKA after human responses are collected.
2. Ask whether high-disagreement prompts reveal semantic, temporal, social, or
   artifact-driven differences between frames.
3. Extend the method from memorability to attention, agency, social warmth,
   threat/safety, and self-other overlap.
4. Use the validated selector before training any generator, so reward-model
   circularity does not define the science.

## References

- Hohwy. The Self-Evidencing Brain. Nous 2016.
- Fields, Friston, Glazebrook, and Levin. A Free Energy Principle for Generic
  Quantum Systems. Progress in Biophysics and Molecular Biology 2022.
- Fields and Glazebrook. Separability, Contextuality, and the Quantum Frame
  Problem. International Journal of Theoretical Physics 2023.
- Dahl, Lutz, and Davidson. Reconstructing and deconstructing the self. Trends
  in Cognitive Sciences 2015.
