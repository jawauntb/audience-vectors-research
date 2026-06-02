# Theory Bridge: Generated Media As Cognitive Intervention

This note is deliberately broader than the NeurIPS selector paper. It captures
why the project is bigger than "find a memorability vector" without overloading
the empirical manuscript.

## Core Reframe

Generated media can be treated as an intervention on a human observer's
generative model.

```text
generator -> stimulus -> observer representation -> memory / attention / action
```

The current project measures one tractable axis of that intervention:
memorability. TRIBE, V-JEPA, CLIP, and human judgments are different
representation frames over the same stimulus set. A selector is useful when a
scalar readout in one frame induces an ordering that survives in another frame,
especially human behavior.

## Why The Active-Inference / Emptiness Literature Helps

The useful lesson is anti-reification:

- A boundary is a modelling choice, not automatically an ontological fact.
- A vector is a coordinate/readout in a representation frame, not the essence of
  the property.
- A persona axis is a useful partition only if it survives changes of prompt,
  model, sign, basis, and human endpoint.
- A generated video's "effect" is not inside the video alone; it is the relation
  between the stimulus and the observer's current generative model.

So the deeper scientific question becomes:

> Which cognitive effects of media are stable enough to be recovered across
> representation frames, and which are artifacts of a particular model's
> factorisation?

## Mathematical Shape

Let `X` be the space of videos and prompts. We observe several representation
maps:

```text
f_T : X -> V_T      TRIBE / brain-aligned frame
f_J : X -> V_J      V-JEPA / self-supervised video frame
f_C : X -> V_C      CLIP / prompt-preservation frame
h   : X -> Y        human memory behavior
```

The current memorability direction is a scalar readout:

```text
l_T : V_T -> R
```

The submission-grade question is whether the induced ordering
`l_T(f_T(x))` predicts `h(x)` better than comparable readouts in `V_J` and
`V_C`.

The broader theory question is whether there is a latent cognitive functional:

```text
m : X -> R
```

that is approximately recoverable across frames:

```text
m ~= l_T o f_T ~= l_J o f_J ~= h
```

For the social-cohesion follow-on, memorability is replaced by a boundary /
other-regard / cooperation functional. The same machinery applies, but the human
endpoint changes.

## Next Experiments This Suggests

1. **Representation-frame invariance.** For each candidate set, compute TRIBE,
   V-JEPA, CLIP, and human similarity/order matrices. Compare them with RSA/CKA
   and prompt-clustered behavioral outcomes.
2. **Cognitive-effect taxonomy.** Add axes beyond memorability: surprise,
   agency, social warmth, threat/safety, self-other overlap, and action intent.
3. **Boundary-shift stimuli.** For the social-cohesion project, generate or
   curate paired stimuli that differ in whether they reinforce separation,
   cooperation, or shared agency, then collect behavioral and eventually EEG/fMRI
   endpoints.
4. **Selector before steering.** First prove that a readout selects human-visible
   effects. Only then amortize the selector into a generator with LoRA/DPO.
5. **Failure-case geometry.** When TRIBE and V-JEPA disagree, inspect whether the
   difference is semantic, temporal, social, object-centric, or artifact-driven.

## How To Use This In Papers

- Main NeurIPS selector paper: one paragraph in discussion/future work, no
  grand metaphysics.
- Theory/manifesto paper: central frame, with active inference, measurement
  frames, and generated media as cognitive intervention.
- Product deck: "cognitive media selector" that ranks generated candidates by
  predicted memory, attention, and social response before human spend.
