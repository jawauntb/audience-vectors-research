# Persona Vectors And Computational Intervention Notes

**Status:** future-work note, started 2026-06-02.
**Use in main paper:** inspiration and framing only. Do not cite this as
evidence for the memorability selector until it has its own experiments.

## Why This Is Relevant

Spencer's critique is useful: TRIBE can be an interface to brain-like media
features, but it is still a learned middleman. Persona Vectors is useful for a
different reason: it gives a clean recipe for turning natural-language traits
into contrastive activation directions that can be monitored, steered, and used
to screen training data.

The bridge is:

```text
Persona Vectors:
trait description -> contrastive prompts -> activation direction -> monitor / steer / filter

Memorability selector:
content axis -> contrastive stimuli -> brain/video direction -> score / select / audit
```

Our current BMD direction is not a model-persona vector. It is closer to a
content-response readout in a brain-aligned representation. The shared idea is
that a high-level behavioral or cognitive property can be operationalized as a
direction, then stress-tested by prediction, intervention, and data filtering.

## Guardrail

The "computational drugs" phrase should be treated as an analogy for controlled
activation-space or network-space interventions. This is not drug-design work,
not a clinical claim, and not a plan for synthesizing biological compounds.

## Research Leads From The Conversation

1. **LLM computational interventions.** Start inside open LLMs: build persona
   vectors for prosociality, coercion, de-escalation, agency support,
   sycophancy, and hallucination. Test monitoring, inference-time steering,
   preventative steering, and projection-difference data screening.

2. **Remove the middleman.** Use image-paired EEG or fMRI datasets to learn
   direct stimulus-to-brain fingerprints for affective or social scenarios.
   Compare those fingerprints with TRIBE/BMD and V-JEPA readouts. If the EEG
   direction predicts human judgments directly, TRIBE becomes one frame among
   several rather than the privileged interface.

3. **TRIBE as a linker, not an oracle.** Treat TRIBE as one representation frame
   that can link audiovisual content to cortical-response predictions. The
   right validation question is whether orderings survive frame transfer:
   TRIBE -> EEG/fMRI -> human behavior.

4. **Network interventions.** The computational-drug analogy is strongest when
   phrased as network pharmacology: change a distributed network state with a
   small intervention and measure downstream effects. In LLMs the intervention
   can be an activation vector or training-time steering. In biological systems
   it would require real experimental perturbation and domain supervision.

5. **Side project candidate.** Build a small "prosocial media intervention"
   prototype: given dialogue/video clips, score coercion, de-escalation,
   prosociality, and hallucination pressure with LLM persona vectors and compare
   those scores with TRIBE/V-JEPA memorability and affect features.

## Concrete Next Experiments

1. Implement the Persona Vectors extraction pipeline for Qwen or Llama using
   traits relevant to this program: prosocial, coercive, sycophantic,
   hallucination-prone, de-escalating, and agency-supporting.

2. Run a projection-difference screen over dialogue or transcript datasets to
   find samples likely to induce coercion, sycophancy, or hallucination after
   finetuning.

3. Decompose those persona directions with SAEs where available, separating
   stylistic features from causal content features.

4. Compare clip-level TRIBE/BMD scores, V-JEPA scores, transcript persona-vector
   projections, and EEG/fMRI fingerprints on a matched set of videos or
   conversations.

5. Only after the above works, consider a broader "computational interventions"
   paper: not computational pharmacology, but a general framework for safe,
   measurable interventions on model and media representations.

## Useful Sources To Track

- [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509)
  establishes the activation-vector monitor/steer/filter pattern for LLM traits.
- [Human EEG recordings for 1,854 concepts presented in rapid serial visual presentation streams](https://www.nature.com/articles/s41597-021-01102-7)
  is a large image-paired EEG anchor based on THINGS concepts.
- [A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758)
  connects high-temporal-resolution EEG with visual object recognition and
  model-representation analyses.
- [The fly connectome reveals a path to the effectome](https://www.nature.com/articles/s41586-024-07982-0)
  is the strongest perturbative-connectomics bridge for the Drosophila part of
  Spencer's framing.
- [Network pharmacology: the next paradigm in drug discovery](https://www.nature.com/articles/nchembio.118)
  is the clean analogy source for treating interventions as distributed network
  state changes rather than single-target toggles.

## Main-Paper Boundary

This material belongs in future work or a theory note unless we run new
experiments. The memorability-selector paper should stay focused on:

- BMD/TRIBE readout quality;
- V-JEPA/CLIP/quality baselines;
- fold-safe TRIBE-internal patch sensitivity;
- human validation of generated-video selection.
