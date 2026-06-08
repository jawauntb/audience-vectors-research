# Research Foundations

This is the claim map for a submission-grade version of the project. The goal
is to cite the right literatures without pretending the current work proves more
than it does.

## 1. Human Memorability Is Stable Enough To Model

Classic image memorability work established that memorability is not merely
idiosyncratic taste. Isola et al. showed that image memorability varies
consistently across observers, and that compact interpretable visual attributes
can explain part of this variation.

Use in paper:

- Justifies treating memorability as a measurable viewer-response axis.
- Supports the phrase "intrinsic memorability" carefully, as population-level
  consistency rather than universal individual response.
- Gives historical grounding for why memory is a meaningful target for content
  selection.

Key citation:

- Isola, Parikh, Torralba, Oliva. "Understanding the Intrinsic Memorability of
  Images." NeurIPS 2011.
  https://papers.nips.cc/paper_files/paper/2011/hash/286674e3082feb7e5afb92777e48821f-Abstract.html

## 2. Video Memorability Has Its Own Datasets And Decay Structure

Memento10k extends memorability to dynamic events and models memory decay over
time. It combines visual and semantic information, making it directly relevant
to short generated videos.

Use in paper:

- Establishes that video memorability is not just image memorability averaged
  over frames.
- Motivates multimodal baselines: visual encoder, caption/text encoder, and
  joint video-text metrics.
- Supports a delayed-recognition human eval as the strongest validation if we
  choose to go beyond pairwise preference.

Key citation:

- Newman et al. "Multimodal Memorability: Modeling Effects of Semantics and
  Decay on Video Memorability." ECCV 2020.
  https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2535_ECCV_2020_paper.php

## 3. BOLD Moments Links Naturalistic Video, fMRI, And Memorability

BOLD Moments sampled 1,102 three-second naturalistic videos from Memento10k and
collected fMRI responses from ten subjects. The dataset includes object, scene,
action, sentence, transcription, memorability score, and memorability decay
metadata. The original paper reports that memorability scores correlate with
brain responses beyond early visual cortex, including parietal cortex.

Use in paper:

- This is the empirical anchor for our brain-aligned memorability direction.
- The paper should clearly distinguish the 1,102 available BMD clips from our
  1,022 analyzable clips after exclusions.
- BMD justifies ROI/fMRI language, but ROI claims must remain exploratory unless
  normalized and cross-subject validated.

Key citation:

- Lahner et al. "Modeling short visual events through the BOLD moments video
  fMRI dataset and metadata." Nature Communications 2024.
  https://www.nature.com/articles/s41467-024-50310-3

## 4. TRIBE Provides A Brain-Encoding Substrate, Not A Human Oracle

TRIBE v2 is a multimodal brain encoding model that predicts fMRI brain responses
to naturalistic video, audio, and text. The 2026 arXiv paper describes it as a
tri-modal foundation model for in-silico neuroscience, trained across large
naturalistic and experimental fMRI corpora.

Use in paper:

- TRIBE is a proxy for population-average fMRI response, not a direct human
  preference oracle.
- The decisive claim should be that a TRIBE-derived signal is useful for
  selecting generated content only after independent human validation.
- TRIBE's value over V-JEPA is not simply higher raw prediction; it is the
  brain-aligned latent space and auditability against fMRI/ROI hypotheses.

Key citation:

- d'Ascoli et al. "A foundation model of vision, audition, and language for
  in-silico neuroscience." arXiv 2026.
  https://arxiv.org/abs/2605.04326

## 4.5. V-JEPA Is The Necessary Non-Brain Video Baseline

V-JEPA 2 is a self-supervised video representation model trained for
understanding, prediction, and planning. It is not brain-aligned, but it is a
strong open video feature space and therefore a fair baseline for asking whether
TRIBE contributes something beyond generic visual-temporal structure.

Use in paper:

- Treat V-JEPA as the strongest current non-brain memorability baseline, not as
  a strawman.
- Phrase the contribution as "brain-aligned selection adds value beyond strong
  video features" only if the human study actually beats V-JEPA.
- Keep the V-JEPA result in the main table even if it ties or beats TRIBE; that
  outcome clarifies the science rather than killing the paper.

Key citation:

- Bardes et al. "V-JEPA 2: Self-Supervised Video Models Enable Understanding,
  Prediction and Planning." arXiv 2025.
  https://arxiv.org/abs/2506.09985

## 5. Video Generation Evaluation Is Already A Crowded Baseline Space

VBench and related benchmarks argue that automatic video-generation evaluation
is difficult because generic metrics do not fully align with human perception.
They decompose video quality into dimensions such as subject consistency,
temporal flicker, motion smoothness, and spatial relationships. FETV similarly
emphasizes fine-grained prompt categories and finds that automatic metrics can
correlate poorly with human judgment.

Use in paper:

- We must compare against standard video quality and text-video alignment
  baselines.
- Our novelty is not "a new generic video metric"; it is a memorability-specific
  brain-aligned selector.
- The paper needs guardrails so the selector does not choose memorable but broken
  videos.

Key citations:

- Huang et al. "VBench: Comprehensive Benchmark Suite for Video Generative
  Models." arXiv 2023.
  https://arxiv.org/abs/2311.17982
- Liu et al. "FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain
  Text-to-Video Generation." NeurIPS Datasets and Benchmarks 2023.
  https://papers.nips.cc/paper_files/paper/2023/hash/c481049f7410f38e788f67c171c64ad5-Abstract-Datasets_and_Benchmarks.html

## 6. Best-of-N And Preference Alignment Are The Right Generation Frame

Best-of-N selection is a test-time alignment method: sample multiple candidates,
score them with a reward/preference model, and return the highest-scoring one.
Diffusion-DPO and VideoDPO adapt preference optimization to image and video
diffusion models, while DenseDPO shows that video preference data can be made
more temporally precise through aligned segment-level comparisons.
Recent trajectory-aware alignment work argues that standard diffusion DPO can
misalign training noise distributions from inference trajectories, which is a
useful caution for our own LoRA/DPO ambitions.

Use in paper:

- Our near-term contribution is selector validation, not full model alignment.
- DPO/LoRA becomes the second paper after the selector is validated.
- Any proxy-labeled DPO must be evaluated against humans to avoid reward-model
  circularity.

Key citations:

- Wallace et al. "Diffusion Model Alignment Using Direct Preference
  Optimization." arXiv 2023.
  https://arxiv.org/abs/2311.12908
- Liu et al. "VideoDPO: Omni-Preference Alignment for Video Diffusion
  Generation." arXiv 2024.
  https://arxiv.org/abs/2412.14167
- Wu et al. "DenseDPO: Fine-Grained Temporal Preference Optimization for Video
  Diffusion Models." NeurIPS 2025 Spotlight.
  https://arxiv.org/abs/2506.03517
- Zhu et al. "Diffusion-APO: Trajectory-Aware Direct Preference Alignment for
  Video Diffusion Transformers." arXiv 2026.
  https://arxiv.org/abs/2605.07503
- Gui et al. "BoNBoN Alignment for Large Language Models and the Sweetness of
  Best-of-n Sampling." NeurIPS 2024.
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/056521a35eacd9d2127b66a7d3c499c5-Abstract-Conference.html

## 7. Representation Frames, Boundaries, And Anti-Reification

The active-inference / free-energy literature gives a useful philosophical
guardrail: an agent accumulates evidence for a generative model, but the
boundaries and factorisations through which evidence is parsed are themselves
model-relative. The quantum-FEP and quantum-frame-problem papers push this into
a strong form: separability and boundaries are not simply self-evident facts
available from within one measurement frame. The 2026 "No self-evidence"
manuscript applies that idea to self/world boundaries and contemplative
realisation; for our purposes, the important lesson is less Buddhist metaphysics
than anti-reification.

Use in paper:

- Treat TRIBE, V-JEPA, CLIP, and human judgments as different measurement or
  representation frames over the same stimulus space.
- Avoid saying a memorability vector is "the" thing memorability is. It is a
  useful readout in a chosen frame.
- Ask whether the ordering induced by a readout is preserved across frames:
  brain-aligned features, self-supervised video features, prompt-alignment
  features, and human behavior.
- Use this to motivate RSA/kernel-alignment style next steps: compare
  representation geometries, not only scalar correlations.
- Keep this material out of the main empirical claim unless framed as
  discussion/future-work context.

Key citations:

- Hohwy. "The Self-Evidencing Brain." Noûs 2016.
  https://research.monash.edu/en/publications/the-self-evidencing-brain/
- Fields, Friston, Glazebrook, Levin. "A Free Energy Principle for Generic
  Quantum Systems." Progress in Biophysics and Molecular Biology 2022.
  https://doi.org/10.1016/j.pbiomolbio.2022.05.006
- Fields, Glazebrook. "Separability, Contextuality, and the Quantum Frame
  Problem." International Journal of Theoretical Physics 2023.
  https://arxiv.org/abs/2304.10010
- Sandved-Smith, Fields, Doctor, Laukkonen, Hohwy. "There is no
  self-evidence: A physics of emptiness realisation." Manuscript/preprint
  supplied by the user, dated May 23, 2026.
- Dahl, Lutz, Davidson. "Reconstructing and deconstructing the self: Cognitive
  mechanisms in meditation practice." Trends in Cognitive Sciences 2015.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4595910/
- Laukkonen, Slagter. "From many to (n)one: Meditation and the plasticity of
  the predictive mind." Neuroscience & Biobehavioral Reviews 2021.
  https://doi.org/10.1016/j.neubiorev.2021.07.021

## The Gap We Can Own

Existing memorability work predicts human memory for natural videos. Existing
video-generation evaluation work scores quality, alignment, and motion. Existing
preference-alignment work learns general human preference. Our wedge is:

> Can a brain-aligned memorability signal improve selection among generated
> videos, and does it generalize beyond standard visual/text-video metrics?

That is narrow enough to test and broad enough to matter.

The larger program is:

> Generated media as controlled intervention on human generative models: which
> stimuli become memorable, which reshape attention, and which alter the
> boundaries people use to parse self, other, object, scene, and action?

That broader frame can guide follow-on projects, but the submission-grade
selector paper should still be decided by independent human memory judgments.
