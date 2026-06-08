# Neural-Response-Guided Generation Literature Memo

Date: 2026-06-08.

Purpose: strengthen the citation base for a future neural-response-guided
generated-video direction while preserving the current claim boundary. The
current content-pocket result is compute-proxy progress: exact V-JEPA and CLIP
can help audit the pocket-regime replay residual, and exact V-JEPA transported
prospectively for orange flowers and hanging clothes. That is not human
memorability validation and not measured-BMD grounding.

## Clean Thesis

Prior work has developed four adjacent lines that are now close enough to make
neural-response-guided generated video credible:

- brain-conditioned image/video reconstruction;
- behavioral video memorability benchmarks;
- media-to-predicted-brain-response scoring with models such as TRIBE v2;
- reward, energy, or saliency guidance for generation.

Our current work should sit in the third line for now. TRIBE, V-JEPA, and CLIP
signals can select, screen, and audit generated-video candidates. Human or
BMD-grounded validation is required before making final memorability claims.

## Boundary Language

| Boundary | What it means | What it does not license |
|---|---|---|
| Brain-to-media reconstruction | Decode or reconstruct a stimulus from measured neural activity, usually fMRI. | It does not prove that a generated candidate selected by a proxy score is more memorable. |
| Media-to-predicted-brain-response scoring | Score an existing or generated stimulus with a learned encoding model such as TRIBE v2. | It is not a measured brain response, not a human judgment, and not a preference oracle. |
| Reward/guidance optimization | Use a score, energy, saliency map, or reward model to guide sampling, optimize latents, or choose candidates. | It validates the optimization pattern, not the target construct unless externally validated. |
| Human memorability validation | Measure recognition, delayed recognition, or forced-choice human judgments against a preregistered endpoint. | This is the layer needed before claiming generated videos became more memorable to people. |

Recommended language:

> V-JEPA, CLIP, and TRIBE-like signals are compute-proxy candidate-selection
> and audit signals. They can prioritize which generated videos deserve human
> or BMD validation, but proxy agreement is not equivalent to behavioral
> memorability or measured neural grounding.

## Main-Paper Citation Candidates

### TRIBE v2

Source: [A foundation model of vision, audition, and language for in-silico
neuroscience](https://arxiv.org/abs/2605.04326).

Contribution: TRIBE v2 is the best citation for the brain-predictive foundation
model substrate. It predicts high-resolution brain responses across video,
audio, and language conditions and frames itself as an in-silico neuroscience
model.

Citation decision: cite in the main paper. Replace older GitHub-only TRIBE
references with the 2026 arXiv citation.

Next-step effect: supports describing TRIBE as a surrogate neural-response
model for scoring and audit. It should not be described as a human memorability
model by itself.

### Energy Guided Diffusion

Source: [Energy Guided Diffusion for Generating Neurally Exciting
Images](https://papers.nips.cc/paper_files/paper/2023/hash/67226725b09ca9363637f63f85ed4bba-Abstract-Conference.html).

Contribution: closest precedent for using a neural encoding model as an
objective inside generation. The paper combines a diffusion prior with an
energy landscape from a neuronal encoding model to synthesize neurally exciting
images.

Citation decision: cite in the main paper related work and in any satellite
about neural-response-guided generation.

Next-step effect: legitimizes a future move from selection to reward or energy
guidance. It does not validate our video-pocket scores or human memorability.

### VideoMem

Source: [VideoMem: Constructing, Analyzing, Predicting Short-term and
Long-term Video Memorability](https://arxiv.org/abs/1812.01973).

Contribution: provides a 10,000-video benchmark with short-term and long-term
memorability annotations and a recognition-style measurement protocol.

Citation decision: cite in the main paper when motivating video memorability as
a behavioral benchmark family.

Next-step effect: good target for transfer checks or dataset-access requests.
Use it as behavioral grounding, not as neuroscience grounding.

### MediaEval 2022 Predicting Video Memorability

Source: [Overview of The MediaEval 2022 Predicting Video Memorability
Task](https://arxiv.org/abs/2212.06516).

Contribution: benchmark context for video memorability prediction, including a
dedicated EEG-based subtask and short-term memorability emphasis.

Citation decision: cite in the main paper if discussing benchmark ecosystem and
potential EEG-adjacent validation; otherwise keep as background for evaluation
planning.

Next-step effect: reminds us that the field already distinguishes behavioral
scores, EEG subtasks, and dataset generalization. That distinction should shape
our validation packet.

### MindMem

Source: [MindMem: Multimodal for Predicting Advertisement Memorability Using
LLMs and Deep Learning](https://arxiv.org/abs/2502.18371).

Contribution: applied multimodal advertisement/video memorability predictor
with commercial content framing and Memento10K evaluation.

Citation decision: cite lightly in related work or a satellite paper. Treat as
behavioral/commercial-adjacent, not as neuroscience validation.

Next-step effect: useful for positioning generated-video memorability as
applied and product-relevant. It should not be used to claim BMD grounding.

### GazeFusion

Source: [GazeFusion: Saliency-Guided Image
Generation](https://arxiv.org/abs/2407.04191).

Contribution: saliency-guided image generation with eye-tracked user validation
and model-based saliency analysis. It shows how predicted human attention can
condition generation.

Citation decision: cite in the main paper or satellite work when discussing
attention/saliency-guided generation.

Next-step effect: supports a pattern where human-response predictors guide
generation, but it concerns gaze/attention, not memorability or BMD.

### SGOOL

Source: [Saliency Guided Optimization of Diffusion
Latents](https://arxiv.org/abs/2410.10257).

Contribution: saliency-guided latent optimization pattern for diffusion models,
using saliency to focus optimization on regions viewers are likely to attend.

Citation decision: cite in satellite/future-work framing; include in main paper
only if space allows a broader guidance paragraph.

Next-step effect: reinforces that test-time or latent-space optimization can be
guided by predicted viewer-response maps. It is not neural-response validation.

## Satellite Or Background Citations

### Mind-Video

Source: [Cinematic Mindscapes: High-quality Video Reconstruction from Brain
Activity](https://arxiv.org/abs/2305.11675).

Contribution: fMRI-to-video reconstruction using masked brain modeling,
contrastive learning, Stable Diffusion augmentation, temporal inflation, and
adversarial guidance.

Citation decision: background in main paper; cite in a future satellite's
brain-to-video reconstruction boundary section.

Next-step effect: helps separate reconstructing what a subject watched from
scoring newly generated media.

### NeuroClips

Source: [NeuroClips: Towards High-fidelity and Smooth fMRI-to-Video
Reconstruction](https://arxiv.org/abs/2410.19452).

Contribution: high-fidelity and smooth fMRI-to-video reconstruction with
semantic and perceptual decoders and video diffusion rendering.

Citation decision: background in main paper; stronger citation in satellite
reconstruction context.

Next-step effect: useful for positioning "brain-to-media reconstruction" as a
separate literature from our selector/audit work.

### NeuroCine

Source: [NeuroCine: Decoding Vivid Video Sequences from Human Brain
Activities](https://arxiv.org/abs/2402.01590). The handoff listed this link as
"NeuralFlix"; the primary source title resolves to NeuroCine.

Contribution: fMRI-to-video reconstruction with temporal interpolation,
noise/lag handling, and dependent prior noise in diffusion generation.

Citation decision: background only unless a satellite needs a reconstruction
taxonomy.

Next-step effect: another example of measured-brain-to-video reconstruction,
not media-to-predicted-brain-response scoring.

### SemVideo

Source: [SemVideo: Reconstructs What You Watch from Brain Activity via
Hierarchical Semantic Guidance](https://arxiv.org/abs/2602.21819).

Contribution: hierarchical semantic guidance for fMRI-to-video reconstruction,
using static anchor descriptions, motion narratives, and holistic summaries to
improve coherence.

Citation decision: background in main paper, satellite citation if discussing
semantic guidance for reconstruction.

Next-step effect: suggests a future validation packet could separate content,
motion, and holistic summary descriptors, but it does not change the current
SVD content-pocket path.

### MindDiffuser

Source: [MindDiffuser: Controlled Image Reconstruction from Human Brain
Activity with Semantic and Structural Diffusion](https://arxiv.org/abs/2303.14139).

Contribution: fMRI-derived CLIP and latent features guide controlled image
reconstruction with semantic and structural constraints.

Citation decision: cite as background if the paper discusses fMRI-to-diffusion
guidance; otherwise keep for satellite framing.

Next-step effect: useful precedent for CLIP/latent guidance from brain-derived
signals, but it is image reconstruction rather than generated-video selection.

## Optional Or Held-Aside Sources

- [UniBrain](https://arxiv.org/abs/2308.07428): background only if discussing
  unified image reconstruction and captioning from fMRI.
- [Brain-Streams](https://arxiv.org/abs/2409.12099): background only if
  discussing semantic/perceptual stream decomposition in reconstruction.
- [AGGAN](https://arxiv.org/abs/1903.12296): skip from the main paper. It is
  older attention-guided GAN background; GazeFusion and SGOOL fit the current
  diffusion/saliency story better.
- WIRED MemNet article: do not cite formally. Use primary image memorability
  papers such as Isola et al. instead.

## Program Impact

Main paper:

- Cite TRIBE v2 as a brain-predictive model, VideoMem/MediaEval as behavioral
  video memorability context, and Energy Guided Diffusion/GazeFusion as
  response-guided generation precedents.
- Keep the main claim narrow: a brain-aligned feature space defines a testable
  generated-video selection workflow.

Satellite paper:

- A credible satellite track is now "Neural-response-guided generated video":
  score or guide generated candidates with TRIBE-like neural-response models,
  V-JEPA/CLIP preservation or audit gates, and eventually human/BMD validation.
- The satellite should explicitly compare selection, latent optimization, and
  generator fine-tuning rather than treating them as interchangeable.

Immediate experiments:

- Do not redirect into broad prompt search or alpha/guidance sweeps.
- Continue descriptor-conditioned replication and validation-packet planning
  around orange flowers and hanging clothes.
- Keep aerial beach, city street, and storm beach as hard negatives.
- Treat exact V-JEPA and CLIP as compute-proxy verifiers for the current
  residual, with the prospective claim narrowed to exact V-JEPA for fresh
  orange-flower/hanging-clothes replication unless CLIP is revalidated.

Claim ledger language to preserve:

- Accepted: exact V-JEPA and CLIP are compute-proxy verifiers for the current
  pocket-regime replay residual.
- Accepted: exact V-JEPA transported prospectively for orange flowers and
  hanging clothes.
- Not accepted: proxy scores alone establish behavioral memorability.
- Not accepted: TRIBE, V-JEPA, or CLIP validate measured-BMD grounding for the
  generated candidates.
