# Neural-Response-Guided Generation Feasibility

Date: 2026-06-08

Status: scoped future-work memo. This is a parallel satellite track, not a
replacement for the current content-pocket validation lane.

## Decision

Neural-response-guided generation is feasible as a candidate-selection and
reward-guidance research track, but the safe next move is a no-generation dry
run over existing videos. The track should first compare proxy signals as
rankers and gates before launching any optimizer that spends generation budget.

This memo does not change `CLAIM_LEDGER.md`. It does not claim that TRIBE v2,
V-JEPA, CLIP, memorability predictors, or saliency predictors are validated
human reward models.

## Available Signals

| signal | local status | expected inputs | usable output | differentiability |
|---|---|---|---|---|
| TRIBE/BMD projection | Operational through `TribeService` and replay scoring. `scripts/modal_bo_memorability_replay.py` calls TRIBE on generated MP4s and projects mean TRIBE frames with `score_projection`. | MP4 bytes or Modal-visible video path; unit BMD memorability direction artifact. | Scalar proxy memorability score for reranking and reporting. | Practical black box. TRIBE and the projection are differentiable in principle, but the deployed path exposes scoring, not gradients through a generator. |
| Exact V-JEPA embeddings | Operational for exact MP4s through `VjepaService` and extraction scripts such as `scripts/extract_pocket_replay_vjepa.py`. | MP4 bytes or path; V-JEPA Modal app. | Video embedding usable for centroid margins, classifiers, and agreement audits. | Practical black box. Use as an embedding verifier or disagreement signal, not as a validated reward. |
| CLIP seed/video/prompt similarities | Operational locally in `scripts/audit_content_pocket_embeddings.py` and `scripts/score_wan22_composite_preservation.py`. | Seed image, generated MP4 frames, optional prompt text. | Seed-image preservation, generated-video embedding margins, prompt similarity, and composite guardrails. | Differentiable inside local PyTorch CLIP calls, but practical use is black-box reranking/gating because generator gradients are not exposed. |
| Visual artifact gate | Operational in `audience_vectors.visual_artifact_gate` and replay reports. | Generated MP4. | Blocking quality/validity filter. | Non-differentiable guardrail. Not a neural-response signal, but mandatory before spending human/BMD budget. |
| Behavioral memorability predictors | Citation-level only in this repo for VideoMem, MediaEval, MindMem, and adjacent work. | Would require model weights, dataset licenses, or new predictor implementation. | Possible external benchmark or additional reranking score. | Unknown until a specific predictor is implemented. Not available for an immediate local spike. |
| Saliency/gaze predictors | Citation-level only through GazeFusion/SGOOL-style references. | Would require a chosen saliency model and frame/clip preprocessing. | Possible attention map, attended-region score, or saliency-preservation gate. | Some saliency models are differentiable, but no local guidance implementation exists now. |
| Measured-BMD readout | Conceptually available as a validation endpoint, not as a generator reward in this branch. | Frozen MP4 stimulus set plus measured-BMD or BMD-grounded scoring protocol. | Validation evidence if run through the proper BMD/measured-brain gate. | Not a generation-time reward unless a dedicated measured-BMD scorer is exposed. |

## Pixel-Affecting Controls

Current SVD replay path:

- `seed_idx` selects the seed image and is a real content intervention.
- `noise_seed` and replicate scheduling change stochastic samples.
- `alpha` changes the CLIP-image conditioning patch in `SVDGenerator.generate`
  when a steering vector is supplied.
- `guidance_scale`, `num_frames`, `num_inference_steps`, `motion_bucket_id`,
  and `noise_aug_strength` are passed into Stable Video Diffusion and can affect
  pixels.
- `fps` changes the encoded playback rate. Treat it as output formatting unless
  an analysis depends on temporal sampling.
- Prompt text in the current SVD replay runner is metadata. `SVDGenerator`
  does not accept a prompt, and `modal_bo_memorability_replay.py` does not pass
  one. Prompt rewriting is blocked in this path.

Prompt-conditioned generator paths:

- `Wan22Generator.generate` consumes prompt text, optional seed image, task,
  size, frame count, sample steps, guide scale, shift, seed, and prompt
  extension options.
- `CogVideoXGenerator.generate` consumes prompt text, optional T5-space
  steering vector, alpha, frame count, steps, guidance, and seed.
- Veo prompt demos exist as prompt-conditioned experiments, but they are a
  separate path from the SVD content-pocket runner.

Not currently exposed:

- Gradient access through the generator to optimize pixels, latents, or prompt
  embeddings against TRIBE/V-JEPA/CLIP.
- Saliency-guided latent optimization.
- A validated human-memorability reward model.

## Feasible Loops

### Candidate Reranking

Feasible now. Generate or reuse a candidate pool, score each clip with
TRIBE/BMD, V-JEPA, CLIP preservation, and the visual gate, then compare rankers.
This is the strongest immediate loop because it does not require gradients or
new generator plumbing.

Allowed wording: "proxy reranking" or "candidate-selection dry run."

Blocked wording: "validated memorability optimization."

### No-Generation Agreement Dry Run

Feasible now if the local data lake has the frozen MP4s or other generated
clips. Score existing candidates only, then report where TRIBE, V-JEPA, CLIP,
and artifact gates agree or disagree. This is the safest first spike because it
cannot accidentally optimize into reward-model loopholes.

The frozen content-pocket validation set can be an evaluation target, but the
dry run must remain separate from the human/BMD validation claim.

### Black-Box Candidate Search

Feasible with existing SVD/Wan/CogVideoX wrappers, but should wait until the
dry run shows that a composite proxy would make materially different selections
than TRIBE alone. Search can use Sobol, BO, or evolutionary selection over
candidate generations.

For current SVD, valid search variables are seed image, noise seed, alpha,
guidance, motion/noise/step controls, and recipe neighborhoods. Prompt search is
not valid in the current SVD runner.

### Prompt Search

Feasible only on prompt-conditioned generation paths. Use Wan2.2, CogVideoX, or
Veo if prompt text is the intended intervention. Do not run a prompt-rewrite
tournament in the current SVD replay path.

### Latent Or Guidance Optimization

Not feasible now as a local repo loop. SVD and CogVideoX expose coarse steering
parameters, but not gradients or latent tensors for iterative reward
optimization. Energy-guided diffusion and saliency-guided optimization are
literature precedents, not implemented capabilities in this repo.

### Reward Distillation

Not safe as a next step. Distillation or DPO would need a validated selector,
held-out human/BMD evaluation, preservation gates, and reward-hacking audits.
The current satellite can produce proxy pairs or ranker comparisons, but not a
training claim.

## Smallest Safe Spike

Build a no-generation dry-run report over an existing candidate set.

Recommended source pool:

- primary option: the frozen content-pocket task videos from
  `content_pocket_validation_pairwise_tasks_20260608.json`;
- fallback option: any already generated and scored SVD/Wan candidate pool with
  local MP4s available.

Recommended outputs:

- `neural_response_guided_generation_dry_run_20260608.json`;
- `neural_response_guided_generation_dry_run_20260608.md`.

Minimum procedure:

1. Load the frozen task manifest and verify every referenced MP4 exists.
2. Record visual-gate status and manual-screening status if available.
3. Attach or compute TRIBE/BMD projection scores for every selected and control
   clip.
4. Attach or compute exact V-JEPA embeddings for the same MP4 bytes.
5. Attach CLIP seed-video and prompt-video similarities where seed images and
   prompts are available.
6. Compare single-signal rankers and a preregistered composite:
   `z(TRIBE/BMD) + w_vjepa * z(V-JEPA margin) + w_clip * z(CLIP preservation)`,
   with all weights written before inspecting winners.
7. Report agreement, disagreement, and changed selections. Do not call the
   result validation.

Stop immediately if the run would require regenerating videos, changing frozen
stimuli, adding new claim-ledger language, or interpreting proxy wins as human
memorability.

## Stop Rules And Gates

- No human-memorability language without a human or measured-BMD gate.
- No "TRIBE/V-JEPA/CLIP reward model" language. Use "proxy signal," "ranker,"
  "audit frame," or "candidate-selection signal."
- No prompt-rewrite experiments in SVD unless prompt text reaches the generator.
- No pooling boundary arms into the primary content-pocket validation claim.
- No changing `CLAIM_LEDGER.md` unless a real accepted claim changes.
- Freeze proxy weights, candidate inclusion rules, and visual-gate thresholds
  before scoring a new dry run.
- Preserve failed, rejected, and withheld candidates. Disagreement is part of
  the feasibility result.
- Require manual MP4 screening before any human/BMD launch, even if the proxy
  composite looks strong.

## Launch Blockers

- The clean worktree does not include the local `data/` lake. A dry-run script
  must either run from a checkout with generated MP4s/features available or
  receive explicit data-lake paths.
- TRIBE and V-JEPA scoring depend on deployed Modal app availability and the
  correct BMD direction/feature artifacts.
- CLIP scoring requires the local ML dependency stack and enough memory to
  embed sampled frames.
- Prompt search needs a prompt-conditioned generator path. The current SVD
  runner is not enough.
- Latent optimization needs new generator plumbing and a separate safety gate.

## Relation To Content-Pocket Validation

The content-pocket validation lane remains:

```text
manual MP4 screening -> hosted videos -> human forced-choice or measured-BMD
validation on the frozen 24-task stimulus set
```

This satellite track can score or rerank the same frozen videos only as a
proxy-analysis dry run. It must not modify the frozen stimulus set, replace the
human/BMD gate, or change the content-pocket claim language.

If the dry run is useful, the next decision is whether a composite proxy screen
should help choose future candidate pools before human/BMD validation. The dry
run itself is not validation.
