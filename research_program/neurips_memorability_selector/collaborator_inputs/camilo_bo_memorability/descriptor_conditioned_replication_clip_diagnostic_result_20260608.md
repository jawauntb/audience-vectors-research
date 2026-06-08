# Descriptor-Conditioned Replication CLIP Diagnostic

Date: 2026-06-08

## Question

After the descriptor-conditioned replication partially passed through TRIBE and
exact V-JEPA but not generated-video CLIP, is the CLIP gap explained by the
initial CLIP readout configuration, or should generated-video CLIP remain a
non-accepted prospective verifier for this fresh replication?

## Discovery-Regime Audit

Current regime:

- Artifact types: restored seed images, prompt metadata, generated SVD replay
  videos, TRIBE scores, exact V-JEPA features, CLIP seed/video/text embeddings,
  pocket labels, centroid-margin descriptors, and leakage-aware classifier
  outputs.
- Operations: re-embed the existing generated MP4s with CLIP, increase sampled
  frames from 4 to 8, include prompt-text CLIP similarities, reuse the existing
  exact V-JEPA features, and rerun the same descriptor/classifier gate.
- Gates/verifiers: embedding descriptors do not use TRIBE score as an input.
  Descriptor acceptance remains `separation_auc >= 0.85` and `abs_cohen_d >=
  1.00`; classifier acceptance remains leave-one-pocket-out `roc_auc >= 0.85`
  and `balanced_accuracy >= 0.75`.
- Known limitation: this is still compute-proxy SVD replay evidence. It is not
  human memorability, measured-BMD validation, delayed recognition, or
  prompt-conditioned generation.

Action class: diagnostic search inside the accepted SVD content-pocket regime.
No new videos were generated and no TRIBE scores were recomputed.

## Inputs

- Replay report:
  `data/reports/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608.json`
- Generated videos:
  `data/generated/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608`
- Exact V-JEPA features:
  `data/features/vjepa_descriptor_conditioned_replication_20260608`
- Output summary:
  `descriptor_conditioned_replication_clip_diagnostic_summary_20260608.json`

Command:

```bash
uv run python scripts/audit_content_pocket_embeddings.py \
  --replay-report /Users/jawaun/.codex/worktrees/descriptor-conditioned-replication-run-20260608/isc_mod/data/reports/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608.json \
  --seed-root /Users/jawaun/.codex/worktrees/descriptor-conditioned-replication-run-20260608/isc_mod/research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/original \
  --vjepa-features-dir /Users/jawaun/.codex/worktrees/descriptor-conditioned-replication-run-20260608/isc_mod/data/features/vjepa_descriptor_conditioned_replication_20260608 \
  --include-text \
  --max-video-frames 8 \
  --out-json research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/descriptor_conditioned_replication_clip_diagnostic_summary_20260608.json \
  --out-md research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/descriptor_conditioned_replication_clip_diagnostic_result_20260608.md
```

## Result

The diagnostic audited 30 task-level candidates from 90 scored replication
rows.

| family | feature | positive mean | negative mean | AUC | abs d | r(score) | result |
|---|---|---:|---:|---:|---:|---:|---|
| `vjepa_video` | pocket-held-out centroid margin | 0.0750 | -0.1371 | 1.0000 | 2.8636 | 0.8380 | accepted |
| `clip_prompt_seed` | prompt-seed CLIP cosine | 0.3290 | 0.2939 | 1.0000 | 2.6251 | 0.7923 | ancillary descriptor accepted |
| `clip_seed_image` | pocket-held-out centroid margin | -0.0104 | -0.0996 | 0.8333 | 1.4599 | 0.6038 | not accepted |
| `clip_seed_video` | seed-video CLIP cosine | 0.9815 | 0.9726 | 0.7685 | 1.0653 | 0.3783 | not accepted |
| `clip_prompt_video` | prompt-video CLIP cosine | 0.3340 | 0.3107 | 0.7130 | 1.1498 | 0.5187 | not accepted |
| `clip_video` | pocket-held-out centroid margin | -0.0212 | -0.0970 | 0.6667 | 0.9884 | 0.4608 | not accepted |

Classifier support:

| family | validation | AUC | balanced accuracy | result |
|---|---|---:|---:|---|
| `vjepa_video` | leave-one-pocket-out | 1.0000 | 1.0000 | accepted |
| `clip_seed_image` | leave-one-pocket-out | 0.8333 | 0.7500 | not accepted |
| `clip_video` | leave-one-pocket-out | 0.8333 | 0.5833 | not accepted |

## Interpretation

Exact V-JEPA remains the only accepted generated-video prospective verifier for
the fresh descriptor-conditioned replication.

Increasing CLIP frame sampling from 4 to 8 and adding prompt-text similarities
did not rescue generated-video CLIP. The generated-video CLIP centroid margin
remained below the descriptor gate (`AUC 0.6667`, `abs d 0.9884`), and the
generated-video CLIP leave-one-pocket-out classifier still missed the balanced
accuracy gate (`0.5833`).

The new passing CLIP signal is `clip_prompt_seed`, not `clip_video`. That means
CLIP can help explain or pre-screen prompt/seed alignment for this pocket set,
but it should not be described as a prospective generated-video verifier. It is
an ancillary seed/prompt descriptor, not evidence that generated-video CLIP
transported on the fresh stochastic replication.

## Gate Result

Resolved CLIP diagnostic:

- accepted: exact V-JEPA generated-video verifier;
- accepted as ancillary: CLIP prompt-seed alignment descriptor;
- not accepted: generated-video CLIP prospective verifier.

## Claim Impact

This preserves the narrowed C-019 language: orange flowers and hanging clothes
are fresh-seed TRIBE/V-JEPA-verified compute-proxy candidate pockets, not
two-descriptor V-JEPA+generated-video-CLIP verified pockets.

The diagnostic supports proceeding to a human/BMD validation packet with a
V-JEPA-primary compute screen, while recording the CLIP prompt-seed descriptor
as a seed/prompt sanity check and preserving generated-video CLIP
non-replication as a limitation.

Do not claim human memorability, measured-BMD grounding, delayed recognition,
or prompt-conditioned generation from this diagnostic.
