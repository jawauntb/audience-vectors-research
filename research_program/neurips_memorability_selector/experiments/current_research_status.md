# Current Research Status

Last updated: 2026-06-08

## Confirmed Enough To Treat As Real

- A supervised TRIBE/BMD memorability direction predicts held-out BMD
  memorability at roughly `rho ~= +0.40`.
- BOLD Moments human memorability labels are the ground-truth behavioral target
  for the core `v_mem` direction.
- The completed Prolific best-of-N study supports the broader TRIBE/BMD selector
  signal: 41 raters, 41 passed attention checks, and Study A found humans chose
  the TRIBE-ranked best-of-N winner over a within-seed median variant 290/451
  times, or 64.3% (Wilson 95% CI [0.598, 0.686], binomial p = 1.3e-9).
- Persona vectors are structured, but the Spencer critique was right: they are
  not independent orthogonal axes. Signed cosine structure implies a smaller set
  of shared latent directions.
- Best-of-N and base-or-gated selection are the most practical current workflow.
  Direct continuous steering is not solved.
- The current Wan proxy run is promising but still proxy-only:
  preference-weighted single LoRA improves 20/24 prompts, and base-or-gated
  best-of-4 improves 18/24 prompts under the TRIBE/BMD proxy.
- V-JEPA is now an active baseline for the same current-pilot candidate pool:
  103 unique V-JEPA embeddings, 24/24 seeds covered, 0 missing features.
- BO/SVD generated-video evidence is currently a compute-proxy regime result,
  not a human-memorability result. The latest regenerated-control,
  prompt-transfer, per-prompt Sobol, content-axis audit, restored seed-bank, and
  pocket regime-audit panels show content-pocket behavior: orange flowers and
  hanging clothes are stable non-jellyfish positive pockets, blue jellyfish
  remains positive, fireworks is visually brittle, and seed-image/content slot
  explains retained SVD replay score variance far better than alpha/guidance
  recipe identity. A follow-up lightweight feature audit did not explain the
  positive pockets strongly enough to accept a descriptor verifier: the best
  near-miss was seed-image colorfulness (AUC 0.8333, abs d 1.8471), below the
  pre-registered AUC >= 0.85 gate. Stronger exact V-JEPA and CLIP embedding
  audits did clear the descriptor-explanation gate. V-JEPA features were
  extracted for 84/84 exact pocket-regime replay MP4s; V-JEPA video centroid
  margin reached AUC 1.0000, abs d 3.2953, and r(score) 0.8871, with
  leave-one-pocket-out classifier balanced accuracy 0.9722. CLIP seed/video
  centroid margins also passed. A descriptor-conditioned replication then
  generated and scored 90/90 fresh-seed SVD clips with 0 visual failures. Orange
  flowers and hanging clothes stayed positive across every retained row, hard
  controls stayed negative, and exact V-JEPA transported prospectively with
  centroid-margin AUC 1.0000 / abs d 2.8636 and leave-one-pocket-out balanced
  accuracy 1.0000. Generated-video CLIP did not replicate prospectively
  (centroid-margin AUC 0.6667, classifier balanced accuracy 0.5833). A targeted
  CLIP diagnostic with 8 sampled frames and prompt-text similarities preserved
  that boundary: generated-video CLIP still failed, while prompt-seed CLIP
  cosine passed only as an ancillary seed/prompt descriptor. This gives the two
  strongest pockets a TRIBE/V-JEPA compute-proxy replication, but human/BMD
  validation remains open and CLIP should not be described as a fresh
  prospective generated-video pass. A follow-up blue jellyfish / old car
  boundary audit generated and scored 90/90 fresh-seed clips with 0 visual
  failures. Both boundary pockets stayed positive and hard controls stayed
  negative, but the verifier split was different: CLIP-side boundary checks
  passed while exact V-JEPA did not clear the pre-registered boundary gate. The
  human/BMD validation packet is therefore tiered: orange flowers and hanging
  clothes are primary TRIBE/V-JEPA candidates; blue jellyfish and old car are
  secondary exploratory boundary arms. The exact validation stimulus set is now
  frozen at 24 pairwise tasks over 45 unique MP4 paths, with no missing selected
  files or missing matched controls. Agent sampled-frame MP4 pre-screening then
  checked all 45 selected stimuli with byte/hash/frame-gate verification,
  produced four contact sheets, and found zero automated failures. Codex
  contact-sheet review found retained candidate subjects and visually distinct
  hard-negative controls, with no obvious sampled-frame text/watermark, frame
  collapse, or attention-check leakage. Final human/IRB-facing screening,
  hosted URLs, and validation still remain open, so this does not change the
  claim status. The stronger two-session recognition-memory path is now further
  along: the flagged filler lure was preserved as rejected evidence, replaced
  with a ceramic-teacups filler pair, refreshed video screening passed 60/60
  generated recognition MP4s, and public GitHub Pages launch assets plus
  Prolific setup notes are ready. No delayed-recognition data have been
  collected yet.

## Newly Built V-JEPA Pilot State

- Augmented manifest:
  `research_program/neurips_memorability_selector/experiments/current_selector_manifest_with_vjepa.json`
- V-JEPA score report:
  `research_program/neurips_memorability_selector/experiments/vjepa_selector_report.json`
- Augmented pairwise tasks:
  `research_program/neurips_memorability_selector/experiments/current_selector_pairwise_tasks_with_vjepa.json`
- Augmented survey:
  `research_program/neurips_memorability_selector/experiments/current_selector_prolific_survey_with_vjepa.html`

Coverage:

```text
unique V-JEPA embeddings: 103
seeds with V-JEPA-selected video path: 24/24
augmented pairwise tasks: 185
missing local video paths in task file: 0
```

Selector overlap:

```text
V-JEPA equals product selector: 7/24 seeds
V-JEPA equals gated selector: 10/24 seeds
```

This is exactly what we want for a useful baseline: V-JEPA is neither identical
to TRIBE/product nor obviously irrelevant.

## Not Yet Proven

- We have not shown that the newer V-JEPA-adjudicated TRIBE-selected generated
  videos are more memorable to humans.
- We have not shown that TRIBE beats V-JEPA, CLIP, or quality baselines in the
  current independent generated-video selector pilot.
- We have not shown that Arthur/Camilo's BO-generated videos improve human
  memorability. Their BO work is compute/control evidence until human-tested,
  and current SVD broadening should target seed-image/content expansion or a
  prompt-conditioned generator path rather than more alpha/guidance-only search.
- We have not validated the stable positive SVD content pockets with human
  behavior or measured-BMD alignment. Orange flowers and hanging clothes now
  have fresh-seed TRIBE/V-JEPA compute-proxy replication, but generated-video
  CLIP did not clear the prospective verifier. Blue jellyfish and old car passed
  fresh TRIBE and CLIP-side boundary checks, but not the exact V-JEPA boundary
  verifier. The validation packet, exact MP4 stimulus freeze, agent
  sampled-frame pre-screen, recognition-memory design, recognition stimulus
  production manifest, and recognition seed-image materialization/screening
  artifacts are assembled. The one flagged filler-lure video was preserved and
  replaced, refreshed screening passed 60/60 generated recognition MP4s, and
  public HTTPS launch assets are ready for Prolific setup. Final human/IRB-facing
  review and delayed Session 2 data remain open, so no human/BMD gate has run.
- We have not shown delayed-recognition memory gains.
- We have not shown that a LoRA or DPO model has actually learned
  memorability; the present model-side results are selector/proxy evidence.
- We have a fold-safe TRIBE-internal hidden-direction intervention on 104
  balanced high/low clips. We have not yet shown a full population-level causal
  mechanism with content stratification and matched-control patches.

## Submission-Critical Next Step

Run the V-JEPA-augmented blinded human pilot. The key tests are:

- `product_vs_vjepa_memorability`
- `gated_vs_vjepa_memorability`
- `product_vs_clip_preservation`
- `gated_vs_clip_preservation`
- `product_vs_base`
- `gated_vs_base`

If TRIBE/product or gated selection loses to V-JEPA, the paper should become a
more honest "brain-aligned and self-supervised video features both expose
memorability-like signals" paper. If it wins, the paper has a strong main
claim: brain-aligned features improve generated-video selection for a cognitive
property beyond standard video-feature baselines.

## After The Pilot

1. Scale from 24 prompts to 50-100 fresh prompts with frozen selector policies.
2. Add a standard video-quality metric or VBench-style score so reviewers cannot
   say the selector only finds artifact-heavy memorable clips.
3. Add a representation-frame analysis: compare TRIBE, V-JEPA, CLIP, and human
   pairwise similarity/order matrices with RSA or CKA, especially on prompts
   where the selectors disagree.
4. Run prompt-clustered bootstrap and mixed-effects logistic analysis.
5. Only then decide whether LoRA/DPO distillation is worth the budget.

## Content-Pocket Validation Packet

The SVD content-pocket validation packet is ready as a separate compute-proxy
artifact:

- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_human_bmd_validation_packet_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_validation_stimuli_manifest_20260608.json`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_validation_pairwise_tasks_20260608.json`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_validation_prolific_survey_20260608.html`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_validation_mp4_screening_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_validation_hosted_video_url_map_template_20260608.json`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_memory_design_20260608.json`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_memory_packet_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_stimulus_production_manifest_20260608.json`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_stimulus_production_manifest_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_materialization_20260608.json`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_materialization_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_screening_result_20260608.json`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_screening_result_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_seed_screening_sheets_20260608/`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_video_generation_result_20260608.json`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_video_generation_result_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_video_screening_20260608.json`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_video_screening_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_video_screening_sheets_20260608/`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_launch_assets_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_hosted_video_url_map_20260608.json`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_recognition_prolific_setup_20260608.md`

Use it only for a human/BMD validation launch. For a direct actual-memory claim,
the stronger path is the two-session recognition-memory packet. The production
manifest now enumerates 60 required seed images and 60 SVD jobs; current status
is `recognition_launch_assets_ready_for_prolific_setup`. The seed-image subgate
passed, the flagged filler lure was preserved and replaced, refreshed video
screening passed 60/60 generated MP4s, and public GitHub Pages/Prolific setup
artifacts are ready. The older forced-choice survey is a weaker
perceived-memorability readout. This does not change the claim boundary: actual
recognition-memory evidence still requires final human/IRB-facing review and
delayed Session 2 data.
