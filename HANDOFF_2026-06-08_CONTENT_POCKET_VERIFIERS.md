# Handoff: Content-Pocket Verifiers And Next Replication

Date: 2026-06-08

Use this note when starting a fresh agent session on the current content-pocket
research thread. It is intentionally focused on what changed, what is now
claimable, and what experiment should run next.

## Quick Start

Read these first, in order:

1. `START_HERE.md`
2. `NEXT_STEPS.md`
3. `CLAIM_LEDGER.md`
4. this handoff

Use a fresh worktree from a fresh fetch/pull of `main`. The repo policy is to
run lints, type checks, targeted tests when needed, then commit, push, open a
PR, and merge finished work.

## Decision

Yes: the next step should be descriptor-conditioned replication.

The quoted plan is directionally right, but one boundary is now stale. Exact
V-JEPA pocket-replay features are available and accepted as a compute-proxy
verifier. The next replication should therefore use both exact V-JEPA and CLIP
centroid-margin logic prospectively, rather than treating V-JEPA as unavailable
or leaving CLIP as the only prospective descriptor.

Do not phrase the current state as:

> V-JEPA is explicitly not claimed because exact pocket-replay features were not available.

The current replacement is:

> Exact V-JEPA and CLIP are accepted compute-proxy verifiers for the current
> pocket-regime replay residual. They are not human memorability or measured-BMD
> validation, but they can guide the next descriptor-conditioned replication.

## Current Accepted State

- C-017 is accepted as a compute-proxy content-pocket finding. Restored
  non-jellyfish content pockets survive local SVD recipe stress tests.
- Strongest positive pockets to consolidate next: orange flowers and hanging
  clothes.
- Boundary/supporting positives: blue jellyfish and old car.
- Hard negative controls: aerial beach, city street, and storm beach.
- Lightweight visual descriptors were not accepted as verifiers. The best
  near-miss was seed-image colorfulness at AUC 0.8333, below the AUC >= 0.85
  pre-registered gate.
- C-018 is accepted: exact V-JEPA and CLIP embedding geometry explain the
  positive-vs-hard-negative pocket residual well enough to become compute-proxy
  verifiers.
- Human memorability, BMD grounding, and prompt-conditioned generation are still
  not proven for these pockets.

Important verifier metrics from the accepted embedding audit:

- Exact V-JEPA video centroid margin: AUC 1.0000, abs d 3.2953, r(score)
  0.8871.
- Exact V-JEPA leave-one-pocket-out classifier: AUC 1.0000, balanced accuracy
  0.9722.
- CLIP seed-image centroid margin: AUC 1.0000, abs d 2.8573, r(score) 0.8541.
- CLIP generated-video centroid margin: AUC 0.8796, abs d 2.0280, r(score)
  0.7620.

## Current Regime Reading

This is search inside the accepted SVD content-pocket regime unless the
prospective verifier changes how candidates are selected.

Current artifact types:

- seed images
- prompts
- SVD/Modal recipes
- generated videos
- visual-gate statuses
- TRIBE scores
- V-JEPA and CLIP embeddings
- centroid-margin verifier values
- pocket labels
- run manifests and result notes
- claim-ledger entries

Current gates:

- complete generation and visual retention
- TRIBE stays positive for target pockets
- hard negatives stay negative under matched recipes
- exact V-JEPA and CLIP margins stay positive for replicated positives
- human or BMD-grounded validation before any final memorability claim

The discovery-relevant move would be turning V-JEPA/CLIP from retrospective
explainers into prospective selection constraints. If that works, the regime
gains a descriptor-conditioned candidate-selection gate.

## Key Artifacts

Committed notes and manifests:

- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/pocket_regime_audit_manifest_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/pocket_regime_audit_result_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_feature_audit_manifest_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_feature_audit_result_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_embedding_audit_manifest_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_embedding_audit_result_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_vjepa_extraction_result_20260608.md`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_embedding_audit_summary_20260608.json`
- `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/content_pocket_vjepa_extraction_summary_20260608.json`

Important scripts:

- `scripts/build_bo_prompt_search_manifest.py`
- `scripts/modal_bo_memorability_replay.py`
- `scripts/restore_bo_seed_bank.py`
- `scripts/audit_content_pocket_features.py`
- `scripts/audit_content_pocket_embeddings.py`
- `scripts/extract_pocket_replay_vjepa.py`

Local data-lake artifacts, intentionally not all committed:

- pocket-regime report:
  `/Users/jawaun/.codex/worktrees/regime-audit-pocket-search-20260608/isc_mod/data/reports/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608.json`
- pocket-regime generated videos:
  `/Users/jawaun/.codex/worktrees/regime-audit-pocket-search-20260608/isc_mod/data/generated/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608/`
- exact V-JEPA pocket features:
  `/Users/jawaun/isc_mod/data/features/vjepa_pocket_regime_audit_20260608`

If a new session cannot see those local paths, inspect the committed manifests
and summaries first. The V-JEPA feature files live outside git because `data/`
is ignored.

## Next Experiment: Descriptor-Conditioned Replication

Question:

Do orange flowers and hanging clothes replicate under fresh stochastic seeds
while preserving positive TRIBE replay score and accepted exact V-JEPA/CLIP
content-pocket margins?

First artifact:

- Create the descriptor-conditioned replication manifest before generation.
- Put it beside the prior Camilo BO memorability audit artifacts.
- Name targets, controls, recipes, stochastic seeds, visual gate, TRIBE gate,
  exact V-JEPA gate, and CLIP gate.
- Manifest created:
  `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/descriptor_conditioned_replication_manifest_20260608.md`.
  The next action is generation from that manifest, not recreating the protocol.

Targets:

- orange flowers
- hanging clothes

Controls:

- aerial beach
- city street
- storm beach

Optional second block, budget permitting:

- blue jellyfish
- old car

Recipe neighborhood:

- Use the local recipe neighborhood that passed the pocket-regime audit,
  currently represented by the Sobol 518-523 x7, reps2, steps50, motion5,
  noise0 run.
- Add fresh stochastic noise seeds. Do not only replay the old seed pool.

Retention:

- Preserve failed, withheld, and visually rejected videos.
- Use complete-candidate retention again: a candidate does not pass by averaging
  away one bad replicate.

Scoring and verifier sequence:

1. Generate from the replication manifest.
2. Apply visual gates and retain rejected artifacts with reasons.
3. Score retained clips with TRIBE.
4. Extract exact V-JEPA features for the new generated MP4 bytes.
5. Run the accepted V-JEPA and CLIP centroid-margin verifier logic on the new
   clips.
6. Summarize per-pocket TRIBE, verifier margins, visual failures, and hard
   negative controls.

Acceptance gate:

- Orange flowers and hanging clothes stay positive in mean TRIBE score across
  fresh stochastic seeds.
- Hard negatives stay negative under matched recipes and seeds.
- Visual gates pass under complete-candidate retention.
- Generated-video V-JEPA and CLIP centroid margins remain positive for the
  replicated positives and do not collapse toward the hard-negative centroid.

If the gate passes:

- Promote orange flowers and hanging clothes to descriptor-verified candidate
  pockets for human/BMD validation.
- Update `CLAIM_LEDGER.md`, `NEXT_STEPS.md`, and current research status.

If the gate fails:

- Narrow C-017/C-018.
- Decide whether V-JEPA/CLIP only explained the old seed pool.
- Preserve the failed replication as a rejected artifact, not as a silent
  discard.

## After This Fork

If the strongest two pockets replicate cleanly:

- Build a human/BMD validation packet.
- Keep the claim language compute-proxy until that gate clears.

If they replicate partially or ambiguously:

- Run a boundary audit for blue jellyfish and old car.
- Demote unstable pockets rather than blending them into a broad success claim.

If they fail:

- Treat C-017 as overfit to the old stochastic pool or local content set.
- Use the failed examples to refine the verifier or switch to a
  prompt-conditioned generator regime.

## Stop Rules

- Do not call proxy-only results human memorability gains.
- Do not claim measured-BMD grounding for these pockets until an explicit BMD
  or human/BMD validation gate clears.
- Do not run a prompt-rewrite tournament in the current SVD runner as if prompt
  text changes the generated pixels.
- Do not broaden alpha/guidance-only search as the main path.
- Do not delete failed candidates or missing clips.

## Handoff Summary For A New Agent

Start from fresh `main`, use the scientific discovery/regime audit lens, and do
descriptor-conditioned replication next. The key update is that exact V-JEPA is
no longer missing. It passed as the strongest compute-proxy verifier, with CLIP
also accepted. The next experiment is not "find any higher score"; it is "test
whether the best two content pockets remain positive under fresh stochastic
replication while V-JEPA and CLIP prospectively agree." Only after that should
the project spend human/BMD validation budget.
