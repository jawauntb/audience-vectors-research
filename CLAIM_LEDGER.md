# Claim Ledger

Last updated: 2026-06-05.

This ledger is the repo's compact audit layer: what we believe, what evidence
supports it, what gate accepted it, what is still tentative, and what should no
longer be said.

Use this file when deciding what can go into the paper, a collaborator note, or
a next experiment. Detailed protocols and raw artifacts still live elsewhere;
this is the reviewer-safe claim map.

## Evidence Hierarchy

```text
human behavior / BMD labels
  > measured fMRI pilots
  > TRIBE/BMD proxy scores
  > V-JEPA/CLIP/R3D proxy scores
  > BO, SVD, LoRA, Modal, and other compute workflows
```

Compute workflows can propose, search, and stabilize candidates. They do not
become human-memorability evidence unless they pass a human or BMD-grounded
gate.

## Status Labels

| status | meaning |
|---|---|
| accepted | The claim passed its stated gate and can be used with the listed scope. |
| tentative | The result is useful but still missing a required gate or has a major caveat. |
| superseded | A better-controlled result replaced the older wording. Do not use the old claim. |
| rejected | The test failed or the claim should not be made. |
| open | Needed work, unresolved provenance, or a missing validation step. |

## Current Claims

| id | status | claim | gate / verifier | scope and caveat |
|---|---|---|---|---|
| C-001 | accepted | The TRIBE/BMD memorability direction predicts BOLD Moments memorability competitively in brain-aligned features. | BMD labels; 5-fold CV and canonical held-out split. | Use for core selector paper. Current claim is competitive brain-aligned prediction, not a clean global V-JEPA win. |
| C-002 | accepted | The TRIBE-projection selector signal survives a human forced-choice check in the earlier best-of-N setting. | Prolific study A: 41 raters, all attention checks passed; 290/451 choices for TRIBE-ranked best-of-N winner; 64.3%; binomial p = 1.3e-9. | Supports the broader TRIBE/BMD selector signal. Does not validate newer V-JEPA-adjudicated pools, BO-generated videos, or delayed-recognition memorability. |
| C-003 | accepted | TRIBE Modal startup/import is fixed for the deployed dev image after pinning `exca==0.5.25`. | Modal rebuild/deploy preflight imported `TribeModel`; post-deploy TRIBE video-byte preflight passed. | Infrastructure claim only. Does not say anything about scientific validity. |
| C-004 | accepted | Fixed-budget SVD replay through Modal is operational after cache warmup. | 32/32 pre-generated replay MP4s completed direct-bytes TRIBE full scoring with longer timeout. | Supports using Modal for compute-side reproduction. Runtime remains a limitation. |
| C-005 | accepted | Single original BO table scores should not be treated as stable point estimates. | Top-2 and top-5 replicated replay panels changed rank order and showed high stochastic variance. | This retires "original top score is a stable winner" language. |
| C-006 | tentative | BO beats the saved Sobol top-5 under equal-count replicated replay. | PR #10 run: 5 BO + 5 Sobol candidates, 3 replicates each, 30/30 full TRIBE scores; BO replay mean 1.3281 vs Sobol replay mean -1.5901. | Tentative until PR #10 is merged and seed coverage is addressed. BO top five are all `fresh24_blue_jellyfish`; this is not broad prompt evidence. |
| C-007 | tentative | BO is a sample-efficient search policy over generated video candidates under proxy objectives. | Reproduced BO replay, replicated panels, and equal-count saved-Sobol comparison. | Say "sample-efficient" or "fixed-budget search"; do not say "wall-clock efficient" or "human-memorability proven." |
| C-008 | superseded | The original `bo07_cand01` score proves a stable best BO candidate. | Superseded by replicated replay. | Do not use. Current top by replay mean differs from original top score. |
| C-009 | superseded | Brain alignment is the active ingredient for all global best-of-N gains. | Superseded by V-JEPA-as-judge parity in some settings. | Current claim: brain alignment is useful for held-out human memorability prediction; generator-side proxy gains need more careful controls. |
| C-010 | open | BO, random/Sobol, and best-of-N are compared under matched seed-image coverage. | Needed: seed-stratified or regenerated tournament panel. | This is the next compute-side gate before stronger BO/control language. |
| C-011 | open | BO-generated videos improve human memorability. | Needed: blinded human study on compute-stabilized candidates. | Do not claim yet. |

## Run Registry

| run id | date | operation | artifact / report | gate result | claim impact |
|---|---:|---|---|---|---|
| R-2026-06-03-tribe-exca-fix | 2026-06-03 | Pin `exca==0.5.25` in TRIBE Modal image and add import preflight. | PR #8; `docs/TRIBE_MODAL_STARTUP_FIX.md`. | Modal deploy and TRIBE preflight passed. | Supports C-003. |
| R-2026-06-03-source-of-truth | 2026-06-03 | Add repo navigation and source-of-truth cleanup. | PR #7; `START_HERE.md`. | Docs merged. | Makes paper/status source explicit. |
| R-2026-06-03-bo-replicates | 2026-06-03 | Add stochastic BO replay replicates. | PR #6; `scripts/modal_bo_memorability_replay.py`; `src/audience_vectors/bo_replay.py`. | Tests passed; merged. | Enables C-005/C-007 gates. |
| R-2026-06-03-top5-replay | 2026-06-03 | Replay top 5 original BO/TRIBE candidates with 3 noise seeds each. | PR #9 note; local report `data/reports/bo_modal_replay_replicates_top5x3_20260603.json`. | 15/15 full TRIBE scores completed. | Supports C-005; weakens original top-rank claim. |
| R-2026-06-05-bo-vs-sobol | 2026-06-05 | Equal-count replay of top 5 BO vs top 5 saved Sobol candidates, 3 replicates each. | PR #10; local report `data/reports/bo_modal_replay_equal_budget_top5_bo_vs_sobol_20260605.json`. | 30/30 full TRIBE scores completed; BO mean 1.3281 vs Sobol mean -1.5901. | Supports C-006 with seed-pocket caveat. |

Local `data/reports/*` and generated MP4s are not committed. They can be cited
only through committed notes, PR descriptions, or regenerated reports.

## Active Gates

| gate id | gate | pass condition | current status |
|---|---|---|---|
| G-001 | Infrastructure gate | Modal generation and TRIBE scoring complete without startup/import failure. | Passed for current dev app. |
| G-002 | Replicate-stability gate | Candidate ranking is summarized by mean/std/SEM across stochastic replays. | Passed for top-2/top-5 panels; stability is weak. |
| G-003 | Equal-budget baseline gate | BO is compared to random/Sobol/best-of-N under equal evaluation count. | Partially passed for saved Sobol top-5 in PR #10; seed coverage caveat remains. |
| G-004 | Seed-stratified tournament gate | BO, random/Sobol, and best-of-N are compared within matched seed-image/prompt strata. | Tooling and BO/Sobol run manifest added; Modal run is blocked on local artifact paths. |
| G-005 | Visual artifact gate | Top candidates are inspected for prompt drift, degenerate motion, and obvious artifacts. | Open for BO panels. |
| G-006 | Human gate | Compute-stabilized generated candidates pass blinded human evaluation. | Open for BO-generated videos. |

## Conceptual Guardrails

The categorical discovery paper is useful as a repo discipline: every research
step should be typed as an artifact, operation, gate, and claim-status update.
In that framing, BO replay is fixed-regime search. A discovery or strong
scientific claim requires a verified transition: new evidence, a new gate, or a
new accepted representational commitment that survives audit.

The autopoiesis/weakness framing is useful as a search discipline: keep enough
slack that the project does not collapse onto one brittle candidate family. For
this repo, that means preserving seed diversity, baseline diversity, and
multiple validation paths until a gate justifies narrowing.

The Wolfram competition/ruliology framing is useful as a validation discipline:
strategy performance can be computationally irreducible, and larger strategies
can win by having specialized pockets for different opponents. For BO/control
work, do not infer "best strategy" from one elegant mechanism or one lucky
candidate. Run tournament panels across strategies, seeds, budgets, and
replicates, then report the distribution.

## Next Move

Run a seed-stratified tournament panel:

- strategies: BO, random/Sobol, best-of-N, and if feasible a cheap CLIP/R3D
  filter before full TRIBE;
- strata: matched seed images/prompts, including but not limited to
  `fresh24_blue_jellyfish`;
- budget: same number of generated clips and full TRIBE scores per strategy;
- summary: per-stratum mean/std/SEM, pooled mixed-effect summary if appropriate,
  wall-clock cost, and visual artifact inspection notes.

Only after this should the BO/control satellite claim move beyond "tentative."
