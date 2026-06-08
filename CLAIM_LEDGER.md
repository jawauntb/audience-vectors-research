# Claim Ledger

Last updated: 2026-06-08.

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
| C-006 | tentative | BO beats the saved Sobol top-5 under equal-count replicated replay. | PR #10 run: 5 BO + 5 Sobol candidates, 3 replicates each, 30/30 full TRIBE scores; BO replay mean 1.3281 vs Sobol replay mean -1.5901. | Seed-stratified replay later produced a mixed result: BO won fireworks, Sobol won jellyfish, and pooled means were close. Keep this claim limited to the saved top-5 panel; do not generalize across prompts. |
| C-007 | tentative | BO is a sample-efficient search policy over generated video candidates under proxy objectives. | Reproduced BO replay, replicated panels, and equal-count saved-Sobol comparison. | Say "sample-efficient" or "fixed-budget search"; do not say "wall-clock efficient" or "human-memorability proven." |
| C-008 | superseded | The original `bo07_cand01` score proves a stable best BO candidate. | Superseded by replicated replay. | Do not use. Current top by replay mean differs from original top score. |
| C-009 | superseded | Brain alignment is the active ingredient for all global best-of-N gains. | Superseded by V-JEPA-as-judge parity in some settings. | Current claim: brain alignment is useful for held-out human memorability prediction; generator-side proxy gains need more careful controls. |
| C-010 | rejected | Saved-table BO/Sobol comparison supports a stronger broad BO/control claim under matched seed-image coverage. | Seed-stratified BO/Sobol replay on 2026-06-05: 12/12 full TRIBE scores; BO won fireworks, Sobol won jellyfish, pooled means were close; visual artifact gate failed. A tuned visual-gated one-replicate panel later passed 4/4 clips and Sobol beat BO in both matched strata. A replicated visual-gated panel on 2026-06-07 generated 12/12 clips but failed the blocking visual gate because `sobol_007` replicate 2 repeatedly collapsed. A visual-first replacement pool later retained 4/7 complete candidates and scored 12/21 rows, but both fireworks Sobol candidates were withheld. A regenerated-control run on 2026-06-08 generated 24/24 clips, withheld one BO fireworks candidate under complete-candidate visual-first retention, scored 21/21 retained rows, and preserved matched BO/control coverage in both selected strata; the result was still mixed by stratum. | Do not use stronger BO/control language from the saved table. The regenerated-control path fixes the missing-control protocol gap for a small two-stratum panel, but it does not establish a broad BO/control or human-memorability claim. |
| C-011 | open | BO-generated videos improve human memorability. | Needed: blinded human study on compute-stabilized candidates. | Do not claim yet. |

## Run Registry

| run id | date | operation | artifact / report | gate result | claim impact |
|---|---:|---|---|---|---|
| R-2026-06-03-tribe-exca-fix | 2026-06-03 | Pin `exca==0.5.25` in TRIBE Modal image and add import preflight. | PR #8; `docs/TRIBE_MODAL_STARTUP_FIX.md`. | Modal deploy and TRIBE preflight passed. | Supports C-003. |
| R-2026-06-03-source-of-truth | 2026-06-03 | Add repo navigation and source-of-truth cleanup. | PR #7; `START_HERE.md`. | Docs merged. | Makes paper/status source explicit. |
| R-2026-06-03-bo-replicates | 2026-06-03 | Add stochastic BO replay replicates. | PR #6; `scripts/modal_bo_memorability_replay.py`; `src/audience_vectors/bo_replay.py`. | Tests passed; merged. | Enables C-005/C-007 gates. |
| R-2026-06-03-top5-replay | 2026-06-03 | Replay top 5 original BO/TRIBE candidates with 3 noise seeds each. | PR #9 note; local report `data/reports/bo_modal_replay_replicates_top5x3_20260603.json`. | 15/15 full TRIBE scores completed. | Supports C-005; weakens original top-rank claim. |
| R-2026-06-05-bo-vs-sobol | 2026-06-05 | Equal-count replay of top 5 BO vs top 5 saved Sobol candidates, 3 replicates each. | PR #10; local report `data/reports/bo_modal_replay_equal_budget_top5_bo_vs_sobol_20260605.json`. | 30/30 full TRIBE scores completed; BO mean 1.3281 vs Sobol mean -1.5901. | Supports C-006 with seed-pocket caveat. |
| R-2026-06-05-seed-stratified | 2026-06-05 | Seed-stratified saved-table BO vs Sobol replay across fireworks and jellyfish prompt strata, 3 replicates each. | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/seed_stratified_tournament_result_20260605.md`; local report `data/reports/bo_modal_replay_seed_stratified_20260605.json`. | 12/12 full TRIBE scores completed; BO won fireworks, Sobol won jellyfish; pooled means nearly tied; visual gate failed. | Rejects broad BO/control language from the saved-table seed-stratified panel; supports a regenerated matched baseline or visual-quality gate next. |
| R-2026-06-05-visual-gated-smoke | 2026-06-05 | One-replicate seed-stratified BO/Sobol generation smoke with automated visual gate enabled as a blocking verifier. | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/visual_gated_smoke_result_20260605.md`; local report `data/reports/bo_visual_gated_smoke_20260605.json`. | 4/4 videos generated; 4/4 failed the visual gate; upload and TRIBE scoring were skipped by `--fail-on-visual-artifacts`. | Confirms G-005 is now operational as a blocking verifier; score comparisons should remain blocked until generation passes it. |
| R-2026-06-05-tuned-visual-gated | 2026-06-05 | Tune SVD replay settings and run a one-replicate seed-stratified BO/Sobol panel with visual gate and full TRIBE scoring. | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/tuned_visual_gated_tribe_result_20260605.md`; local report `data/reports/bo_visual_gated_tribe_steps50_motion5_noise0_20260605.json`. | 50 SVD steps, motion bucket 5, noise 0 passed 4/4 visual gate and completed 4/4 full TRIBE scores; Sobol beat BO in both matched strata. | Validates tuned visual-gated replay path; further rejects broad saved-table BO/control language and motivates replicated visual-gated replay. |
| R-2026-06-07-replicated-visual-gated-blocked | 2026-06-07 | Run the tuned seed-stratified BO/Sobol panel with 3 stochastic replicates per matched candidate under the blocking visual gate. | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/replicated_visual_gated_blocked_result_20260607.md`; local report `data/reports/bo_visual_gated_tribe_replicates3_steps50_motion5_noise0_20260607.json`. | 12/12 videos generated; 1/12 failed visual gate (`sobol_007` replicate 2, seed `20007`); upload and TRIBE scoring were skipped. Probes at 75 steps, motion buckets 3/2, and noise 0.005 did not remove the same Sobol collapse. | Blocks the human-panel path from the one-replicate result; motivates visual-first replacement or resampling before scoring. |
| R-2026-06-07-visual-first-max2 | 2026-06-07 | Run seed-stratified BO/Sobol replay with a visual-first complete-candidate retention pool (`max_evals=2`, 3 replicates each). | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/visual_first_retention_result_20260607.md`; local report `data/reports/bo_visual_first_complete_candidates_max2_reps3_steps50_motion5_noise0_20260607.json`. | 21/21 videos generated; 3/21 failed visual gate; complete-candidate retention kept 4/7 candidates and scored 12/21 rows. Retained candidates: `bo06_cand01`, `bo07_cand01`, `bo04_cand01`, `sobol_005`. Withheld candidates: `bo09_cand01`, `sobol_007`, `sobol_008`. | Visual-first retention works as a protocol, but the saved-table pool lacks any retained fireworks Sobol candidate, so it still cannot support a complete matched BO/Sobol claim. |
| R-2026-06-08-regenerated-control-preview | 2026-06-08 | Add regenerated Sobol control selection and dry-run the next matched visual-first panel. | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/regenerated_visual_controls_manifest_20260608.md`; dry-run report `/tmp/bo_regenerated_controls_preview.json`. | Dry-run loaded 32 saved-table trials, selected 4 saved BO anchors, appended 4 deterministic unscored Sobol controls across fireworks and jellyfish, and expanded to 24 planned replay jobs. Full Modal generation/scoring not run in this gate. | Establishes the next regenerated-control protocol. Does not upgrade BO/control claims until visual-first retention and TRIBE scoring complete. |
| R-2026-06-08-regenerated-visual-controls | 2026-06-08 | Run the regenerated-control visual-first panel with 2 saved BO anchors and 2 deterministic regenerated Sobol controls per selected prompt stratum, 3 replicates each. | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/regenerated_visual_controls_result_20260608.md`; local report `data/reports/bo_regenerated_visual_controls_topbo2_regensobol2_reps3_steps50_motion5_noise0_20260608.json`. | 24/24 clips generated; 1/24 failed the visual gate (`bo09_cand01` replicate 1); complete-candidate retention kept 7/8 candidates and scored 21/21 retained rows. Matched BO/control coverage survived in fireworks and jellyfish. BO won jellyfish; regenerated Sobol won fireworks. | Passes the small regenerated-control structural gate and unblocks candidate-set preparation, but the scientific result remains mixed and proxy-only. |
| R-2026-06-08-next-foundation-audit | 2026-06-08 | Audit saved-table prompt coverage and preflight the next balanced regenerated-control stress test. | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/next_research_foundation_manifest_20260608.md`; dry-run reports `/tmp/top_bo_per_prompt_all.json`, `/tmp/saved_matched_prompt_all.json`, and `/tmp/regenerated_controls_max3_start128_preflight.json`. | Saved BO replay coverage is limited to two prompt strata: 3 fireworks candidates and 17 jellyfish candidates. A balanced max-3 regenerated-control dry-run selected 6 saved BO anchors, appended 6 fresh regenerated Sobol controls from index 128+, and expanded to 36 planned replay jobs. | Clarifies that the saved table can support a within-table stress test, but true broad prompt evidence requires a new BO/search panel over more seed prompts. |

Local `data/reports/*` and generated MP4s are not committed. They can be cited
only through committed notes, PR descriptions, or regenerated reports.

## Active Gates

| gate id | gate | pass condition | current status |
|---|---|---|---|
| G-001 | Infrastructure gate | Modal generation and TRIBE scoring complete without startup/import failure. | Passed for current dev app. |
| G-002 | Replicate-stability gate | Candidate ranking is summarized by mean/std/SEM across stochastic replays. | Passed for top-2/top-5 panels; stability is weak. |
| G-003 | Equal-budget baseline gate | BO is compared to random/Sobol/best-of-N under equal evaluation count. | Partially passed for saved Sobol top-5 in PR #10; seed coverage caveat remains. |
| G-004 | Seed-stratified tournament gate | BO, random/Sobol, and best-of-N are compared within matched seed-image/prompt strata. | Saved-table BO/Sobol panel completed for two matched strata; result is mixed and not sufficient for stronger BO/control language. |
| G-005 | Visual artifact gate | Top candidates are inspected for prompt drift, degenerate motion, and obvious artifacts. | Automated gate is operational as both a blocking verifier and visual-first retention filter. The regenerated-control run generated 24/24 clips, failed 1/24 rows, and retained 7/8 complete candidates before scoring. |
| G-006 | Regenerated-control gate | Deterministic unscored controls are generated for each selected BO stratum, visual-first retention keeps complete candidates before TRIBE scoring, and at least one BO and one control remain in a matched stratum. | Passed for the small 2026-06-08 fireworks/jellyfish panel: regenerated controls covered all target strata, complete-candidate retention kept matched BO/control coverage in both strata, and 21/21 retained rows completed full TRIBE scoring. |
| G-007 | Human gate | Compute-stabilized generated candidates pass blinded human evaluation. | Open. A human panel can now be prepared only for visually retained matched candidates, with explicit caveats that the regenerated-control result is small, mixed, and proxy-only. |
| G-008 | Prompt-coverage gate | BO/control comparisons cover enough prompt or seed strata that a pooled result is not just a seed-pocket artifact. | Open. The saved 3-objective table covers only two BO prompt strata, so broader evidence requires either a new BO/search panel or a clearly labeled within-table stress test. |

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

Choose the next research fork:

1. Broader regenerated-control foundation:
   - first run the balanced max-3 within-table stress test in
     `next_research_foundation_manifest_20260608.md`;
   - treat it as stability evidence, not broad prompt evidence;
   - for true prompt broadening, run a new BO/search panel beyond the current
     fireworks/jellyfish saved-table strata;
   - include BO, regenerated Sobol/random, saved Sobol where visual-retained,
     and if feasible best-of-N or a cheap CLIP/R3D/VBench-style filter;
   - keep equal generated clips and full TRIBE scores per strategy;
   - preserve visual-first complete-candidate retention before scoring;
   - report per-stratum mean/std/SEM and a mixed-effects summary only after
     there is enough stratum coverage.
2. Human-panel candidate freeze:
   - freeze only visually retained matched candidates from
     `regenerated_visual_controls_result_20260608.md`;
   - make the panel small and explicitly exploratory;
   - avoid broad BO/control wording because the compute proxy result is mixed;
   - include the withheld `bo09_cand01` provenance as an exclusion note.

For the stronger paper-facing claim, regenerate a broader matched baseline with
a visual-quality gate:

- strategies: BO, random/Sobol, best-of-N, and if feasible a cheap CLIP/R3D or
  VBench-style filter before full TRIBE;
- strata: matched seed images/prompts across more than the two saved-table
  strata;
- budget: same number of generated clips and full TRIBE scores per strategy;
- generation: use settings that preserve subject identity past the first frame;
- summary: per-stratum mean/std/SEM, pooled mixed-effect summary if appropriate,
  wall-clock cost, and visual artifact inspection notes before any human study.

Only after this should the BO/control satellite claim move beyond "tentative."
