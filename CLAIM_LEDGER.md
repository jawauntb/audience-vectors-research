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
| C-012 | accepted | Current BO replay evidence is prompt-pocket behavior, not broad strategy dominance. | Max-3 regenerated-control stress test on 2026-06-08: 36/36 clips generated; 2/36 visual failures; complete-candidate retention kept 10/12 candidates and scored 30/30 retained rows. Jellyfish BO retained 3/3 candidates and averaged 1.7443 vs regenerated Sobol 1.0575. Fireworks BO retained only 1/3 candidates and scored -3.9426, while regenerated Sobol averaged -5.1029. | Compute-proxy claim only. It supports "BO exploits a stable jellyfish pocket while fireworks remains visually brittle/low-scoring"; it does not support broad prompt-level BO superiority or human memorability. |
| C-013 | accepted | The top saved BO parameter recipes do not transfer across prompt strata. | Prompt-transfer stress test on 2026-06-08: top saved BO recipes `bo07_cand01`, `bo04_cand01`, and `bo02_cand01` were retargeted across five image-backed prompt slots and compared with matched Sobol alpha/guidance controls. 30/30 clips generated; 2/30 failed visual gate; 28/28 retained rows completed full TRIBE. BO-transfer averaged -3.5444, Sobol-transfer averaged -3.0223, and blue jellyfish was the only positive prompt slot for both policies. | Compute-proxy claim only. Supports "saved high-scoring BO recipes are jellyfish-pocket recipes, not reusable global steering/guidance policies." It does not replace a true per-prompt BO/search panel. |
| C-014 | accepted | In the prompt-broadened replay regime, prompt/seed identity dominates alpha/guidance recipe choice. | Per-prompt Sobol search on 2026-06-08: 8 shared Sobol alpha/guidance points were generated across each of five image-backed prompt slots. 40/40 clips generated; 2/40 fireworks rows failed visual gate; complete-candidate retention scored 38/38 retained rows. Prompt-only additive model explained R2 = 0.9196 of retained TRIBE score variance, while Sobol recipe index alone explained R2 = 0.0062 and alpha/guidance/interaction alone explained R2 = 0.0042. The top eight retained candidates were all blue jellyfish rows. | Compute-proxy claim only. This is a regime-diagnostic claim, not a human-memorability claim. It says the next broadening axis should be prompt/seed/content search, not more BO over alpha/guidance alone. |
| C-015 | accepted | Prompt rewriting is not an active intervention in the current SVD replay runner; SVD content search is currently seed-image limited. | SVD content-axis audit on 2026-06-08: `scripts/audit_bo_content_axes.py` found 24 prompt catalog rows but only 5 locally available seed images, with 19 missing seed images. AST audit found `SVDGenerator.generate` does not accept prompt text and `scripts/modal_bo_memorability_replay.py` does not pass prompt text into the SVD Modal generation call. | Infrastructure/regime claim. It blocks prompt-rewrite experiments under the current SVD replay path. Valid content broadening requires seed-bank restoration/expansion or a prompt-conditioned generator path. |
| C-016 | accepted | Fixed-recipe SVD replay confirms that seed-image/content slot, not alpha/guidance recipe identity, is the dominant broadening variable in the current replay regime. | Fixed-recipe seed-content probe on 2026-06-08: Sobol recipes 516 and 517 were replayed across the five local seed images with 2 stochastic reps each. 20/20 clips generated; both fireworks candidates were withheld by complete-candidate visual-first retention after one row per candidate failed `tail_sharpness_collapse`; 16/16 retained rows completed full TRIBE after the Modal TRIBE image was repinned to `transformers==4.56.1` and stale tasks were stopped. Seed-content slot explained retained-score R2 = 0.9494, recipe-only explained R2 = 0.0026, and blue jellyfish was the only positive retained slot. | Compute-proxy regime claim only. It strengthens C-014/C-015 and says the next valid SVD broadening move is seed-bank expansion or seed selection, not prompt rewriting or more alpha/guidance-only search. It does not establish human memorability or global generator superiority. |

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
| R-2026-06-08-max3-regenerated-controls | 2026-06-08 | Run the balanced max-3 regenerated-control stress test: 3 saved BO anchors and 3 regenerated Sobol controls per selected prompt stratum, 3 replicates each. | `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/max3_regenerated_visual_controls_result_20260608.md`; local report `data/reports/bo_regenerated_visual_controls_max3_regensobol3_reps3_steps50_motion5_noise0_start128_20260608.json`. | 36/36 clips generated; 2/36 failed visual gate (`bo09_cand01` replicate 1 and `bo03_cand01` replicate 2); complete-candidate retention kept 10/12 candidates and scored 30/30 retained rows. BO numerically beat regenerated Sobol in both retained strata, but fireworks remained low-scoring and BO retained only 1/3 fireworks candidates. | Supports C-012: current evidence is prompt-pocket behavior. The stable positive signal is jellyfish; broad prompt-level BO superiority remains unproven. |
| R-2026-06-08-prompt-transfer-stress | 2026-06-08 | Test whether the top saved BO alpha/guidance recipes transfer across all locally image-backed prompt slots, with matched Sobol-transfer controls. | `scripts/build_bo_prompt_transfer_manifest.py`; `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/prompt_transfer_stress_test_result_20260608.md`; local reports `data/reports/bo_prompt_transfer_trial_table_top3x5_sobol3_20260608.json` and `data/reports/bo_prompt_transfer_top3x5_sobol3_reps1_steps50_motion5_noise0_20260608.json`. | 30/30 clips generated; 2/30 failed visual gate, both fireworks; complete-candidate retention kept 28/30 candidates and all retained rows completed full TRIBE. BO-transfer was negative outside jellyfish and averaged -3.5444 overall; Sobol-transfer averaged -3.0223. | Supports C-013: saved BO recipes are not portable global recipes. The next broadening step must run per-prompt search or a cheap prefilter, not transfer old BO anchors. |
| R-2026-06-08-per-prompt-sobol-search | 2026-06-08 | Run a prompt-local Sobol search panel with 8 shared alpha/guidance points across each of five image-backed prompt slots. | `scripts/build_bo_prompt_search_manifest.py`; `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/per_prompt_sobol_search_result_20260608.md`; local reports `data/reports/bo_prompt_search_trial_table_sobol8x5_20260608.json` and `data/reports/bo_prompt_search_sobol8x5_reps1_steps50_motion5_noise0_20260608.json`. | 40/40 clips generated; 2/40 fireworks rows failed visual gate; complete-candidate retention kept 38/40 candidates and all retained rows completed full TRIBE. Blue jellyfish was the only positive prompt slot, with mean 1.6597 and best candidate `sobol_prompt_search_517_slot03` at 2.9734. Prompt identity explained nearly all retained score variance. | Supports C-014 and upgrades the next-step diagnosis: alpha/guidance-only search is exhausted as a broadening axis under the current replay regime. |
| R-2026-06-08-svd-content-axis-audit | 2026-06-08 | Audit which content axes the current SVD replay runner can actually manipulate. | `scripts/audit_bo_content_axes.py`; `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/svd_content_axes_audit_20260608.md`; local report `data/reports/bo_svd_content_axes_audit_20260608.json`. | Prompt catalog has 24 rows but only 5 available local seed images; 19 seed images are missing. The SVD Modal generator path is image-conditioned and does not accept/pass prompt text. | Supports C-015. Prevents invalid prompt-rewrite-only SVD experiments and redirects content broadening toward seed-bank expansion or prompt-conditioned generators. |
| R-2026-06-08-seed-content-fixed-recipe | 2026-06-08 | Replay two fixed Sobol alpha/guidance recipes across all five available SVD seed-image slots with two stochastic reps each. | `scripts/build_bo_prompt_search_manifest.py`; `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/seed_content_fixed_recipe_probe_result_20260608.md`; local reports `data/reports/bo_seed_content_recipe_probe_trial_table_sobol516_517_x5_reps2_20260608.json`, `data/reports/bo_seed_content_recipe_probe_sobol516_517_x5_reps2_steps50_motion5_noise0_20260608.json`, and `data/reports/bo_seed_content_recipe_probe_sobol516_517_x5_reps2_steps50_motion5_noise0_rescored_20260608.json`. | 20/20 clips generated; complete-candidate visual-first retention withheld both fireworks candidates and scored 16/16 retained rows. Blue jellyfish averaged 2.2202 and was the only positive slot; seed-content slot explained R2 = 0.9494 of retained score variance; recipe identity explained R2 = 0.0026. | Supports C-016 and turns the content-axis audit into a scored seed-content result. |

Local `data/reports/*` and generated MP4s are not committed. They can be cited
only through committed notes, PR descriptions, or regenerated reports.

## Active Gates

| gate id | gate | pass condition | current status |
|---|---|---|---|
| G-001 | Infrastructure gate | Modal generation and TRIBE scoring complete without startup/import failure. | Passed for current dev app. |
| G-002 | Replicate-stability gate | Candidate ranking is summarized by mean/std/SEM across stochastic replays. | Passed for top-2/top-5 panels; stability is weak. |
| G-003 | Equal-budget baseline gate | BO is compared to random/Sobol/best-of-N under equal evaluation count. | Partially passed for saved Sobol top-5 in PR #10; seed coverage caveat remains. |
| G-004 | Seed-stratified tournament gate | BO, random/Sobol, and best-of-N are compared within matched seed-image/prompt strata. | Saved-table BO/Sobol panel completed for two matched strata; result is mixed and not sufficient for stronger BO/control language. |
| G-005 | Visual artifact gate | Top candidates are inspected for prompt drift, degenerate motion, and obvious artifacts. | Automated gate is operational as both a blocking verifier and visual-first retention filter. The max-3 regenerated-control run generated 36/36 clips, failed 2/36 rows, and retained 10/12 complete candidates before scoring. |
| G-006 | Regenerated-control gate | Deterministic unscored controls are generated for each selected BO stratum, visual-first retention keeps complete candidates before TRIBE scoring, and at least one BO and one control remain in a matched stratum. | Passed for the max-3 2026-06-08 fireworks/jellyfish panel: regenerated controls covered all target strata, complete-candidate retention kept matched BO/control coverage in both strata, and 30/30 retained rows completed full TRIBE scoring. |
| G-007 | Human gate | Compute-stabilized generated candidates pass blinded human evaluation. | Open. A human panel can now be prepared only for visually retained matched candidates, with explicit caveats that the regenerated-control result is small, mixed, and proxy-only. |
| G-008 | Prompt-coverage gate | BO/control comparisons cover enough prompt or seed strata that a pooled result is not just a seed-pocket artifact. | Still open for BO/control claims. The prompt-transfer, per-prompt Sobol, and fixed-recipe seed-content panels now cover five image-backed seed slots, but all show that the positive signal is jellyfish-specific. Seed-content identity, not alpha/guidance recipe choice, explains the broad prompt-broadened replay score structure. The SVD content-axis audit shows prompt rewriting is not active in the current SVD path and the local seed bank has only 5 usable images. |
| G-009 | Content-axis gate | The next broadening experiment must manipulate a content variable that the generator actually consumes. | Active. Under current SVD replay, valid content variables are seed-image selection and seed-bank expansion. The fixed-recipe seed-content panel passed this gate at small scale and found seed-content R2 = 0.9494. Prompt rewriting requires switching to a prompt-conditioned generator or changing generator plumbing first. |

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

Stop spending broadening budget on alpha/guidance-only BO. The max-3
regenerated-control stress test found a stable jellyfish pocket and a
weak/brittle fireworks stratum; the prompt-transfer stress test showed that the
best saved BO recipes do not transfer beyond jellyfish; the per-prompt Sobol
search then showed that prompt identity explains the retained replay scores far
better than the shared alpha/guidance recipe; and the fixed-recipe seed-content
probe confirmed that seed-image/content slot remains dominant even when recipe
choice is held to two shared Sobol points.

- strategies: seed-image expansion, seed selection, or a prompt-conditioned
  generator path; prompt rewriting alone is invalid for current SVD replay
  because prompt text is metadata-only;
- strata: matched seed images/prompts across more than the two saved-table
  strata, with explicit non-jellyfish candidate generation;
- budget: same number of generated clips and full TRIBE scores per content
  strategy;
- generation: use settings that preserve subject identity past the first frame;
- search: optimize or prefilter within each prompt stratum instead of retargeting
  old jellyfish-pocket recipes;
- summary: per-stratum mean/std/SEM, pooled mixed-effect summary if appropriate,
  wall-clock cost, and visual artifact inspection notes before any human study.

Only after G-008 passes should the BO/control satellite claim move beyond
"prompt-pocket behavior."
