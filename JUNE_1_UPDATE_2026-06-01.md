# June 1 Canonical Update

Prepared: June 1, 2026
Project root: [/Users/jawaun/isc_mod](/Users/jawaun/isc_mod)

This is the single navigable update for what changed, what we learned, what is
still unproven, and what to do next.

## TL;DR

Today moved the project from "interesting research artifact" toward a cleaner
research program plus usable product surface.

The core research state is now:

- TRIBE/BMD memorability is a real predictive signal, roughly `rho ~= +0.40`.
- Spencer's simple Fourier/positional-artifact worry is weakened: output-space
  temporal DC, mean pooling, and full tensor readouts are similar, and zeroing
  learned time-position embeddings does not collapse the ordering.
- The harder mechanistic story is more interesting: TRIBE's encoder non-DC
  sequence structure is load-bearing, and removing one learned hidden
  memorability direction collapses the balanced-subset readout from early
  attention residuals through the final encoder.
- V-JEPA is now a real baseline, not a strawman. It covers all 24 current Wan
  seeds and differs enough from TRIBE/product choices to make the human pilot
  meaningful.
- Wan2.2 selection is promising as a product workflow, but still proxy-only:
  the current product selector improves `18/24` seeds under the TRIBE/BMD proxy.
- The submission-grade blocker is still independent human validation of the
  V-JEPA-augmented selector.
- The analyzer is live and now has raw TRIBE dimensions, BMD memorability,
  NOVA-inspired affect proxy, multi-input intake, batch ranking, and
  copy/download reports.

Live analyzer: [https://jawaun--video-analyzer.modal.run](https://jawaun--video-analyzer.modal.run)

## What Was Built Today

### 1. NeurIPS-Style Research Program Split

The research is now organized as a main paper plus satellite papers instead of
one overloaded manuscript.

Program folder:
[/Users/jawaun/isc_mod/research_program/neurips_memorability_selector](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector)

Main paper:

- [paper.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/main_selector_paper/paper.md)
- [paper.pdf](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/main_selector_paper/paper.pdf)
- [paper.html](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/main_selector_paper/paper.html)

Satellite papers:

- [01_mechanistic_audit.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/satellite_papers/01_mechanistic_audit.md)
- [02_audience_axes.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/satellite_papers/02_audience_axes.md)
- [03_reward_distillation.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/satellite_papers/03_reward_distillation.md)
- [04_representation_frames.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/satellite_papers/04_representation_frames.md)
- [05_affect_aware_media_selection.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/satellite_papers/05_affect_aware_media_selection.md)

Generated program site:

- [site/index.html](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/site/index.html)
- [site/manifest.json](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/site/manifest.json)

Submission status:

- [submission_status_2026-06-01.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/submissions/submission_status_2026-06-01.md)
- [neurips_readiness_checklist.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/submissions/neurips_readiness_checklist.md)

### 2. Packages And Share Assets

Full split package:

- [neurips_memorability_selector_split_package_2026-06-01.zip](/Users/jawaun/isc_mod/data/reports/neurips_memorability_selector_split_package_2026-06-01.zip)

Overleaf package:

- [main_selector_overleaf_2026-06-01.zip](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/packages/main_selector_overleaf_2026-06-01.zip)
- [README_OVERLEAF.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/overleaf/main_selector/README_OVERLEAF.md)
- [main.tex](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/overleaf/main_selector/main.tex)

Professor / IRB packet:

- [professor_irb_packet_2026-06-01.zip](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/packages/professor_irb_packet_2026-06-01.zip)
- [professor_irb_brief.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/irb/professor_irb_brief.md)
- [irb_protocol_draft.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/irb/irb_protocol_draft.md)
- [consent_form_draft.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/irb/consent_form_draft.md)
- [prolific_launch_checklist.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/irb/prolific_launch_checklist.md)

### 3. V-JEPA-Augmented Human Pilot Assets

The current pilot now compares TRIBE/product and gated selection against V-JEPA,
CLIP/preservation, base, single-LoRA, and raw-best variants.

Key files:

- [current_selector_manifest_with_vjepa.json](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/current_selector_manifest_with_vjepa.json)
- [vjepa_selector_report.json](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/vjepa_selector_report.json)
- [current_selector_pairwise_tasks_with_vjepa.json](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/current_selector_pairwise_tasks_with_vjepa.json)
- [current_selector_prolific_survey_with_vjepa.html](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/current_selector_prolific_survey_with_vjepa.html)
- [selector_human_eval_protocol.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/selector_human_eval_protocol.md)
- [pilot_runbook.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/pilot_runbook.md)

Pilot scale:

- `24` seeds
- `185` pairwise tasks
- `0` missing local video paths
- V-JEPA selected video path exists for `24/24` seeds
- Product selector equals V-JEPA selector on `7/24` seeds
- Gated selector equals V-JEPA selector on `10/24` seeds

Prolific launch review assets prepared today:

- [prolific_launch_assets_2026-06-01/README.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/prolific_launch_assets_2026-06-01/README.md)
- [professor_irb_summary.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/prolific_launch_assets_2026-06-01/professor_irb_summary.md)
- [launch_copy.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/prolific_launch_assets_2026-06-01/launch_copy.md)
- [response_schema.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/prolific_launch_assets_2026-06-01/response_schema.md)

### 4. Analyzer Product Polish

Analyzer source:
[/Users/jawaun/isc_mod/src/audience_vectors/modal_app/functions/video_analyzer_site.py](/Users/jawaun/isc_mod/src/audience_vectors/modal_app/functions/video_analyzer_site.py)

Live:
[https://jawaun--video-analyzer.modal.run](https://jawaun--video-analyzer.modal.run)

What the analyzer now supports:

- Upload videos, images, and text files.
- Paste public URLs for video/image/text pages where cloud download is allowed.
- Paste raw text/copy directly.
- Run multiple items in one batch without losing earlier jobs.
- Segment videos into windows.
- Show raw TRIBE dimensions:
  - overall
  - attention
  - emotion
  - memory
  - visual
  - language
  - cognitive ease
  - hook score for videos
- Keep BMD/human-label memorability separate from TRIBE's raw memory dimension.
- Show palette, motion, edge, density, and natural-language commentary.
- Show NOVA-inspired affect proxy:
  - happy
  - anger
  - fear
  - sadness
  - disgust
  - neutral
- Rank completed batch candidates with a product selector score.
- Copy or download selector reports as Markdown.
- Download full batch JSON.
- Use favicon, apple-touch icon, and OG/share image routes.

Important claim boundary:

The affect section is explicitly a media proxy from TRIBE plus visual statistics.
It is not EEG PSD decoding, and it is not validated emotion induction.

Product-facing support files touched:

- [app.py](/Users/jawaun/isc_mod/src/audience_vectors/modal_app/app.py)
- [image_factory.py](/Users/jawaun/isc_mod/src/audience_vectors/modal_app/image_factory.py)
- [tribe_predictor.py](/Users/jawaun/isc_mod/src/audience_vectors/modal_app/functions/tribe_predictor.py)
- [tribe_service.py](/Users/jawaun/isc_mod/src/audience_vectors/services/tribe_service.py)

### 5. Affect / NOVA-Inspired Branch

This was kept separate from the main memorability paper.

Project branch:
[/Users/jawaun/isc_mod/research_program/affect_aware_media_selector/README.md](/Users/jawaun/isc_mod/research_program/affect_aware_media_selector/README.md)

Satellite:
[/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/satellite_papers/05_affect_aware_media_selection.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/satellite_papers/05_affect_aware_media_selection.md)

Useful framing:

- Product proxy now: affect-like readout from media features.
- Research validation later: human affect labels, then EEG PSD or differential
  entropy if using NOVA-style recordings.
- Do not fold EEG into the current memorability IRB unless it becomes a separate
  protocol or amendment.

## Results And Learnings

### A. Main Memorability Signal

Confirmed:

- TRIBE/BMD memorability direction is a real predictive signal, around
  `rho ~= +0.40`.
- Audit numbers include 5-fold CV `+0.403 +/- 0.061` and canonical split
  `+0.478`, CI `[+0.302, +0.627]`.

Best references:

- [critical_research_audit_2026-05-25.md](/Users/jawaun/isc_mod/data/reports/critical_research_audit_2026-05-25.md)
- [current_research_status.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/current_research_status.md)

Reviewer-safe claim:

> We have a stable predictive memorability readout in TRIBE/BMD-aligned features.

Not safe:

> TRIBE has proven a causal human memorability mechanism.

### B. Spencer Fourier / Positional Critique

The simple version of the critique is weakened:

- TRIBE output-space full tensor, temporal DC, and mean-pooled readouts are very
  similar.
- The standard direction has about `0.898 +/- 0.009` energy in temporal DC on the
  native 4-bin subset.
- Zeroing or scaling the learned TRIBE time-position table changes projection
  values but does not collapse ranking or the high-low gap.
- Rotary frequency scaling also preserves rank ordering.

The more interesting version remains alive:

- Encoder hidden states have load-bearing non-DC sequence structure.
- Removing non-DC sequence variation collapses the balanced-subset high-low gap.
- Removing a single learned hidden memorability direction collapses the readout
  from early post-attention residual layers through the final encoder.
- This is still an in-sample 24-clip high/low intervention, so a larger fold-safe
  patch is required before treating it as population mechanistic evidence.

Best references:

- [tribe_fourier_critique_review.md](/Users/jawaun/isc_mod/data/reports/tribe_fourier_critique_review.md)
- [tribe_layerwise_encoder_localization.md](/Users/jawaun/isc_mod/data/reports/tribe_layerwise_encoder_localization.md)
- [tribe_layerwise_direction_patch.md](/Users/jawaun/isc_mod/data/reports/tribe_layerwise_direction_patch.md)
- [spectral_positional_memorability_probe_summary.md](/Users/jawaun/isc_mod/data/reports/spectral_positional_memorability_probe_summary.md)

Spoken update version:

> The easy artifact story did not survive. Time-position embeddings and rotary
> frequency ablations do not explain away the readout. But the signal is not a
> clean position-free scalar either. It depends on non-DC encoder sequence
> structure, and a learned hidden memorability direction becomes load-bearing
> early in the encoder.

### C. V-JEPA Baseline

Confirmed:

- V-JEPA direction exists over BMD labels and performs around TRIBE-level as a
  memorability baseline.
- Reported V-JEPA full-CV mean Spearman is `+0.395 +/- 0.037`.
- The current Wan pilot has V-JEPA features for all current candidates needed
  for the selector comparison.

Best references:

- [cv_vjepa_n1026.md](/Users/jawaun/isc_mod/data/reports/cv_vjepa_n1026.md)
- [vjepa_baseline_plan.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/vjepa_baseline_plan.md)
- [vjepa_selector_report.json](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/vjepa_selector_report.json)

Claim boundary:

Do not claim TRIBE clearly beats V-JEPA. Current evidence says they are close.
The human pilot is needed to decide whether TRIBE/product selection adds value
beyond V-JEPA.

### D. Wan2.2 Product Selector

Current proxy result:

- Preference-weighted product policy improves `18/24` seeds under the TRIBE/BMD
  proxy.
- Mean/median proxy lift: `+2.8170 / +2.4324`.
- Raw best-of-N has higher proxy lift but can select worse semantic drift.
- Product-safe workflow is base-or-gated best-of-N, not unconstrained
  maximization.

Best references:

- [wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0_product_selector.md](/Users/jawaun/isc_mod/data/reports/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0_product_selector.md)
- [wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_selector_tuning.md](/Users/jawaun/isc_mod/data/reports/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_selector_tuning.md)

Practical takeaway:

> The near-term product is a generation-ranking workflow: generate a small
> candidate set, score each candidate, preserve the prompt/seed, reject drift,
> and keep the base when the selected candidate is worse.

Not proven:

> A LoRA or DPO model has actually learned human memorability.

### E. Human Validation

Current state:

- A V-JEPA-augmented pilot survey exists.
- Human responses have not been collected for this version.
- The current `selector_pairwise_analysis.json` has `n_participants = 0` and
  `n_responses = 0`.

Reference:

- [selector_pairwise_analysis.json](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/selector_pairwise_analysis.json)

The hard anti-circularity rule:

> TRIBE/BMD can define a selector policy, but the human-validation endpoint must
> be independent human judgment or delayed recognition. It cannot be another
> TRIBE score.

### F. YouTube / Cloud Download Bug Claim

Observed behavior:

- Local YouTube download/scoring can work on a laptop.
- Modal/cloud URL scoring can fail because YouTube blocks unauthenticated cloud
  downloaders.

Current product handling:

- The analyzer now returns a specific message explaining that YouTube blocked
  the cloud downloader and suggesting direct MP4 upload or configuring yt-dlp
  cookies/PO-token env vars.

Relevant code:

- [video_analyzer_site.py YouTube handling](/Users/jawaun/isc_mod/src/audience_vectors/modal_app/functions/video_analyzer_site.py)

Practical answer:

> It worked locally because your laptop/browser/network context is different.
> The Modal worker is a cloud downloader and can be treated as an unauthenticated
> bot by YouTube. Uploading MP4 directly is the reliable path unless we configure
> cookies/PO-token support.

## Verification Done Today

Local checks run after analyzer polish:

```bash
.venv/bin/ruff format src/audience_vectors/modal_app/functions/video_analyzer_site.py scripts/build_neurips_selector_site.py
.venv/bin/ruff check src/audience_vectors/modal_app/functions/video_analyzer_site.py scripts/build_neurips_selector_site.py
.venv/bin/python -m py_compile src/audience_vectors/modal_app/functions/video_analyzer_site.py
.venv/bin/pyright --pythonpath .venv/bin/python src/audience_vectors/modal_app/functions/video_analyzer_site.py scripts/build_neurips_selector_site.py
```

Result:

- Ruff format/check: passed.
- Python compile: passed.
- Pyright: `0 errors, 0 warnings, 0 informations`.
- Extracted browser script parsed successfully with `bun` via `new Function(...)`.

Deploy:

```bash
.venv/bin/modal deploy -m audience_vectors.modal_app.app
```

Result:

- Modal deployed successfully.
- Web function:
  [https://jawaun--video-analyzer.modal.run](https://jawaun--video-analyzer.modal.run)

Live UI shell check:

- `Batch selector`: present.
- `Selector formula`: present.
- `Copy report`: present.
- `Download report`: present.
- `Affect proxy`: present.
- `NOVA-inspired`: present.

Live API smoke check:

```text
POST /api/analyze
input: short text prompt
status: HTTP 200
filename: pasted text
verdict: revise
overall: 5.9
attention: 5.7
affect: neutral
segments: 1
```

## What Is Not Proven Yet

Do not claim these:

- TRIBE/product-selected videos are more memorable to humans.
- TRIBE clearly beats V-JEPA.
- Wan LoRA or DPO has learned memorability.
- Persona directions are independent orthogonal audience axes.
- Affect proxy is EEG emotion decoding.
- Current layerwise TRIBE patch is a population-level causal mechanism.

Safe framing:

- TRIBE/BMD and V-JEPA expose memorability-like predictive signals.
- TRIBE/BMD product selection is promising under proxy metrics.
- V-JEPA is the right current baseline.
- Human validation is now the decisive next step.
- Mechanistic probes moved the critique from "simple positional artifact" to
  "hidden sequence-structured memorability direction."

## Next Steps

### Highest Priority

Run the V-JEPA-augmented blinded human pilot after faculty/IRB review.

Needed before launch:

- Faculty/PI review.
- IRB exemption, approval, or written determination.
- Final consent form with institutional contacts.
- Hosted video URLs instead of local paths.
- Stimulus screening.
- Compensation and completion code.
- Frozen exclusion rules.
- Chrome/Safari dry run.

Relevant folder:
[/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/prolific_launch_assets_2026-06-01](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/prolific_launch_assets_2026-06-01)

### Research Next

1. Launch/collect/analyze human pairwise pilot.
2. If signal survives, scale from 24 prompts to 50-100 fresh prompts.
3. Add a video-quality/VBench-style baseline so reviewers cannot say the
   selector just finds artifact-heavy clips.
4. Run fold-safe larger internal TRIBE hidden-direction patching.
5. Compare TRIBE, V-JEPA, CLIP, and human preference matrices with RSA/CKA.
6. Only then decide whether LoRA/DPO distillation is worth the budget.

### Product Next

1. Use the analyzer as the practical selector workflow.
2. Add persistent job history/share links if this becomes a real tool.
3. Add hosted asset ingestion so YouTube/URL failures do not block workflows.
4. Add a small active-learning export: top wins, worst failures, ambiguous
   candidates, and recommended next generations.
5. Keep affect as exploratory until human affect labels or EEG validation exist.

## Best One-Sentence Status

We have a credible brain-aligned memorability selector program with a live
product analyzer and meaningful mechanistic progress, but the paper becomes
submission-grade only after the V-JEPA-augmented human pilot shows independent
human preference or recognition gains.
