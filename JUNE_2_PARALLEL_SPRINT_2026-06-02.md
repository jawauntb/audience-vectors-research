# June 2 Parallel Sprint Update

Prepared: June 2, 2026
Root: [/Users/jawaun/isc_mod](/Users/jawaun/isc_mod)

## Summary

We ran the next-step plan in parallel across product, Prolific/IRB preflight,
mechanistic follow-up, and packaging.

What completed:

- Analyzer product polish shipped and deployed.
- Server-backed analyzer share links now work.
- Local browser history and snapshot exports are available.
- Active-learning export was added to batch selector reports.
- V-JEPA Prolific launch-preflight assets were created and validated.
- Professor/IRB checklist/brief were updated with the hosted-survey requirements.
- Fold-safe TRIBE hidden-direction patch experiment was prepared, unblocked with
  a 104-clip hidden-cache expansion, and completed across 5 folds.
- The research split package was rebuilt and now includes the launch-preflight
  assets plus the fold-safe runbook/report.

Remaining blocker:

The fold-safe mechanistic cache blocker is resolved. The remaining
submission-critical blocker is external human validation: the V-JEPA-augmented
Prolific pilot still needs hosted video URLs, stimulus screening, and
faculty/IRB determination before launch.

## Product Analyzer

Live:
[https://jawaun--video-analyzer.modal.run](https://jawaun--video-analyzer.modal.run)

Source:
[/Users/jawaun/isc_mod/src/audience_vectors/modal_app/functions/video_analyzer_site.py](/Users/jawaun/isc_mod/src/audience_vectors/modal_app/functions/video_analyzer_site.py)

Added:

- Local history panel for recent completed analyses.
- Per-result snapshot copy/download.
- Batch active-learning export.
- Server-backed share links:
  - `POST /api/runs`
  - `GET /api/runs/{id}`
  - page hydration from `?run=<id>`
- Modal JSON snapshot volume: `audience-analyzer-runs-v1`.

Live verification:

- UI shell contains `Local history`, `Share link`, `Active-learning`,
  `hydrateSharedRun`, and `/api/runs`.
- Share API smoke:
  - `POST /api/runs`: `HTTP 200`
  - returned id: `5ba81fc3c4044b57`
  - `GET /api/runs/5ba81fc3c4044b57`: `HTTP 200`
  - fetched payload contained `1` ranked item, filename `share-smoke`.

Claim boundary:

Server-backed sharing persists analysis JSON snapshots, including filenames,
input text/URLs if present, scores, and segment summaries. This is useful for
collaboration, but it should be treated as non-private unless we add auth and
deletion controls.

## Prolific / IRB Preflight

New launch-preflight folder:

- [launch_preflight/README.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/launch_preflight/README.md)
- [launch_preflight/build_hosted_survey.py](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/launch_preflight/build_hosted_survey.py)

New/updated launch assets:

- [hosted_video_url_map.template.json](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/prolific_launch_assets_2026-06-01/hosted_video_url_map.template.json)
- [task_randomization_freeze.json](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/prolific_launch_assets_2026-06-01/task_randomization_freeze.json)
- [prolific launch README](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/prolific_launch_assets_2026-06-01/README.md)

Updated IRB/professor docs:

- [prolific_launch_checklist.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/irb/prolific_launch_checklist.md)
- [professor_irb_brief.md](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/irb/professor_irb_brief.md)

Preflight state:

- V-JEPA augmented task pool is frozen for review.
- Hosted URL map template has `103` video slots.
- It intentionally has `0` final hosted URLs and `0` screened videos today.
- The study is still not launched.

Still blocked before Prolific:

- Stable HTTPS video hosting.
- Stimulus screening.
- Faculty/PI and IRB exemption/approval/determination.
- Final compensation, completion code, Prolific settings, exclusion rules, and
  response export path.
- Browser dry-run with hosted assets.

## Mechanistic Fold-Safe Patch

New runnable script/report:

- [scripts/tribe_foldsafe_direction_patch.py](/Users/jawaun/isc_mod/scripts/tribe_foldsafe_direction_patch.py)
- [tribe_foldsafe_direction_patch.md](/Users/jawaun/isc_mod/data/reports/tribe_foldsafe_direction_patch.md)
- [tribe_foldsafe_direction_patch.json](/Users/jawaun/isc_mod/data/reports/tribe_foldsafe_direction_patch.json)
- [tribe_layerwise_encoder_hidden_capture_104.md](/Users/jawaun/isc_mod/data/reports/tribe_layerwise_encoder_hidden_capture_104.md)
- [tribe_layerwise_encoder_hidden_capture_104.json](/Users/jawaun/isc_mod/data/reports/tribe_layerwise_encoder_hidden_capture_104.json)

Result:

- Status: complete.
- Scored TRIBE feature clips found: `1022`.
- Clips with all requested layerwise hidden caches: `104`.
- Balanced cache coverage: `52 low + 52 high`; missing `0`.
- Fold-safe split:
  - train: `40 low + 40 high`
  - held-out eval: `12 low + 12 high`
  - folds: `5`
- Mean held-out baseline ρ across folds: `+0.602`.
- Mean patched held-out ρ by layer: `+0.054` to `+0.200`.
- Mean high-low gap ratio by layer: about `0.135` to `0.212`.
- Interpretation: the learned hidden memorability direction remains
  patch-sensitive under a fold-safe split, not just on the old 24-clip subset.

Commands run:

```bash
uv run python scripts/tribe_layerwise_encoder_localization.py \
  --annotations data/raw/bold_moments/annotations.json \
  --feature-dir data/features/tribe \
  --output-dir data/features/tribe_layerwise_encoder \
  --n-each 52 \
  --capture-only \
  --capture-concurrency 4 --timeout 300 \
  --out-json data/reports/tribe_layerwise_encoder_hidden_capture_104.json \
  --out-md data/reports/tribe_layerwise_encoder_hidden_capture_104.md
```

```bash
uv run python scripts/tribe_foldsafe_direction_patch.py \
  --annotations data/raw/bold_moments/annotations.json \
  --feature-dir data/features/tribe \
  --hidden-dir data/features/tribe_layerwise_encoder \
  --n-train-each 40 --n-eval-each 12 --folds 5 \
  --alphas 1.0 --concurrency 6
```

## Package Rebuild

Regenerated package:

- [neurips_memorability_selector_split_package_2026-06-01.zip](/Users/jawaun/isc_mod/data/reports/neurips_memorability_selector_split_package_2026-06-01.zip)

Package now includes:

- `research_program/.../experiments/launch_preflight/README.md`
- `research_program/.../experiments/launch_preflight/build_hosted_survey.py`
- `hosted_video_url_map.template.json`
- `task_randomization_freeze.json`
- `data/reports/tribe_foldsafe_direction_patch.md`
- `scripts/tribe_foldsafe_direction_patch.py`

The package builder was updated to exclude `__pycache__` and `.pyc` files:

- [build_neurips_selector_site.py](/Users/jawaun/isc_mod/scripts/build_neurips_selector_site.py)

## Verification

Commands/checks run:

```bash
.venv/bin/ruff check \
  src/audience_vectors/modal_app/functions/video_analyzer_site.py \
  research_program/neurips_memorability_selector/experiments/launch_preflight/build_hosted_survey.py \
  scripts/tribe_foldsafe_direction_patch.py \
  scripts/build_neurips_selector_site.py

.venv/bin/python -m py_compile \
  src/audience_vectors/modal_app/functions/video_analyzer_site.py \
  research_program/neurips_memorability_selector/experiments/launch_preflight/build_hosted_survey.py \
  scripts/tribe_foldsafe_direction_patch.py \
  scripts/build_neurips_selector_site.py

.venv/bin/pyright --pythonpath .venv/bin/python \
  src/audience_vectors/modal_app/functions/video_analyzer_site.py \
  research_program/neurips_memorability_selector/experiments/launch_preflight/build_hosted_survey.py \
  scripts/tribe_foldsafe_direction_patch.py \
  scripts/build_neurips_selector_site.py
```

Results:

- Ruff: passed.
- Python compile: passed.
- Pyright: `0 errors, 0 warnings, 0 informations`.
- Extracted analyzer browser script parses.
- Hosted-survey preflight JSON validates with `jq`.
- Live analyzer deployed successfully.
- Live share API smoke passed.

## Next Concrete Move

The research-critical path is now:

1. Host the `103` video assets and fill
   [hosted_video_url_map.template.json](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/prolific_launch_assets_2026-06-01/hosted_video_url_map.template.json).
2. Screen every hosted stimulus and mark the map accordingly.
3. Generate hosted survey HTML with
   [build_hosted_survey.py](/Users/jawaun/isc_mod/research_program/neurips_memorability_selector/experiments/launch_preflight/build_hosted_survey.py).
4. Send the professor/IRB packet for determination.
5. Use the completed fold-safe mechanistic result as supporting evidence, but
   keep the main submission claim gated on human validation.
