# Pilot Runbook

This runbook converts the current 24-prompt Wan selector result into a human
validation pilot.

## Build Current Manifest

```bash
uv run python scripts/build_selector_human_eval_manifest.py
```

Output:

```text
research_program/neurips_memorability_selector/experiments/current_selector_manifest.json
```

## Build Pairwise Tasks

```bash
uv run python scripts/build_selector_pairwise_tasks.py
```

Output:

```text
research_program/neurips_memorability_selector/experiments/current_selector_pairwise_tasks.json
```

For the current stronger pilot, use the V-JEPA-augmented manifest:

```bash
uv run python scripts/build_selector_pairwise_tasks.py \
  --manifest research_program/neurips_memorability_selector/experiments/current_selector_manifest_with_vjepa.json \
  --out research_program/neurips_memorability_selector/experiments/current_selector_pairwise_tasks_with_vjepa.json
```

## Build Survey HTML

```bash
uv run python scripts/build_selector_prolific_survey.py
```

Output:

```text
research_program/neurips_memorability_selector/experiments/current_selector_prolific_survey.html
```

For the V-JEPA pilot:

```bash
uv run python scripts/build_selector_prolific_survey.py \
  --tasks research_program/neurips_memorability_selector/experiments/current_selector_pairwise_tasks_with_vjepa.json \
  --out research_program/neurips_memorability_selector/experiments/current_selector_prolific_survey_with_vjepa.html
```

The V-JEPA HTML samples 24 trials per participant from the 185 available pairwise
tasks, balancing across comparison families when possible. It embeds local video
paths relative to the project root, so it works as a local pilot. For a real
Prolific launch, host the videos and replace `ASSET_BASE` in the generated HTML
or add an upload step that rewrites URLs.

## Analyze Responses

Put downloaded response JSON files here:

```text
research_program/neurips_memorability_selector/experiments/responses/
```

Then run:

```bash
uv run python scripts/analyze_selector_pairwise.py
```

Outputs:

```text
research_program/neurips_memorability_selector/experiments/selector_pairwise_analysis.json
research_program/neurips_memorability_selector/experiments/selector_pairwise_analysis.md
```

## Current Pilot Arms

- `product_vs_base`: product selector against base generation.
- `product_vs_single_lora`: product selector against single LoRA.
- `product_vs_raw_best`: product selector against raw TRIBE best-of-N.
- `product_vs_clip_seed_image`: product selector against seed-image CLIP
  preservation.
- `product_vs_clip_prompt`: product selector against prompt CLIP alignment.
- `product_vs_clip_preservation`: product selector against a weighted CLIP-only
  preservation selector.
- `product_vs_vjepa_memorability`: product selector against the V-JEPA
  memorability selector.
- `gated_vs_base`: gated best-of-N against base generation.
- `gated_vs_clip_preservation`: gated best-of-N against the weighted CLIP-only
  preservation selector.
- `gated_vs_vjepa_memorability`: gated best-of-N against the V-JEPA
  memorability selector.

Some rows are intentionally skipped when both policies choose the same video.

Current task counts:

```text
gated_vs_base: 24
product_vs_single_lora: 24
product_vs_clip_prompt: 20
product_vs_clip_preservation: 19
product_vs_clip_seed_image: 19
product_vs_base: 18
product_vs_vjepa_memorability: 17
gated_vs_clip_preservation: 17
gated_vs_vjepa_memorability: 14
product_vs_raw_best: 13
```

## What This Pilot Can Prove

- Whether the current product selector is visibly/humanly better than base on
  the same prompt set.
- Whether the preservation gate avoids the most obvious proxy reward failures.
- Whether the survey mechanics work before we spend money on a larger study.

## What This Pilot Cannot Prove

- Submission-grade generalization.
- Actual superiority over V-JEPA/CLIP unless independent human responses are
  collected for the augmented survey.
- Actual delayed recognition memorability.

## Next Upgrade

Generate at least 50-100 fresh prompts with all selector baselines scored on the
same candidate pool. Then run the predeclared selector comparison from
`selector_human_eval_protocol.md`.
