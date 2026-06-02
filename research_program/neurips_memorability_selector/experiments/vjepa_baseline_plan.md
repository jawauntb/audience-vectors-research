# V-JEPA Selector Baseline

The current pilot now includes V-JEPA memorability selection over the same
candidate pool as the TRIBE/product selector.

## Current Status

- BMD-trained V-JEPA direction exists:

```text
data/models/vectors/facebook__vjepa2-vitl-fpc64-256__vjepa_mean_pool__bmd_memorability_n1026.npz
```

- Current Wan selector candidates have V-JEPA embeddings in:

```text
data/features/vjepa_wan22_selector_pref_weighted_r16_s300/
```

- Extraction completed for all current-pilot missing embeddings:

```text
[done] extracted 88/88 missing embeddings
```

- Scoring completed for all seeds:

```text
[done] complete seeds: 24/24; missing features: 0
```

- The augmented manifest is:

```text
research_program/neurips_memorability_selector/experiments/current_selector_manifest_with_vjepa.json
```

- The V-JEPA pairwise task file is:

```text
research_program/neurips_memorability_selector/experiments/current_selector_pairwise_tasks_with_vjepa.json
```

- The V-JEPA survey file is:

```text
research_program/neurips_memorability_selector/experiments/current_selector_prolific_survey_with_vjepa.html
```

## Why This Matters

V-JEPA is the strongest non-brain baseline in the current paper. A NeurIPS
reviewer will not be satisfied by product-vs-CLIP alone because CLIP mostly
tests prompt/seed preservation, not video memorability. The decisive human study
should compare:

- TRIBE + preservation gate
- V-JEPA memorability selector
- CLIP preservation selector
- random/base

## Extraction Command

```bash
uv run python scripts/extract_selector_vjepa_features.py \
  --concurrency 4 \
  --transport bytes
```

Bytes transport was used because Modal volume reads were stale/inconsistent for
newly uploaded generated videos. Volume transport remains available for future
BMD-style workflows.

## Scoring Command

```bash
uv run python scripts/score_selector_vjepa_baseline.py \
  --augmented-manifest research_program/neurips_memorability_selector/experiments/current_selector_manifest_with_vjepa.json
```

## Human-Eval Build Commands

```bash
uv run python scripts/build_selector_pairwise_tasks.py \
  --manifest research_program/neurips_memorability_selector/experiments/current_selector_manifest_with_vjepa.json \
  --out research_program/neurips_memorability_selector/experiments/current_selector_pairwise_tasks_with_vjepa.json

uv run python scripts/build_selector_prolific_survey.py \
  --tasks research_program/neurips_memorability_selector/experiments/current_selector_pairwise_tasks_with_vjepa.json \
  --out research_program/neurips_memorability_selector/experiments/current_selector_prolific_survey_with_vjepa.html
```

Current coverage:

```text
unique V-JEPA embeddings: 103
seeds with V-JEPA-selected video path: 24/24
product selector equals V-JEPA selector: 7/24
gated selector equals V-JEPA selector: 10/24
augmented pairwise tasks: 185
```
