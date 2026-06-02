# TRIBE Fold-Safe Hidden Direction Patch

## Status

- Status: **blocked**.
- Blocker: insufficient layerwise hidden cache for requested fold-safe split: need at least 104 clips with all requested targets, found 24. Balanced expansion is missing 40 low-tail and 40 high-tail clips.

## Setup

- Scored TRIBE feature clips found: **1022**.
- Clips with all requested layerwise hidden caches: **24**.
- Balanced hidden-cache coverage: **24 / 104** (12 low + 12 high ready; missing 40 low + 40 high).
- Train clips per fold: **80** (40 low + 40 high).
- Held-out eval clips per fold: **24** (12 low + 12 high).
- Folds: **5**.
- Alphas: `1.0`.
- Fold-safe rule: hidden direction, output readout, and reported patch metrics use disjoint train/eval clips within each fold.

## Hidden Cache Expansion

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

This expands the hidden cache to the fold-safe requirement of 52 low + 52 high clips.

## Fold-Safe Patch Rerun

```bash
uv run python scripts/tribe_foldsafe_direction_patch.py \
  --annotations data/raw/bold_moments/annotations.json \
  --feature-dir data/features/tribe \
  --hidden-dir data/features/tribe_layerwise_encoder \
  --n-train-each 40 --n-eval-each 12 --folds 5 \
  --alphas 1.0 --concurrency 6
```
