# TRIBE Fold-Safe Hidden Direction Patch

## Status

- Status: **complete**.

## Setup

- Scored TRIBE feature clips found: **1022**.
- Clips with all requested layerwise hidden caches: **104**.
- Balanced hidden-cache coverage: **104 / 104** (52 low + 52 high ready; missing 0 low + 0 high).
- Train clips per fold: **80** (40 low + 40 high).
- Held-out eval clips per fold: **24** (12 low + 12 high).
- Folds: **5**.
- Alphas: `1.0`.
- Fold-safe rule: hidden direction, output readout, and reported patch metrics use disjoint train/eval clips within each fold.

## Layerwise Results

| fold | target | alpha | train hidden rho | eval baseline rho | eval patch rho | eval gap ratio | |Δproj| / std |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `attn00_post_resid` | +1.000 | +0.649 | +0.485 | +0.013 | +0.041 | 0.813 |
| 1 | `attn02_post_resid` | +1.000 | +0.551 | +0.485 | +0.069 | +0.119 | 0.778 |
| 1 | `attn04_post_resid` | +1.000 | +0.553 | +0.485 | +0.038 | +0.131 | 1.389 |
| 1 | `attn06_post_resid` | +1.000 | +0.589 | +0.485 | +0.079 | +0.128 | 0.756 |
| 1 | `attn08_post_resid` | +1.000 | +0.590 | +0.485 | +0.073 | +0.160 | 2.629 |
| 1 | `attn10_post_resid` | +1.000 | +0.598 | +0.485 | +0.026 | +0.179 | 4.944 |
| 1 | `attn12_post_resid` | +1.000 | +0.603 | +0.485 | +0.078 | +0.220 | 7.863 |
| 1 | `attn14_post_resid` | +1.000 | +0.601 | +0.485 | +0.075 | +0.249 | 12.954 |
| 1 | `final_encoder` | +1.000 | +0.541 | +0.485 | +0.079 | +0.109 | 1.253 |
| 2 | `attn00_post_resid` | +1.000 | +0.658 | +0.632 | +0.223 | +0.219 | 0.716 |
| 2 | `attn02_post_resid` | +1.000 | +0.663 | +0.632 | +0.047 | +0.110 | 2.224 |
| 2 | `attn04_post_resid` | +1.000 | +0.665 | +0.632 | +0.090 | +0.122 | 1.767 |
| 2 | `attn06_post_resid` | +1.000 | +0.671 | +0.632 | +0.010 | +0.092 | 3.038 |
| 2 | `attn08_post_resid` | +1.000 | +0.674 | +0.632 | -0.086 | +0.029 | 5.378 |
| 2 | `attn10_post_resid` | +1.000 | +0.671 | +0.632 | -0.070 | +0.012 | 8.020 |
| 2 | `attn12_post_resid` | +1.000 | +0.656 | +0.632 | -0.192 | -0.032 | 12.231 |
| 2 | `attn14_post_resid` | +1.000 | +0.642 | +0.632 | -0.004 | +0.033 | 18.579 |
| 2 | `final_encoder` | +1.000 | +0.645 | +0.632 | +0.149 | +0.218 | 0.853 |
| 3 | `attn00_post_resid` | +1.000 | +0.666 | +0.640 | +0.344 | +0.221 | 0.713 |
| 3 | `attn02_post_resid` | +1.000 | +0.590 | +0.640 | +0.216 | +0.228 | 0.781 |
| 3 | `attn04_post_resid` | +1.000 | +0.593 | +0.640 | +0.277 | +0.272 | 1.138 |
| 3 | `attn06_post_resid` | +1.000 | +0.609 | +0.640 | +0.223 | +0.190 | 0.708 |
| 3 | `attn08_post_resid` | +1.000 | +0.611 | +0.640 | +0.110 | +0.111 | 2.636 |
| 3 | `attn10_post_resid` | +1.000 | +0.616 | +0.640 | +0.053 | +0.054 | 4.531 |
| 3 | `attn12_post_resid` | +1.000 | +0.609 | +0.640 | -0.010 | +0.005 | 7.374 |
| 3 | `attn14_post_resid` | +1.000 | +0.604 | +0.640 | +0.010 | +0.010 | 12.961 |
| 3 | `final_encoder` | +1.000 | +0.583 | +0.640 | +0.233 | +0.201 | 0.940 |
| 4 | `attn00_post_resid` | +1.000 | +0.673 | +0.559 | +0.117 | +0.128 | 0.862 |
| 4 | `attn02_post_resid` | +1.000 | +0.601 | +0.559 | +0.101 | +0.136 | 1.318 |
| 4 | `attn04_post_resid` | +1.000 | +0.586 | +0.559 | +0.154 | +0.180 | 1.057 |
| 4 | `attn06_post_resid` | +1.000 | +0.613 | +0.559 | +0.134 | +0.108 | 1.046 |
| 4 | `attn08_post_resid` | +1.000 | +0.613 | +0.559 | +0.085 | +0.110 | 2.623 |
| 4 | `attn10_post_resid` | +1.000 | +0.609 | +0.559 | +0.117 | +0.122 | 4.801 |
| 4 | `attn12_post_resid` | +1.000 | +0.600 | +0.559 | +0.090 | +0.134 | 7.951 |
| 4 | `attn14_post_resid` | +1.000 | +0.608 | +0.559 | +0.096 | +0.192 | 13.429 |
| 4 | `final_encoder` | +1.000 | +0.551 | +0.559 | +0.123 | +0.194 | 1.414 |
| 5 | `attn00_post_resid` | +1.000 | +0.621 | +0.694 | +0.304 | +0.251 | 0.725 |
| 5 | `attn02_post_resid` | +1.000 | +0.530 | +0.694 | +0.374 | +0.385 | 0.970 |
| 5 | `attn04_post_resid` | +1.000 | +0.532 | +0.694 | +0.324 | +0.356 | 1.568 |
| 5 | `attn06_post_resid` | +1.000 | +0.553 | +0.694 | +0.312 | +0.284 | 1.037 |
| 5 | `attn08_post_resid` | +1.000 | +0.557 | +0.694 | +0.325 | +0.322 | 1.279 |
| 5 | `attn10_post_resid` | +1.000 | +0.551 | +0.694 | +0.342 | +0.329 | 2.515 |
| 5 | `attn12_post_resid` | +1.000 | +0.522 | +0.694 | +0.307 | +0.349 | 4.285 |
| 5 | `attn14_post_resid` | +1.000 | +0.526 | +0.694 | +0.323 | +0.372 | 7.843 |
| 5 | `final_encoder` | +1.000 | +0.507 | +0.694 | +0.217 | +0.314 | 1.277 |

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

This expanded the hidden cache to the completed fold-safe requirement of 52 low + 52 high clips.

## Fold-Safe Patch Rerun

```bash
uv run python scripts/tribe_foldsafe_direction_patch.py \
  --annotations data/raw/bold_moments/annotations.json \
  --feature-dir data/features/tribe \
  --hidden-dir data/features/tribe_layerwise_encoder \
  --n-train-each 40 --n-eval-each 12 --folds 5 \
  --alphas 1.0 --concurrency 6
```
