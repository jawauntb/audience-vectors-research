# TRIBE Layerwise Encoder Hidden Capture

## Status

- Status: **complete**.
- Clips requested: **104** balanced BMD clips (52 low + 52 high).
- Cached clips reused: **24**.
- Newly captured clips: **80**.

## Target Cache Counts

| target | cached selected clips |
|---|---:|
| `attn00_post_resid` | 104 |
| `attn02_post_resid` | 104 |
| `attn04_post_resid` | 104 |
| `attn06_post_resid` | 104 |
| `attn08_post_resid` | 104 |
| `attn10_post_resid` | 104 |
| `attn12_post_resid` | 104 |
| `attn14_post_resid` | 104 |
| `final_encoder` | 104 |

## Next Step

```bash
uv run python scripts/tribe_foldsafe_direction_patch.py \
  --annotations data/raw/bold_moments/annotations.json \
  --feature-dir data/features/tribe \
  --hidden-dir data/features/tribe_layerwise_encoder \
  --n-train-each 40 --n-eval-each 12 --folds 5 \
  --alphas 1.0 --concurrency 6
```
