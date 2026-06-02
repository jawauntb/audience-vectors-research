# Mechanistic Audit Of A TRIBE Memorability Readout

**Draft status:** satellite paper draft, regenerated 2026-06-02.
**Core purpose:** answer the Spencer-style mechanistic critique without
overclaiming population-level causality.

## Abstract

Linear readouts in large representation models can be predictive for accidental
reasons. A memorability direction learned in TRIBE could reflect temporal
position artifacts, a generic nuisance axis, or intertwined nonlinear factors
rather than a stable viewer-response signal. We audit this concern using
Fourier decompositions of saved TRIBE outputs, direct patches to learned
time-position machinery, layerwise encoder hooks, and a transparent AlexNet
sanity check. The final TRIBE output is mostly temporal-DC, and direct
time-position and rotary-frequency patches do not collapse memorability
ordering on a balanced 24-clip sensitivity set. However, non-DC encoder hidden
structure is load-bearing: removing the learned hidden memorability direction
from early attention residuals through the final encoder sharply disrupts the
readout. A new fold-safe run expands the hidden cache to
104 balanced clips and uses disjoint
train/eval splits; after patching, layerwise held-out rho drops from mean
+0.602 to a patched range of
+0.054 to +0.200. The
reviewer-safe conclusion is that the BMD/TRIBE readout is not explained by a
simple learned-position-table artifact, but it remains a sequence-dependent
model readout, not a fully isolated population-level causal feature.

## Why This Exists

The first paper can say "this direction predicts memorability." A mechanistic
reviewer will ask whether the vector is merely the easiest linear basis in a
messy model. This satellite paper answers that question directly and keeps the
main selector paper cleaner.

## Tests

1. Fourier-decompose the learned output-space direction over TRIBE time bins.
2. Compare full tensor, temporal-DC, nonzero-temporal, and mean-pooled readouts.
3. Patch `_model.time_pos_embed` and rotary inverse frequencies during inference.
4. Hook encoder layers 0, 2, 4, ..., 14 and final encoder output.
5. Remove either non-DC sequence content or the one learned hidden
   high-minus-low memorability direction at each hook.
6. Replicate the compact-direction pattern in AlexNet conv5, where forward
   patching is transparent.

## Headline Results

- Saved TRIBE output readout: full-tensor rho about +0.401, mean-pooled rho
  about +0.405, temporal-DC rho about +0.405, nonzero-temporal rho about +0.297.
- Learned time-position ablation preserves ordering on the 24-clip set:
  baseline rho about +0.677, time-position scale 0 rho about +0.703.
- Rotary-frequency zeroing also preserves ordering: rho about +0.685.
- Non-DC encoder removal collapses the high/low gap most strongly at final
  encoder: patch rho about +0.097, gap ratio about +0.014.
- Direction-only hidden patch is sharper: first collapse appears at
  `attn00_post_resid`, and final encoder removal gives patch rho about -0.105
  with gap ratio about +0.004.
- Fold-safe hidden-direction patching is now complete on
  104 balanced clips. Across nine
  hook targets, mean held-out baseline rho is
  +0.602; mean patched rho ranges from
  +0.054 to +0.200, and
  the remaining high/low gap ratio ranges from +0.135
  to +0.212.
- AlexNet conv5 gives a transparent sanity check: learned-direction ablation
  drops rho from about +0.386 to +0.018, and forward patching weakens fc7 from
  about +0.432 to +0.212.

## Fold-Safe Hidden Patch

The fold-safe run expanded the hidden cache to
52 low-memorability plus
52 high-memorability clips. Each of
5 folds trains a hidden high-minus-low direction on
40 low plus 40
high clips, then patches 12 low plus
12 high held-out clips. The hidden direction,
output readout, and reported patch metrics are disjoint within each fold.

| Hook target | Train hidden rho | Held-out baseline rho | Held-out patch rho | Gap ratio |
|---|---:|---:|---:|---:|
| `attn00_post_resid` | +0.653 | +0.602 | +0.200 | +0.172 |
| `attn02_post_resid` | +0.587 | +0.602 | +0.161 | +0.196 |
| `attn04_post_resid` | +0.586 | +0.602 | +0.177 | +0.212 |
| `attn06_post_resid` | +0.607 | +0.602 | +0.152 | +0.160 |
| `attn08_post_resid` | +0.609 | +0.602 | +0.102 | +0.146 |
| `attn10_post_resid` | +0.609 | +0.602 | +0.094 | +0.139 |
| `attn12_post_resid` | +0.598 | +0.602 | +0.054 | +0.135 |
| `attn14_post_resid` | +0.596 | +0.602 | +0.100 | +0.171 |
| `final_encoder` | +0.565 | +0.602 | +0.160 | +0.207 |

## Layerwise Summary

The strongest localization result is not "position does not matter." The better
claim is:

```text
simple learned-position-table artifact: weakened
hidden sequence dependence: real
population-level TRIBE mechanism: not yet proven
```

The layerwise artifacts are stored in:

- `data/reports/tribe_fourier_critique_review.md`
- `data/reports/tribe_layerwise_encoder_localization.md`
- `data/reports/tribe_layerwise_direction_patch.md`
- `data/reports/tribe_layerwise_encoder_hidden_capture_104.md`
- `data/reports/tribe_foldsafe_direction_patch.md`

## Reviewer-Safe Interpretation

The mechanistic audit supports the selector paper by removing an easy dismissal:
the readout is not merely the temporal position table or a mean-pooling artifact.
But it also narrows the live concern. The model uses sequence structure
internally, and the fold-safe hidden-direction patch shows the learned hidden
direction remains load-bearing across disjoint train/eval folds. That supports
the layerwise effect as an intervention result, while population-level
generalization and content-stratified claims should still be framed cautiously.

## Next Controls

The 104-clip fold-safe patch is complete. The next mechanistic controls are:

- report prompt/content stratification;
- compare random hidden directions and matched-norm patches;
- repeat across multiple balanced subsets.
- replicate the intervention in an open brain-encoder or a transparent video
  model where hidden hooks are easier to audit.

These controls would turn the current fold-safe intervention into a stronger
population-level mechanistic claim.

## References

- Lahner et al. BOLD Moments. Nature Communications 2024.
- Meta/Facebook Research. TRIBE v2 repository.
- Spencer critique thread and follow-up Fourier/position discussion, May 2026.
