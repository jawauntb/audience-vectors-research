# Audience Vectors Project Handoff

Date: 2026-05-26

This archive is intended for someone who wants to try active learning,
alternative ranking methods, preference optimization, or other methods for
optimizing video generation with the audience-vector reward signal.

## What Is Included

- Paper/site artifacts:
  - `data/reports/paper.html`
  - `data/reports/paper.pdf`
  - `data/reports/arena_demo.html`
  - `data/reports/video_gallery/`
- Core reports:
  - `data/reports/critical_research_audit_2026-05-25.md`
  - `data/reports/wan22_pref_weighted_lora_2026-05-25.md`
  - `data/reports/wan22_product_selector_runnable_example.md`
  - `data/reports/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_selector_tuning.md`
- Product-selector artifacts:
  - `scripts/run_wan22_product_selector.py`
  - `scripts/tune_wan22_selector.py`
  - `scripts/summarize_wan22_product_selector.py`
  - `scripts/score_wan22_composite_preservation.py`
  - `scripts/eval_wan22_best_of_n.py`
- Wan2.2 LoRA artifacts:
  - `src/audience_vectors/modal_app/functions/wan22_lora_trainer.py`
  - `scripts/build_wan22_preference_weighted_sft_dataset.py`
  - `data/training/wan22_lora_pref_weighted_winners_50_margin05/`
  - `data/training/wan22_lora_smoke_outputs/wan22_tribe_proxy_pref_weighted_r16_s300/`
- Generated evaluation videos:
  - `data/generated/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_s12/`
  - `data/generated/wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0/`
  - Other generated clips under `data/generated/`
- Seed images/prompts:
  - `data/training/wan22_lora_eval_fresh_picsum_24/`
- Scored candidate pools:
  - TRIBE score reports under `data/reports/*results.json`
  - CLIP preservation reports under `data/reports/*composite*.json`
  - selector JSON/Markdown reports under `data/reports/*product_selector*`

## Current Best Product Workflow

The strongest practical workflow is a selector, not direct one-shot steering:

1. Generate a base video and several LoRA candidates.
2. Score each candidate with the TRIBE/BMD audience-vector projection.
3. Apply a semantic preservation guardrail using CLIP seed-image/prompt metrics.
4. Fall back to the base clip when the gated LoRA candidate scores below base.

Run the current selector on existing artifacts:

```bash
uv run python scripts/run_wan22_product_selector.py \
  --preset pref-weighted-r16-s300
```

Run the proxy-only selector tuning sweep:

```bash
uv run python scripts/tune_wan22_selector.py
```

## Latest Wan2.2 Result Snapshot

On 24 fresh image-seeded prompts:

| method | improved seeds | mean lift | median lift |
|---|---:|---:|---:|
| Single preference-weighted LoRA | 20/24 | +1.274 | +1.218 |
| Raw best-of-4 | 20/24 | +3.561 | +3.451 |
| Base-or-gated best-of-4, seed-drop <= 0.08 | 18/24 | +2.817 | +2.432 |
| Base-or-raw best-of-4, no preservation gate | 20/24 | +3.674 | +3.451 |

The selector tuning sweep suggests the next human-facing decision is whether a
looser preservation gate, such as seed-drop <= 0.16, preserves enough semantic
intent while recovering more proxy lift.

## Important Caveats

- These Wan2.2 selector results are TRIBE/BMD proxy-scored.
- The preference-weighted LoRA is winner-only SFT, not true diffusion DPO.
- The current product selector is not yet human-validated.
- The preservation gate is intentionally conservative and may suppress useful
  high-reward candidates.
- Existing Prolific data supports the broader audience-vector ranking signal,
  but this exact selector/gate tradeoff needs a small human validation pass.

## Suggested Next Experiments

- Active-learning loop: select uncertain or high-disagreement candidate pairs
  for human labeling, then tune the selector weights/gates.
- Compare gate policies: seed-drop <= 0.08 vs <= 0.16 vs no gate.
- Add VLM or perceptual quality checks as additional selector features.
- Train a proper preference model from human labels and distill it into a video
  LoRA or DPO-style adapter.
- Test transfer to another open video model with easier activation/weight access.

## Archive Exclusions

The full handoff zip excludes machine-local or redundant files:

- `.git/`
- `.venv/`
- Python caches and test caches
- `.DS_Store`
- Nested `.zip` files, because their contents are already represented by the
  expanded project artifacts.
