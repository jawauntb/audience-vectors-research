# Attention-Capture TRIBE Modal-Volume Extraction

## Verdict

- Ready: False
- App: `audience-vectors-dev`
- Modal volume: `attention-capture-features-v1`
- Output prefix: `attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610`
- Event mode: `full`
- Jobs: 3
- Written: 0
- Cached: 0
- Errors: 3
- Shape mismatches: 0
- Claim boundary: This report verifies Modal-side feature extraction and storage only. It does not validate the Phase 1 capture-score hypothesis.

## Blocking Reasons

- 3 feature extraction jobs failed

## Error Preview

- `dhf1k_146` RuntimeError: Ratio of unmatched words is 0.3333 on 3 words while AddSentenceToWords.max_unmatched_ratio=0.05
- `dhf1k_203` RuntimeError: Ratio of unmatched words is 0.2500 on 4 words while AddSentenceToWords.max_unmatched_ratio=0.05
- `dhf1k_262` RuntimeError: Ratio of unmatched words is 0.3333 on 3 words while AddSentenceToWords.max_unmatched_ratio=0.05

## Output Preview

| dhf1k_146 | error | 0 x 0 | /attention-capture-features/attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610/dhf1k_146.npz |
| dhf1k_203 | error | 0 x 0 | /attention-capture-features/attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610/dhf1k_203.npz |
| dhf1k_262 | error | 0 x 0 | /attention-capture-features/attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610/dhf1k_262.npz |
