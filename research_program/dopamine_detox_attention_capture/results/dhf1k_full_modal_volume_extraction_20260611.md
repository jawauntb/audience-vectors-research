# Attention-Capture TRIBE Modal-Volume Extraction

## Verdict

- Ready: False
- App: `audience-vectors-dev`
- Modal volume: `attention-capture-features-v1`
- Output prefix: `attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610`
- Event mode: `full`
- Jobs: 350
- Written: 38
- Cached: 309
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

| dhf1k_003 | cached | 15 x 20484 | /attention-capture-features/attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610/dhf1k_003.npz |
| dhf1k_004 | cached | 31 x 20484 | /attention-capture-features/attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610/dhf1k_004.npz |
| dhf1k_007 | cached | 18 x 20484 | /attention-capture-features/attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610/dhf1k_007.npz |
| dhf1k_008 | cached | 19 x 20484 | /attention-capture-features/attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610/dhf1k_008.npz |
| dhf1k_009 | cached | 8 x 20484 | /attention-capture-features/attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610/dhf1k_009.npz |
| dhf1k_012 | cached | 33 x 20484 | /attention-capture-features/attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610/dhf1k_012.npz |
| dhf1k_013 | cached | 12 x 20484 | /attention-capture-features/attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610/dhf1k_013.npz |
| dhf1k_016 | cached | 22 x 20484 | /attention-capture-features/attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610/dhf1k_016.npz |
| dhf1k_017 | cached | 23 x 20484 | /attention-capture-features/attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610/dhf1k_017.npz |
| dhf1k_018 | cached | 22 x 20484 | /attention-capture-features/attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610/dhf1k_018.npz |
