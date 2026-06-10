# Attention-Capture Feature Cache Audit

## Verdict

- Feature dir: data/features/tribe_dhf1k_attention_audio_only
- Ready for reuse: True
- NPZ files: 516
- Expected sample ids: 516
- Missing expected sample ids: 0
- Extra sample ids: 0
- Bad NPZ files: 0
- Shape mismatches: 0
- Total bytes: 786743319
- Aggregate SHA-256: 990c7605e215799d8d0afbb7a3fedaa7e4436af17e20703eb2c38f19747be776
- Archive URI: n/a
- Rerun commands: 3
- Ready for reproduction: True
- Claim boundary: This audit verifies cached TRIBE feature artifact integrity and manifest coverage. It does not validate attentional capture.

## Blocking Reasons

- none

## Counts

- Event modes: audio-only=516
- Transports: bytes=516
- Frame shapes: 10x20484=24, 11x20484=17, 12x20484=16, 13x20484=18, 14x20484=18, 15x20484=41, 16x20484=31, 17x20484=21, 18x20484=33, 19x20484=36, 20x20484=36, 21x20484=23, 22x20484=16, 23x20484=11, 24x20484=13, 25x20484=21, 26x20484=17, 27x20484=9, 28x20484=10, 29x20484=15, 30x20484=17, 31x20484=20, 32x20484=6, 33x20484=4, 34x20484=3, 35x20484=3, 36x20484=2, 37x20484=4, 38x20484=1, 39x20484=4, 40x20484=1, 41x20484=1, 43x20484=1, 52x20484=1, 7x20484=3, 8x20484=8, 9x20484=11

## Reproduction Path

- Archive URI: n/a
- `uv run python scripts/extract_attention_capture_tribe_features.py --source-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_extremes_20260608.csv --output-dir data/features/tribe_dhf1k_attention_audio_only --sample-id-column sample_id --media-path-column video_path --transport bytes --event-mode audio-only --concurrency 8`
- `uv run python scripts/extract_attention_capture_tribe_features.py --source-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_20260609.csv --output-dir data/features/tribe_dhf1k_attention_audio_only --sample-id-column sample_id --media-path-column video_path --transport bytes --event-mode audio-only --concurrency 8`
- `uv run python scripts/audit_attention_capture_feature_cache.py --feature-dir data/features/tribe_dhf1k_attention_audio_only --display-feature-dir data/features/tribe_dhf1k_attention_audio_only --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_audio_only_manifest_20260609.json --manifest research_program/dopamine_detox_attention_capture/phase1_dhf1k_fixation_density_audio_only_manifest_20260609.json --output-json research_program/dopamine_detox_attention_capture/results/dhf1k_audio_only_feature_cache_audit_20260610.json --output-md research_program/dopamine_detox_attention_capture/results/dhf1k_audio_only_feature_cache_audit_20260610.md`

## File Preview

| path | sample_id | shape | event_mode | size | sha256 prefix |
|---|---|---|---|---:|---|
| dhf1k_002.npz | dhf1k_002 | 27x20484 | audio-only | 2046030 | d0a264ac7494 |
| dhf1k_003.npz | dhf1k_003 | 15x20484 | audio-only | 1140317 | 9ca9790ef2af |
| dhf1k_004.npz | dhf1k_004 | 31x20484 | audio-only | 2362308 | b41f9db21d9d |
| dhf1k_005.npz | dhf1k_005 | 16x20484 | audio-only | 1219233 | eb056db79d89 |
| dhf1k_007.npz | dhf1k_007 | 18x20484 | audio-only | 1372740 | 16cac6559c80 |
| dhf1k_008.npz | dhf1k_008 | 19x20484 | audio-only | 1448692 | a2a867dad3e2 |
| dhf1k_009.npz | dhf1k_009 | 8x20484 | audio-only | 605990 | 87b94dcf43f8 |
| dhf1k_010.npz | dhf1k_010 | 20x20484 | audio-only | 1527134 | a6e8f55a67ab |
| dhf1k_011.npz | dhf1k_011 | 24x20484 | audio-only | 1825551 | 1a26f17c17db |
| dhf1k_012.npz | dhf1k_012 | 33x20484 | audio-only | 2509614 | 9f7a8c409a81 |
| dhf1k_013.npz | dhf1k_013 | 12x20484 | audio-only | 914780 | 5ff81756461f |
| dhf1k_014.npz | dhf1k_014 | 15x20484 | audio-only | 1143385 | ec5f33eb0c86 |
| dhf1k_015.npz | dhf1k_015 | 34x20484 | audio-only | 2587737 | c7aa0f89dec5 |
| dhf1k_016.npz | dhf1k_016 | 22x20484 | audio-only | 1673292 | 32d7866db12f |
| dhf1k_017.npz | dhf1k_017 | 23x20484 | audio-only | 1753575 | 5aac82da4762 |
| dhf1k_018.npz | dhf1k_018 | 22x20484 | audio-only | 1667999 | d7d20222708e |
| dhf1k_019.npz | dhf1k_019 | 30x20484 | audio-only | 2285391 | 647dc45b614b |
| dhf1k_020.npz | dhf1k_020 | 19x20484 | audio-only | 1443721 | cb10dbff8b66 |
| dhf1k_021.npz | dhf1k_021 | 17x20484 | audio-only | 1291368 | 0a479a2314d3 |
| dhf1k_022.npz | dhf1k_022 | 15x20484 | audio-only | 1145665 | b3b7a348be27 |
