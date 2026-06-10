# DHF1K Modal Media Audit

- Labels CSV: `research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_20260609.csv`
- Modal path CSV: `research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_modal_20260610.csv`
- Modal volume: `bmd-videos-v1`
- Modal prefix: `/bmd-videos/attention_capture/DHF1K`
- Expected videos: **350**
- Found videos: **350**
- Missing videos: **0**
- Zero-byte videos: **0**
- Ready for full feature extraction: **True**
- Claim boundary: This Modal CPU audit checks whether the DHF1K media files needed for full-mode TRIBE extraction are mounted. It does not score TRIBE features or validate attentional capture.

## Blocking Reasons

- none

## Missing Preview

- none

## Found Preview

- `dhf1k_003` -> `/bmd-videos/attention_capture/DHF1K/video/003.AVI` (3,381,814 bytes)
- `dhf1k_004` -> `/bmd-videos/attention_capture/DHF1K/video/004.AVI` (6,245,522 bytes)
- `dhf1k_007` -> `/bmd-videos/attention_capture/DHF1K/video/007.AVI` (3,024,320 bytes)
- `dhf1k_008` -> `/bmd-videos/attention_capture/DHF1K/video/008.AVI` (3,870,350 bytes)
- `dhf1k_009` -> `/bmd-videos/attention_capture/DHF1K/video/009.AVI` (1,560,378 bytes)
- `dhf1k_012` -> `/bmd-videos/attention_capture/DHF1K/video/012.AVI` (6,860,040 bytes)
- `dhf1k_013` -> `/bmd-videos/attention_capture/DHF1K/video/013.AVI` (2,605,584 bytes)
- `dhf1k_016` -> `/bmd-videos/attention_capture/DHF1K/video/016.AVI` (2,082,520 bytes)
- `dhf1k_017` -> `/bmd-videos/attention_capture/DHF1K/video/017.AVI` (4,572,822 bytes)
- `dhf1k_018` -> `/bmd-videos/attention_capture/DHF1K/video/018.AVI` (4,698,590 bytes)
- `dhf1k_019` -> `/bmd-videos/attention_capture/DHF1K/video/019.AVI` (6,135,516 bytes)
- `dhf1k_020` -> `/bmd-videos/attention_capture/DHF1K/video/020.AVI` (4,108,480 bytes)
- `dhf1k_021` -> `/bmd-videos/attention_capture/DHF1K/video/021.AVI` (3,512,906 bytes)
- `dhf1k_024` -> `/bmd-videos/attention_capture/DHF1K/video/024.AVI` (4,220,490 bytes)
- `dhf1k_025` -> `/bmd-videos/attention_capture/DHF1K/video/025.AVI` (1,828,576 bytes)
- `dhf1k_030` -> `/bmd-videos/attention_capture/DHF1K/video/030.AVI` (2,955,592 bytes)
- `dhf1k_032` -> `/bmd-videos/attention_capture/DHF1K/video/032.AVI` (3,035,068 bytes)
- `dhf1k_034` -> `/bmd-videos/attention_capture/DHF1K/video/034.AVI` (3,552,704 bytes)
- `dhf1k_035` -> `/bmd-videos/attention_capture/DHF1K/video/035.AVI` (4,153,888 bytes)
- `dhf1k_036` -> `/bmd-videos/attention_capture/DHF1K/video/036.AVI` (4,021,642 bytes)
- `dhf1k_037` -> `/bmd-videos/attention_capture/DHF1K/video/037.AVI` (3,206,834 bytes)
- `dhf1k_038` -> `/bmd-videos/attention_capture/DHF1K/video/038.AVI` (7,958,590 bytes)
- `dhf1k_041` -> `/bmd-videos/attention_capture/DHF1K/video/041.AVI` (3,850,942 bytes)
- `dhf1k_043` -> `/bmd-videos/attention_capture/DHF1K/video/043.AVI` (4,069,610 bytes)
- `dhf1k_046` -> `/bmd-videos/attention_capture/DHF1K/video/046.AVI` (5,526,542 bytes)

## Full-Mode Extraction Command

Run this only after the audit reports ready:

```bash
uv run --extra modal python scripts/extract_attention_capture_tribe_features.py --source-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_modal_20260610.csv --output-dir data/features/tribe_dhf1k_attention_full --sample-id-column sample_id --media-path-column video_path --transport path --event-mode full --concurrency 8
```
