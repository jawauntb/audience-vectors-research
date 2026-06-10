# DHF1K Full-Mode Modal-Volume Checkpoint

- Generated: `2026-06-10T22:02:09.813150+00:00`
- Modal volume: `attention-capture-features-v1`
- Output prefix: `attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610`
- Persisted: **307 / 350**
- Remaining: **43**
- Known error IDs from interrupted run: `dhf1k_146, dhf1k_203, dhf1k_262`
- Claim boundary: Checkpoint only. It verifies Modal-volume persistence coverage, not Phase 1 hypothesis validity.

## Missing Sample IDs

- `dhf1k_146`
- `dhf1k_203`
- `dhf1k_262`
- `dhf1k_616`
- `dhf1k_617`
- `dhf1k_620`
- `dhf1k_622`
- `dhf1k_625`
- `dhf1k_626`
- `dhf1k_627`
- `dhf1k_629`
- `dhf1k_630`
- `dhf1k_636`
- `dhf1k_638`
- `dhf1k_639`
- `dhf1k_641`
- `dhf1k_643`
- `dhf1k_644`
- `dhf1k_647`
- `dhf1k_650`
- `dhf1k_651`
- `dhf1k_652`
- `dhf1k_654`
- `dhf1k_655`
- `dhf1k_656`
- `dhf1k_660`
- `dhf1k_662`
- `dhf1k_663`
- `dhf1k_666`
- `dhf1k_668`
- `dhf1k_669`
- `dhf1k_673`
- `dhf1k_674`
- `dhf1k_676`
- `dhf1k_677`
- `dhf1k_680`
- `dhf1k_686`
- `dhf1k_687`
- `dhf1k_689`
- `dhf1k_691`
- `dhf1k_697`
- `dhf1k_699`
- `dhf1k_700`

## Resume Command

```bash
set -a; source /Users/jawaun/isc_mod/.env; set +a; uv run --extra modal python scripts/extract_attention_capture_tribe_features_modal_volume.py --source-csv research_program/dopamine_detox_attention_capture/dhf1k_attention_labels_fixation_density_extremes_modal_20260610.csv --output-json research_program/dopamine_detox_attention_capture/results/dhf1k_full_modal_volume_extraction_20260610.json --output-md research_program/dopamine_detox_attention_capture/results/dhf1k_full_modal_volume_extraction_20260610.md --sample-id-column sample_id --media-path-column video_path --event-mode full --concurrency 8
```
