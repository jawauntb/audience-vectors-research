# Visual-Gated BO/Sobol Smoke Result

Last updated: 2026-06-05.

This note records the first replay run after wiring the automated visual
artifact gate into `scripts/modal_bo_memorability_replay.py`. Raw reports and
generated MP4s are local ignored artifacts and are intentionally not committed.

## Run

The artifact-required preflight selected the same four one-replicate
seed-stratified BO/Sobol candidates used by the saved-table smoke: BO/Sobol for
the fireworks stratum and BO/Sobol for the jellyfish stratum.

```bash
BO_MEM_STEERING_ARTIFACT=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/Users/jawaun/isc_mod/data/artifacts/camilo_bo_memorability/v_mem.npz \
uv run --extra modal python scripts/modal_bo_memorability_replay.py \
  --selection seed-stratified-bo-vs-sobol \
  --stratify-by prompt \
  --max-evals 1 \
  --replicates 1 \
  --num-inference-steps 4 \
  --generation-timeout 1800 \
  --tribe-mode skip \
  --fail-on-visual-artifacts \
  --report-path data/reports/bo_visual_gated_smoke_20260605.json \
  --output-dir data/generated/bo_visual_gated_smoke_20260605
```

The run generated 4/4 videos, then exited nonzero as expected because the visual
gate failed. Because `--fail-on-visual-artifacts` was enabled, the script wrote
the report and skipped Modal volume upload and TRIBE scoring.

Local ignored artifacts:

- `data/reports/bo_visual_gated_smoke_20260605.json`
- `data/generated/bo_visual_gated_smoke_20260605/*.mp4`

## Gate Result

Top-level report fields confirmed that the visual gate became an actual
blocking verifier:

- `visual_gate_blocked_scoring: true`
- `visual_artifact_gate.n_videos: 4`
- `visual_artifact_gate.n_failed: 4`
- no generated row received a `modal_video_path`
- no generated row received a TRIBE score or status

| label | policy | stratum | visual flags | tail sharpness ratio | tail contrast ratio |
|---|---|---|---|---:|---:|
| `bo_replay_00_bo06_cand01` | BO | fireworks | `tail_sharpness_collapse`, `tail_contrast_collapse` | 0.1421 | 0.4545 |
| `bo_replay_01_sobol_007` | Sobol | fireworks | `tail_sharpness_collapse`, `tail_contrast_collapse` | 0.1283 | 0.3636 |
| `bo_replay_02_bo07_cand01` | BO | jellyfish | `tail_sharpness_collapse` | 0.2293 | 1.2691 |
| `bo_replay_03_sobol_005` | Sobol | jellyfish | `tail_sharpness_collapse` | 0.1434 | 1.6787 |

## Claim Impact

This run validates the gate behavior, not the candidate videos. The next replay
workflow can now reject degenerate generations before upload/TRIBE scoring. The
generation settings still fail the visual-quality criterion, so BO/Sobol score
comparisons should remain blocked until a generated panel passes this gate.

Reviewer-safe wording:

```text
After adding an automated visual artifact gate, a one-replicate seed-stratified
BO/Sobol smoke generated 4/4 videos but failed the gate for all four. The replay
script wrote a report and skipped upload/TRIBE scoring, confirming that visual
quality now acts as a blocking verifier rather than a post-hoc note.
```
