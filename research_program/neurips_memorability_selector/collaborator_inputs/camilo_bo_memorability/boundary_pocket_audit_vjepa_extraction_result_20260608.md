# Content-Pocket V-JEPA Extraction Result - 2026-06-08

## Discovery-Regime Audit

Question: can the missing exact V-JEPA artifact family be populated for the pocket-regime replay videos, so the embedding audit can test V-JEPA without using mismatched features?

Current regime:

- Artifact types: pocket-regime replay report rows, exact generated MP4s, V-JEPA `.npz` feature files, extraction status rows, and embedding-audit inputs.
- Operations: upload each exact local MP4 by bytes to the Modal `VjepaPredictor`, save one feature file by generated-video stem, and report exact feature coverage.
- Gates/verifiers: coverage is complete only if every scored replay-video row has a cached or newly written feature file.

## Result

- Replay report: `data/reports/bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_noise0_20260608.json`
- Output dir: `data/features/vjepa_boundary_pocket_audit_20260608`
- Jobs: 90
- Features available: 90
- Coverage complete: **True**
- Status counts: written: 90

## Next Move

Rerun `scripts/audit_content_pocket_embeddings.py` with `--vjepa-features-dir` pointing at this output directory. Only then may the claim ledger say whether V-JEPA passed or failed for the exact pocket replay videos.
