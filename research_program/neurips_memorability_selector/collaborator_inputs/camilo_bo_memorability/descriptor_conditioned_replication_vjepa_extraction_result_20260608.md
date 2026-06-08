# Descriptor-Conditioned Replication V-JEPA Extraction Result - 2026-06-08

## Discovery-Regime Audit

Question: can exact V-JEPA features be populated for the fresh descriptor-conditioned replication videos, so the embedding verifier can test V-JEPA on the new MP4 bytes without reusing the older pocket-regime features?

Current regime:

- Artifact types: descriptor-conditioned replication report rows, exact generated MP4s, V-JEPA `.npz` feature files, extraction status rows, and embedding-audit inputs.
- Operations: upload each exact local MP4 by bytes to the Modal `VjepaPredictor`, save one feature file by generated-video stem, and report exact feature coverage.
- Gates/verifiers: coverage is complete only if every scored replay-video row has a cached or newly written feature file.

## Result

- Replay report: `data/reports/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608.json`
- Output dir: `data/features/vjepa_descriptor_conditioned_replication_20260608`
- Jobs: 90
- Features available: 90
- Coverage complete: **True**
- Status counts: written: 90

## Next Move

The embedding verifier was run with this directory in
`descriptor_conditioned_replication_embedding_result_20260608.md`. Use the
controlling replication result note,
`descriptor_conditioned_replication_result_20260608.md`, for the final gate
interpretation.
