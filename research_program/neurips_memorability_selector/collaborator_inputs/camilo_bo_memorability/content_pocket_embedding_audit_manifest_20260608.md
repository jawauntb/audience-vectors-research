# Content-Pocket Embedding Audit Manifest - 2026-06-08

## Discovery-Regime Audit

Question: do stronger CLIP/V-JEPA-style embeddings explain the stable positive
content pockets that survived the pocket-regime audit after lightweight visual
descriptors failed their gate?

Current regime:

- Artifact types: restored SVD seed images, generated SVD replay videos, TRIBE
  replay scores, pocket labels, CLIP seed-image embeddings, CLIP frame-pooled
  generated-video embeddings, optional exact V-JEPA generated-video embeddings,
  descriptor metrics, leakage-aware classifier results, and claim-ledger
  entries.
- Operations: load the exact pocket-regime replay report, map task rows back to
  restored seed slots, encode seed images and deterministic generated-video
  frame samples with CLIP, optionally load exact V-JEPA `.npz` embeddings by
  generated-video stem, aggregate replicate videos to task-level embeddings,
  compute pocket-held-out centroid margins, train leave-one-pocket-out
  classifiers, and update accepted/rejected claims.
- Gates/verifiers: no descriptor may use TRIBE replay score as an input.
  Acceptance requires either a non-score embedding descriptor or a
  leakage-aware classifier to separate positive pockets from hard negative
  controls.
- Known limitations: this is still compute-proxy evidence. Passing the
  embedding gate does not prove human memorability, delayed recognition, or a
  prompt-conditioned generation mechanism. V-JEPA may only be claimed if exact
  features exist for these replay-video stems.

Action class:

- Search inside the current compute-proxy regime, with discovery relevance if
  an embedding descriptor becomes an accepted verifier for the content-pocket
  residual.

Experiment:

- Source replay report:
  `data/reports/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608.json`
- Script: `scripts/audit_content_pocket_embeddings.py`
- Output summary:
  `content_pocket_embedding_audit_summary_20260608.json`
- Output note:
  `content_pocket_embedding_audit_result_20260608.md`
- Positive targets: `fresh24_orange_flowers`, `fresh24_hanging_clothes`,
  `fresh24_blue_jellyfish`, `fresh24_old_car`.
- Negative controls: `fresh24_aerial_beach`, `fresh24_city_street`,
  `fresh24_storm_beach`.
- Stress tests: pocket-held-out centroid margins and leave-one-pocket-out
  classifiers so repeated recipes inside the same seed/content pocket cannot
  fake generalization.

Gate:

- Descriptor acceptance: `separation_auc >= 0.85` and `abs_cohen_d >= 1.00`.
- Classifier acceptance: leave-one-pocket-out `roc_auc >= 0.85` and
  `balanced_accuracy >= 0.75`.
- Withheld/rejected rule: if exact V-JEPA feature coverage is incomplete, do not
  mix in mismatched Wan/BMD V-JEPA features; record V-JEPA as missing for this
  replay audit.

Expected interpretation:

- If accepted, update C-017/C-018 and G-010 to say the content-pocket residual
  now has an accepted embedding verifier, while keeping human/BMD validation
  open.
- If rejected, keep C-017 scoped as a stable but black-box TRIBE compute-proxy
  content-pocket finding and prioritize exact V-JEPA, measured-BMD, or human
  validation before further blind replication.
