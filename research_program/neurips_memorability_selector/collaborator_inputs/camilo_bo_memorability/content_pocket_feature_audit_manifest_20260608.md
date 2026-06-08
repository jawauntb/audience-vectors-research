# Content-Pocket Feature Audit Manifest - 2026-06-08

## Discovery-Regime Audit

Question: can lightweight visual descriptors explain the stable positive
content pockets from the pocket-regime audit, or should C-017 remain a
black-box TRIBE compute-proxy pocket finding?

Current regime:

- Artifact types: restored seed images, generated SVD replay videos, TRIBE
  replay scores, visual-gate status, seed/video visual descriptors, positive
  pocket labels, hard negative control labels, and result notes.
- Operations: join the pocket-regime replay report to restored seed images,
  sample generated-video frames, compute descriptor families, compare positive
  pockets against hard negative controls, and preserve accepted/rejected
  explanations.
- Gates/verifiers: descriptor separation cannot use TRIBE score as an input
  feature. A descriptor family is accepted only if it separates positive pockets
  from hard negative controls with `separation_auc >= 0.85` and
  `abs_cohen_d >= 1.00`.
- Known limitation: lightweight descriptors are not CLIP, V-JEPA, or human
  evidence. Passing this gate would only justify a stronger mechanistic probe,
  not a human memorability claim.

Action class: search inside the current compute-proxy regime. It becomes a
discovery-relevant regime addition only if the descriptor becomes an accepted
verifier or artifact class for content-pocket consolidation.

## Inputs

- Replay report:
  `data/reports/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608.json`
- Generated video directory:
  `data/generated/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608`
- Restored seed root:
  `research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/original`
- Positive targets:
  `fresh24_orange_flowers`, `fresh24_hanging_clothes`,
  `fresh24_blue_jellyfish`, `fresh24_old_car`
- Hard negative controls:
  `fresh24_aerial_beach`, `fresh24_city_street`, `fresh24_storm_beach`

## Descriptor Families

- Seed-image descriptors: colorfulness, brightness, saturation, RGB means,
  warm/cool balance, hue-region fractions, edge/texture proxies, entropy, and
  center/border brightness contrast.
- Generated-video descriptors: the same descriptor set averaged over a
  deterministic five-frame sample per generated replay video, then averaged to
  task level across stochastic replicates.

## Command

```bash
uv run python scripts/audit_content_pocket_features.py \
  --replay-report data/reports/bo_pocket_regime_audit_sobol518_523_x7_reps2_steps50_motion5_noise0_20260608.json
```

In a clean worktree, pass `--replay-report` to the local data-lake copy of that
ignored report.

## Acceptance Rule

Accept a descriptor-level explanation only if at least one seed-image or
generated-video descriptor satisfies both:

- `separation_auc >= 0.85` for positive pockets versus hard negative controls;
- `abs_cohen_d >= 1.00`.

If no descriptor clears the gate, preserve the near-miss descriptors but keep
C-017 scoped as a black-box compute-proxy content-pocket finding pending a
stronger CLIP/V-JEPA/human/BMD verifier.
