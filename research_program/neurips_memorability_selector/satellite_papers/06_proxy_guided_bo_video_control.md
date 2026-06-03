# Multi-Objective Proxy-Guided Search For Brain-Aligned Memorability In Generated Video

**Draft status:** collaborator intake / satellite stub, started 2026-06-03.
**Use in main paper:** future-work/control evidence only until reproduced and
human-validated.

## Core Idea

The collaborator BO prototype treats video generation as an expensive
black-box optimization problem. It searches over SVD-XT steering parameters and
seed choices, then scores each candidate with proxy objectives:

- TRIBE/BMD memorability projection;
- CLIP prompt/image fidelity;
- R3D-18 similarity to neutral reference videos as a lightweight quality
  proxy.

The interesting claim is not that this is cheap. The interesting claim is that
Bayesian optimization may be **sample-efficient** under a fixed evaluation
budget.

## Current Evidence

Imported collaborator assets live in:

```text
research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/
```

The imported 3-objective run table reports:

- `32` total evaluations;
- `12` Sobol initial evaluations;
- `10` BO rounds with batch size `2`;
- best observed TRIBE proxy score about `6.15`;
- saved hypervolume history and figure artifacts.

The generated videos and model-weight artifacts are not committed.

## Claim Boundary

Safe:

```text
BO is a plausible sample-efficient proxy search policy for generated-video
candidate selection.
```

Unsafe:

```text
BO proves controllable human memorability.
```

The `v_mem_CLIP` bridge is the critical provenance item. The result should not
be treated as evidence until the exact construction from cortical `v_mem` is
audited and reproduced.

## Compute-Cost Framing

If the reported runtime is `32` evaluations in about `9` hours on an RTX 5080,
that is a real limitation but not fatal. The paper should say:

```text
BO is sample-efficient but not yet wall-clock efficient.
```

Report:

- total evaluations;
- wall-clock runtime;
- hardware;
- average minutes per evaluation;
- whether generation or scoring dominates;
- fixed-budget comparison against random/Sobol/best-of-N.

## Modal Reproduction Plan

Use:

```bash
uv run python scripts/modal_bo_memorability_replay.py --dry-run
```

Then with external artifacts:

```bash
BO_MEM_STEERING_ARTIFACT=/path/to/tribe_clip_adapter.pt \
BO_MEM_CORTICAL_VMEM=/path/to/v_mem.npz \
uv run python scripts/modal_bo_memorability_replay.py \
  --selection top-tribe \
  --max-evals 2 \
  --num-inference-steps 8
```

Scale after smoke validation:

1. 2-4 eval Modal smoke run.
2. 32-eval fixed-table reproduction.
3. Equal-budget random/Sobol/best-of-N baseline.
4. Top-video visual inspection.
5. Optional larger prompt/seed replication.

## Main-Paper Boundary

This satellite should not block the human pilot. It can strengthen the compute
story after reproduction, but the main selector claim still needs independent
human validation against V-JEPA, CLIP/preservation, and base/random baselines.
