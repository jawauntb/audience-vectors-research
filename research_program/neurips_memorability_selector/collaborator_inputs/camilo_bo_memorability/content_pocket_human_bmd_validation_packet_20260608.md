# Content-Pocket Human/BMD Validation Packet

Date: 2026-06-08

## Purpose

Prepare the next validation spend for the SVD content-pocket result without
overclaiming compute-proxy evidence. The packet tests whether the strongest
TRIBE/V-JEPA compute-proxy pockets transfer to human memorability behavior or
measured-BMD grounding.

This packet is V-JEPA-caveated:

- `fresh24_orange_flowers` and `fresh24_hanging_clothes` are primary
  TRIBE/V-JEPA-verified compute-proxy candidates.
- `fresh24_blue_jellyfish` and `fresh24_old_car` are secondary boundary arms.
  They passed fresh TRIBE replication and CLIP-side boundary checks, but they
  did not pass the fresh boundary V-JEPA verifier.
- No pocket in this packet is a proven human-memorability or measured-BMD
  result yet.

## Source Evidence

Primary candidate evidence:

- `descriptor_conditioned_replication_manifest_20260608.md`
- `descriptor_conditioned_replication_result_20260608.md`
- `descriptor_conditioned_replication_vjepa_extraction_result_20260608.md`
- `descriptor_conditioned_replication_embedding_result_20260608.md`
- `descriptor_conditioned_replication_clip_diagnostic_result_20260608.md`

Boundary-arm evidence:

- `boundary_pocket_audit_manifest_20260608.md`
- `boundary_pocket_audit_result_20260608.md`
- `boundary_pocket_audit_vjepa_extraction_result_20260608.md`
- `boundary_pocket_audit_embedding_result_20260608.md`

Frozen validation stimuli:

- `content_pocket_validation_stimuli_manifest_20260608.json`
- `content_pocket_validation_stimuli_manifest_20260608.md`
- `content_pocket_validation_pairwise_tasks_20260608.json`
- `content_pocket_validation_prolific_survey_20260608.html`
- `content_pocket_validation_mp4_screening_20260608.json`
- `content_pocket_validation_mp4_screening_20260608.md`
- `content_pocket_validation_hosted_video_url_map_template_20260608.json`
- `content_pocket_validation_screening_sheets_20260608/`
- `content_pocket_recognition_memory_design_20260608.json`
- `content_pocket_recognition_memory_packet_20260608.md`
- `content_pocket_recognition_stimulus_production_manifest_20260608.json`
- `content_pocket_recognition_stimulus_production_manifest_20260608.md`

Local data-lake inputs:

- Primary replay report:
  `data/reports/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608.json`
- Primary generated videos:
  `data/generated/bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_steps50_motion5_noise0_20260608`
- Boundary replay report:
  `data/reports/bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_noise0_20260608.json`
- Boundary generated videos:
  `data/generated/bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_noise0_20260608`

## Candidate Tiers

| tier | pocket | role | compute status | packet use |
|---|---|---|---|---|
| primary | `fresh24_orange_flowers` | positive candidate | TRIBE positive; exact V-JEPA accepted prospectively; generated-video CLIP not accepted | primary human/BMD success gate |
| primary | `fresh24_hanging_clothes` | positive candidate | TRIBE positive; exact V-JEPA accepted prospectively; generated-video CLIP not accepted | primary human/BMD success gate |
| secondary | `fresh24_blue_jellyfish` | boundary positive | TRIBE positive; CLIP boundary verifier accepted; exact V-JEPA not accepted in boundary audit | exploratory generality arm |
| secondary | `fresh24_old_car` | boundary positive | TRIBE positive; CLIP boundary verifier accepted; exact V-JEPA not accepted in boundary audit | exploratory generality arm |
| control | `fresh24_aerial_beach` | hard negative | negative in primary and boundary audits | matched control |
| control | `fresh24_city_street` | hard negative | negative in primary and boundary audits | matched control |
| control | `fresh24_storm_beach` | hard negative | negative in primary and boundary audits | matched control |

## Stimulus Construction Rules

Use complete-candidate retained clips only.

- Preserve every generated, rejected, withheld, and failed artifact in the local
  data lake.
- For primary analysis, draw positive stimuli only from orange flowers and
  hanging clothes.
- Match hard-negative controls by Sobol recipe index and stochastic replicate
  schedule when possible.
- Keep blue jellyfish and old car in a separately labeled exploratory block.
  Do not pool them into the primary success test.
- Prefer task-level candidates with positive mean TRIBE score and no failed
  stochastic replicate.
- Preserve the three stochastic replicates for each chosen task-level candidate
  until the final survey/BMD stimulus set is frozen.
- Before launch, manually inspect the selected MP4s for semantic subject
  retention, frame collapse, text/watermark artifacts, and obvious
  attention-check leakage.

## Frozen Stimulus Set

The 2026-06-08 freeze selected the top two complete retained task-level
candidates per pocket in each analysis tier:

| tier | pocket | selected Sobol recipes |
|---|---|---|
| primary | `fresh24_orange_flowers` | `519`, `520` |
| primary | `fresh24_hanging_clothes` | `521`, `520` |
| exploratory boundary | `fresh24_blue_jellyfish` | `519`, `521` |
| exploratory boundary | `fresh24_old_car` | `522`, `518` |

Each selected candidate keeps all three stochastic replicates. Replicates are
paired with hard negatives from the same Sobol recipe index:

- `rep00` versus `fresh24_aerial_beach`
- `rep01` versus `fresh24_city_street`
- `rep02` versus `fresh24_storm_beach`

Frozen task pool:

- 24 blinded pairwise tasks.
- 12 primary orange/hanging tasks.
- 12 exploratory blue/old boundary tasks.
- 45 unique MP4 paths because matched controls can be reused when selected
  pockets share a Sobol recipe.
- No missing selected MP4s or missing matched controls.
- Task payload SHA-256:
  `b151326f1e120d7d6c6440c97e9341784bb3b25a1393b8ca7a8481fbcb3cef6c`.
- Agent sampled-frame MP4 pre-screening found 0 byte/hash/frame-gate failures
  across 45 stimuli and produced 4 contact sheets. Codex contact-sheet review
  found retained candidate subjects and visually distinct hard-negative
  controls, with no obvious sampled-frame text/watermark, frame collapse, or
  attention-check leakage. Final human/IRB-facing screening is still required.

## Recognition-Memory Upgrade

The direct actual-memory readout is now specified in:

- `content_pocket_recognition_memory_design_20260608.json`
- `content_pocket_recognition_memory_packet_20260608.md`

This design uses sparse Session 1 exposure followed by delayed Session 2
old-vs-lure recognition. Each participant sees only one old target from each
analysis arm, then later chooses the exact old clip against a newly generated
same-category lure.

Why this matters:

- It tests actual recognition memory rather than perceived memorability.
- It directly addresses the risk that orange-flower or hanging-clothes clips
  are too visually similar by requiring distinct same-category lures.
- It keeps hard negatives in the same old-vs-lure format, so primary positives
  must beat false-familiarity pressure rather than simply look more appealing.

Launch state:

- Designed: 15 frozen old targets, 15 required lure seed-image requests, six
  sparse forms, target 300 usable delayed participants, minimum 200 usable
  delayed participants before interpreting the gate.
- Production manifest built: 15 analysis lures, 25 filler old targets, 20
  filler lures, 60 seed-image requests, and 60 SVD generation jobs.
- Not launchable: 60 seed images, 60 generated MP4s, image/video screening
  sheets, hosted URLs, and final Prolific setup are still missing.

## Weaker Perceived-Memorability Pilot

Primary question:

Do humans judge orange-flower and hanging-clothes clips as more memorable than
matched hard negative controls?

Design:

- Blinded forced choice, positive pocket clip versus matched hard-negative
  control.
- Within-participant randomized side/order.
- At least two task-level candidates per primary pocket.
- Include all three hard-negative control pockets across the packet.
- Add attention checks that are not visually confusable with the target
  content-pocket clips.

Primary gate:

- Pooled primary-pocket choices exceed 50% under a pre-registered binomial or
  mixed-effects logistic test.
- Each primary pocket has a positive estimated effect direction.
- Report pocket-level effects separately even if the pooled test passes.

Secondary boundary gate:

- Analyze blue jellyfish and old car separately from the primary result.
- Treat positive results as generality evidence and negative or mixed results
  as a boundary on the content-pocket regime.

## Recommended BMD Or Measured-Brain Gate

Primary question:

Do primary candidate pockets show stronger measured-BMD or BMD-grounded
memorability-direction support than matched hard negatives?

Design:

- Use the frozen primary stimuli and matched hard-negative controls.
- Score or measure against the BMD memorability direction using the same
  preprocessing and split discipline as the accepted core selector work.
- Keep any TRIBE-only result labeled as proxy evidence unless measured-BMD or
  held-out BMD labels are explicitly involved.

Primary gate:

- Orange flowers and hanging clothes separate from hard negatives in the
  BMD-grounded readout.
- Pocket-level effects have the same sign for both primary pockets.
- Secondary boundary arms are analyzed separately and do not rescue a failed
  primary gate.

## Claim Language If The Packet Passes

Allowed:

- "Orange flowers and hanging clothes were TRIBE/V-JEPA compute-proxy
  candidates before human/BMD validation."
- "The human/BMD packet validated the primary content-pocket candidates under
  the pre-registered gate."
- "Blue jellyfish and old car were exploratory boundary arms."

Not allowed:

- "All four pockets were V-JEPA-verified in fresh replication."
- "Generated-video CLIP prospectively verified orange flowers and hanging
  clothes."
- "Proxy-only scores prove human memorability."
- "The current SVD prompt text caused the content-pocket effect."

## Next Action

For an actual human memorability claim, use the recognition-memory packet:
materialize the production manifest seed images, screen/contact-sheet them,
generate matched lure and filler MP4s, screen/contact-sheet the MP4s, freeze the
complete recognition set, and then run the two-session delayed Prolific study.

Alternative lower-claim paths:

1. run the blinded forced-choice survey as perceived-memorability evidence, or
2. run a measured-BMD/BMD-grounded transfer report.

Keep the compute-proxy packet and the human/BMD result as separate artifacts so
the claim ledger can distinguish candidate selection from validation.
