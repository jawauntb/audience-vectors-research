# Generated-Video Content Pockets Predict Delayed Human Recognition Memory

**Draft status:** standalone satellite manuscript draft, started 2026-06-12.
**Result status:** Wave 2 human recognition-memory data analyzed; narrow claim
only.
**Use in main paper:** human-behavior validation for the SVD content-pocket
regime, not a replacement for the broader TRIBE/V-JEPA selector pilot.

## Abstract

Brain-predictive and self-supervised video models can rank generated videos, but
proxy scores do not by themselves establish that people remember the selected
clips. We test a narrow content-pocket prediction from a Stable Video Diffusion
replay regime: two positive content pockets, orange flowers and hanging clothes,
were first stabilized by TRIBE/BMD replay residuals and exact V-JEPA verifier
geometry against hard negative controls. We then froze an old-vs-lure delayed
recognition task and ran a two-wave Prolific study. In the complete Wave 2
sample, 62 participants submitted 25 forced-choice recognition trials each.
Excluding media-error flagged trials, primary positive pockets were recognized
on 114/123 trials (92.7%, Wilson 95% CI [86.7%, 96.1%]), compared with 150/186
hard-negative-control trials (80.6%, [74.4%, 85.7%]). The paired
participant-level primary-positive minus hard-negative contrast was +11.7
percentage points (bootstrap 95% CI [+4.4, +19.4], sign-flip p = 0.00425).
Hanging clothes was individually robust (58/61, 95.1%); orange flowers was high
in absolute recognition (56/62, 90.3%) but weaker as a standalone contrast
against the hard-negative pool. These results support a narrow claim:
compute-proxy content pockets can predict delayed human recognition-memory
advantage for generated videos. They do not establish broad human memorability,
measured-BMD grounding, prompt-conditioned generation control, or a general
BO/SVD optimization result.

## Figures

![Delayed recognition accuracy by content arm](../collaborator_inputs/camilo_bo_memorability/figures/content_pocket_recognition_accuracy_20260612.svg)

![Primary pockets exceed hard controls](../collaborator_inputs/camilo_bo_memorability/figures/content_pocket_recognition_contrast_20260612.svg)

![Pocket-specific lift vs hard controls](../collaborator_inputs/camilo_bo_memorability/figures/content_pocket_recognition_arm_contrasts_20260612.svg)

## 1. Introduction

Generated-video pipelines increasingly make the search problem cheap enough to
ask cognitive questions: which generated clips will people remember? The
project's broader selector work uses TRIBE/BMD, V-JEPA, and CLIP-like
representations to rank candidate videos, but a proxy-selected clip is not a
human-memory result until it survives a behavioral endpoint.

This paper isolates one small, auditable step. A collaborator SVD replay audit
found that score residuals were dominated by content pockets rather than by
alpha/guidance recipe identity. Orange flowers and hanging clothes emerged as
the strongest non-jellyfish positive pockets. Aerial beach, city street, and
storm beach were retained as hard negative controls. The compute-side result was
not accepted as human memorability; it was accepted as a candidate-selection
regime that justified a direct old-vs-lure recognition study.

The contribution here is therefore intentionally modest:

1. We translate a proxy content-pocket finding into a frozen two-session human
   recognition-memory task.
2. We preserve same-category lures and hard negative controls so the task is
   not merely "flowers are familiar."
3. We show that the primary positive pockets are recognized more accurately
   than hard negative controls after a delayed Session 2 wave.
4. We keep the result's scope narrow: exact recognition for this content-pocket
   packet, not broad memorability or generator steering.

## 2. Prior Compute-Proxy Regime

The accepted compute regime before the human study consisted of:

- seed images and image-conditioned SVD generations;
- fixed Sobol recipe neighborhoods;
- visual gates and complete-candidate retention;
- TRIBE/BMD replay residuals;
- exact V-JEPA video features and centroid-margin/classifier verifiers;
- CLIP diagnostics, with generated-video CLIP failing the fresh prospective
  verifier and prompt/seed CLIP retained only as an ancillary descriptor.

The key compute-side claims were:

- orange flowers and hanging clothes are fresh-seed TRIBE/V-JEPA-verified
  compute-proxy candidate pockets;
- hard negative controls remain negative under matched recipe neighborhoods;
- the result is content-pocket behavior inside the current SVD replay regime,
  not broad BO/control superiority;
- no human memorability or measured-BMD claim is allowed without a human/BMD
  validation gate.

The present study is that first human-behavior gate.

## 3. Human Recognition-Memory Study

### 3.1 Stimuli

The analysis arms were:

| Arm | Group | Role |
|---|---|---|
| `orange_flowers` | primary positive | TRIBE/V-JEPA-verified candidate pocket |
| `hanging_clothes` | primary positive | TRIBE/V-JEPA-verified candidate pocket |
| `aerial_beach` | hard negative control | matched negative content pocket |
| `city_street` | hard negative control | matched negative content pocket |
| `storm_beach` | hard negative control | matched negative content pocket |

Each participant saw one old target from each analysis arm in Session 1. Session
2 presented the old target against a newly generated same-category lure. The
sparse-form design prevented any participant from seeing multiple old targets
from the same analysis arm. Fillers were included to stabilize the cover task
and reduce the salience of the five analysis arms.

### 3.2 Procedure

Session 1 was a viewing/cover-task session. Session 2 was a delayed
forced-choice recognition task: for each pair, participants chose which clip
they had seen in Session 1. Participant IDs were hashed into one of six forms,
and the same deterministic form assignment was used across both sessions.

The endpoint is exact old-vs-lure recognition accuracy. This is stronger than a
preference or "which is more memorable" judgment, but narrower than long-term
free recall or measured neural validation.

### 3.3 Data Integrity

The Wave 2 Prolific export contained 64 rows: 60 approved, 2 awaiting review,
and 2 timed out. The webhook export contained complete real Session 2 payloads
for all 62 completed submissions and none for the two timed-out submissions.
There were no duplicate real Prolific IDs in complete Session 2 payloads.

The Wave 1 webhook inbox was capped before the paid upgrade, so the full Session
1 JSON payload set is not available. This is a limitation. The retained Wave 1
subset contains 24 complete payloads; 21 overlap with complete Wave 2
participants, and all 21 have matching deterministic form assignments. The
human result is therefore reported as Prolific-confirmed Wave 1 exposure plus
complete Session 2 recognition, with partial Wave 1 JSON retention noted as a
provenance caveat.

## 4. Results

Primary analyses exclude media-error flagged trials. The no-media-error table
is the manuscript-facing table; all-trial sensitivity is preserved in the
companion result note.

| Group | Correct / n | Accuracy | Wilson 95% CI | Exact p vs 0.5 |
|---|---:|---:|---:|---:|
| Primary positives | 114 / 123 | 0.927 | [0.867, 0.961] | 2.68e-24 |
| Hard negative controls | 150 / 186 | 0.806 | [0.744, 0.857] | 9.63e-18 |
| Unrelated fillers | 991 / 1238 | 0.800 | [0.777, 0.822] | 7.69e-106 |
| Orange flowers | 56 / 62 | 0.903 | [0.805, 0.955] | 2.97e-11 |
| Hanging clothes | 58 / 61 | 0.951 | [0.865, 0.983] | 3.29e-14 |
| Aerial beach | 52 / 62 | 0.839 | [0.728, 0.910] | 5.71e-08 |
| City street | 48 / 62 | 0.774 | [0.656, 0.860] | 1.74e-05 |
| Storm beach | 50 / 62 | 0.806 | [0.691, 0.886] | 1.21e-06 |

At the participant level, the primary-positive minus hard-negative contrast was
positive:

```text
+11.7 percentage points
bootstrap 95% CI: [+4.4, +19.4]
sign-flip permutation p = 0.00425
complete paired participants after media-error exclusion: 61
```

The all-trial sensitivity analysis gives the same qualitative result:

```text
+12.1 percentage points
bootstrap 95% CI: [+4.8, +19.9]
sign-flip permutation p = 0.00287
complete paired participants: 62
```

The pocket-specific paired contrasts show why the packet-level claim is the
right claim. Hanging clothes exceeded the hard-control pool by +14.2 percentage
points (bootstrap 95% CI [+6.6, +21.9], sign-flip p = 0.00069). Orange flowers
was high in absolute recognition but weaker as a standalone contrast: +9.7
percentage points, bootstrap 95% CI [-0.5, +19.4], sign-flip p = 0.08285.

### 4.1 Interpretation

The hard negative controls were also recognized well above chance. This matters:
the result is not "only the positive pockets are memorable." The supported
claim is a relative advantage: the two primary positive pockets were recognized
more accurately than the hard-negative-control pool under the same sparse
old-vs-lure design.

Hanging clothes is individually robust against the control pool. Orange flowers
is highly recognized in absolute terms, but its standalone positive-vs-control
contrast is weaker in this Wave 2 sample. The honest packet-level claim is
therefore stronger than the individual orange-flowers claim.

## 5. Discussion

This study upgrades the content-pocket line from compute-proxy evidence to a
narrow human behavioral result. The result is scientifically useful because the
selection path was prospective at the level that matters for claims:

```text
TRIBE/V-JEPA content-pocket candidates -> frozen old-vs-lure stimuli -> delayed human recognition endpoint
```

The result does not imply that any arbitrary TRIBE-high generated video is more
memorable to humans. It says that, in this SVD content-pocket packet, the
primary positive content pockets selected by the compute regime were later
recognized more accurately than hard negative controls.

The most important scientific reading is that content identity can be a real
memorability-bearing axis in generated-video candidate selection. That should
shape the next selector paper: content-pocket controls, same-category lures, and
human old-vs-lure endpoints are necessary to avoid overclaiming from proxy
scores.

## 6. Limitations

First, the originally drafted full response-analysis plan named a larger minimum
usable sample. This Wave 2 result is strong enough to draft around, but it
should be presented as the first human recognition-memory validation wave rather
than as a large-sample final confirmation.

Second, the full Wave 1 webhook payload set was not retained because the webhook
inbox was capped before upgrade. Prolific completion and deterministic form
assignment support the exposure record, and the retained Wave 1 subset shows
zero form mismatches, but complete Session 1 JSON provenance is absent.

Third, the task measures two-alternative exact recognition against same-category
lures. It does not measure free recall, long-term memory, engagement,
preference, quality, emotion, virality, or commercial outcomes.

Fourth, the positive pockets are content categories. This result does not prove
that alpha/guidance recipes, BO search, or prompt text caused memorability
improvements. In the current SVD runner, prompt text is metadata-only for
generation.

Fifth, BMD and measured-fMRI grounding remain separate gates. TRIBE/V-JEPA
helped choose the packet, but this human study is behavioral recognition
evidence, not measured neural validation.

## 7. Claim Contract

Allowed now:

- The content-pocket packet has delayed human old-vs-lure recognition evidence.
- Primary positives, pooled across orange flowers and hanging clothes, exceed
  hard negative controls in participant-level recognition accuracy.
- Hanging clothes is individually robust in this wave.
- Orange flowers has high absolute recognition and supports the pooled primary
  result, but is weaker as a standalone contrast.
- The result validates a content-pocket candidate-selection regime, not a broad
  generator-control mechanism.

Not allowed:

- "Human memorability is solved."
- "BO-generated videos are broadly more memorable to humans."
- "TRIBE/V-JEPA scores are human memory."
- "Prompt rewriting controls memorability in the current SVD runner."
- "The result is measured-BMD or fMRI validation."

## 8. Provenance

Committed protocol and launch artifacts live under:

```text
research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/
```

Key protocol files:

- `content_pocket_recognition_memory_packet_20260608.md`
- `content_pocket_recognition_response_analysis_plan_20260608.md`
- `content_pocket_recognition_launch_assets_20260608.md`
- `content_pocket_recognition_session1_prolific_20260608.html`
- `content_pocket_recognition_session2_prolific_20260608.html`

Wave 2 result note and charts:

- `content_pocket_recognition_response_analysis_result_20260612.md`
- `figures/content_pocket_recognition_accuracy_20260612.svg`
- `figures/content_pocket_recognition_contrast_20260612.svg`
- `figures/content_pocket_recognition_arm_contrasts_20260612.svg`

Raw Prolific/webhook exports are intentionally kept in ignored local `data/`
paths and should not be committed because they contain participant metadata.
