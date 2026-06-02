# Professor / IRB Brief

Date: 2026-06-01

## Project Title

Brain-Aligned Memorability Signals for Generated Video Selection

## One-Sentence Summary

We have built a computational selector that ranks generated video candidates by
a brain-aligned memorability signal; the next scientific step is a minimal-risk
online human study testing whether humans judge the selected clips as more
memorable than clips selected by non-brain baselines such as V-JEPA and CLIP.

## Why We Need IRB Guidance

The current evidence is computational and proxy-scored. We should not claim that
the selector improves human memorability until we collect independent human
responses. Because this requires online participants viewing videos and making
judgments, we need IRB review or an institutional determination that the study
is exempt/minimal risk.

## Current Scientific Status

Confirmed:

- TRIBE/BMD features predict BOLD Moments memorability at Spearman rho about
  +0.403 across five folds.
- V-JEPA is a strong baseline at about +0.395, so the study is not built around
  a weak comparator.
- TRIBE and V-JEPA disagree enough on generated candidates to make a human
  adjudication meaningful.
- The current Wan2.2 selector improves 18/24 prompts under the TRIBE/BMD proxy
  when using a conservative base-or-gated best-of-4 policy.

Not yet proven:

- TRIBE-selected generated videos are more memorable to humans.
- TRIBE beats V-JEPA, CLIP, or video-quality baselines in independent judgment.
- The selector is product-ready.
- Any LoRA/DPO model has learned human memorability.

## Proposed First Human Study

Participants recruited through Prolific view pairs of short videos generated for
the same prompt or image seed. They answer:

> Which video do you think would be more memorable if you saw many such clips?

The primary endpoint is whether humans choose the TRIBE+gate selected clip over
the strongest non-brain baseline at a rate above chance, using prompt-clustered
confidence intervals.

The current preflight packet freezes the V-JEPA-augmented task pool and survey
randomization metadata for review. The study is not launch-ready until the video
stimuli are hosted at stable HTTPS URLs, every hosted stimulus is screened, and
faculty/IRB status plus Prolific operational settings are finalized.

## Why This Is Minimal Risk

- Participants are adults.
- The task is a short online media-rating task.
- The study does not collect medical, clinical, biometric, political, sexual,
  religious, financial, or other sensitive personal data.
- Stimuli should be screened to exclude graphic violence, explicit sexual
  content, political persuasion, medical claims, and other sensitive content.
- Participants can stop at any time.
- The main risks are boredom, fatigue, mild visual discomfort, and ordinary
  online privacy risks.

## What We Need From A Professor

1. Advice on whether this should be submitted as exempt or expedited/minimal
   risk under the institution's IRB categories.
2. Review of the consent language and recruitment wording.
3. Confirmation that Prolific IDs and response data are handled acceptably.
4. Help naming the institutional PI or faculty sponsor, if required.
5. Feedback on whether the first study should measure predicted memorability
   only, or whether we should also include a delayed-recognition follow-up.

## Attachments In This Folder

- `irb_protocol_draft.md` - full protocol draft.
- `consent_form_draft.md` - participant consent draft.
- `prolific_launch_checklist.md` - operational checklist before launch.
- `professor_email_draft.md` - short email to send with the packet.
- `../experiments/prolific_launch_assets_2026-06-01/task_randomization_freeze.json`
  - prelaunch task/randomization metadata snapshot.

## Recommended Ask

Ask the professor to help convert this into an IRB submission and to advise
whether the first Prolific pilot can run under minimal-risk/exempt review.
