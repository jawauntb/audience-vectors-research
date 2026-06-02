# Human Evaluation Summary For Faculty / IRB Review

## Project Summary

We are studying whether model-derived video selection scores can help choose
generated clips that humans later judge as more memorable. The current pilot
compares generated-video selector policies on the same prompts. Participants see
two short videos from the same prompt and answer which one they expect would be
more memorable if encountered among many similar clips.

## Scientific Motivation

Prior computational work suggests a stable memorability-like direction in
brain-encoding model features. The unresolved question is whether this direction
is useful as a practical selector for generated videos when compared against
non-brain baselines such as CLIP preservation and V-JEPA memorability proxies.

## Participant Task

Participants complete a short online pairwise-preference survey. Each trial
shows two videos generated from the same underlying prompt. Participants answer a
single forced-choice question about which clip is more memorable. The current
survey samples 24 trials per participant from a frozen 185-task pool.

## Data Collected

- Prolific participant identifier or anonymous study identifier, depending on
  final platform configuration
- Trial-level left/right video assignment
- Participant choice
- Trial timing and browser-completion metadata if enabled by the survey host
- Attention-check responses if added before launch

No medical, biometric, EEG, fMRI, or clinical data is collected in this pilot.

## Risk Level

The planned pilot is minimal risk if the video stimuli are screened to exclude
sensitive, graphic, sexual, medical, political, or distressing material. The task
is a short media-preference judgment and does not ask for private personal
information beyond standard study-platform metadata.

## Analysis Plan

The primary endpoint is prompt-clustered win rate of the TRIBE/BMD gated
selector against the strongest non-brain baseline. Planned analyses include:

- Prompt-clustered bootstrap confidence interval
- Mixed-effects logistic regression with selector policy as a fixed effect
- Random intercepts for prompt and, when participant identifiers are available,
  participant
- Per-prompt failure-case review

## Current Limitations

The current computational scores are not independent human validation. The pilot
only becomes evidence after participant responses are collected and analyzed
against predeclared comparisons. Delayed recognition would be a stronger memory
endpoint but is not the first pilot because it is more expensive and operationally
heavier.
