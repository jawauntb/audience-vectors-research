# Affect-Aware Media Selection From Brain-Aligned And EEG-Inspired Signals

**Draft status:** satellite project seed, started 2026-06-01.
**Core purpose:** extend the memorability-selector program into affect while
keeping the current main paper clean.

## Abstract

Memorability is only one cognitive effect of generated media. A richer selector
should also estimate affect: whether a video is likely to feel happy, angry,
fearful, sad, aversive, or neutral. The NOVA dataset motivates a tractable
signal-processing baseline for this direction: compute EEG power spectral
density features from video-evoked recordings and train simple supervised
classifiers for emotion labels. Our current analyzer does not collect EEG, so it
cannot perform NOVA-style decoding directly. Instead, it exposes a
NOVA-inspired media proxy using TRIBE engagement dimensions plus palette,
motion, edge, density, arousal, and valence proxies. The scientific plan is to
validate whether these media-derived affect estimates align with EEG-derived or
human-rated affect labels, then test whether affect and memorability can be
jointly optimized during generated-video selection.

## Why This Is Separate From The Main Paper

The main paper is about memorability selection. NOVA is about EEG affect
decoding. Mixing them into one headline would create a reviewer problem: we
would be asking one paper to prove memory, emotion, brain alignment, and
generation selection at the same time. The right structure is:

- main paper: generated-video memorability selection;
- satellite paper: affect-aware media selection;
- product feature: affect proxy for exploratory creative diagnosis;
- future validation: EEG or human affect labels.

## Current Product Implementation

The Modal analyzer now returns an `affect_profile` in each analysis summary and
an `affect_proxy` for each segment. The classes are:

- happy;
- anger;
- fear;
- sadness;
- disgust;
- neutral.

The implementation is intentionally labeled:

```text
NOVA-inspired media proxy from TRIBE dimensions plus palette/motion/density;
not EEG PSD decoding.
```

This lets users diagnose affective shape without pretending the site has access
to neural recordings.

## Proper NOVA-Style Research Version

When EEG data are available, the correct pipeline is:

```text
video stimulus
  -> EEG recording
  -> preprocessing / artifact handling
  -> power spectral density by channel and frequency
  -> supervised ridge or logistic-ridge emotion classifier
  -> emotion class probabilities
```

Differential entropy can be included as a baseline, but PSD should remain the
default richer feature because it preserves more frequency-specific structure.

## Scientific Questions

1. Do TRIBE-derived emotion and media-derived affect proxies predict human
   affect labels?
2. Do they predict EEG-derived NOVA-style emotion classes?
3. Are high-memorability clips also high-arousal, or can memory and affect be
   separated?
4. Does affect-aware gating improve generated-video selection without reducing
   prompt preservation or visual quality?
5. Where do TRIBE, V-JEPA, CLIP, EEG, and human affect ratings agree or
   disagree as representation frames?

## Minimum Validation Plan

1. Curate or generate prompt-conditioned candidate sets.
2. Score each candidate with:
   - TRIBE/BMD memorability;
   - TRIBE raw emotion;
   - NOVA-inspired media affect proxy;
   - V-JEPA and CLIP baselines.
3. Collect human affect labels on Prolific using the six NOVA-like classes.
4. Compare model predictions to human labels with accuracy, macro-F1, and
   prompt-clustered uncertainty.
5. Only after behavioral validation, compare against EEG-derived labels or run a
   small EEG/fMRI follow-up.

## IRB Note

Do not add EEG to the current memorability IRB pilot. EEG changes the study
burden, recruitment, privacy, equipment, and risk profile. Keep the first IRB
packet focused on online pairwise memorability judgments. Submit affect labels
as a behavioral amendment or separate minimal-risk study. Submit EEG as a later,
more involved protocol.

## Product Direction

Affect-aware selection becomes useful once it can support queries like:

```text
pick the most memorable variant that is happy rather than fearful
pick the highest-arousal opener that does not drift off prompt
find segments where attention spikes but affect turns aversive
```

This is product-relevant, but it should not be marketed as neural emotion
decoding until EEG validation exists.
