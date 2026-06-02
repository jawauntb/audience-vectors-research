# Affect-Aware Media Selector

Started: 2026-06-01

## North Star

Build a selector that ranks generated media by affective response while keeping
memorability, prompt preservation, and quality under control.

The first deployed version is a media-derived proxy. A true NOVA-style version
requires EEG recordings and emotion labels.

## Why Now

The memorability project already has:

- TRIBE/BMD memorability scoring;
- raw TRIBE engagement dimensions;
- video segmentation;
- palette, motion, edge, and density analysis;
- a product analyzer site;
- a human-validation path.

NOVA suggests a natural next axis: video-evoked affect. Spencer's proposal maps
well onto the next research branch:

```text
EEG -> power spectral density -> ridge/logistic-ridge classifier -> emotion
```

## Current Product State

The analyzer site now returns:

- six-class affect proxy: happy, anger, fear, sadness, disgust, neutral;
- arousal proxy;
- valence proxy;
- per-window affect labels;
- aggregate affect profile;
- explicit caveat that this is not EEG PSD decoding.

## Research Stages

### Stage 0: Product Proxy

Use media-only features:

- TRIBE attention/emotion/memory/visual/language/cognitive ease;
- palette warmth;
- motion;
- edge density;
- visual density;
- saturation and contrast.

Output a clear affect readout for creative debugging.

### Stage 1: Human Affect Labels

Collect Prolific ratings with six classes:

- happy;
- anger;
- fear;
- sadness;
- disgust;
- neutral.

Evaluate proxy accuracy, macro-F1, and calibration.

### Stage 2: NOVA-Style EEG Baseline

When EEG data are available:

- preprocess EEG;
- compute PSD by channel/frequency;
- train ridge or logistic-ridge classifier;
- compare PSD vs differential entropy;
- compare EEG-derived affect to TRIBE/V-JEPA/CLIP media features.

### Stage 3: Multi-Objective Selector

Select generated clips under constraints:

```text
maximize memorability
target affect class or arousal/valence region
preserve prompt/seed
avoid quality failures
```

## Immediate Next Tasks

1. Add affect labels to analyzer output. Done.
2. Add affect section to UI. Done.
3. Save example JSON from several videos for calibration.
4. Draft a Prolific affect-labeling protocol.
5. Decide whether affect labels belong in the current IRB as an amendment or in
   a separate behavioral protocol.

## Claim Contract

Allowed now:

- "The analyzer exposes a NOVA-inspired media affect proxy."
- "The proxy is useful for creative diagnosis and hypothesis generation."

Not allowed yet:

- "The analyzer decodes emotion from EEG."
- "The affect labels are neurologically validated."
- "The model can reliably induce emotion in humans."
- "Affect-aware selection is validated for product decisions."
