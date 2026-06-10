# Phase 1 Capture-Score Preflight

## Verdict

- Mechanical ready: True
- Claim update allowed: True
- Claim ready: True
- Samples: 350
- ROI masks: research_program/dopamine_detox_attention_capture/results/destrieux_roi_masks_disjoint_20260608.npz
- Provenance ready: True
- Claim boundary: Preflight verifies manifest mechanics and label variance only. It does not validate attentional capture.

## Blocking Reasons

- none

## Warnings

- none

## Label Groups

| dataset | n | finite | distinct | std | ready |
|---|---:|---:|---:|---:|---|
| DHF1K | 350 | 350 | 350 | 0.0001 | True |

## Feature Audit

- Feature-path samples: 350
- Existing features: 350
- Missing features: 0
- Shape mismatches: 0

## Provenance Audit

- Required: True
- Ready: True
- Alignment audit sha256: fde522251a93544973b9b1638ea53313452b4a699eb92b42531c5c2900f1f2d4
- Alignment audit path: research_program/dopamine_detox_attention_capture/results/phase1_dhf1k_fixation_density_audio_only_alignment_20260609.json
- Label audit ready: True

## Scoring Audit

- Attempted: True
- Reason skipped: n/a
- Invalid capture denominators: 48
- Valid capture denominators: 302
