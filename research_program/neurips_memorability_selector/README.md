# NeurIPS-Grade Memorability Selector Program

Last manually reconciled 2026-06-03.

This folder now contains a split version of the original exploratory
audience-vector project:

- a main selector manuscript;
- five satellite manuscripts;
- a committed navigation page;
- rendered HTML pages;
- the current V-JEPA-augmented evaluation scaffold.

## What Is Finished

- Main manuscript source: `main_selector_paper/paper.md`
- Main manuscript HTML: `main_selector_paper/paper.html`
- Fold-safe hidden patch: `experiments/tribe_foldsafe_direction_patch.md`
- Expanded hidden cache report:
  `experiments/tribe_layerwise_encoder_hidden_capture_104.md`
- Program navigation page: `index.html`

Generated PDFs, zip packages, raw responses, generated videos, and model weights
belong in the local data lake, not in this committed folder. On the main
workstation those artifacts usually live under `/Users/jawaun/isc_mod/data/`.

## Existing Human Evidence

There is already human behavioral evidence for the broader TRIBE/BMD selector
signal:

- BOLD Moments human memorability labels supervise and evaluate the core
  `v_mem` direction.
- The completed Prolific best-of-N study had 41 raters, all passing attention
  checks. Humans preferred the TRIBE-ranked best-of-N winner over the
  within-seed median variant 290/451 times, or 64.3%, with Wilson 95% CI
  [0.598, 0.686] and binomial p = 1.3e-9.

Scope boundary: that study supports the earlier selector signal. It does not
complete the newer V-JEPA-adjudicated generated-video selector pilot, the
collaborator BO/video-control satellite, or delayed-recognition validation.

## What The Main Paper Claims Now

TRIBE/BMD features expose a compact brain-aligned memorability readout that is
competitive with V-JEPA on BOLD Moments and useful enough to define a generated
video selector. A fold-safe TRIBE-internal hidden-direction patch on
104 balanced clips supports the
mechanistic readout as load-bearing under disjoint train/eval intervention. The
generated-video selector is promising under the proxy metric. The earlier
Prolific result supports the general TRIBE/BMD selector signal, but the current
V-JEPA-adjudicated selector pool still needs independent human validation.

## What It Does Not Claim Yet

- TRIBE-selected generated videos are more memorable to humans.
- TRIBE beats V-JEPA or CLIP in independent human judgment.
- Direct generator steering is solved.
- Synthetic persona axes are independent real audience segments.
- The fold-safe hidden patch proves population-level causality.

## Current Strongest Numbers

| Result | Current status | Number | Claim use |
|---|---|---:|---|
| TRIBE/BMD memorability prediction | confirmed on BMD CV | +0.403 +/- 0.061 | brain-aligned signal exists |
| V-JEPA memorability prediction | confirmed baseline | +0.395 +/- 0.037 | TRIBE is competitive, not dominant |
| Persona-axis overlap | reviewer-corrected | mean abs cos 0.434, rank 3.56/12 | personas are not independent axes |
| TRIBE hidden-direction patch | fold-safe 104-clip intervention | baseline rho +0.602 -> patched rho +0.054 to +0.200; gap ratio +0.135 to +0.212 | mechanistic patch-sensitivity, not population proof |
| Wan selector proxy gain | proxy-only | 18/24 improved, mean lift +2.817 | product workflow candidate, not behavioral proof |

## Folder Map

- `main_selector_paper/` - main submission candidate.
- `satellite_papers/` - mechanistic audit, audience axes, reward distillation,
  and representation-frame theory note.
- `experiments/` - manifests, survey HTML, protocols, and representation audits.
- `index.html` - committed navigation page for the split program.
- `submissions/` - readiness notes.
- `future_work/` - research leads that are inspiring but not yet evidence for
  the main paper.

## Next Decisive Step

Run the V-JEPA-augmented blinded human pilot. If TRIBE+gate beats V-JEPA and
CLIP under prompt-clustered human judgments, the main selector paper has its
backbone. If it ties or loses, the honest paper becomes a comparison of
brain-aligned and self-supervised video frames for memorability-like signals.
