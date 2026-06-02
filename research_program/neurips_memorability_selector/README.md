# NeurIPS-Grade Memorability Selector Program

Regenerated 2026-06-01.

This folder now contains a split version of the original exploratory
audience-vector project:

- a main selector manuscript;
- five satellite manuscripts;
- a local navigation site;
- rendered PDFs and HTML pages;
- the current V-JEPA-augmented evaluation scaffold.

## What Is Finished

- Main manuscript source: `main_selector_paper/paper.md`
- Main manuscript PDF: `main_selector_paper/paper.pdf`
- Main manuscript HTML: `main_selector_paper/paper.html`
- Site index: `site/index.html`
- Split package zip: `data/reports/neurips_memorability_selector_split_package_2026-06-01.zip`

## What The Main Paper Claims Now

TRIBE/BMD features expose a compact brain-aligned memorability readout that is
competitive with V-JEPA on BOLD Moments and useful enough to define a generated
video selector. The generated-video selector is promising under the proxy metric
but still needs independent human validation.

## What It Does Not Claim Yet

- TRIBE-selected generated videos are more memorable to humans.
- TRIBE beats V-JEPA or CLIP in independent human judgment.
- Direct generator steering is solved.
- Synthetic persona axes are independent real audience segments.

## Current Strongest Numbers

| Result | Current status | Number | Claim use |
|---|---|---:|---|
| TRIBE/BMD memorability prediction | confirmed on BMD CV | +0.403 +/- 0.061 | brain-aligned signal exists |
| V-JEPA memorability prediction | confirmed baseline | +0.395 +/- 0.037 | TRIBE is competitive, not dominant |
| Persona-axis overlap | reviewer-corrected | mean abs cos 0.434, rank 3.56/12 | personas are not independent axes |
| Wan selector proxy gain | proxy-only | 18/24 improved, mean lift +2.817 | product workflow candidate, not behavioral proof |

## Folder Map

- `main_selector_paper/` - main submission candidate.
- `satellite_papers/` - mechanistic audit, audience axes, reward distillation,
  and representation-frame theory note.
- `experiments/` - manifests, survey HTML, protocols, and representation audits.
- `site/` - rendered local website and PDFs.
- `submissions/` - readiness notes.

## Next Decisive Step

Run the V-JEPA-augmented blinded human pilot. If TRIBE+gate beats V-JEPA and
CLIP under prompt-clustered human judgments, the main selector paper has its
backbone. If it ties or loses, the honest paper becomes a comparison of
brain-aligned and self-supervised video frames for memorability-like signals.
