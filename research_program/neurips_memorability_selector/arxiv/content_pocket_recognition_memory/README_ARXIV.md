# Content-Pocket Recognition Memory arXiv Package

This folder is a self-contained arXiv-style source package for:

**Compute-Selected Generated-Video Content Pockets Predict Delayed Human Recognition**

Files:

- `main.tex`: preprint source.
- `references.bib`: local bibliography.
- `figures/recognition_accuracy.png`: arm-level recognition plot.
- `figures/primary_vs_hard_lift.png`: pooled primary-vs-hard-control lift.
- `figures/pocket_specific_lift.png`: individual pocket contrast.

Local compile command, when a TeX toolchain is installed:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The current Codex environment did not have `pdflatex`, `latexmk`, `xelatex`, or
`tectonic`, so this package was not locally compiled here. The companion
presentation-style PDF remains at:

```text
research_program/neurips_memorability_selector/satellite_papers/08_content_pocket_recognition_memory.pdf
```

Claim boundary:

- Allowed: narrow delayed old-vs-lure human recognition evidence for the pooled
  orange-flowers/hanging-clothes content-pocket packet.
- Not allowed: broad memorability proof, measured-BMD/fMRI grounding,
  prompt-conditioned generation control, or a general BO/SVD optimization claim.
