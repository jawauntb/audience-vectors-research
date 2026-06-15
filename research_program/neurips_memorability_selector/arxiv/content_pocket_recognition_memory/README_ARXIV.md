# Content-Pocket Recognition Memory arXiv Package

This folder is a self-contained arXiv-style source package for:

**Compute-Selected Generated-Video Content Pockets Predict Delayed Human Recognition**

Files:

- `main.tex`: preprint source.
- `main.pdf`: compiled preprint PDF.
- `references.bib`: local bibliography.
- `figures/recognition_accuracy.png`: arm-level recognition plot.
- `figures/primary_vs_hard_lift.png`: pooled primary-vs-hard-control lift.
- `figures/pocket_specific_lift.png`: individual pocket contrast.

Local compile command with Tectonic:

```bash
tectonic -X compile main.tex
```

Equivalent classic TeX sequence:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

This package was locally compiled with `tectonic 0.16.9` on 2026-06-15. The
compile completed with only a bibliography underfull-box warning. The companion
presentation-style PDF remains at:

```text
research_program/neurips_memorability_selector/satellite_papers/08_content_pocket_recognition_memory.pdf
```

Claim boundary:

- Allowed: narrow delayed old-vs-lure human recognition evidence for the pooled
  orange-flowers/hanging-clothes content-pocket packet.
- Not allowed: broad memorability proof, measured-BMD/fMRI grounding,
  prompt-conditioned generation control, or a general BO/SVD optimization claim.

Reproducibility boundary:

- Link the paper to the project repository:
  `https://github.com/jawauntb/audience-vectors-research`.
- Keep raw Prolific/webhook exports out of git because they contain participant
  metadata.
- Release generated old/lure stimulus videos as a separate artifact bundle
  with stable URLs or a DOI when submitting externally; do not embed videos in
  the PDF appendix.
- Use the committed design, stimulus, screening, hosted-URL, response-plan,
  aggregate-result, and figure artifacts as the participant-safe reproducibility
  packet.
