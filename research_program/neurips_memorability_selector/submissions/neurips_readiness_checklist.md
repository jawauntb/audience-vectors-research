# NeurIPS Readiness Checklist

## Scientific Claim

- [ ] The paper has one primary claim.
- [ ] Every headline number supports that claim.
- [ ] All exploratory results are moved to appendix or satellite tracks.
- [ ] Claims distinguish proxy-scored, human-rated, and fMRI-grounded evidence.

## Evaluation

- [ ] Frozen data split.
- [ ] Frozen selector policies.
- [ ] Independent human endpoint.
- [ ] Baselines include random, CLIP/text-video, V-JEPA, and quality/preservation.
- [ ] Prompt-clustered uncertainty is reported.
- [ ] Failure cases are shown.

## Reproducibility

- [ ] One command rebuilds main tables from released artifacts.
- [ ] Dataset exclusions are listed.
- [ ] Generated-video candidate manifest is versioned.
- [ ] Survey randomization seed and attention checks are documented.
- [ ] Code/data availability statement is written.

## Reviewer Risk

- [ ] No stale numbers from older Wan runs.
- [ ] No "orthogonal personas" language.
- [ ] No broad TRIBE-internal causality language from 24-clip patching.
- [ ] No claim that LoRA improves human memorability without human validation.
- [ ] No hidden product pitch in the main paper.

## Packaging

- [ ] NeurIPS LaTeX style.
- [ ] Main content under page limit.
- [ ] Appendix contains full protocols, prompts, survey screenshots, and
      statistical details.
- [ ] Website/demo is optional and not required to understand the paper.
