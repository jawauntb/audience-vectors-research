# Selector Human Evaluation Protocol

This is the decisive experiment for a NeurIPS-grade paper.

## Primary Question

When choosing among multiple generated videos for the same prompt, does a
TRIBE/BMD memorability selector choose clips that humans judge as more memorable
than clips chosen by non-brain baselines?

## Experimental Unit

Prompt-conditioned candidate set.

Example:

- prompt/image seed: `fresh24_aerial_beach`
- candidates: base, LoRA single, LoRA best-of-4 variants, optional other model
  variants
- selector policies choose one candidate per prompt

## Selector Policies

Minimum:

1. Random candidate.
2. CLIP/text-video alignment winner.
3. V-JEPA memorability/linear baseline winner.
4. TRIBE memorability winner.
5. TRIBE + preservation gate winner.

Optional:

6. Aesthetic/video-quality winner.
7. Wan LoRA selector winner.
8. Ensemble: TRIBE memorability + quality + prompt preservation.

## Human Task Options

### Option A: Fast Pairwise Preference

Show two videos from the same prompt and ask:

> Which video do you think would be more memorable if you saw many such clips?

This is cheaper and closer to our existing Prolific setup, but it measures
predicted memorability rather than actual memory.

### Option B: Delayed Recognition

Phase 1: show a stream of target and filler clips.

Phase 2: after a delay, show repeated and lure clips, ask whether each was seen
before.

This is scientifically stronger and closer to Memento10k/BMD, but costs more.

Recommended path:

- Run Option A first as a power/calibration study.
- If TRIBE beats baselines, run a smaller Option B confirmatory study.

## Primary Endpoint

Prompt-clustered win rate of TRIBE+gate selected clips against the strongest
non-brain baseline.

Primary hypothesis:

`P(human chooses TRIBE+gate over baseline) > 0.5`.

## Sample Size Sketch

Pilot:

- 50 prompts.
- 2-3 pairwise comparisons per prompt.
- 20 raters per comparison.
- Rough total: 2,000-3,000 judgments.

Submission-grade:

- 100 prompts.
- 3-5 selector comparisons per prompt.
- 20-30 raters per comparison.
- Rough total: 6,000-15,000 judgments depending on design.

## Statistical Tests

- Prompt-clustered bootstrap confidence interval.
- Mixed-effects logistic regression:
  - fixed effect: selector policy
  - random intercept: prompt
  - optional random intercept: rater
- Report per-prompt win-rate distribution, not only pooled votes.
- Correct for multiple selector comparisons.

## Anti-Circularity Rule

No claim of human improvement can rely on TRIBE-scored gains alone. TRIBE may
train or select, but the final endpoint must be independent human judgment or
delayed recognition.

## Study Arms For Current 24-Seed Pilot

Using existing artifacts, the first pilot can compare:

- base vs TRIBE+gate product selection
- random LoRA candidate vs TRIBE+gate product selection
- V-JEPA and CLIP proxy winners vs TRIBE+gate product selection

This is underpowered for a final paper but useful to test survey mechanics and
estimate effect size.

Current implementation note: V-JEPA scores are now computed for the same
24-seed candidate pool, producing `product_vs_vjepa_memorability` and
`gated_vs_vjepa_memorability` pairwise arms in the augmented survey.
