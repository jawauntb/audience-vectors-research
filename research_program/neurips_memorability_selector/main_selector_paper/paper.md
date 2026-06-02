# Brain-Aligned Memorability Signals Improve Video Generation Selection

**Draft status:** split-program manuscript, regenerated 2026-06-02.
**Intended venue shape:** NeurIPS main track or Evaluations and Datasets, pending
independent selector validation.
**Primary caveat:** the strongest generation result is still proxy-scored. The
human endpoint is designed and scaffolded, but not yet complete.

## Abstract

Generated videos can now be sampled cheaply, but creators still need to decide
which candidate is worth showing to humans. Standard evaluation metrics target
visual quality, prompt alignment, or generic preference; they do not directly
ask what a viewer will remember. We study whether a brain-aligned video
representation can supply a practical memorability selector for generated
videos. Using 1,022 analyzable clips from the 1,102-clip BOLD Moments dataset,
we learn a supervised memorability readout in TRIBE-predicted cortical-response
features and compare it with self-supervised video features, prompt-preservation
signals, and generation-time selector policies. TRIBE predicts held-out BOLD
Moments memorability at Spearman rho +0.403
+/- 0.061, comparable
to the V-JEPA baseline at rho +0.395
+/- 0.037. On a
Wan2.2 candidate pool, TRIBE/BMD and V-JEPA selectors overlap only partially,
creating a useful adjudication setting for blinded human evaluation. The current
paper therefore makes a narrow claim: brain-aligned features expose a compact,
auditable memorability signal and define a testable generated-video selection
workflow. A fold-safe hidden-direction patch on
104 balanced high/low BMD clips
further shows that removing the learned hidden direction disrupts held-out
TRIBE readouts, reducing layerwise held-out rho from mean
+0.602 to a patched range of
+0.054 to +0.200. It does
not yet claim that TRIBE-selected
generated videos are more memorable to humans, nor that direct generator
steering is solved.

## Big Picture

The useful product and scientific unit is no longer "a magic memorability
vector." It is a selector:

```text
prompt or seed -> N generated candidates -> representation scores -> gated winner
```

If the selector's ordering transfers to human behavior, it becomes a practical
way to optimize cognitive properties of generated media without immediately
training a generator. Memorability is the first axis because it has established
datasets, behavioral labels, and a plausible neural substrate.

## Contributions

1. We learn a compact memorability readout from TRIBE features on BOLD Moments.
2. We benchmark it against V-JEPA and simple non-brain baselines rather than
   treating generic video features as strawmen.
3. We define a generation-selection protocol with preservation gates so the
   selector cannot simply choose off-prompt but high-scoring clips.
4. We complete a fold-safe hidden-direction patching audit inside TRIBE, using
   disjoint train/eval clips and expanded layerwise hidden caches.
5. We package the human-evaluation protocol and V-JEPA-augmented candidate
   manifest needed to test the decisive claim.
6. We separate confirmed representation results from proxy-only generation
   results and direct-steering negatives.

## Current Evidence Ledger

| Result | Current status | Number | Claim use |
|---|---|---:|---|
| TRIBE/BMD memorability prediction | confirmed on BMD CV | +0.403 +/- 0.061 | brain-aligned signal exists |
| V-JEPA memorability prediction | confirmed baseline | +0.395 +/- 0.037 | TRIBE is competitive, not dominant |
| Persona-axis overlap | reviewer-corrected | mean abs cos 0.434, rank 3.56/12 | personas are not independent axes |
| TRIBE hidden-direction patch | fold-safe 104-clip intervention | baseline rho +0.602 -> patched rho +0.054 to +0.200; gap ratio +0.135 to +0.212 | mechanistic patch-sensitivity, not population proof |
| Wan selector proxy gain | proxy-only | 18/24 improved, mean lift +2.817 | product workflow candidate, not behavioral proof |

## Method

### Dataset

BOLD Moments contains 1,102 short naturalistic videos sampled from Memento10k,
with fMRI responses and memorability metadata. Our current TRIBE analysis uses
1,022 analyzable clips after feature and URL exclusions. We keep this distinction
explicit because reviewers will care about the denominator.

### Brain-Aligned Feature Space

TRIBE v2 maps video, audio, and text representations to predicted cortical
surface responses. We mean-pool the saved TRIBE output per clip, producing a
brain-aligned feature vector. This is not a literal brain scan and not a human
preference oracle. It is a learned encoding-model representation whose value is
that it can be compared against fMRI and ROI hypotheses.

### Memorability Readout

For each train fold, we sort clips by BOLD Moments memorability, take the top
and bottom 30 percent, and form a unit contrastive direction:

```text
v_mem = mean(high memorability) - mean(low memorability)
```

Held-out clips are scored by projection onto this direction. All claims about
the selector should be read as claims about this specific readout in this
specific representation frame.

### Baselines

The required baselines are:

- random candidate selection;
- CLIP or text-video prompt-preservation selection;
- V-JEPA memorability selection;
- video-quality or temporal-consistency selection where feasible;
- TRIBE/BMD memorability selection;
- TRIBE/BMD with preservation or quality gates.

V-JEPA is especially important because it performs nearly as well as TRIBE on
global BMD memorability prediction. The paper's claim cannot be "TRIBE simply
beats visual features." The better question is whether brain alignment adds
selection value beyond a strong self-supervised video frame.

## Results To Date

### BMD Prediction

TRIBE reaches mean held-out Spearman rho
+0.403 +/- 0.061
across five folds. V-JEPA reaches
+0.395 +/- 0.037.
The old "TRIBE is 1.9x V-JEPA" headline is retired. TRIBE remains interesting
because the signal is brain-aligned and mechanistically auditable, not because
it has a clean global lead on this prediction task.

### Directional Compactness

The fold-safe linear ablation result supports dominant-axis compactness. Removing
the learned TRIBE direction and retraining on residual features drops performance
from roughly +0.403
to -0.016. Random
direction ablations leave the signal intact. Nonlinear residual probes recover
some signal, so the safe interpretation is "dominant linear readout," not
"memorability is literally one-dimensional."

The newer hidden-state audit strengthens this point without overclaiming it.
We expanded the layerwise hidden cache to
52 low-memorability and
52 high-memorability clips, then ran
5 fold-safe interventions. Each fold trained hidden
directions on 40 low plus
40 high clips and patched
12 low plus
12 high held-out clips. Across nine hook targets,
mean held-out baseline rho was +0.602; after
removing the learned hidden direction, mean patched rho ranged from
+0.054 to +0.200, and the
remaining high-minus-low gap ratio ranged from +0.135
to +0.212. This is a fold-safe TRIBE-internal
intervention result on tail clips, not a proof of full population-level
causality.

### Generated-Video Selector

The current Wan2.2 proxy experiment evaluates 24 fresh prompt/image seeds. The
best product-style rule compares base clips to LoRA and best-of-N candidates,
uses CLIP-style preservation gates, and keeps the base when the gated candidate
is worse. Under the TRIBE/BMD proxy, base-or-gated best-of-4 improves
18/24 seeds with mean lift
+2.817.
This is a promising selector-policy result, not independent behavioral evidence.

### Representation-Frame Audit

On the current candidate pool, TRIBE and V-JEPA are related but not identical.
Their scalar score Spearman correlation is
+0.632. Within prompt-conditioned seed
groups, their mean rank agreement is
+0.443, and their top-1 selector
agreement is 45.8%. This is exactly the useful
regime: the baselines disagree often enough that human evaluation can adjudicate
between them.

## Human Evaluation Plan

The decisive experiment is a blinded within-prompt pairwise study. Each trial
shows two videos generated for the same prompt or image seed. Participants choose
which clip they expect to be more memorable. The primary endpoint is the
prompt-clustered win rate of TRIBE+gate selected clips against the strongest
non-brain baseline, especially V-JEPA.

Current implementation assets:

- `experiments/current_selector_manifest_with_vjepa.json`
- `experiments/current_selector_pairwise_tasks_with_vjepa.json`
- `experiments/current_selector_prolific_survey_with_vjepa.html`
- `experiments/selector_human_eval_protocol.md`

No independent responses have been collected for the V-JEPA-augmented selector pilot yet.

## Failure Modes Reviewers Will Notice

- The generated-video result is proxy-scored until human validation is complete.
- V-JEPA may tie or beat TRIBE in human judgments. That would change the paper
  into a representation-frame comparison rather than a brain-aligned-selector
  win.
- Memorability is not engagement, virality, attention, quality, emotion, or
  commercial effectiveness.
- Direct continuous steering is not solved. Current evidence favors ranking and
  gating candidate generations.
- The TRIBE readout is a coordinate in a model frame, not an ontological essence
  of memory.
- The fold-safe hidden patch supports load-bearing hidden-direction sensitivity
  in TRIBE, but it is still a high/low-tail intervention rather than a full
  causal account of human memorability.

## Submission-Ready Claim Contract

Allowed now:

- TRIBE/BMD features contain a compact memorability readout.
- TRIBE is competitive with V-JEPA for BMD memorability prediction.
- Fold-safe hidden-direction patching shows the TRIBE-internal hidden direction
  is load-bearing on disjoint held-out high/low clips.
- The current generated-video selector is promising under the TRIBE/BMD proxy.
- TRIBE and V-JEPA disagree enough on generated candidates to justify a human
  adjudication study.

Requires human validation:

- TRIBE-selected generated videos are more memorable to humans.
- TRIBE improves selection beyond V-JEPA, CLIP, and quality baselines.
- The selector is product-ready.

Cut or avoid:

- "TRIBE is a preference oracle."
- "Persona axes are independent."
- "Direct steering is solved."
- "LoRA learned human memorability."
- "The hidden-direction patch proves population-level causality."

## References

- Isola, Parikh, Torralba, and Oliva. Understanding the Intrinsic Memorability
  of Images. NeurIPS 2011.
- Newman et al. Multimodal Memorability: Modeling Effects of Semantics and Decay
  on Video Memorability. ECCV 2020.
- Lahner et al. Modeling short visual events through the BOLD Moments video fMRI
  dataset and metadata. Nature Communications 2024.
- Meta/Facebook Research. TRIBE v2 repository.
- Bardes et al. V-JEPA 2: Self-Supervised Video Models Enable Understanding,
  Prediction and Planning. arXiv 2025.
- Huang et al. VBench: Comprehensive Benchmark Suite for Video Generative
  Models. arXiv 2023.
- Liu et al. FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain
  Text-to-Video Generation. NeurIPS Datasets and Benchmarks 2023.
- Wallace et al. Diffusion Model Alignment Using Direct Preference Optimization.
  arXiv 2023.
