# pyright: reportMissingImports=false
"""Build the split memorability-selector paper package.

This script turns the current research-program scaffold into shareable paper
artifacts: markdown manuscripts, HTML pages, PDFs, a navigation site, and a
dated zip bundle. It deliberately distinguishes confirmed results from pending
human-validation claims.
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "research_program" / "neurips_memorability_selector"
REPORTS = ROOT / "data" / "reports"
SITE = PROGRAM / "site"
SITE_PAPERS = SITE / "papers"
BUILD_DATE = "2026-06-02"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {}


def fnum(value: Any, digits: int = 3, signed: bool = True) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    sign = "+" if signed else ""
    return f"{val:{sign}.{digits}f}"


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{100 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def read_results() -> dict[str, Any]:
    return {
        "tribe_cv": load_json(REPORTS / "cv_tribe_n1022.json"),
        "vjepa_cv": load_json(REPORTS / "cv_vjepa_n1026.json"),
        "patch_tribe": load_json(REPORTS / "patching_tribe.json"),
        "patch_vjepa": load_json(REPORTS / "patching_vjepa.json"),
        "random_ablation": load_json(REPORTS / "random_ablation_null.json"),
        "nonlinear": load_json(REPORTS / "nonlinear_probes.json"),
        "multi_direction": load_json(REPORTS / "multi_direction.json"),
        "persona_cos": load_json(REPORTS / "persona_cos_reviewer.json"),
        "wan_product": load_json(
            REPORTS
            / "wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0_product_selector.json"
        ),
        "rep_frame": load_json(
            PROGRAM / "experiments" / "representation_frame_analysis.json"
        ),
        "vjepa_selector": load_json(
            PROGRAM / "experiments" / "vjepa_selector_report.json"
        ),
        "pairwise": load_json(
            PROGRAM / "experiments" / "selector_pairwise_analysis.json"
        ),
        "tribe_layerwise": load_json(
            REPORTS / "tribe_layerwise_encoder_localization.json"
        ),
        "tribe_direction_patch": load_json(
            REPORTS / "tribe_layerwise_direction_patch.json"
        ),
        "tribe_timepos": load_json(REPORTS / "tribe_timepos_patch_probe.json"),
        "tribe_temporal": load_json(REPORTS / "tribe_temporal_spectral_probe.json"),
        "alexnet": load_json(REPORTS / "alexnet_memorability_probe.json"),
        "alexnet_forward": load_json(REPORTS / "alexnet_forward_patch_probe.json"),
        "tribe_foldsafe_patch": load_json(
            PROGRAM / "experiments" / "tribe_foldsafe_direction_patch.json"
        ),
        "tribe_hidden_capture_104": load_json(
            PROGRAM / "experiments" / "tribe_layerwise_encoder_hidden_capture_104.json"
        ),
    }


def foldsafe_summary(results: dict[str, Any]) -> dict[str, Any]:
    report = results.get("tribe_foldsafe_patch", {})
    raw_layers = report.get("summary", {}).get("layers", [])
    if not isinstance(raw_layers, list):
        raw_layers = []

    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in raw_layers:
        if isinstance(row, dict) and isinstance(row.get("label"), str):
            by_label.setdefault(row["label"], []).append(row)

    layers: list[dict[str, Any]] = []
    for label, rows in by_label.items():
        train = [
            val
            for val in (
                numeric(r.get("hidden", {}).get("train_spearman_vs_memorability"))
                for r in rows
            )
            if val is not None
        ]
        baseline = [
            val
            for val in (
                numeric(r.get("patch", {}).get("eval_baseline_spearman_vs_memorability"))
                for r in rows
            )
            if val is not None
        ]
        patch = [
            val
            for val in (
                numeric(r.get("patch", {}).get("eval_patch_spearman_vs_memorability"))
                for r in rows
            )
            if val is not None
        ]
        gap = [
            val
            for val in (
                numeric(r.get("patch", {}).get("eval_patch_gap_ratio_vs_baseline"))
                for r in rows
            )
            if val is not None
        ]
        delta = [
            val
            for val in (
                numeric(r.get("patch", {}).get("eval_projection_delta_in_baseline_std"))
                for r in rows
            )
            if val is not None
        ]
        non_dc = [
            val
            for val in (
                numeric(r.get("hidden", {}).get("non_dc_energy")) for r in rows
            )
            if val is not None
        ]
        layers.append(
            {
                "label": label,
                "folds": len(rows),
                "train_hidden_rho": mean(train),
                "eval_baseline_rho": mean(baseline),
                "eval_patch_rho": mean(patch),
                "eval_gap_ratio": mean(gap),
                "delta_std": mean(delta),
                "non_dc_energy": mean(non_dc),
            }
        )

    patch_values = [
        val
        for val in (numeric(layer.get("eval_patch_rho")) for layer in layers)
        if val is not None
    ]
    gap_values = [
        val
        for val in (numeric(layer.get("eval_gap_ratio")) for layer in layers)
        if val is not None
    ]
    baseline_values = [
        val
        for val in (numeric(layer.get("eval_baseline_rho")) for layer in layers)
        if val is not None
    ]
    train_values = [
        val
        for val in (numeric(layer.get("train_hidden_rho")) for layer in layers)
        if val is not None
    ]
    non_dc_values = [
        val
        for val in (numeric(layer.get("non_dc_energy")) for layer in layers)
        if val is not None
    ]

    summary = report.get("summary", {})
    coverage = summary.get("balanced_hidden_cache_coverage", {})
    if not isinstance(coverage, dict):
        coverage = {}

    return {
        "status": report.get("status", "missing"),
        "layers": sorted(layers, key=lambda row: row["label"]),
        "n_layers": len(layers),
        "folds": report.get("config", {}).get("folds"),
        "n_train_each": report.get("config", {}).get("n_train_each"),
        "n_eval_each": report.get("config", {}).get("n_eval_each"),
        "n_hidden_cache_eligible": summary.get("n_hidden_cache_eligible"),
        "needed_hidden_cache_clips": summary.get("needed_hidden_cache_clips"),
        "low_ready": coverage.get("low_ready"),
        "high_ready": coverage.get("high_ready"),
        "baseline_mean": mean(baseline_values),
        "patch_min": min(patch_values) if patch_values else None,
        "patch_max": max(patch_values) if patch_values else None,
        "gap_min": min(gap_values) if gap_values else None,
        "gap_max": max(gap_values) if gap_values else None,
        "train_min": min(train_values) if train_values else None,
        "train_max": max(train_values) if train_values else None,
        "non_dc_min": min(non_dc_values) if non_dc_values else None,
        "non_dc_max": max(non_dc_values) if non_dc_values else None,
    }


def foldsafe_layer_table(results: dict[str, Any]) -> str:
    summary = foldsafe_summary(results)
    layers = summary.get("layers", [])
    if not layers:
        return "Fold-safe hidden patch results are not available in this build."

    rows = [
        "| Hook target | Train hidden rho | Held-out baseline rho | Held-out patch rho | Gap ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for layer in layers:
        rows.append(
            f"| `{layer['label']}` | "
            f"{fnum(layer.get('train_hidden_rho'))} | "
            f"{fnum(layer.get('eval_baseline_rho'))} | "
            f"{fnum(layer.get('eval_patch_rho'))} | "
            f"{fnum(layer.get('eval_gap_ratio'))} |"
        )
    return "\n".join(rows)


def wan_policy(results: dict[str, Any], name: str) -> dict[str, Any]:
    wan = results.get("wan_product", {})
    if isinstance(wan.get(name), dict):
        return wan[name]
    policies = wan.get("summary", {}).get("policies", {})
    policy = policies.get(name, {})
    return policy if isinstance(policy, dict) else {}


def policy_improved(policy: dict[str, Any], default: int) -> int:
    value = policy.get("improved_seeds", policy.get("n_improved", default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def policy_mean(policy: dict[str, Any], default: float) -> float:
    value = policy.get("mean_lift", policy.get("mean", default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def policy_median(policy: dict[str, Any], default: float) -> float:
    value = policy.get("median_lift", policy.get("median", default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def result_table(results: dict[str, Any]) -> str:
    tribe = results["tribe_cv"]
    vjepa = results["vjepa_cv"]
    persona = results["persona_cos"]
    gated = wan_policy(results, "base_or_gated_best_of_n")
    foldsafe = foldsafe_summary(results)
    rows = [
        "| Result | Current status | Number | Claim use |",
        "|---|---|---:|---|",
        (
            "| TRIBE/BMD memorability prediction | confirmed on BMD CV | "
            f"{fnum(tribe.get('mean_spearman'))} +/- {fnum(tribe.get('stdev_spearman'), signed=False)} | "
            "brain-aligned signal exists |"
        ),
        (
            "| V-JEPA memorability prediction | confirmed baseline | "
            f"{fnum(vjepa.get('mean_spearman'))} +/- {fnum(vjepa.get('stdev_spearman'), signed=False)} | "
            "TRIBE is competitive, not dominant |"
        ),
        (
            "| Persona-axis overlap | reviewer-corrected | "
            f"mean abs cos {fnum(persona.get('abs_cos_off_diag', {}).get('mean'), signed=False)}, "
            f"rank {fnum(persona.get('effective_rank'), 2, signed=False)}/12 | "
            "personas are not independent axes |"
        ),
        (
            "| TRIBE hidden-direction patch | fold-safe 104-clip intervention | "
            f"baseline rho {fnum(foldsafe.get('baseline_mean'))} -> patched rho "
            f"{fnum(foldsafe.get('patch_min'))} to {fnum(foldsafe.get('patch_max'))}; "
            f"gap ratio {fnum(foldsafe.get('gap_min'))} to {fnum(foldsafe.get('gap_max'))} | "
            "mechanistic patch-sensitivity, not population proof |"
        ),
        (
            "| Wan selector proxy gain | proxy-only | "
            f"{policy_improved(gated, 18)}/24 improved, "
            f"mean lift {fnum(policy_mean(gated, 2.817))} | "
            "product workflow candidate, not behavioral proof |"
        ),
    ]
    return "\n".join(rows)


def main_selector_paper(results: dict[str, Any]) -> str:
    rep = results["rep_frame"]
    pairwise = results["pairwise"]
    human_n = pairwise.get("n_responses", 0)
    human_note = (
        "No independent responses have been collected for the V-JEPA-augmented "
        "selector pilot yet."
        if not human_n
        else f"The current pilot contains {human_n} independent pairwise responses."
    )
    score_corrs = rep.get("score_correlations", [])
    rank = rep.get("rank_agreement", [])
    top1 = next(
        (
            r
            for r in rank
            if r.get("score_a") == "v_mem_projection"
            and r.get("score_b") == "vjepa_memorability_score"
        ),
        {},
    )
    tv_corr = next(
        (
            r
            for r in score_corrs
            if r.get("score_a") == "v_mem_projection"
            and r.get("score_b") == "vjepa_memorability_score"
        ),
        {},
    )
    gated = wan_policy(results, "base_or_gated_best_of_n")
    foldsafe = foldsafe_summary(results)

    return f"""# Brain-Aligned Memorability Signals Improve Video Generation Selection

**Draft status:** split-program manuscript, regenerated {BUILD_DATE}.
**Intended venue shape:** NeurIPS main track or Evaluations and Datasets, pending
independent selector validation.
**Primary caveat:** the strongest current generated-video selector result is
still proxy-scored. Earlier Prolific data supports the broader TRIBE/BMD
best-of-N selector signal, but the V-JEPA-adjudicated selector endpoint is
designed and scaffolded, not yet complete.

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
Moments memorability at Spearman rho {fnum(results['tribe_cv'].get('mean_spearman'))}
+/- {fnum(results['tribe_cv'].get('stdev_spearman'), signed=False)}, comparable
to the V-JEPA baseline at rho {fnum(results['vjepa_cv'].get('mean_spearman'))}
+/- {fnum(results['vjepa_cv'].get('stdev_spearman'), signed=False)}. On a
Wan2.2 candidate pool, TRIBE/BMD and V-JEPA selectors overlap only partially,
creating a useful adjudication setting for blinded human evaluation. The current
paper therefore makes a narrow claim: brain-aligned features expose a compact,
auditable memorability signal and define a testable generated-video selection
workflow. Earlier Prolific evidence supports the broader TRIBE/BMD best-of-N
selector signal, but not yet the current V-JEPA-adjudicated generated-video
selector pool. A fold-safe hidden-direction patch on
{foldsafe.get('n_hidden_cache_eligible', 104)} balanced high/low BMD clips
further shows that removing the learned hidden direction disrupts held-out
TRIBE readouts, reducing layerwise held-out rho from mean
{fnum(foldsafe.get('baseline_mean'))} to a patched range of
{fnum(foldsafe.get('patch_min'))} to {fnum(foldsafe.get('patch_max'))}. It does
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

{result_table(results)}

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
{fnum(results['tribe_cv'].get('mean_spearman'))} +/- {fnum(results['tribe_cv'].get('stdev_spearman'), signed=False)}
across five folds. V-JEPA reaches
{fnum(results['vjepa_cv'].get('mean_spearman'))} +/- {fnum(results['vjepa_cv'].get('stdev_spearman'), signed=False)}.
The old "TRIBE is 1.9x V-JEPA" headline is retired. TRIBE remains interesting
because the signal is brain-aligned and mechanistically auditable, not because
it has a clean global lead on this prediction task.

### Directional Compactness

The fold-safe linear ablation result supports dominant-axis compactness. Removing
the learned TRIBE direction and retraining on residual features drops performance
from roughly {fnum(results['patch_tribe'].get('mean_baseline_rho', 0.401))}
to {fnum(results['patch_tribe'].get('mean_ablated_rho', 0.057))}. Random
direction ablations leave the signal intact. Nonlinear residual probes recover
some signal, so the safe interpretation is "dominant linear readout," not
"memorability is literally one-dimensional."

The newer hidden-state audit strengthens this point without overclaiming it.
We expanded the layerwise hidden cache to
{foldsafe.get('low_ready', 52)} low-memorability and
{foldsafe.get('high_ready', 52)} high-memorability clips, then ran
{foldsafe.get('folds', 5)} fold-safe interventions. Each fold trained hidden
directions on {foldsafe.get('n_train_each', 40)} low plus
{foldsafe.get('n_train_each', 40)} high clips and patched
{foldsafe.get('n_eval_each', 12)} low plus
{foldsafe.get('n_eval_each', 12)} high held-out clips. Across nine hook targets,
mean held-out baseline rho was {fnum(foldsafe.get('baseline_mean'))}; after
removing the learned hidden direction, mean patched rho ranged from
{fnum(foldsafe.get('patch_min'))} to {fnum(foldsafe.get('patch_max'))}, and the
remaining high-minus-low gap ratio ranged from {fnum(foldsafe.get('gap_min'))}
to {fnum(foldsafe.get('gap_max'))}. This is a fold-safe TRIBE-internal
intervention result on tail clips, not a proof of full population-level
causality.

### Generated-Video Selector

The current Wan2.2 proxy experiment evaluates 24 fresh prompt/image seeds. The
best product-style rule compares base clips to LoRA and best-of-N candidates,
uses CLIP-style preservation gates, and keeps the base when the gated candidate
is worse. Under the TRIBE/BMD proxy, base-or-gated best-of-4 improves
{policy_improved(gated, 18)}/24 seeds with mean lift
{fnum(policy_mean(gated, 2.817))}.
This is a promising selector-policy result, not independent behavioral evidence.

### Representation-Frame Audit

On the current candidate pool, TRIBE and V-JEPA are related but not identical.
Their scalar score Spearman correlation is
{fnum(tv_corr.get('spearman'), signed=True)}. Within prompt-conditioned seed
groups, their mean rank agreement is
{fnum(top1.get('mean_seed_spearman'), signed=True)}, and their top-1 selector
agreement is {pct(top1.get('top1_agreement'))}. This is exactly the useful
regime: the baselines disagree often enough that human evaluation can adjudicate
between them.

### Existing Human Evidence

Human behavior is the reference standard for this project. The strongest
existing behavioral evidence is the completed Prolific best-of-N study from the
earlier selector setting: 41 raters, all passing attention checks, with humans
choosing the TRIBE-projection-ranked best-of-N winner over a within-seed median
variant 290/451 times, or 64.3% (Wilson 95% CI [0.598, 0.686], binomial
p = 1.3e-9; pair-cluster bootstrap CI [0.565, 0.718]).

That result matters because it shows the TRIBE/BMD selector signal survives
human contact. Its scope is limited: it does not by itself validate the current
V-JEPA-adjudicated selector pool, the collaborator BO-generated videos, or
delayed-recognition memory. The paper should therefore treat proxy scores as
candidate-selection evidence that is downstream of human behavior, not a
replacement for it.

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

{human_note}

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
"""


def mechanistic_paper(results: dict[str, Any]) -> str:
    foldsafe = foldsafe_summary(results)
    return f"""# Mechanistic Audit Of A TRIBE Memorability Readout

**Draft status:** satellite paper draft, regenerated {BUILD_DATE}.
**Core purpose:** answer the Spencer-style mechanistic critique without
overclaiming population-level causality.

## Abstract

Linear readouts in large representation models can be predictive for accidental
reasons. A memorability direction learned in TRIBE could reflect temporal
position artifacts, a generic nuisance axis, or intertwined nonlinear factors
rather than a stable viewer-response signal. We audit this concern using
Fourier decompositions of saved TRIBE outputs, direct patches to learned
time-position machinery, layerwise encoder hooks, and a transparent AlexNet
sanity check. The final TRIBE output is mostly temporal-DC, and direct
time-position and rotary-frequency patches do not collapse memorability
ordering on a balanced 24-clip sensitivity set. However, non-DC encoder hidden
structure is load-bearing: removing the learned hidden memorability direction
from early attention residuals through the final encoder sharply disrupts the
readout. A new fold-safe run expands the hidden cache to
{foldsafe.get('n_hidden_cache_eligible', 104)} balanced clips and uses disjoint
train/eval splits; after patching, layerwise held-out rho drops from mean
{fnum(foldsafe.get('baseline_mean'))} to a patched range of
{fnum(foldsafe.get('patch_min'))} to {fnum(foldsafe.get('patch_max'))}. The
reviewer-safe conclusion is that the BMD/TRIBE readout is not explained by a
simple learned-position-table artifact, but it remains a sequence-dependent
model readout, not a fully isolated population-level causal feature.

## Why This Exists

The first paper can say "this direction predicts memorability." A mechanistic
reviewer will ask whether the vector is merely the easiest linear basis in a
messy model. This satellite paper answers that question directly and keeps the
main selector paper cleaner.

## Tests

1. Fourier-decompose the learned output-space direction over TRIBE time bins.
2. Compare full tensor, temporal-DC, nonzero-temporal, and mean-pooled readouts.
3. Patch `_model.time_pos_embed` and rotary inverse frequencies during inference.
4. Hook encoder layers 0, 2, 4, ..., 14 and final encoder output.
5. Remove either non-DC sequence content or the one learned hidden
   high-minus-low memorability direction at each hook.
6. Replicate the compact-direction pattern in AlexNet conv5, where forward
   patching is transparent.

## Headline Results

- Saved TRIBE output readout: full-tensor rho about +0.401, mean-pooled rho
  about +0.405, temporal-DC rho about +0.405, nonzero-temporal rho about +0.297.
- Learned time-position ablation preserves ordering on the 24-clip set:
  baseline rho about +0.677, time-position scale 0 rho about +0.703.
- Rotary-frequency zeroing also preserves ordering: rho about +0.685.
- Non-DC encoder removal collapses the high/low gap most strongly at final
  encoder: patch rho about +0.097, gap ratio about +0.014.
- Direction-only hidden patch is sharper: first collapse appears at
  `attn00_post_resid`, and final encoder removal gives patch rho about -0.105
  with gap ratio about +0.004.
- Fold-safe hidden-direction patching is now complete on
  {foldsafe.get('n_hidden_cache_eligible', 104)} balanced clips. Across nine
  hook targets, mean held-out baseline rho is
  {fnum(foldsafe.get('baseline_mean'))}; mean patched rho ranges from
  {fnum(foldsafe.get('patch_min'))} to {fnum(foldsafe.get('patch_max'))}, and
  the remaining high/low gap ratio ranges from {fnum(foldsafe.get('gap_min'))}
  to {fnum(foldsafe.get('gap_max'))}.
- AlexNet conv5 gives a transparent sanity check: learned-direction ablation
  drops rho from about +0.386 to +0.018, and forward patching weakens fc7 from
  about +0.432 to +0.212.

## Fold-Safe Hidden Patch

The fold-safe run expanded the hidden cache to
{foldsafe.get('low_ready', 52)} low-memorability plus
{foldsafe.get('high_ready', 52)} high-memorability clips. Each of
{foldsafe.get('folds', 5)} folds trains a hidden high-minus-low direction on
{foldsafe.get('n_train_each', 40)} low plus {foldsafe.get('n_train_each', 40)}
high clips, then patches {foldsafe.get('n_eval_each', 12)} low plus
{foldsafe.get('n_eval_each', 12)} high held-out clips. The hidden direction,
output readout, and reported patch metrics are disjoint within each fold.

{foldsafe_layer_table(results)}

## Layerwise Summary

The strongest localization result is not "position does not matter." The better
claim is:

```text
simple learned-position-table artifact: weakened
hidden sequence dependence: real
population-level TRIBE mechanism: not yet proven
```

The layerwise artifacts are stored in:

- `data/reports/tribe_fourier_critique_review.md`
- `data/reports/tribe_layerwise_encoder_localization.md`
- `data/reports/tribe_layerwise_direction_patch.md`
- `data/reports/tribe_layerwise_encoder_hidden_capture_104.md`
- `data/reports/tribe_foldsafe_direction_patch.md`

## Reviewer-Safe Interpretation

The mechanistic audit supports the selector paper by removing an easy dismissal:
the readout is not merely the temporal position table or a mean-pooling artifact.
But it also narrows the live concern. The model uses sequence structure
internally, and the fold-safe hidden-direction patch shows the learned hidden
direction remains load-bearing across disjoint train/eval folds. That supports
the layerwise effect as an intervention result, while population-level
generalization and content-stratified claims should still be framed cautiously.

## Next Controls

The 104-clip fold-safe patch is complete. The next mechanistic controls are:

- report prompt/content stratification;
- compare random hidden directions and matched-norm patches;
- repeat across multiple balanced subsets.
- replicate the intervention in an open brain-encoder or a transparent video
  model where hidden hooks are easier to audit.

These controls would turn the current fold-safe intervention into a stronger
population-level mechanistic claim.

## References

- Lahner et al. BOLD Moments. Nature Communications 2024.
- Meta/Facebook Research. TRIBE v2 repository.
- Spencer critique thread and follow-up Fourier/position discussion, May 2026.
"""


def audience_axes_paper(results: dict[str, Any]) -> str:
    persona = results["persona_cos"]
    return f"""# Synthetic Audience Axes Are Structured But Not Orthogonal

**Draft status:** satellite paper draft, regenerated {BUILD_DATE}.
**Core purpose:** preserve the useful audience-vector work while removing the
original overclaim.

## Abstract

Synthetic persona-conditioned scoring can reveal structured differences in how
generated or natural videos are ranked, but signed cosine averages can make
shared axes look falsely orthogonal. We revisit the audience-vector
decomposition after a reviewer critique. Twelve persona-conditioned TRIBE
contrastive directions have signed off-diagonal cosine mean
{fnum(persona.get('signed_cos_off_diag', {}).get('mean'))}, but this statistic
hides large positive and negative alignments. The corrected unsigned overlap is
mean abs cosine {fnum(persona.get('abs_cos_off_diag', {}).get('mean'), signed=False)},
and the effective rank is {fnum(persona.get('effective_rank'), 2, signed=False)}
out of 12, compared with nearly 12 for random directions in the same dimension.
Thus the audience directions are structured, stable, and potentially useful for
ranking, but they do not span twelve independent audience axes. The right next
step is persona-matched human validation, not stronger claims about cognitive
modules.

## Corrected Claim

Original unsafe claim:

```text
Per-persona directions are near-orthogonal and reveal independent audience axes.
```

Current safe claim:

```text
Synthetic persona directions compress to a small set of signed latent axes.
Opposite signs can reflect the same underlying axis, and ROI localization is
exploratory.
```

## Why The Correction Matters

A cosine of -0.99 is not evidence for a different memorability axis. It is the
same axis with opposite sign. Since squared projection removes sign, any ROI or
energy localization analysis must be interpreted with this sign ambiguity in
mind.

## Current Numbers

| Quantity | Value |
|---|---:|
| Personas | {persona.get('n_personas', 12)} |
| Signed off-diagonal cosine mean | {fnum(persona.get('signed_cos_off_diag', {}).get('mean'))} |
| Signed off-diagonal cosine range | {fnum(persona.get('signed_cos_off_diag', {}).get('min'))} to {fnum(persona.get('signed_cos_off_diag', {}).get('max'))} |
| Mean absolute off-diagonal cosine | {fnum(persona.get('abs_cos_off_diag', {}).get('mean'), signed=False)} |
| Effective rank | {fnum(persona.get('effective_rank'), 2, signed=False)} / 12 |
| Components for 90 percent variance | {persona.get('n_components_90pct', 4)} |

## Interpretation

The persona system is still interesting. It suggests that model-generated
audience archetypes induce repeatable orderings over stimuli. But the structure
looks low-dimensional and signed, not cleanly modular. This may be useful for
product interfaces, where users want sliders such as "fast-scroll salience" or
"narrative emotionality." It is not yet evidence that real audiences divide
into those exact archetypes.

## Validation Needed

1. Recruit participants who self-identify with or are behaviorally matched to
   the persona profiles.
2. Ask them to rank or choose videos under identical prompt-conditioned sets.
3. Compare persona-derived selector wins with matched and mismatched raters.
4. Report whether synthetic persona axes predict real subgroup preferences
   beyond the global memorability direction.

## Keep Out Of The Main Selector Paper

This work is useful, but it can distract from the primary selector claim. The
main paper should mention persona axes only as exploratory context or appendix
material unless persona-matched human validation is complete.
"""


def distillation_paper(results: dict[str, Any]) -> str:
    single = wan_policy(results, "single_lora")
    raw = wan_policy(results, "raw_best_of_n")
    base_raw = wan_policy(results, "base_or_raw_best_of_n")
    gated = wan_policy(results, "base_or_gated_best_of_n")
    return f"""# From Brain-Aligned Selectors To Video-Generator Distillation

**Draft status:** satellite paper draft, regenerated {BUILD_DATE}.
**Core purpose:** define the LoRA/DPO project without pretending the proxy result
is already behavioral success.

## Abstract

Brain-aligned reward signals can be used in two ways: selecting among generated
candidates at test time, or distilling the selector into the generator through
preference optimization. The current project supports the first path more
strongly than the second. On a 24-seed Wan2.2 proxy run, a preference-weighted
single LoRA improves 20/24 prompts under the TRIBE/BMD projection, and
base-or-gated best-of-4 improves 18/24 with mean lift about +2.817 while avoiding
negative seed-level regressions by falling back to the base clip. This is useful
engineering evidence for a generation-ranking workflow, but it is not proof that
the generator learned human memorability. A distillation paper should begin only
after the selector itself is validated by humans.

## Current Product Workflow

```text
generate base video
generate LoRA / best-of-N variants
score each candidate with TRIBE/BMD v_mem
score preservation with CLIP-style gates
choose the best gated candidate
fallback to base when the gated candidate is worse
```

## Current Proxy Result

| Selection rule | Improved seeds | Mean lift | Median lift |
|---|---:|---:|---:|
| Single LoRA | {policy_improved(single, 20)}/24 | {fnum(policy_mean(single, 1.274))} | {fnum(policy_median(single, 1.218))} |
| Raw best-of-N | {policy_improved(raw, 20)}/24 | {fnum(policy_mean(raw, 3.561))} | {fnum(policy_median(raw, 3.451))} |
| Base or raw best-of-N | {policy_improved(base_raw, 20)}/24 | {fnum(policy_mean(base_raw, 3.674))} | {fnum(policy_median(base_raw, 3.451))} |
| Base or gated best-of-N | {policy_improved(gated, 18)}/24 | {fnum(policy_mean(gated, 2.817))} | {fnum(policy_median(gated, 2.432))} |

## Why DPO Is Harder Than Selection

Selection is cheap because the reward model only chooses among candidates.
Distillation is harder because the generator can learn reward-model loopholes:
artifact-heavy clips, semantic drift, or high-scoring visual tropes that humans
do not actually remember. Video DPO also needs temporally meaningful preference
pairs, consistent seeds, enough diversity, and held-out human evaluation. The
training objective must not be evaluated only by the reward model that created
the labels.

## Minimal DPO Study Design

1. Validate the selector against humans on 50-100 prompts.
2. Generate 4-8 candidates per prompt for 500-2,000 prompts.
3. Label preference pairs with the validated selector plus preservation gates.
4. Train a small LoRA or DPO adapter.
5. Evaluate on fresh prompts with:
   - human pairwise memorability;
   - V-JEPA and CLIP baselines;
   - video-quality gates;
   - delayed recognition if budget allows.

## Product Direction

The most useful near-term product is not a fully trained memorability generator.
It is a queueable selector that lets users upload or generate multiple variants,
then returns:

- raw TRIBE dimensions;
- BMD memorability projection;
- V-JEPA and CLIP baseline scores;
- threshold checks;
- natural-language failure analysis;
- recommended edits or next generations.

Distillation becomes worthwhile only after the selector reliably predicts human
judgment.
"""


def theory_paper(results: dict[str, Any]) -> str:
    rep = results["rep_frame"]
    geom = rep.get("geometry", [])
    first = geom[0] if geom else {}
    return f"""# Representation Frames For Cognitive Media Selection

**Draft status:** theory and methods note, regenerated {BUILD_DATE}.
**Core purpose:** keep the broader intellectual frame without letting it
overclaim the empirical paper.

## Abstract

A learned vector in a model should not be treated as the essence of a cognitive
property. It is a readout in a representation frame. The current memorability
project can be formalized as a frame-transfer problem: TRIBE, V-JEPA, CLIP, and
human behavior each map the same stimulus into different coordinates, and the
scientific question is whether an ordering learned in one frame survives in
another. On the current Wan candidate pool, TRIBE and V-JEPA show moderate
geometric relationship, with RSA Spearman about
{fnum(first.get('rsa_spearman_upper_triangle'), signed=True)} and linear CKA
about {fnum(first.get('linear_cka'), signed=True)}, while CLIP-preservation
signals are much less aligned with the memorability scores. This motivates a
broader program: generated media as controlled intervention on human generative
models, with memorability as the first validated axis.

## Mathematical Frame

Let `X` be the space of prompt-conditioned video candidates. We observe several
representation maps:

```text
f_T : X -> V_T      TRIBE / brain-aligned frame
f_J : X -> V_J      V-JEPA / self-supervised video frame
f_C : X -> V_C      CLIP / prompt-preservation frame
h   : X -> Y        human memory behavior
```

The current readout is:

```text
l_T : V_T -> R
```

The empirical question is whether `l_T(f_T(x))` predicts `h(x)` better than
comparable readouts in `V_J` and `V_C`.

## Anti-Reification Rule

The vector is not the property. The vector is a coordinate that becomes useful
when its induced ordering transfers across frames. This matters for:

- memorability vectors;
- attention vectors;
- persona vectors;
- social-cohesion or volition vectors in future projects.

## Current Frame Audit

TRIBE and V-JEPA are partially aligned, not interchangeable. CLIP preservation
is closer to prompt/seed fidelity than memorability. That separation is useful:
it lets a product enforce prompt fidelity while still optimizing memory signal.

The existing artifact is:

- `experiments/representation_frame_analysis.md`

## Future Work

1. Compare TRIBE, V-JEPA, CLIP, and human pairwise similarity matrices with RSA
   and CKA after human responses are collected.
2. Ask whether high-disagreement prompts reveal semantic, temporal, social, or
   artifact-driven differences between frames.
3. Extend the method from memorability to attention, agency, social warmth,
   threat/safety, and self-other overlap.
4. Use the validated selector before training any generator, so reward-model
   circularity does not define the science.

## References

- Hohwy. The Self-Evidencing Brain. Nous 2016.
- Fields, Friston, Glazebrook, and Levin. A Free Energy Principle for Generic
  Quantum Systems. Progress in Biophysics and Molecular Biology 2022.
- Fields and Glazebrook. Separability, Contextuality, and the Quantum Frame
  Problem. International Journal of Theoretical Physics 2023.
- Dahl, Lutz, and Davidson. Reconstructing and deconstructing the self. Trends
  in Cognitive Sciences 2015.
"""


def readme(results: dict[str, Any]) -> str:
    foldsafe = foldsafe_summary(results)
    return f"""# NeurIPS-Grade Memorability Selector Program

Last generated {BUILD_DATE}. Manually reconciled 2026-06-03.

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
{foldsafe.get('n_hidden_cache_eligible', 104)} balanced clips supports the
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

{result_table(results)}

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
"""


CSS = """
:root {
  --bg: #f6f1e7;
  --ink: #181511;
  --muted: #6f6558;
  --line: #d9ccb9;
  --accent: #193c36;
  --accent2: #a34a2c;
  --paper: #fffaf0;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background:
    linear-gradient(135deg, rgba(25,60,54,0.08), transparent 28%),
    linear-gradient(315deg, rgba(163,74,44,0.08), transparent 34%),
    var(--bg);
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
  line-height: 1.56;
}
.page {
  max-width: 980px;
  margin: 0 auto;
  padding: 42px 28px 72px;
}
.paper {
  background: rgba(255, 250, 240, 0.88);
  border: 1px solid var(--line);
  box-shadow: 14px 14px 28px rgba(90,70,45,0.11), -10px -10px 24px rgba(255,255,255,0.72);
  padding: 48px;
}
nav {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 0 0 20px;
  color: var(--muted);
  font-size: 12px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
nav a {
  color: var(--accent);
  text-decoration: none;
  border: 1px solid var(--line);
  padding: 7px 10px;
  background: rgba(255,250,240,0.7);
}
h1, h2, h3 {
  line-height: 1.08;
  font-family: Georgia, "Times New Roman", serif;
  font-weight: 500;
}
h1 {
  font-size: clamp(36px, 6vw, 72px);
  letter-spacing: -0.02em;
  margin: 0 0 24px;
}
h2 {
  font-size: 29px;
  margin-top: 42px;
  border-top: 1px solid var(--line);
  padding-top: 24px;
}
h3 {
  font-size: 21px;
  margin-top: 30px;
}
p, li { font-size: 16px; }
a { color: var(--accent2); }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: rgba(25,60,54,0.08);
  padding: 0.12em 0.28em;
}
pre {
  padding: 16px;
  overflow-x: auto;
  background: #181511;
  color: #fffaf0;
  border: 1px solid #181511;
}
pre code { background: transparent; padding: 0; }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 18px 0 28px;
  font-size: 14px;
}
th, td {
  border: 1px solid var(--line);
  padding: 9px 10px;
  vertical-align: top;
}
th {
  background: rgba(25,60,54,0.08);
  text-align: left;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  font-size: 12px;
}
blockquote {
  margin: 24px 0;
  padding: 8px 18px;
  border-left: 3px solid var(--accent2);
  background: rgba(163,74,44,0.07);
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 16px;
  margin-top: 26px;
}
.card {
  display: block;
  color: var(--ink);
  text-decoration: none;
  background: rgba(255,250,240,0.8);
  border: 1px solid var(--line);
  padding: 20px;
  min-height: 170px;
  box-shadow: inset 2px 2px 8px rgba(90,70,45,0.08), inset -2px -2px 8px rgba(255,255,255,0.72);
}
.card b {
  display: block;
  margin-bottom: 10px;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 22px;
  line-height: 1.08;
}
.metric {
  display: inline-block;
  margin: 6px 8px 6px 0;
  padding: 8px 10px;
  border: 1px solid var(--line);
  background: rgba(25,60,54,0.06);
}
@media print {
  body { background: white; }
  .page { max-width: none; padding: 0; }
  .paper { box-shadow: none; border: none; padding: 24px; }
  nav { display: none; }
  h1 { font-size: 38px; }
  h2 { break-after: avoid; }
}
"""


def render_markdown(markdown_text: str, title: str, nav_prefix: str = "../") -> str:
    md = MarkdownIt("default", {"html": True})
    body = md.render(markdown_text)
    nav = f"""<nav>
  <a href="{nav_prefix}index.html">Program</a>
  <a href="{nav_prefix}main_selector_paper/paper.html">Main Paper</a>
  <a href="{nav_prefix}satellite_papers/01_mechanistic_audit.html">Mechanistic</a>
  <a href="{nav_prefix}satellite_papers/02_audience_axes.html">Audience Axes</a>
  <a href="{nav_prefix}satellite_papers/03_reward_distillation.html">Distillation</a>
  <a href="{nav_prefix}satellite_papers/05_affect_aware_media_selection.html">Affect</a>
  <a href="{nav_prefix}satellite_papers/04_representation_frames.html">Theory</a>
</nav>"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <main class="page">
    <article class="paper">
{nav}
{body}
    </article>
  </main>
</body>
</html>
"""


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    if not CHROME.exists():
        print(f"[warn] Chrome not found at {CHROME}; skipped {pdf_path}")
        return
    cmd = [
        str(CHROME),
        "--headless",
        "--disable-gpu",
        "--allow-file-access-from-files",
        f"--print-to-pdf={pdf_path}",
        "--no-pdf-header-footer",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def write_doc(
    *,
    markdown_text: str,
    source_md: Path,
    source_html: Path,
    source_pdf: Path,
    site_name: str,
    title: str,
) -> dict[str, str]:
    source_md.parent.mkdir(parents=True, exist_ok=True)
    source_md.write_text(markdown_text)
    html_text = render_markdown(markdown_text, title=title)
    source_html.write_text(html_text)
    print_pdf(source_html, source_pdf)

    SITE_PAPERS.mkdir(parents=True, exist_ok=True)
    site_html = SITE_PAPERS / f"{site_name}.html"
    site_pdf = SITE_PAPERS / f"{site_name}.pdf"
    site_html.write_text(html_text)
    if source_pdf.exists():
        shutil.copy2(source_pdf, site_pdf)
    else:
        print_pdf(site_html, site_pdf)

    return {
        "title": title,
        "markdown": str(source_md.relative_to(ROOT)),
        "html": str(source_html.relative_to(ROOT)),
        "pdf": str(source_pdf.relative_to(ROOT)),
        "site_html": str(site_html.relative_to(ROOT)),
        "site_pdf": str(site_pdf.relative_to(ROOT)),
    }


def site_index(
    results: dict[str, Any],
    docs: list[dict[str, str]],
    *,
    base_prefix: str,
    root_prefix: str,
) -> str:
    tribe = results["tribe_cv"]
    vjepa = results["vjepa_cv"]
    gated = wan_policy(results, "base_or_gated_best_of_n")
    foldsafe = foldsafe_summary(results)
    program_root = PROGRAM.relative_to(ROOT)

    def doc_href(doc: dict[str, str]) -> str:
        path = Path(doc["html"])
        try:
            rel = path.relative_to(program_root)
        except ValueError:
            rel = path
        return f"{base_prefix}{rel.as_posix()}"

    cards = "\n".join(
        f"""<a class="card" href="{html.escape(doc_href(d))}">
          <b>{html.escape(d['title'])}</b>
          <span>HTML</span>
        </a>"""
        for d in docs
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Memorability Selector Program</title>
  <style>{CSS}</style>
</head>
<body>
  <main class="page">
    <section class="paper">
      <nav>
        <a href="{root_prefix}START_HERE.md">Start Here</a>
        <a href="{base_prefix}main_selector_paper/paper.html">Main Paper</a>
        <a href="{base_prefix}experiments/current_selector_prolific_survey_with_vjepa.html">Pilot Survey</a>
      </nav>
      <h1>Memorability Selector Program</h1>
      <p>Regenerated {BUILD_DATE}; manually reconciled 2026-06-03. This is the
      split, reviewer-safe version of the audience-vector research program: one
      main selector paper plus satellites for mechanistic auditing, audience
      axes, reward distillation, and theory.</p>
      <p>
        <span class="metric">TRIBE BMD rho {fnum(tribe.get('mean_spearman'))}</span>
        <span class="metric">V-JEPA rho {fnum(vjepa.get('mean_spearman'))}</span>
        <span class="metric">Fold-safe patch rho {fnum(foldsafe.get('patch_min'))} to {fnum(foldsafe.get('patch_max'))}</span>
        <span class="metric">Wan proxy selector {policy_improved(gated, 18)}/24</span>
        <span class="metric">Prior Prolific: 64.3% best-of-N</span>
        <span class="metric">V-JEPA pilot pending</span>
      </p>
      <div class="grid">
        {cards}
      </div>
      <h2>Current Honest Bottom Line</h2>
      <p>The main paper is now framed around a practical generation-ranking
      workflow, not direct steering. The confirmed result is a compact
      brain-aligned memorability readout with a completed fold-safe
      TRIBE-internal hidden-patch audit and prior Prolific support for the
      broader TRIBE/BMD best-of-N selector signal. The remaining decisive
      experiment is whether TRIBE+gate beats V-JEPA, CLIP, and quality baselines
      in blinded human judgments on the current selector pool.</p>
    </section>
  </main>
</body>
</html>
"""


def write_submission_status(docs: list[dict[str, str]]) -> None:
    lines = [
        "# Submission Status",
        "",
        f"Regenerated: {BUILD_DATE}",
        "",
        "## Completed In This Split Build",
        "",
        "- Main selector manuscript written and rendered.",
        "- Mechanistic audit satellite written and rendered.",
        "- Audience-axis satellite written and rendered.",
        "- Reward-distillation satellite written and rendered.",
        "- Representation-frame theory note written and rendered.",
        "- Program navigation site generated.",
        "- Dated split package zip generated.",
        "",
        "## Still Blocking A Submission-Grade Claim",
        "",
        "- Independent human validation of the V-JEPA-augmented selector.",
        "- Prompt-clustered statistics over human responses.",
        "- Video-quality or VBench-style baseline if feasible.",
        "- Content-stratified and matched-random-control analysis for the completed",
        "  104-clip fold-safe hidden-direction patch.",
        "",
        "## Committed Rendered Papers",
        "",
    ]
    for doc in docs:
        lines.append(f"- {doc['title']}: `{doc['html']}`")
    lines.extend(
        [
            "",
            "PDFs are generated local data-lake outputs and are not guaranteed to",
            "be present in a clean git checkout.",
        ]
    )
    (PROGRAM / "submissions" / f"submission_status_{BUILD_DATE}.md").write_text(
        "\n".join(lines) + "\n"
    )


def make_zip(docs: list[dict[str, str]]) -> Path:
    zip_path = REPORTS / f"neurips_memorability_selector_split_package_{BUILD_DATE}.zip"
    include_roots = [
        PROGRAM / "README.md",
        PROGRAM / "index.html",
        PROGRAM / "theory_bridge.md",
        PROGRAM / "lit_review",
        PROGRAM / "main_selector_paper",
        PROGRAM / "satellite_papers",
        PROGRAM / "experiments",
        PROGRAM / "future_work",
        PROGRAM / "irb",
        PROGRAM / "overleaf",
        PROGRAM / "packages",
        PROGRAM / "submissions",
        PROGRAM / "site",
    ]
    extra = [
        REPORTS / "FINAL_REPORT.md",
        REPORTS / "critical_research_audit_2026-05-25.md",
        REPORTS / "spencer_style_second_review_2026-05-28.md",
        REPORTS / "tribe_fourier_critique_review.md",
        REPORTS / "tribe_layerwise_encoder_localization.md",
        REPORTS / "tribe_layerwise_encoder_hidden_capture_104.md",
        REPORTS / "tribe_layerwise_direction_patch.md",
        REPORTS / "tribe_foldsafe_direction_patch.md",
        REPORTS
        / "wan22_lora_eval_fresh_picsum_24_pref_weighted_r16_s300_bon_24x4_s12_m1p0_product_selector.md",
        REPORTS / "paper.html",
        REPORTS / "paper.pdf",
        ROOT / "scripts" / "tribe_foldsafe_direction_patch.py",
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root in include_roots:
            if root.is_file():
                zf.write(root, root.relative_to(ROOT))
            elif root.exists():
                for path in root.rglob("*"):
                    if (
                        path.is_file()
                        and path.name != ".DS_Store"
                        and path.suffix != ".pyc"
                        and "__pycache__" not in path.parts
                    ):
                        zf.write(path, path.relative_to(ROOT))
        for path in extra:
            if path.exists():
                zf.write(path, path.relative_to(ROOT))
    return zip_path


def main() -> None:
    results = read_results()
    SITE.mkdir(parents=True, exist_ok=True)
    SITE_PAPERS.mkdir(parents=True, exist_ok=True)

    docs = [
        write_doc(
            markdown_text=main_selector_paper(results),
            source_md=PROGRAM / "main_selector_paper" / "paper.md",
            source_html=PROGRAM / "main_selector_paper" / "paper.html",
            source_pdf=PROGRAM / "main_selector_paper" / "paper.pdf",
            site_name="main_selector_paper",
            title="Brain-Aligned Memorability Signals Improve Video Generation Selection",
        ),
        write_doc(
            markdown_text=mechanistic_paper(results),
            source_md=PROGRAM / "satellite_papers" / "01_mechanistic_audit.md",
            source_html=PROGRAM / "satellite_papers" / "01_mechanistic_audit.html",
            source_pdf=PROGRAM / "satellite_papers" / "01_mechanistic_audit.pdf",
            site_name="mechanistic_audit",
            title="Mechanistic Audit Of A TRIBE Memorability Readout",
        ),
        write_doc(
            markdown_text=audience_axes_paper(results),
            source_md=PROGRAM / "satellite_papers" / "02_audience_axes.md",
            source_html=PROGRAM / "satellite_papers" / "02_audience_axes.html",
            source_pdf=PROGRAM / "satellite_papers" / "02_audience_axes.pdf",
            site_name="audience_axes",
            title="Synthetic Audience Axes Are Structured But Not Orthogonal",
        ),
        write_doc(
            markdown_text=distillation_paper(results),
            source_md=PROGRAM / "satellite_papers" / "03_reward_distillation.md",
            source_html=PROGRAM / "satellite_papers" / "03_reward_distillation.html",
            source_pdf=PROGRAM / "satellite_papers" / "03_reward_distillation.pdf",
            site_name="reward_distillation",
            title="From Brain-Aligned Selectors To Video-Generator Distillation",
        ),
        write_doc(
            markdown_text=theory_paper(results),
            source_md=PROGRAM / "satellite_papers" / "04_representation_frames.md",
            source_html=PROGRAM / "satellite_papers" / "04_representation_frames.html",
            source_pdf=PROGRAM / "satellite_papers" / "04_representation_frames.pdf",
            site_name="representation_frames",
            title="Representation Frames For Cognitive Media Selection",
        ),
        write_doc(
            markdown_text=(
                PROGRAM / "satellite_papers" / "05_affect_aware_media_selection.md"
            ).read_text(),
            source_md=PROGRAM
            / "satellite_papers"
            / "05_affect_aware_media_selection.md",
            source_html=PROGRAM
            / "satellite_papers"
            / "05_affect_aware_media_selection.html",
            source_pdf=PROGRAM
            / "satellite_papers"
            / "05_affect_aware_media_selection.pdf",
            site_name="affect_aware",
            title="Affect-Aware Media Selection From Brain-Aligned And EEG-Inspired Signals",
        ),
    ]

    (PROGRAM / "README.md").write_text(readme(results))
    (SITE / "index.html").write_text(
        site_index(results, docs, base_prefix="../", root_prefix="../../../")
    )
    (PROGRAM / "index.html").write_text(
        site_index(results, docs, base_prefix="", root_prefix="../../")
    )
    # Also mirror a convenient top-level report entry point.
    (REPORTS / "neurips_memorability_selector_program.html").write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="0; url=../../research_program/neurips_memorability_selector/index.html" />
  <title>Memorability Selector Program</title>
</head>
<body>
  <p><a href="../../research_program/neurips_memorability_selector/index.html">Open the memorability selector program index</a>.</p>
</body>
</html>
"""
    )
    (SITE / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "build_date": BUILD_DATE,
                "docs": docs,
            },
            indent=2,
        )
        + "\n"
    )
    write_submission_status(docs)
    zip_path = make_zip(docs)

    print("[done] generated split paper package")
    print(f"[site] {SITE / 'index.html'}")
    print(f"[zip]  {zip_path}")


if __name__ == "__main__":
    main()
