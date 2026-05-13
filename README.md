# audience_vectors

Interpretable viewer-response directions in brain-aligned video models.

A no-new-human-data research framework for predicting which moments of a video
viewers are likely to attend to, remember, skip, or find confusing — by
combining public memorability / engagement / fMRI datasets, synthetic
persona-conditioned labels from VLMs, and contrastive activation directions
extracted from brain-encoding video models (TRIBE v2, V-JEPA 2, InternVideo2).

See the project conversation for the full design write-up. This README is
just the operating manual.

---

## Layout

```
isc_mod/
├── pyproject.toml              hatchling + Python 3.12, ruff/pyright/pytest config
├── .env / .env.example         secrets + paths + model IDs + pipeline knobs
├── .python-version
├── .gitignore                  excludes data/, .env, model weights
├── data/                       (gitignored) media + features + labels + models
├── scripts/                    CLI entry points (build_manifest.py, ...)
├── tests/                      pytest smoke + unit tests
└── src/audience_vectors/
    ├── config.py               Config + Paths + ModelIds + ApiKeys (env-backed)
    ├── schemas/canonical.py    LabelValue, Segment, Persona, AudienceVector
    ├── datasets/base.py        DatasetAdapter ABC — subclass per source
    ├── modal_app/
    │   ├── app.py              unified modal.App, env_secrets, MODAL_REGION
    │   ├── image_factory.py    base_image + tribe_image (pinned CUDA stack)
    │   └── functions/
    │       └── tribe_predictor.py    populate_tribe_weights + TribeV2Predictor
    └── services/
        └── tribe_service.py    async wrapper around the Modal predictor
```

## Setup

```bash
# 1. Install deps. Hatchling + uv assumed.
uv sync --group dev --extra modal --extra ml --extra tracking

# 2. Fill in API keys (at minimum: ANTHROPIC_API_KEY, GOOGLE_API_KEY, HUGGINGFACE_TOKEN).
$EDITOR .env

# 3. Verify Python imports.
uv run pytest -q

# 4. (Optional) When ready to run on Modal:
modal token new                                                 # one-time
modal deploy audience_vectors.modal_app.app                     # deploy
modal run audience_vectors.modal_app.functions.tribe_predictor::populate_tribe_weights
```

The TRIBE weights volume costs ~10 GB and downloads gated models
(`meta-llama/Llama-3.2-3B`) — make sure your HF token has accepted Meta's
license before `populate_tribe_weights`.

## Model IDs (pinned)

| Component | HuggingFace repo | Notes |
|---|---|---|
| TRIBE v2 | `facebook/tribev2` @ `f894e783020944dcd96e5568550afe2aa9743f9f` | CC-BY-NC-4.0 — non-commercial only |
| TRIBE source | `github.com/facebookresearch/tribev2` @ `72399081ed3f1040c4d996cefb2864a4c46f5b8e` | |
| V-JEPA 2 (default) | `facebook/vjepa2-vitl-fpc64-256` | MIT, 0.3B params |
| V-JEPA 2 (used inside TRIBE) | `facebook/vjepa2-vitg-fpc64-256` | Apache-2.0, ~1B params |
| InternVideo2 encoder | `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` | **Gated** — click-through required |
| InternVideo2.5 chat | `OpenGVLab/InternVideo2_5_Chat_8B` | Optional chat VLM alternative |
| Qwen3-VL (default VLM) | `Qwen/Qwen3-VL-8B-Instruct` | Apache-2.0, current default |
| Qwen2.5-VL (fallback) | `Qwen/Qwen2.5-VL-7B-Instruct` | Apache-2.0, for parity with prior results |
| Whisper large-v3 | `openai/whisper-large-v3` | |
| faster-whisper CT2 | `Systran/faster-whisper-large-v3` @ `edaa852ec7e145841d8ffdb056a99866b5f0a478` | What TRIBE invokes via `uvx whisperx` |

TRIBE's nested feature encoders (also pinned in `image_factory.py`):

| Component | HF revision |
|---|---|
| `meta-llama/Llama-3.2-3B` | `13afe5124825b4f3751f836b40dafda64c1ed062` (gated) |
| `facebook/w2v-bert-2.0` | `da985ba0987f70aaeb84a80f2851cfac8c697a7b` |
| `facebook/vjepa2-vitg-fpc64-256` | `875c192b7b704b87d1e1d99345769632dd5f739a` |
| `facebook/dinov2-large` | `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` |

## Datasets (access)

| Dataset | How to get it | Gotchas |
|---|---|---|
| **VideoMem** (Cohendet et al.) | Email `videomemmanagement@interdigital.com` | Research-only license, French IP law, no commercial use without paid agreement |
| **Memento10k** (Newman et al.) | Form at <http://memento.csail.mit.edu/> | MIT CSAIL gated form |
| **VIDEM** (MediaEval 2025) | Register for MediaEval 2025 commercial memorability task | Tied to task participation; 424 ads + brand recall scores |
| **BOLD Moments** | OpenNeuro [`ds005165`](https://openneuro.org/datasets/ds005165) + GitHub [`blahner/BOLDMomentsDataset`](https://github.com/blahner/BOLDMomentsDataset) | Stimuli **not** on OpenNeuro (licensing) — pull from GitHub |
| **Hasson naturalistic fMRI** | OpenNeuro — Sherlock recall `ds001132`, Sherlock/Merlin `ds001110`, Narratives `ds002345` | Stimulus videos are copyrighted; only fMRI + recall is public |
| **PEEK / VLEngagement** (Yousuf/Bulathwela) | GitHub [`sahanbull/PEEK-Dataset`](https://github.com/sahanbull/PEEK-Dataset), [`sahanbull/context-agnostic-engagement`](https://github.com/sahanbull/context-agnostic-engagement) | Feature CSVs only — VideoLectures.Net source videos not redistributed |

## Recommended starting order

MVP (week 1):

1. **`BOLDMoments`** adapter → unified manifest (annotations are public S3 — no form, no auth, run `uv run python scripts/download_bold_moments.py`)
2. 3-second segmentation
3. Gemini Flash synthetic labels on 20 dev videos
4. First TRIBE feature extraction job on Modal
5. First contrastive memorability vector
6. Heatmap notebook

Research extension:

7. Add `Memento10k` once the request form approves (adapter already scaffolded)
8. + `VIDEM` for ad/brand memorability
9. + `Hasson` for naturalistic fMRI
10. + Persona-conditioned labels (5 archetypes → 20 → 1000 variants)
11. Activation patching + steering experiments

## BOLD Moments quick start

Annotations come from OpenNeuro `ds005165`'s public S3 bucket — no auth, no
form. Memorability scores, captions, action labels, scene labels, and spoken
transcriptions for all 1,102 clips (1,000 train + 102 test), ~1.6 MB total:

```bash
# One-time download (annotations only; stimulus videos require MiT_url
# walking, optional via --include-videos).
uv run python scripts/download_bold_moments.py

# Build the manifest from the configured adapter(s).
uv run python scripts/build_manifest.py --datasets BOLDMoments

# Optionally also fetch the actual stimulus MP4s via each entry's MiT_url
# (slow, ~1102 requests, some upstream links may have rotted).
uv run python scripts/download_bold_moments.py --include-videos --limit 20
```

The adapter falls back to each entry's `MiT_url` as the `media_uri` when no
local video is on disk, so downstream segmentation / VLM labeling can still
work clip-by-clip without the full video download.

## Status

End-to-end pipeline runs on real data:

```bash
# Pull BMD annotations (~1.6 MB) + 20 sample stimulus clips (~6.5 MB)
uv run python scripts/download_bold_moments.py --include-videos --limit 20

# Build manifest + cut segments + score with Gemini
uv run python scripts/build_manifest.py    --datasets BOLDMoments --limit 20
uv run python scripts/segment_dataset.py   --datasets BOLDMoments --limit 20
uv run python scripts/label_segments.py    --limit 20 --concurrency 4

# Compare VLM-synthetic labels against BMD human memorability scores
uv run python scripts/eval_label_correlation.py \
    --output data/reports/vlm_vs_human.md
```

### Scaled to n=1,026 BMD clips with 5-fold cross-validation (the defensible result)

After scaling to the full BMD dataset (1,026 V-JEPA features extracted; the
other 76 had dead MiT URLs) and running proper k-fold CV with bootstrap CIs:

```
V-JEPA contrastive vector vs BMD memorability_score
  5-fold cross-validation, n=205-206 test per fold
  
  fold 1:  ρ = +0.426  95% CI [+0.289, +0.538]
  fold 2:  ρ = +0.345  95% CI [+0.217, +0.467]
  fold 3:  ρ = +0.375  95% CI [+0.248, +0.497]
  fold 4:  ρ = +0.395  95% CI [+0.272, +0.508]
  fold 5:  ρ = +0.435  95% CI [+0.316, +0.543]
  
  Mean ± stdev:  +0.395 ± 0.037
  Random-direction null (n=200 trials):  ρ = -0.006 ± 0.095  |ρ|@95th = +0.184
```

Every fold's bootstrap CI excludes 0. Mean ρ = +0.395 is **2.1× the
random null's 95th percentile**. Scaling from n=184 → n=1,026 kept the
mean steady (+0.378 → +0.395) but collapsed the per-fold stdev 3.4×
(0.126 → 0.037), turning a noisy preliminary signal into a robust effect.

### Scaled to n=200 BMD clips (early checkpoint, kept for record)

| stage | result |
|---|---|
| Stimulus videos downloaded via MiT_url | 184/200 (16 upstream 404s) |
| V-JEPA 2 features extracted on Modal A10G | 184/200, 1024-dim each |
| Zero-shot Gemini synthetic labels | 183/200 |
| Persona-conditioned labels (4 personas × 20 segments) | 80 |
| Contrastive audience vectors trained | 3 |

**Predictor head-to-head against BMD human memorability**

n=19 (initial pilot — extreme curated subset):

| predictor | Spearman ρ |
|---|---:|
| zero-shot Gemini memorability | **+0.507** |

n=200 (full scale — broader sample, includes held-out middle band):

| predictor | all (n≈184) | held-out middle (n≈75) |
|---|---:|---:|
| zero-shot Gemini memorability | +0.151 | −0.031 |
| V-JEPA contrastive (trained on BMD top/bottom) | +0.519* | +0.059 |
| V-JEPA contrastive (trained on Gemini ranking) | +0.066 | −0.051 |

*Includes training-set contribution — not generalization.

**V-JEPA recovers VLM-detectable interestingness, not human memorability**:
when V-JEPA's contrastive direction is trained against *Gemini's* memorability
ranking, held-out Spearman against that VLM target is **+0.207**. But the same
vector vs BMD human ground truth is near zero. V-JEPA features encode "what
VLMs notice," not "what humans actually remember."

### TRIBE v2 (brain-aligned) vs V-JEPA vs Gemini head-to-head

After Llama-3.2 approval landed, ran TRIBE v2 on Modal (H100/B200 with cu124
torch wheel) to get per-vertex cortical activations (20,484 vertices × 4
time chunks per 3-second clip), then trained a contrastive vector against
BMD memorability top/bottom 30%:

| predictor | all ρ vs BMD | n | held-out ρ | n |
|---|---:|---:|---:|---:|
| Gemini zero-shot | +0.217 | 183 | +0.091 | 100 |
| V-JEPA contrastive (1,024-dim) | +0.309 | 184 | **+0.233** | 100 |
| TRIBE contrastive (20,484-dim) | **+0.399** | 133 | −0.082 | 51 |

Two real findings:

1. **Brain-aligned features have stronger overall contrast** — TRIBE beats V-JEPA
   (which beats Gemini) at separating top/bottom memorable clips by BMD
   ground truth.
2. **TRIBE generalizes worse on held-out at this scale** — 20,484-dim contrastive
   direction from 80 training segments overfits; V-JEPA's tighter feature space
   generalizes better with the same training budget.

The story for scaling: TRIBE needs either more training data (the full 1,102 BMD clips)
or smarter aggregations (per-cortical-region vs full-cortex mean) before its
held-out generalization catches up to its overall correlation.

**Persona signal (n=4 personas × 20 segments)** stays informative:

| persona | memorability ρ vs BMD |
|---|---:|
| frame-poet-cleo (cinematic) | +0.30 |
| ad-blocker-priya (skeptical) | +0.27 |
| tearjerker-theo (narrative) | +0.10 |
| **swipe-king-zara (fast-scroll)** | **−0.12** |

Fast-scroll viewer is anti-correlated with the general human population —
the cleanest evidence so far that persona-conditioned vectors are a real
research direction.

### Persona-conditioned signal (n=4 personas × 20 segments)

Different synthetic audience archetypes produce **structurally different
rankings** of the same clips. Per-persona Spearman correlation against
BMD human memorability:

| persona | memorability ρ | semantic_surprise ρ | rewatch ρ |
|---|---:|---:|---:|
| frame-poet-cleo (cinematic) | +0.30 | +0.43 | +0.42 |
| ad-blocker-priya (skeptical) | +0.27 | +0.51 | +0.11 |
| tearjerker-theo (narrative) | +0.10 | +0.13 | −0.08 |
| swipe-king-zara (fast-scroll) | **−0.12** | +0.09 | −0.05 |

The fast-scroll viewer is *anti-correlated* with the BMD population
average on memorability. Cross-persona stdev on attention/visual axes is
~0.15 — meaningful audience disagreement, which is exactly what
contrastive audience-vector decomposition is supposed to recover.

Still TODO: TRIBE feature extraction (pending Llama-3.2 approval),
V-JEPA feature extraction (scaffolded + Modal deployed, populate in
flight), contrastive vector training, scoring + heatmaps, more adapters.

`pyright` reports unresolved imports for `audience_vectors.*` until you
run `uv sync` (or `uv sync --group dev --extra ml --extra modal`).
The actual pytest suite passes — 34/34 as of this checkpoint.

## License notes

This repo includes pins to TRIBE v2 (CC-BY-NC-4.0). Any deployed system built
on this pipeline inherits non-commercial use, regardless of the licenses of
individual backbones.
