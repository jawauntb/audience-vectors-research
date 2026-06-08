# audience_vectors

Interpretable viewer-response directions in brain-aligned video models.

A research framework for predicting which moments of a video viewers are likely
to attend to, remember, skip, or find confusing — by combining public
memorability / engagement / fMRI datasets, synthetic persona-conditioned labels
from VLMs, contrastive activation directions extracted from brain-encoding video
models (TRIBE v2, V-JEPA 2, InternVideo2), and a lightweight Prolific validation
pass.

## Current Source Of Truth

The project is now split into a codebase, a research-program folder, a live
analyzer product surface, and a local data lake. Start here first:

- [Start Here](START_HERE.md)
- [Claim ledger](CLAIM_LEDGER.md)
- [Current research status](research_program/neurips_memorability_selector/experiments/current_research_status.md)
- [Current main paper source](research_program/neurips_memorability_selector/main_selector_paper/paper.md)
- [Current main paper HTML](research_program/neurips_memorability_selector/main_selector_paper/paper.html)
- [Publishing and artifact policy](docs/PUBLISHING.md)

The older `data/reports/paper.html` and `data/reports/paper.pdf` files are
local data-lake artifacts, not committed source-of-truth documents. They may
exist on the main workstation under `/Users/jawaun/isc_mod/data/reports/`, but
GitHub intentionally excludes generated PDFs/zips, raw Prolific exports, model
weights, and generated videos.

Live analyzer:
[https://jawaun--video-analyzer.modal.run](https://jawaun--video-analyzer.modal.run)

Current research-program folder:
[`research_program/neurips_memorability_selector`](research_program/neurips_memorability_selector)

Current strongest blocker:

- Human validation scope: the earlier Prolific best-of-N study supports the
  broader TRIBE/BMD selector signal, but the newer V-JEPA-augmented generated
  selector pilot is prepared and not launched.
- Mechanistic validation: the fold-safe TRIBE hidden-direction patch now runs on
  the planned 104 balanced cache-eligible clips. Hidden-cache coverage is
  complete at 52 low + 52 high clips across all requested encoder targets.

GitHub artifact policy:

- Commit source code, scripts, tests, Markdown/HTML research docs, protocols,
  templates, and lightweight JSON manifests.
- Do not commit raw data, model weights, generated videos, generated PDFs/zips,
  Prolific response exports, local caches, or vendored third-party repos.
- Keep the repo private while it references gated models, TRIBE non-commercial
  terms, unpublished human-study materials, and local generated media.

## Status (updated 2026-06-03, post critical source-of-truth cleanup)

Latest committed paper source:
`research_program/neurips_memorability_selector/main_selector_paper/paper.md`.
Latest committed readable render:
`research_program/neurips_memorability_selector/main_selector_paper/paper.html`.

Local generated reports, PDFs, demos, and bundles live under
`/Users/jawaun/isc_mod/data/reports/` on the main workstation when available.
They are not committed.

### Headline (one line)

The TRIBE-projection direction survives human contact: on a Prolific pairwise
forced-choice study (n=41, 100% attention-check pass), humans prefer the
TRIBE-projection-ranked best-of-N winner over the within-seed median variant
**64.3% of the time** (290/451, 95% Wilson CI [0.598, 0.686], binomial
p = 1.3 × 10⁻⁹; pair-cluster bootstrap CI [0.565, 0.718]; per-pair t-test
p = 0.0057). The previously open "metric-internal" caveat is substantially
addressed for that earlier selector setting. This does not automatically validate
the newer V-JEPA-adjudicated selector pool, BO-generated videos, or delayed
recognition memorability.

### Full numbers, all bootstrapped or controlled

- **Prediction (BMD memorability):** TRIBE 5-fold CV ρ = **+0.403 ± 0.061**
  (n=1022); canonical BMD test split ρ = **+0.478** (n=93, 95% CI
  [+0.30, +0.63]). The critical audit retires the old "1.9× V-JEPA" headline:
  the current V-JEPA full-CV run is comparable at **+0.395 ± 0.037**.
  The updated claim is competitive prediction in a brain-aligned feature space,
  not a clean global-baseline win.
- **Compactness:** label-permutation null (n=1000) → z = +9.79; random-direction
  ablation control (n=200) leaves ρ = +0.405 ± 0.004 while fold-safe v_mem
  ablation leaves ρ = +0.057 (z = −78.8). Nonlinear probes show residual signal
  (random forest +0.414 → +0.196), so the right claim is dominant-axis, not
  literal one-dimensionality. Directions 2–10 are weak (mean ρ = −0.007, range
  [−0.059, +0.060]).
- **Stability:** disjoint-halves cos = +0.963.
- **Measured-fMRI pilot:** BMD sub-01 measured beta estimates recover a
  memorability direction aligned with TRIBE's v_mem (cos = +0.336) and predictive
  of BMD memorability (5-fold CV ρ = +0.449). This is a positive single-subject
  pilot; all-subject aggregation is still needed.
- **TRIBE Fourier/position audit:** saved TRIBE output tensors do not support a
  simple temporal-position artifact explanation. Native 4-bin clips: full tensor
  ρ = +0.401 ± 0.031, temporal-DC-only ρ = +0.405 ± 0.027, nonzero-temporal
  ρ = +0.297 ± 0.064; all 1022 clips after 3→4 resampling replicate the pattern.
  Modal introspection confirms the internal model has learned temporal positional
  embeddings (`_model.time_pos_embed`, shape [1, 1024, 1152]) and rotary positional
  machinery. Directly scaling that learned table on 24 balanced clips leaves the
  memorability readout intact: scale 0.0 gives ρ = +0.703 vs baseline ρ = +0.677
  and retains 80.9% of the high-low gap. A deeper encoder hook shows the nuance:
  zeroing rotary `inv_freq` preserves ordering (ρ = +0.685), but collapsing the
  encoder output to sequence-DC nearly removes the high-low gap (ρ = +0.097; gap
  ratio 0.013). Layerwise post-attention hooks localize this dependence from the
  first attention residual onward: layer 0 non-DC removal already leaves only
  9.7% of the high-low gap, and the final encoder leaves 1.4%. The sharper
  direction-only patch is stronger: removing the learned hidden memorability
  direction at layer 0 gives ρ = −0.030 and gap ratio 0.057; removing it at the
  final encoder gives ρ = −0.105 and gap ratio 0.004. The larger fold-safe
  hidden-patch run has now completed on 104 balanced cache-eligible clips:
  baseline held-out ρ averages +0.602 across folds, while patching the learned
  hidden memorability direction drops eval ρ to roughly +0.054 to +0.200
  depending on layer, with high-low gap ratios around 0.135 to 0.212. So the
  simple learned-position artifact is weakened, while the learned hidden
  memorability direction is patch-sensitive under a fold-safe split.
- **Open-model sanity checks:** transparent AlexNet conv5 layer-5 features predict
  BMD memorability at ρ = +0.386; offline ablation drops prediction to ρ = +0.018
  while random-direction ablations leave ≈ +0.382. A true AlexNet forward-pass
  patch before fc6/fc7/logits weakens downstream readouts (fc7 +0.432 → +0.212;
  logits +0.386 → +0.128). A small open CLIP frame-encoder pilot also shows signed
  block-level steering works, though removal is inconclusive.
- **Generator-side (TRIBE metric):** best-of-N lift +2.07 SVD-XT (95% CI
  [+0.95, +3.19]) and +1.53 Veo 3 Fast (CI [+0.70, +2.40]). The open-weight
  Wan2.2 follow-up is now a small LoRA/product-selector result:
  the preference-weighted single LoRA improves **20/24** fresh image-seeded prompts,
  and base-or-gated best-of-4 improves **18/24** with mean lift **+2.817** and
  median lift **+2.432** under the TRIBE/BMD projection. This is proxy-scored and
  still needs human validation.
- **Human validation (Prolific):** Best-of-N pooled 64.3% (pair-cluster
  bootstrap CI [0.565, 0.718], per-pair t-test p = 0.0057);
  α-steering 1-pair 63.4% (n.s.); persona-pair pooled 73.2% (p = 2.3×10⁻¹¹,
  with honest caveat that persona-matched raters are still needed).

### Honest caveats (now reviewer-corrected)

- Persona "decomposition" is ~4 effective axes, not 12 — the signed cosine
  mean of +0.02 was bimodal, the correct metric is |cos| = 0.43 (§5.4).
- α-steering is not significant across 9 seeds (t-test p=0.44); the 1-pair
  Prolific result is descriptive only.
- Cross-domain transfer drops to ρ = +0.10–0.13 for indoor↔outdoor scenes.
- Brain-alignment is NOT the active ingredient for global best-of-N
  (V-JEPA-as-judge matches TRIBE). It is clearly useful for held-out human
  memorability prediction; the V-JEPA persona comparison is currently
  inconclusive rather than a clean TRIBE win.
- TRIBE has real internal temporal positional machinery. The Fourier result and
  direct `time_pos_embed` scale patch weaken a simple time-position artifact
  critique. Rotary-frequency zeroing also preserves ordering, but encoder non-DC
  sequence structure matters for the readout from the first post-attention residual
  onward. Direction-only patches show that removing the learned hidden
  memorability direction sharply disrupts the readout on a 24-clip high/low
  subset, so the honest remaining
  caveat is hidden sequence entanglement rather than a simple positional-table
  artifact.
- Wan LoRA/product-selector gains are not behavioral evidence yet; they are
  TRIBE/BMD proxy gains that need blinded human comparison against base and
  random LoRA variants.
- BO-generated SVD videos are not yet broad strategy evidence. The completed
  max-3 regenerated-control stress test shows prompt-pocket behavior: a stable
  positive jellyfish pocket and a brittle, low-scoring fireworks pocket.
- A prompt-transfer stress test retargeted the top saved BO recipes across five
  image-backed prompt slots. The recipes did not transfer: only jellyfish stayed
  positive, and matched Sobol-transfer controls slightly beat BO-transfer
  overall.
- A per-prompt Sobol search over the same five image-backed prompt slots found
  that prompt/seed identity dominates alpha/guidance choice: prompt-only R2 =
  0.9196 on retained TRIBE scores, while recipe-only R2 = 0.0062. All top eight
  retained rows were blue jellyfish.
- A follow-up SVD content-axis audit found that prompt text is metadata-only in
  the current SVD replay path and that only 5/24 catalog seed images are locally
  available. Prompt rewriting is therefore not a valid SVD replay intervention
  until the generator path changes.

### 2026-06-08 Max-3 Regenerated Visual Controls Result

The balanced max-3 stress test generated 36/36 requested clips. Two BO
fireworks clips failed the automated visual gate (`bo09_cand01` replicate 1 and
`bo03_cand01` replicate 2), so complete-candidate retention withheld those two
candidate families before scoring. The retained panel kept 10/12 candidates and
scored 30/30 retained rows with full TRIBE.

Retained stratum result:

| stratum | BO retained candidates | BO mean | regenerated Sobol candidates | Sobol mean | interpretation |
|---|---:|---:|---:|---:|---|
| fireworks | 1/3 | -3.9426 | 3/3 | -5.1029 | BO is less bad, but the stratum is visually brittle and low scoring |
| jellyfish | 3/3 | 1.7443 | 3/3 | 1.0575 | stable positive BO pocket |

The compute-side finding is prompt-pocket behavior, not broad strategy
dominance: BO reliably exploits the jellyfish replay pocket, while fireworks
remains weak under the current tuned SVD/TRIBE replay for both BO and controls.
The committed run note is
`research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/max3_regenerated_visual_controls_result_20260608.md`.
True broad prompt evidence still requires a new BO/search panel over additional
seed prompts.

### 2026-06-08 Prompt-Transfer Stress Test Result

The prompt-transfer stress test asked whether the top saved BO recipes were
portable alpha/guidance policies. It retargeted `bo07_cand01`, `bo04_cand01`,
and `bo02_cand01` across five locally image-backed prompt slots, with three
matched Sobol alpha/guidance controls per slot.

The run generated 30/30 clips. Two fireworks rows failed the visual gate, so
complete-candidate retention kept 28/30 rows for full TRIBE scoring.

| prompt slot | BO-transfer mean | Sobol-transfer mean | interpretation |
|---|---:|---:|---|
| fireworks | -4.0593 | -3.9195 | visually brittle and negative for both policies |
| ocean cliffs | -7.5721 | -8.0602 | very negative for both policies |
| concert stage | -2.4617 | -0.3998 | Sobol-transfer is less bad; BO recipes do not help |
| blue jellyfish | 1.0982 | 1.2100 | only positive prompt slot; Sobol-transfer slightly higher |
| forest canopy | -4.8986 | -4.2410 | negative for both policies |

This is evidence against portable BO recipe transfer. The saved high-scoring BO
recipes are jellyfish-pocket recipes, not reusable global steering/guidance
policies. The committed run note is
`research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/prompt_transfer_stress_test_result_20260608.md`.

### 2026-06-08 Per-Prompt Sobol Search Result

The per-prompt search panel tested eight shared Sobol alpha/guidance points
across each of the five locally image-backed prompt slots. It generated 40/40
clips; two fireworks rows failed the visual gate, so complete-candidate
retention scored 38/40 retained candidates with full TRIBE.

| prompt slot | scored / requested | mean TRIBE | best score |
|---|---:|---:|---:|
| fireworks | 6 / 8 | -3.9068 | -1.9761 |
| ocean cliffs | 8 / 8 | -8.1526 | -6.8832 |
| concert stage | 8 / 8 | -1.3808 | -0.1553 |
| blue jellyfish | 8 / 8 | 1.6597 | 2.9734 |
| forest canopy | 8 / 8 | -3.7246 | -1.7318 |

Prompt identity explained almost all retained score variance (prompt-only R2 =
0.9196), while Sobol recipe index alone explained almost none (R2 = 0.0062).
The top eight retained candidates were all blue jellyfish. The committed run
note is
`research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/per_prompt_sobol_search_result_20260608.md`.

The new operating conclusion is that alpha/guidance-only search is exhausted as
a broadening axis under the current replay regime. For current SVD replay, the
next search should add seed-image selection or seed-bank expansion before more
BO over alpha/guidance. Prompt rewriting only becomes a valid content variable
after switching to, or plumbing in, a prompt-conditioned generator path.

### 2026-06-08 SVD Content-Axis Audit

The content-axis audit checked what the current SVD replay runner can actually
change. It found 24 prompt catalog rows but only 5 locally available seed
images, and the SVD Modal generation call does not accept or pass prompt text.
The prompt field is currently provenance/stratification metadata, not a
generation-conditioning variable.

An attempted 20-job fixed-recipe seed-content replicate panel dry-ran correctly,
but full generation was blocked immediately by the Modal workspace billing-cycle
spend limit. No scored replay result was produced by that attempted probe.

The committed audit note is
`research_program/neurips_memorability_selector/collaborator_inputs/camilo_bo_memorability/svd_content_axes_audit_20260608.md`.

The next valid content-broadening move is seed-bank restoration/expansion under
SVD, or a switch to a prompt-conditioned generator such as CogVideoX, Wan2.2, or
Veo before running prompt-rewrite tournaments.

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

## Historical Checkpoints (Superseded)

The checkpoints below are retained for provenance. The source of truth is the
2026-06-03 status block above plus `START_HERE.md` and
`research_program/neurips_memorability_selector/main_selector_paper/paper.md`.
The older `data/reports/paper.*` artifacts are local generated outputs, not the
committed navigation spine.

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

### Scaled to n=1,026 BMD clips with 5-fold cross-validation (early V-JEPA checkpoint)

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

### TRIBE v2 (brain-aligned) vs V-JEPA vs Gemini head-to-head (historical small-n checkpoint, superseded)

After Llama-3.2 approval landed, ran TRIBE v2 on Modal (H100/B200 with cu124
torch wheel) to get per-vertex cortical activations (20,484 vertices × 4
time chunks per 3-second clip), then trained a contrastive vector against
BMD memorability top/bottom 30%:

| predictor | all ρ vs BMD | n | held-out ρ | n |
|---|---:|---:|---:|---:|
| Gemini zero-shot | +0.217 | 183 | +0.091 | 100 |
| V-JEPA contrastive (1,024-dim) | +0.309 | 184 | **+0.233** | 100 |
| TRIBE contrastive (20,484-dim) | **+0.399** | 133 | −0.082 | 51 |

This checkpoint is kept for archaeology only; the full-CV results above supersede
it and retire any global "TRIBE beats V-JEPA" claim. Two local findings from the
small-n run were:

1. **Brain-aligned features had stronger small-n overall contrast** — in this
   early subset, TRIBE separated top/bottom memorable clips better in all-sample
   correlation, but this was not held-out evidence of a global V-JEPA win.
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

Current TODOs are the forward-looking items in the paper: multi-subject fMRI
aggregation, persona-matched raters, cross-dataset transfer, and a real open
video-brain model patching experiment beyond the AlexNet/CLIP pilots. The old
extraction TODOs above have been completed or superseded by the full paper
pipeline.

## License notes

This repo includes pins to TRIBE v2 (CC-BY-NC-4.0). Any deployed system built
on this pipeline inherits non-commercial use, regardless of the licenses of
individual backbones.
