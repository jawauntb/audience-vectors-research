# Active Research Control Doc

Last updated: 2026-06-08.

Purpose: keep the research loop explicit enough that we can continue across
sessions without drifting. This is the working queue for what to do next, why it
matters, what gate decides whether it counts, and which artifacts should be
updated after each run.

Update rule: every substantial research turn should update this file, the
current status doc, or the claim ledger if the accepted state changes. If a run
only refines tactics, update this file. If a run changes what can be claimed,
update `CLAIM_LEDGER.md` and
`research_program/neurips_memorability_selector/experiments/current_research_status.md`
too.

## Current Accepted State

- Human behavior remains the strongest validation layer. The Prolific Study A
  result supports the TRIBE/BMD selector signal, but it does not validate every
  later compute-only candidate pool.
- BO/SVD/LoRA/Modal work is compute-proxy evidence until a human or
  BMD-grounded gate validates the selected videos.
- Recent SVD replay and pocket-regime audit results show that seed-image and
  content identity dominate the score structure. Alpha/guidance-only broadening
  is not the current path to research progress.
- Accepted compute-proxy finding C-017: stable non-jellyfish content pockets
  survive local recipe stress tests. Orange flowers and hanging clothes are the
  most important positive pockets to consolidate next. Aerial beach, city
  street, and storm beach are current hard negative controls.
- Prompt text is metadata-only in the current SVD runner. A prompt-rewrite
  tournament should wait until we use a prompt-conditioned generator path where
  the prompt changes the actual pixels.

## Current Regime

- Artifact types: seed images, prompts, SVD/Modal recipes, generated videos,
  TRIBE scores, memorability deltas, visual-gate statuses, run manifests,
  pocket labels, claim-ledger entries, and research notes.
- Operations: restore seed-bank frames, generate fixed-recipe or Sobol replay
  videos, score with TRIBE, run visual gates, aggregate by seed/content pocket,
  compare against hard negative controls, and summarize accepted/rejected
  claims.
- Gates/verifiers: successful generation, no visual-gate failure, positive
  mean TRIBE delta for candidate pockets, negative control separation, enough
  stochastic replication to avoid one-off seed luck, and human/BMD validation
  before final memorability claims.
- Store: committed manifests and result notes live under
  `research_program/neurips_memorability_selector/`; raw videos and generated
  score reports live in the local `data/` lake and are intentionally not all
  committed.
- Known limitation: the current pocket finding is a compute-proxy content
  finding, not a proven human memorability mechanism.

## Active Objective

Turn the content-pocket result from "interesting compute proxy residual" into a
stable candidate-selection regime. The next several steps should either
consolidate the positive pockets, explain them with interpretable descriptors,
or show that the signal is too model-internal to trust without immediate human
validation.

## Active Queue

| Priority | Task | Why | Gate | Next artifact | Status |
|---|---|---|---|---|---|
| P0 | Feature/embedding audit of positive vs negative content pockets | Tests whether the accepted pocket residual has an interpretable visual or embedding basis | Positives separate from hard negatives in at least one descriptor family, or we explicitly mark the pocket as black-box TRIBE content specificity | Content-pocket feature audit manifest and result note | Next |
| P1 | Stochastic replication around orange flowers and hanging clothes | Consolidates the best non-jellyfish positives before spending human-validation budget | Positive mean persists across new stochastic seeds, visual gates pass, and hard negatives remain negative under matched recipes | Expanded pocket replication manifest and result note | Queued |
| P1 | Blue jellyfish and old car boundary audit | Checks whether weaker positives are stable enough to keep or should be demoted | Stable positive mean with acceptable variance, or demote to exploratory/supporting only | Boundary audit note | Queued |
| P2 | Prompt-conditioned generator transition | Moves from metadata-only prompt text to a generator where prompt operations can actually change content | Prompt interventions change generated video content while preserving visual validity and improving candidate selection | Prompt-conditioned generator manifest | Queued |
| P2 | V-JEPA-augmented candidate screen | Reintroduces the broader selector stack only after content pockets are stabilized | V-JEPA adjudication improves or de-risks candidate ranking against TRIBE-only selection | Selector-stack comparison note | Queued |
| Human-owned | Human pilot or delayed-recognition study | Needed before final human memorability claims | Participants prefer selected candidates or delayed recognition improves under pre-registered analysis | Prolific/IRB packet and analysis report | Parked until compute screen |
| Async | Memento10k, VideoMem, or measured-BMD transfer checks | Tests cross-dataset and measured-brain grounding | External dataset correlation or measured-fMRI direction alignment clears pre-registered threshold | Dataset transfer report | Parked |

## Immediate Next Experiment

Question: are the stable positive pockets explainable by interpretable visual or
embedding features, or are they only TRIBE score islands?

Action class: search inside the current compute-proxy regime. It becomes a
small discovery move only if it creates a new accepted descriptor, verifier, or
artifact class that survives the hard negative controls.

Positive targets: orange flowers, hanging clothes, blue jellyfish, old car.

Negative controls: aerial beach, city street, storm beach.

Suggested descriptors:

- CLIP or image/video embedding distance from positive and negative seed
  centroids.
- Basic visual features: color histograms, edge density, optical-flow or frame
  change magnitude if available, object/text labels if already accessible.
- Existing TRIBE score, recipe id, Sobol index, and visual-gate status as
  covariates.

Acceptance gate:

- At least one descriptor family separates accepted positives from hard
  negatives without using the memorability score directly; or
- the result note explicitly rejects descriptor-level explanation and narrows
  C-017 to a black-box compute-proxy pocket finding.

Required outputs:

- a manifest naming inputs, descriptors, positives, controls, and withheld
  cases;
- a result note with accepted and rejected explanations;
- claim-ledger update only if the result changes C-017's scope.

## Stop Rules

- Do not call proxy-only improvements human memorability gains.
- Do not broaden alpha/guidance-only search until content identity has been
  controlled or deliberately accepted as the main mechanism.
- Do not run prompt-rewrite tournaments in the current SVD runner as if prompt
  text changes the generated pixels.
- Do not delete failed candidates. Rejected alternatives explain what the
  accepted pocket is not.
- If a positive pocket fails replication, demote it instead of averaging it into
  a broader success story.

## Reference: External Human And Dataset Tasks From Earlier Plan

The sections below preserve the older click-by-click launch material. They are
still useful, but they are not the active Codex queue until the compute screen
above produces a candidate set worth spending human or dataset-access effort on.

---

## 1. Prolific human study

### What it tests

17 video pairs across 3 sub-studies:

- **A. Best-of-N validation** (11 pairs). For each seed/prompt, the TRIBE-picked
  best-of-N winner is shown next to a median-projection variant. Tests whether
  TRIBE's projection-ranked winners are actually more memorable to humans.
  Pooled binomial test vs 50% chance. **The main result.**
- **B. α-steering validation** (1 pair). The single seed where α-steering gave
  ρ=+1.0 — α=+10 clip vs α=−10 clip. Descriptive only (n=1 pair).
- **C. Persona-winner validation** (5 pairs). For each seed, the two most-
  disagreeing personas' winners. Audience-decomposition test.

Plus 2 attention-check pairs to filter inattentive raters.

### Files in this repo

- `data/reports/prolific_survey.html` — the survey page (standalone)
- `data/reports/prolific_stimuli.json` — the 17 stimulus pairs (already inlined
  in the survey HTML; this is just for reference)
- `scripts/analyze_prolific.py` — analysis script
- `data/generated/svd_best_of_n/*.mp4` + `data/generated/veo_best_of_n/*.mp4` —
  the video stimuli (already exist)

### Steps

1. **Host the survey + videos.** Cheapest: a free Cloudflare Pages or Netlify
   project. Drag-drop the `data/reports/prolific_survey.html` AND the
   `data/generated/svd_best_of_n/` + `data/generated/veo_best_of_n/` folders
   in a structure where the survey is at `/reports/prolific_survey.html` and
   the videos are at `/generated/svd_best_of_n/...` and
   `/generated/veo_best_of_n/...` (so the relative `../generated/` paths in
   the survey HTML work).

   Alternative: just upload the `audience_vectors_share.zip` to Netlify Drop
   and link to `reports/prolific_survey.html` inside it.

2. **Set up a Google Form to collect responses.**
   - Create a Form with one question: "Response data" → Paragraph (long text).
   - Click Send → `< >` (Pre-fill) → put any text in the field → Get Link.
   - From the URL, grab the form ID (after `/d/e/`) and the entry ID
     (`entry.NNNNNNNNN`).
   - Edit `prolific_survey.html` lines 121–125:
     - Replace `REPLACE_ME_PROLIFIC_CODE` with the completion code you make in
       Prolific (step 3).
     - Set `RESPONSE_URL` to `https://docs.google.com/forms/d/e/<FORM_ID>/formResponse`
     - Set `RESPONSE_FIELD` to your `entry.NNNN` ID.
   - Re-upload the edited HTML.

3. **Create the Prolific study.**
   - https://app.prolific.com → New Study.
   - Title: "Quick video memory pairs (10 min)"
   - Description: copy the block below.
   - External study link: URL where you hosted the survey.
   - Completion URL: `https://app.prolific.com/submissions/complete?cc=YOURCODE`
     (Prolific gives you the code). Put the SAME code in
     `PROLIFIC_COMPLETION_CODE` in the survey HTML.
   - Estimated completion time: 10 min.
   - Reward per participant: **$2.00** (which is $12/hour, above Prolific's
     fair-pay floor).
   - Total participants: **30** (gives ~80% power to detect a 65% vs 50%
     binomial effect on the pooled A study).
   - Screeners: "Fluent in English" + "Approval rate ≥ 95%".
   - Total cost: 30 × $2 + Prolific fee (~33%) ≈ **$80**.

4. **Launch.** Results typically come back in 6–24 hours.

5. **Collect responses.** Download the Google Form responses as JSON or CSV,
   put each rater's JSON blob in `data/raw/prolific_responses/`, then run:

   ```bash
   .venv/bin/python scripts/analyze_prolific.py
   ```

   It will print pooled and per-pair binomial p-values and write
   `data/reports/prolific_analysis.json`.

### Prolific study description (copy verbatim)

> **Quick video memory pairs (10 min)**
>
> In this study you'll watch about 20 short AI-generated video clips, side
> by side in pairs. For each pair, pick the clip you think is *more
> memorable* — the one you'd more likely remember if asked tomorrow.
>
> There are no right or wrong answers — we want your honest first impression.
>
> You will receive a completion code at the end to paste into Prolific. The
> study takes ~10 minutes and pays $2.00.
>
> Two of the pairs are attention checks (you'll be told to pick a specific
> side). Please read carefully.

### Power note

- Pooled study A: 11 pairs × 30 raters = 330 trials. Detecting a 60% / 40%
  effect at α=0.05 needs ~190 trials → we are well-powered for a real effect.
- Study C: 5 pairs × 30 raters = 150 trials, ~30 per pair. Underpowered for
  per-pair test; only pooled "persona A vs persona B" agreement makes sense.
- Study B: 1 pair × 30 raters = 30 trials. Power for 60% vs 50%: ~25%.
  Descriptive only.

---

## 2. Memento10k + VideoMem access requests

Both are request-only. You submit a form, they email you a download link
within 1–2 weeks.

### Memento10k

- **URL:** http://memento.csail.mit.edu/ → scroll to "Memento10k Dataset"
- **Action:** the page says "Download the code for the Memento project here"
  but the dataset itself requires emailing the authors. The standard
  contact is **memento@csail.mit.edu** (per the project's published papers).
- **Email draft:**

```
Subject: Memento10k dataset access request — academic research on
brain-aligned video memorability

Hi Memento team,

I'm conducting research on interpretable directions for video memorability
derived from brain-aligned video models (TRIBE v2). On the BOLD Moments
Dataset, we find a single contrastive direction in TRIBE-predicted cortical
space that predicts human memorability at Spearman ρ = +0.48 on the
canonical test split.

To test cross-dataset transfer of this direction, I'd like to request access
to Memento10k. Specifically I'd run TRIBE on the Memento10k clips, project
onto the BMD-derived memorability direction, and compare the projection's
correlation with Memento10k human memorability scores. This is a within-paper
control, not a redistribution.

Affiliation: General Intelligence Company (independent research)
Email: jawaun@generalintelligencecompany.com
Use: academic / research only. I'll cite Newman et al. 2020 and follow any
distribution restrictions.

Thanks,
Jawaun
```

### VideoMem (the MediaEval predicting-media-memorability dataset)

- **URL:** https://www.di.unito.it/~constantin/videomem/ (or the MediaEval
  page for the year the task ran)
- **Action:** Look for the year's task page (e.g. "Predicting Media
  Memorability 2020/2021"). Each year's data has a separate signup.
- **Email draft (same template, swap dataset names):**

```
Subject: VideoMem / Predicting Media Memorability dataset access — academic
research on brain-aligned video memorability

Hi,

I'm requesting access to the VideoMem dataset for a within-paper cross-dataset
transfer control. The setup: a single contrastive direction trained on TRIBE
predictions over BOLD Moments memorability labels (ρ = +0.48 canonical test
split) and re-evaluated against VideoMem human memorability scores. No
redistribution; academic citation only.

Affiliation: General Intelligence Company (independent research)
Email: jawaun@generalintelligencecompany.com

Thanks!
Jawaun
```

### What to do with the data once it arrives

For each dataset:
1. Download videos.
2. Run TRIBE on them (modal predictor in `src/audience_vectors/modal_app/`).
3. Project features onto `v_mem` (computed from BMD; saved in
   `data/reports/final_analyses.json`).
4. Compute Spearman ρ vs the dataset's memorability scores.
5. The expected outcome: if the direction generalizes, you get ρ > 0.2;
   if it doesn't (consistent with §5.3.7's indoor↔outdoor result), it
   stays around +0.1.

---

## Decision tree after results come back

| outcome | what it means | next step |
|---|---|---|
| Prolific A pooled p < 0.05 AND humans favor TRIBE winners > 60% | brain projection corresponds to real human memorability; §6.7 best-of-N claim is validated | **DPO on open T2V** — this is the steering pipeline that's worth building |
| Prolific A pooled p > 0.05 OR humans at 50/50 | brain projection is metric-internal; the +2.07 lift doesn't map to humans | reframe paper as "TRIBE-internal ranking" only; cut the production-recipe framing |
| Prolific C: persona pair-by-pair varies > chance | audience decomposition is real | persona-specific TRIBE-judged best-of-N becomes a defensible product story |
| Prolific C at chance | persona decomposition is model-defined only | drop or downgrade the audience-arena claim |
| Memento10k transfer ρ > 0.25 | direction generalizes across datasets | strong publication-ready claim |
| Memento10k transfer ρ < 0.15 | direction is BMD-content-specific | reframe scope to "TRIBE+BMD memorability axis"; honest |

Run both in parallel — Memento10k correspondence is async and shouldn't block
the Prolific batch.

---

## 3. BMD fMRI joint analysis (closes the "predicted-not-measured" gap)

### What it tests

The single highest-leverage compute-only test left: does **measured human brain
activity** (real fMRI from BOLD Moments) give the same memorability direction
that TRIBE's *predicted* activations do?

Specifically:
- Train v_mem_measured the same contrastive way, but on per-clip beta estimates
  from real fMRI (BMD authors published them on OpenNeuro: ds005165, in
  fsaverage space, ~4 GB per hemisphere per subject).
- Compute cos(v_mem_measured, v_mem_TRIBE). If > 0.5 in a 20,484-dim space,
  TRIBE's predicted direction reflects real neural structure.
- Also: does v_mem_measured predict BMD memorability at ρ ≈ +0.40? If yes,
  the compactness claim is grounded in measured brain data, not just in TRIBE.

If both tests pass, the §8 limitation "Predicted activations, not measured fMRI"
is closed and the framework's claims gain a major external grounding. If they
fail, we honestly disclose that TRIBE's direction is a property of the
brain-aligned model, not necessarily a property of real brains.

### Files

- `scripts/bmd_fmri_pilot.py` — full pipeline for one subject. Downloads,
  resamples fsaverage→fsaverage5 (TRIBE's space), computes v_mem_measured,
  reports cos and CV ρ.

### Requirements

- **Disk:** ~8 GB peak per subject during processing (script auto-cleans
  after — use `--keep-files` to retain). 10 subjects sequentially keeps peak
  at 8 GB. (Your disk was 21 GB free when I built this — should be fine.)
- **Bandwidth:** ~8 GB download per subject. ~10 min on a typical connection.
- **Compute:** all CPU. ~2 min per subject for the actual analysis after
  download.
- **Python deps:** `nilearn`, `nibabel` (both already in the venv).

### Steps

1. **Run sub-01 first as the pilot:**
   ```bash
   .venv/bin/python scripts/bmd_fmri_pilot.py --subject 01
   ```
   This will:
   - Download `sub-01_organized_betas_task-train_hemi-{left,right}_normalized.pkl`
     from OpenNeuro (~8 GB)
   - Resample fsaverage (163,842 verts/hemi) → fsaverage5 (10,242 verts/hemi)
     to match TRIBE's space (20,484 total)
   - Compute v_mem_measured the standard contrastive way
   - Compute cos(v_mem_measured, v_mem_TRIBE) — the headline number
   - 5-fold CV ρ on measured brain
   - Save `data/reports/fmri_pilot_sub01.json` + `_vmem.npz`
   - Auto-delete the raw 8 GB pkls

2. **If sub-01 looks reasonable (cos > 0.2, CV ρ > 0.2), expand to all 10:**
   ```bash
   for s in 02 03 04 05 06 07 08 09 10; do
     .venv/bin/python scripts/bmd_fmri_pilot.py --subject $s
   done
   ```
   Sequential, ~30 min total.

3. **Aggregate across subjects:** mean of 10 per-subject v_mems, compute
   final cos vs TRIBE and CV ρ.

### Possible outcomes

| cos(v_mem_measured, v_mem_TRIBE) | CV ρ measured | meaning |
|---|---|---|
| > 0.5 | > 0.30 | TRIBE's direction is brain-grounded. §8 limitation closed. Major win. |
| 0.2–0.5 | 0.15–0.30 | partial overlap; TRIBE adds noise but captures real signal |
| < 0.2 | < 0.15 | TRIBE's v_mem is a TRIBE-model-internal direction, not aligned with real brain. Honest disclosure required. |
| any | any but with negative correlation | unexpected; check resampling step, clip-ID alignment |

### Caveats

- **Resampling assumption:** the script assumes the standard FreeSurfer ico
  convention that fsaverage5 vertices = first 10,242 vertices of fsaverage.
  It verifies this against nilearn's `fetch_surf_fsaverage` on launch. If
  that check fails, it falls back to nearest-neighbor sphere mapping (slower
  but bulletproof).
- **Clip-ID alignment:** BMD organized_betas pickles use the BMD authors'
  clip naming (likely `vid_idx0001` style or numeric index). The script
  tries multiple patterns; if it aligns < 100 clips, it prints sample IDs
  so you can adjust.
- **Per-subject noise:** single-subject fMRI is noisy. 10-subject aggregation
  is what closes the limitation.
