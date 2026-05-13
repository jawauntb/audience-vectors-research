"""Approach A: caption-mediated memorability steering.

Mine the Gemini segment descriptions to find what distinguishes high-mem from
low-mem clips at the *language* level. The output is a 'memorability prompt
rule' — a set of mutators a user could apply to a base T2V prompt to bias
generation toward memorable content.

This is the cheap, language-only version of brain-direction-conditioned video
generation: the brain (TRIBE → BMD memorability) selects which captions to
imitate, then we extract their textual signature.

Steps:
  1. Load BMD memorability ground truth per clip.
  2. Load Gemini segment descriptions.
  3. Split top 30% vs bottom 30%.
  4. tfidf n-gram analysis (1,2,3-grams) — which phrases are over-represented
     in memorable captions?
  5. Compute per-clip embedding via sentence-transformers; train a logistic
     probe to predict memorability from caption embedding. Report ROC AUC.
  6. Derive a contrastive 'memorability caption direction' in embedding space.
  7. Save the rule: top-K phrases to inject + the direction vector for later
     CLIP-conditioning use in Approach B.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def _bmd() -> dict[str, float]:
    ann = json.loads(Path("data/raw/bold_moments/annotations.json").read_text())
    return {
        f"bmd_vid_idx{eid}": float(e["memorability_score"])
        for eid, e in ann.items() if "memorability_score" in e
    }


def main() -> None:
    print("[A] loading captions + ground truth")
    g = pl.read_parquet("data/labels/synthetic_gemini.parquet")
    # take one description per segment
    g = g.unique(subset=["segment_id"]).filter(pl.col("reason").is_not_null())
    bmd = _bmd()
    rows = []
    for r in g.iter_rows(named=True):
        sid = r["segment_id"]
        vid = sid.split("_seg_")[0]
        if vid in bmd:
            rows.append({"sid": sid, "vid": vid, "caption": r["reason"], "mem": bmd[vid]})
    print(f"[A] {len(rows)} clips with caption + BMD")

    # top/bottom 30%
    rows.sort(key=lambda r: r["mem"])
    n_each = int(len(rows) * 0.30)
    low = rows[:n_each]
    high = rows[-n_each:]
    print(f"[A] split top/bottom 30% → low={len(low)} (mem ≤ {low[-1]['mem']:.3f}), "
          f"high={len(high)} (mem ≥ {high[0]['mem']:.3f})")

    # ===== tfidf n-gram analysis =====
    captions = [r["caption"] for r in rows]
    labels = np.array([1 if r["mem"] >= high[0]["mem"] else 0 if r["mem"] <= low[-1]["mem"] else -1
                       for r in rows])
    mask = labels >= 0
    cap_sub = [captions[i] for i, m in enumerate(mask) if m]
    lab_sub = labels[mask]

    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_df=0.7, stop_words="english")
    X = vec.fit_transform(cap_sub)
    terms = np.asarray(vec.get_feature_names_out())

    # per-term: ratio of mean-in-high vs mean-in-low
    high_mask = lab_sub == 1
    low_mask = lab_sub == 0
    mean_high = np.asarray(X[high_mask].mean(axis=0)).ravel()
    mean_low = np.asarray(X[low_mask].mean(axis=0)).ravel()
    eps = 1e-6
    log_ratio = np.log((mean_high + eps) / (mean_low + eps))
    # only consider terms with non-trivial frequency
    freq_mask = (mean_high + mean_low) > 0.002
    candidate_idx = np.where(freq_mask)[0]
    candidate_terms = terms[candidate_idx]
    candidate_lr = log_ratio[candidate_idx]
    order = np.argsort(-candidate_lr)
    top_memorable_phrases = [
        {"term": candidate_terms[i], "log_ratio": float(candidate_lr[i])}
        for i in order[:25]
    ]
    bottom = np.argsort(candidate_lr)
    top_anti_phrases = [
        {"term": candidate_terms[i], "log_ratio": float(candidate_lr[i])}
        for i in bottom[:25]
    ]

    # ===== logistic probe on tfidf =====
    print("[A] logistic probe on tfidf (5-fold CV)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_i, test_i in kf.split(cap_sub):
        # build a smaller vectorizer per fold to avoid leakage
        vec_f = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_df=0.7, stop_words="english")
        Xt = vec_f.fit_transform([cap_sub[i] for i in train_i])
        Xv = vec_f.transform([cap_sub[i] for i in test_i])
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xt, lab_sub[train_i])
        probs = clf.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(lab_sub[test_i], probs))
    auc_mean, auc_std = float(np.mean(aucs)), float(np.std(aucs))
    print(f"[A] tfidf logistic probe: AUC = {auc_mean:.3f} ± {auc_std:.3f}")

    # ===== embedding-based direction =====
    print("[A] caption embeddings (sentence-transformers / all-MiniLM-L6-v2)")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        st_path = os.environ.get("ST_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(st_path)
        embs = model.encode(captions, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[A] sentence-transformers unavailable ({exc!r}); falling back to tfidf-svd")
        from sklearn.decomposition import TruncatedSVD
        Xfull = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.8).fit_transform(captions)
        svd = TruncatedSVD(n_components=128, random_state=42)
        embs = svd.fit_transform(Xfull)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    mems = np.asarray([r["mem"] for r in rows], dtype=np.float32)
    order_mem = np.argsort(mems)
    n_e = int(len(rows) * 0.30)
    low_e = embs[order_mem[:n_e]].mean(axis=0)
    high_e = embs[order_mem[-n_e:]].mean(axis=0)
    cap_direction = high_e - low_e
    cap_direction = cap_direction / (np.linalg.norm(cap_direction) + 1e-12)

    # CV on this direction
    print("[A] caption-direction probe (5-fold CV)")
    proj_aucs = []
    for train_i, test_i in kf.split(rows):
        train_embs = embs[train_i]
        train_mems = mems[train_i]
        o = np.argsort(train_mems)
        ne = int(len(train_i) * 0.30)
        v = train_embs[o[-ne:]].mean(axis=0) - train_embs[o[:ne]].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        test_proj = embs[test_i] @ v
        test_mems = mems[test_i]
        med = np.median(test_mems)
        bin_lab = (test_mems > med).astype(int)
        proj_aucs.append(roc_auc_score(bin_lab, test_proj))
    proj_auc_mean = float(np.mean(proj_aucs))
    proj_auc_std = float(np.std(proj_aucs))
    print(f"[A] caption-direction AUC = {proj_auc_mean:.3f} ± {proj_auc_std:.3f}")

    out = {
        "n_clips": len(rows),
        "split_threshold_low": low[-1]["mem"],
        "split_threshold_high": high[0]["mem"],
        "tfidf_probe_auc_mean": auc_mean,
        "tfidf_probe_auc_std": auc_std,
        "caption_direction_auc_mean": proj_auc_mean,
        "caption_direction_auc_std": proj_auc_std,
        "top_memorable_phrases": top_memorable_phrases,
        "top_anti_memorable_phrases": top_anti_phrases,
        "embedding_dim": int(embs.shape[1]),
    }
    Path("data/reports/approach_a.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(
        "data/reports/approach_a_direction.npz",
        direction=cap_direction.astype(np.float32),
        embeddings=embs.astype(np.float32),
        sample_ids=np.asarray([r["sid"] for r in rows]),
        mems=mems.astype(np.float32),
    )
    print("[done] wrote data/reports/approach_a.json + approach_a_direction.npz")


if __name__ == "__main__":
    main()
