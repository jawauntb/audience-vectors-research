"""K-fold cross-validation + bootstrap CIs for contrastive vector training.

This is the stats-rigor layer that turns single-point Spearman numbers
into mean ± stdev across folds, with bootstrap CIs on every estimate.
The methodology:

  K-fold protocol
  ---------------
  For K=5 folds:
    - Hold out one fold as test
    - Train contrastive direction on the remaining 4 folds:
        * top top_k_frac of the train portion = positive set
        * bottom top_k_frac of the train portion = negative set
    - Project all test-fold segments onto the direction
    - Spearman(test_gt, test_projections) = fold metric
  Report mean and stdev across the K folds.

  Bootstrap CI
  ------------
  For each fold's Spearman, resample test-set pairs with replacement
  B times and recompute Spearman each time. The 2.5/97.5 percentiles
  are the 95% CI.

This is the difference between "we got ρ=+0.23" and "we got ρ=+0.23
[95% CI +0.11, +0.35] across 5 folds (n=180 per fold)."
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    rx = sorted(range(n), key=lambda i: xs[i])
    ry = sorted(range(n), key=lambda i: ys[i])
    rank_x = [0] * n
    rank_y = [0] * n
    for r, i in enumerate(rx):
        rank_x[i] = r
    for r, i in enumerate(ry):
        rank_y[i] = r
    mx = sum(rank_x) / n
    my = sum(rank_y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rank_x, rank_y))
    dx = (sum((a - mx) ** 2 for a in rank_x)) ** 0.5
    dy = (sum((b - my) ** 2 for b in rank_y)) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def _load_features_vector(features_dir: Path, sample_id: str) -> np.ndarray | None:
    p = features_dir / f"{sample_id}.npz"
    if not p.exists():
        return None
    feat = np.load(p, allow_pickle=False)
    if "frames" in feat.files:
        arr = np.asarray(feat["frames"], dtype=np.float32)
        return arr.mean(axis=0) if arr.ndim == 2 else arr
    if "embedding" in feat.files:
        return np.asarray(feat["embedding"], dtype=np.float32)
    return None


@dataclass
class FoldResult:
    fold_idx: int
    n_train: int
    n_test: int
    n_pos: int
    n_neg: int
    spearman: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float


@dataclass
class CrossValSummary:
    mean_spearman: float
    stdev_spearman: float
    median_spearman: float
    overall_ci_low: float
    overall_ci_high: float
    folds: list[FoldResult]


def _bootstrap_ci(
    xs: list[float],
    ys: list[float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng: random.Random,
) -> tuple[float, float]:
    if len(xs) < 3:
        return (0.0, 0.0)
    paired = list(zip(xs, ys))
    estimates: list[float] = []
    n = len(paired)
    for _ in range(n_resamples):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        sx = [p[0] for p in sample]
        sy = [p[1] for p in sample]
        estimates.append(_spearman(sx, sy))
    estimates.sort()
    lo = estimates[int(n_resamples * (alpha / 2))]
    hi = estimates[int(n_resamples * (1 - alpha / 2))]
    return lo, hi


def cross_validate_contrastive(
    *,
    scored_segments: list[tuple[str, float]],
    features_dir: Path,
    k: int = 5,
    top_k_frac: float = 0.30,
    min_set_size: int = 3,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
) -> CrossValSummary:
    """K-fold CV of contrastive-direction training.

    For each fold, train on the (K-1)/K segments by taking top/bottom
    `top_k_frac` of the training set, mean their features, take the
    difference. Then project test-fold features onto this direction
    and compute Spearman vs ground truth.
    """
    # Filter to segments with features on disk.
    available: list[tuple[str, float, np.ndarray]] = []
    for sample_id, score in scored_segments:
        vec = _load_features_vector(features_dir, sample_id)
        if vec is not None:
            available.append((sample_id, score, vec))
    if len(available) < k * 2 * min_set_size:
        raise ValueError(
            f"need at least {k * 2 * min_set_size} segments with features, "
            f"got {len(available)}"
        )

    rng = random.Random(seed)
    shuffled = available[:]
    rng.shuffle(shuffled)

    fold_size = len(shuffled) // k
    folds: list[FoldResult] = []

    for fold_idx in range(k):
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size if fold_idx < k - 1 else len(shuffled)
        test = shuffled[test_start:test_end]
        train = shuffled[:test_start] + shuffled[test_end:]

        train_sorted = sorted(train, key=lambda t: -t[1])
        n_tail = max(min_set_size, int(len(train_sorted) * top_k_frac))
        positives = train_sorted[:n_tail]
        negatives = train_sorted[-n_tail:]

        pos_mean = np.mean([t[2] for t in positives], axis=0)
        neg_mean = np.mean([t[2] for t in negatives], axis=0)
        direction = pos_mean - neg_mean
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            direction = direction / norm

        test_gt = [t[1] for t in test]
        test_proj = [float(np.dot(t[2], direction)) for t in test]
        rho = _spearman(test_gt, test_proj)
        lo, hi = _bootstrap_ci(test_gt, test_proj, n_resamples=bootstrap_resamples, rng=rng)

        folds.append(FoldResult(
            fold_idx=fold_idx,
            n_train=len(train),
            n_test=len(test),
            n_pos=len(positives),
            n_neg=len(negatives),
            spearman=rho,
            bootstrap_ci_low=lo,
            bootstrap_ci_high=hi,
        ))
        logger.info(
            "fold %d/%d  n_test=%d  ρ=%+.3f  [%+.3f, %+.3f]",
            fold_idx + 1, k, len(test), rho, lo, hi,
        )

    rhos = [f.spearman for f in folds]
    rhos_sorted = sorted(rhos)
    mean_r = sum(rhos) / len(rhos)
    var = sum((r - mean_r) ** 2 for r in rhos) / max(1, len(rhos) - 1)
    stdev_r = var ** 0.5
    median_r = rhos_sorted[len(rhos_sorted) // 2]

    # 95% interval over the K fold estimates themselves (small-K caveat).
    fold_lo = rhos_sorted[0] if k <= 5 else rhos_sorted[int(0.025 * k)]
    fold_hi = rhos_sorted[-1] if k <= 5 else rhos_sorted[int(0.975 * k)]

    return CrossValSummary(
        mean_spearman=mean_r,
        stdev_spearman=stdev_r,
        median_spearman=median_r,
        overall_ci_low=fold_lo,
        overall_ci_high=fold_hi,
        folds=folds,
    )


def random_baseline_spearman(
    *,
    scored_segments: list[tuple[str, float]],
    features_dir: Path,
    n_trials: int = 200,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Project features onto N random unit vectors. Return mean/stdev/95%-upper
    of Spearman vs ground truth — this is the null-hypothesis distribution
    a real contrastive direction has to beat to be meaningful."""
    available: list[tuple[str, float, np.ndarray]] = []
    for sample_id, score in scored_segments:
        vec = _load_features_vector(features_dir, sample_id)
        if vec is not None:
            available.append((sample_id, score, vec))
    if len(available) < 5:
        raise ValueError(f"need at least 5 segments, got {len(available)}")

    rng = np.random.default_rng(seed)
    dim = available[0][2].shape[0]
    gt = [t[1] for t in available]

    rhos: list[float] = []
    for _ in range(n_trials):
        direction = rng.standard_normal(dim).astype(np.float32)
        direction = direction / np.linalg.norm(direction)
        proj = [float(np.dot(t[2], direction)) for t in available]
        rhos.append(_spearman(gt, proj))
    rhos_sorted = sorted(rhos)
    abs_rhos_sorted = sorted([abs(r) for r in rhos])
    mean_r = sum(rhos) / n_trials
    stdev_r = (sum((r - mean_r) ** 2 for r in rhos) / max(1, n_trials - 1)) ** 0.5
    abs_95 = abs_rhos_sorted[int(0.95 * n_trials)]
    return mean_r, stdev_r, abs_95


def summary_to_dict(summary: CrossValSummary) -> dict:
    return {
        "mean_spearman": summary.mean_spearman,
        "stdev_spearman": summary.stdev_spearman,
        "median_spearman": summary.median_spearman,
        "overall_ci_low": summary.overall_ci_low,
        "overall_ci_high": summary.overall_ci_high,
        "folds": [
            {
                "fold_idx": f.fold_idx,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "n_pos": f.n_pos,
                "n_neg": f.n_neg,
                "spearman": f.spearman,
                "bootstrap_ci_low": f.bootstrap_ci_low,
                "bootstrap_ci_high": f.bootstrap_ci_high,
            }
            for f in summary.folds
        ],
    }
