"""Train a contrastive audience vector for one target axis.

By default uses BMD's human `memorability_score` as the target. Splits
top vs bottom `top_k_frac` of segments by score, means their features,
takes the difference, optionally normalizes. Saves the direction + metadata
under `data/models/vectors/`.

Also reports held-out Spearman correlation between vector projections
and ground-truth scores on the middle band of segments — that's the real
indicator of whether the contrastive direction generalizes vs. just
memorizes the training set.

Usage:
    uv run python scripts/train_audience_vectors.py \\
        --features-dir data/features/vjepa \\
        --model-id facebook/vjepa2-vitl-fpc64-256 \\
        --layer vjepa_mean_pool \\
        --target bmd_memorability
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from audience_vectors.activations import ContrastiveVectorTrainer, project_features
from audience_vectors.config import get_config


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


def _load_bmd_memorability_scores() -> dict[str, float]:
    bmd_path = Path("./data/raw/bold_moments/annotations.json")
    if not bmd_path.exists():
        return {}
    with bmd_path.open() as fh:
        ann = json.load(fh)
    return {
        f"bmd_vid_idx{eid}": float(e["memorability_score"])
        for eid, e in ann.items()
        if "memorability_score" in e
    }


def _load_gemini_scores(axis: str) -> dict[str, float]:
    """Load synthetic Gemini scores for `axis` keyed by segment_id."""
    import polars as pl  # noqa: PLC0415

    path = Path("./data/labels/synthetic_gemini.parquet")
    if not path.exists():
        return {}
    df = pl.read_parquet(path)
    return {
        r["segment_id"]: float(r["scores"][axis])
        for r in df.iter_rows(named=True)
        if isinstance(r["scores"], dict) and axis in r["scores"]
    }


def _load_segments(segments_path: Path) -> list[dict]:
    import polars as pl  # noqa: PLC0415

    return pl.read_parquet(segments_path).to_dicts()


def _project_segment(features_path: Path, direction: np.ndarray) -> float | None:
    if not features_path.exists():
        return None
    payload = np.load(features_path, allow_pickle=False)
    if "frames" in payload.files:
        arr = np.asarray(payload["frames"], dtype=np.float32)
        return project_features(arr, direction)
    if "embedding" in payload.files:
        return project_features(np.asarray(payload["embedding"], dtype=np.float32), direction)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, required=True,
                        help="e.g. 'facebook/vjepa2-vitl-fpc64-256' or 'facebook/tribev2'.")
    parser.add_argument("--layer", type=str, required=True,
                        help="Short tag for the feature space (e.g. 'vjepa_mean_pool').")
    parser.add_argument("--target", type=str, default="bmd_memorability",
                        help="Target axis name (used in the vector_id + report).")
    parser.add_argument(
        "--score-source",
        choices=["bmd", "gemini"],
        default="bmd",
        help="Where to read ground truth from: BMD human memorability (default) or Gemini synthetic.",
    )
    parser.add_argument(
        "--gemini-axis",
        default="memorability",
        help="When --score-source=gemini, which axis to use as ground truth.",
    )
    parser.add_argument("--top-k-frac", type=float, default=0.30)
    parser.add_argument("--min-set-size", type=int, default=3)
    parser.add_argument(
        "--segments", type=Path, default=None,
        help="segments.parquet (default: data/training/segments.parquet)",
    )
    args = parser.parse_args()

    cfg = get_config()
    cfg.paths.ensure()
    segments_path = args.segments or (cfg.paths.training / "segments.parquet")
    segs = _load_segments(segments_path)

    # Ground truth source: BMD human memorability OR Gemini synthetic labels.
    scored: list[tuple[str, float]] = []
    if args.score_source == "bmd":
        scores_map = _load_bmd_memorability_scores()
        if not scores_map:
            print("[fail] could not load BMD memorability scores")
            sys.exit(1)
        for s in segs:
            video_id = s["video_id"]
            if video_id in scores_map:
                scored.append((s["sample_id"], scores_map[video_id]))
    else:
        scores_by_seg = _load_gemini_scores(args.gemini_axis)
        if not scores_by_seg:
            print(f"[fail] no Gemini scores for axis={args.gemini_axis!r}")
            sys.exit(1)
        for s in segs:
            if s["sample_id"] in scores_by_seg:
                scored.append((s["sample_id"], scores_by_seg[s["sample_id"]]))
    if len(scored) < 2 * args.min_set_size:
        print(f"[fail] not enough scored segments ({len(scored)})")
        sys.exit(1)

    # Pick the top/bottom tails for training, leave the middle as holdout.
    scored_sorted = sorted(scored, key=lambda x: -x[1])
    n_tail = max(args.min_set_size, int(len(scored_sorted) * args.top_k_frac))
    training = scored_sorted[:n_tail] + scored_sorted[-n_tail:]
    holdout = scored_sorted[n_tail:-n_tail] if len(scored_sorted) > 2 * n_tail else []
    print(f"[plan] training_set={len(training)} holdout={len(holdout)} "
          f"top_k_frac={args.top_k_frac}")

    trainer = ContrastiveVectorTrainer(
        features_dir=args.features_dir,
        vectors_dir=cfg.paths.vectors,
    )
    vector = trainer.train(
        target=args.target,
        scored_segments=training,
        model_id=args.model_id,
        layer=args.layer,
        top_k_frac=0.5,  # training set is already split top/bottom; use both halves
        min_set_size=args.min_set_size,
        notes=f"trained on {len(training)} tail segments from BMD",
    )
    print(f"[trained] {vector.vector_id} dim={vector.dim} "
          f"pos={vector.positive_set_size} neg={vector.negative_set_size}")
    print(f"          direction: {vector.direction_uri}")

    direction = np.load(vector.direction_uri, allow_pickle=False)["direction"]

    # Sanity check: project the training tails — should be well-separated.
    train_proj: list[float] = []
    train_scores: list[float] = []
    for sample_id, gt in training:
        proj = _project_segment(args.features_dir / f"{sample_id}.npz", direction)
        if proj is not None:
            train_proj.append(proj)
            train_scores.append(gt)
    if train_proj:
        rho_train = _spearman(train_scores, train_proj)
        print(f"[train] Spearman(gt, projection) over training tails = {rho_train:+.3f}")

    # Real test: held-out middle band. If the vector generalizes, projection
    # rank should still correlate with BMD ground-truth memorability on
    # segments the trainer never saw.
    if holdout:
        ho_proj: list[float] = []
        ho_scores: list[float] = []
        for sample_id, gt in holdout:
            proj = _project_segment(args.features_dir / f"{sample_id}.npz", direction)
            if proj is not None:
                ho_proj.append(proj)
                ho_scores.append(gt)
        if ho_proj:
            rho_ho = _spearman(ho_scores, ho_proj)
            print(f"[holdout] Spearman(gt, projection) over {len(ho_proj)} middle-band "
                  f"segments = {rho_ho:+.3f}")


if __name__ == "__main__":
    main()
