"""Head-to-head: BMD memorability vs Gemini, V-JEPA contrastive, TRIBE contrastive.

Computes Spearman correlations between BMD's human memorability_score
(ground truth) and three predictors:

  1. zero-shot Gemini memorability (no training)
  2. V-JEPA contrastive vector projection
  3. TRIBE v2 contrastive vector projection

Both vectors are trained the same way — top-30%/bottom-30% split on
BMD memorability — so the comparison isolates the feature space.
Reports overall + held-out (middle band excluded from training tails).

Usage:
    uv run python scripts/eval_tribe_vs_vjepa.py
    uv run python scripts/eval_tribe_vs_vjepa.py --output data/reports/tribe_vs_vjepa.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audience_vectors.activations import project_features


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


def _load_bmd() -> dict[str, float]:
    p = Path("./data/raw/bold_moments/annotations.json")
    with p.open() as fh:
        ann = json.load(fh)
    return {
        f"bmd_vid_idx{eid}": float(e["memorability_score"])
        for eid, e in ann.items()
        if "memorability_score" in e
    }


def _load_gemini_by_seg(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    import polars as pl  # noqa: PLC0415

    df = pl.read_parquet(path)
    out: dict[str, float] = {}
    for r in df.iter_rows(named=True):
        if isinstance(r["scores"], dict) and "memorability" in r["scores"]:
            out[r["segment_id"]] = float(r["scores"]["memorability"])
    return out


@dataclass
class VectorBundle:
    name: str
    direction: np.ndarray
    features_dir: Path
    train_ids: set[str]


def _load_vector(npz_path: Path, features_dir: Path, name: str) -> VectorBundle:
    payload = np.load(npz_path, allow_pickle=False)
    direction = payload["direction"]
    meta = json.loads(npz_path.with_suffix(".json").read_text())
    train_ids = set(meta.get("positive_ids", []) + meta.get("negative_ids", []))
    return VectorBundle(name=name, direction=direction, features_dir=features_dir, train_ids=train_ids)


def _project_one(bundle: VectorBundle, sample_id: str) -> float | None:
    p = bundle.features_dir / f"{sample_id}.npz"
    if not p.exists():
        return None
    feat = np.load(p, allow_pickle=False)
    if "frames" in feat.files:
        return project_features(np.asarray(feat["frames"], dtype=np.float32), bundle.direction)
    if "embedding" in feat.files:
        return project_features(np.asarray(feat["embedding"], dtype=np.float32), bundle.direction)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vjepa-vector", type=Path, default=None,
                        help="V-JEPA contrastive .npz (default: auto-pick most recent BMD-trained).")
    parser.add_argument("--tribe-vector", type=Path, default=None,
                        help="TRIBE contrastive .npz (default: auto-pick most recent).")
    parser.add_argument("--vjepa-features", type=Path,
                        default=Path("./data/features/vjepa"))
    parser.add_argument("--tribe-features", type=Path,
                        default=Path("./data/features/tribe"))
    parser.add_argument("--gemini-labels", type=Path,
                        default=Path("./data/labels/synthetic_gemini.parquet"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    bmd = _load_bmd()
    gem = _load_gemini_by_seg(args.gemini_labels)

    vectors_dir = Path("./data/models/vectors")
    if not args.vjepa_vector:
        candidates = sorted(vectors_dir.glob("*vjepa*bmd_memorability*.npz"))
        if not candidates:
            print("[fail] no V-JEPA BMD-memorability vector under", vectors_dir)
            sys.exit(1)
        args.vjepa_vector = candidates[-1]
    if not args.tribe_vector:
        candidates = sorted(vectors_dir.glob("*tribev2*bmd_memorability*.npz"))
        if candidates:
            args.tribe_vector = candidates[-1]

    bundles: list[VectorBundle] = []
    bundles.append(_load_vector(args.vjepa_vector, args.vjepa_features, "V-JEPA"))
    if args.tribe_vector and args.tribe_vector.exists():
        bundles.append(_load_vector(args.tribe_vector, args.tribe_features, "TRIBE"))
    else:
        print("[note] no TRIBE vector yet — only V-JEPA will be reported")

    # Build aligned rows: one per sample_id that has BMD ground truth.
    sample_ids: set[str] = set()
    for b in bundles:
        sample_ids |= {p.stem for p in b.features_dir.glob("*.npz")}
    sample_ids |= set(gem)
    rows: list[dict[str, object]] = []
    for sample_id in sorted(sample_ids):
        video_id = sample_id.rsplit("_seg_", 1)[0]
        bmd_score = bmd.get(video_id)
        if bmd_score is None:
            continue
        row: dict[str, object] = {
            "sample_id": sample_id,
            "bmd": bmd_score,
            "gemini": gem.get(sample_id),
        }
        in_any_train = False
        for b in bundles:
            row[f"{b.name}_proj"] = _project_one(b, sample_id)
            if sample_id in b.train_ids:
                in_any_train = True
        row["in_train"] = in_any_train
        rows.append(row)

    def _spearman_against_bmd(key: str, holdout_only: bool) -> tuple[float, int]:
        xs, ys = [], []
        for r in rows:
            v = r.get(key)
            if v is None:
                continue
            if holdout_only and r["in_train"]:
                continue
            xs.append(r["bmd"])
            ys.append(float(v))
        return _spearman(xs, ys), len(xs)

    print(f"Total segments with BMD ground truth: {len(rows)}\n")

    print("Spearman ρ vs BMD memorability_score:\n")
    print(f"{'predictor':<28} {'all':>16} {'held-out':>16}")
    print("-" * 64)

    predictors = ["gemini"] + [f"{b.name}_proj" for b in bundles]
    table_rows: list[tuple[str, float, int, float, int]] = []
    for key in predictors:
        rho_all, n_all = _spearman_against_bmd(key, holdout_only=False)
        rho_ho, n_ho = _spearman_against_bmd(key, holdout_only=True)
        pretty = {
            "gemini": "Gemini zero-shot",
            "V-JEPA_proj": "V-JEPA contrastive",
            "TRIBE_proj": "TRIBE contrastive",
        }.get(key, key)
        print(f"{pretty:<28} {rho_all:+.3f} (n={n_all:>4}) {rho_ho:+.3f} (n={n_ho:>4})")
        table_rows.append((pretty, rho_all, n_all, rho_ho, n_ho))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Head-to-head: BMD memorability vs predictors",
            "",
            f"Total segments with BMD ground truth: **n={len(rows)}**",
            "",
            "| predictor | all ρ | n | held-out ρ | n |",
            "|---|---:|---:|---:|---:|",
        ]
        for pretty, rho_all, n_all, rho_ho, n_ho in table_rows:
            lines.append(f"| {pretty} | {rho_all:+.3f} | {n_all} | {rho_ho:+.3f} | {n_ho} |")
        lines += [
            "",
            "Held-out excludes segments in either vector's training tail set.",
            "",
            f"- V-JEPA vector: `{args.vjepa_vector.name}`",
        ]
        if args.tribe_vector:
            lines.append(f"- TRIBE vector: `{args.tribe_vector.name}`")
        args.output.write_text("\n".join(lines) + "\n")
        print(f"\n[done] wrote {args.output}")


if __name__ == "__main__":
    main()
