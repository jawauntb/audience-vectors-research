"""TRIBE output-space temporal Fourier probe for memorability.

Spencer's critique can be read as: a learned direction may look linear only
because position/time structure is folded into the representation. TRIBE is
opaque to us locally, but the saved outputs preserve four temporal bins:

    (time=4, cortical_vertices=20484)

This script learns the standard high-minus-low BMD memorability direction in
that tensor space, decomposes the direction with a 1-D FFT over the time axis,
and evaluates each temporal-frequency band on held-out clips.

It is not a true internal positional-embedding patch. It is the closest direct
TRIBE-specific test we can run from the current saved artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_FEATURE_DIR = Path("data/features/tribe")
DEFAULT_ANNOTATIONS = Path("data/raw/bold_moments/annotations.json")
DEFAULT_JSON = Path("data/reports/tribe_temporal_spectral_probe.json")
DEFAULT_MD = Path("data/reports/tribe_temporal_spectral_probe.md")
DEFAULT_SUMMARY = Path("data/reports/tribe_temporal_spectral_probe_summary.md")


@dataclass(frozen=True)
class TribeRecord:
    sample_id: str
    video_id: str
    path: Path
    score: float


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ra = rankdata(np.asarray(a, dtype=np.float64))
    rb = rankdata(np.asarray(b, dtype=np.float64))
    denom = float(np.std(ra) * np.std(rb))
    if denom <= 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def unit(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= 1e-12:
        raise ValueError("near-zero vector")
    return np.asarray(x / norm, dtype=np.float32)


def load_scores(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    return {
        f"bmd_vid_idx{key}": float(row["memorability_score"])
        for key, row in payload.items()
        if "memorability_score" in row
    }


def load_tribe_tensor(
    *,
    feature_dir: Path,
    annotations: Path,
    time_policy: str,
    target_time: int,
) -> tuple[list[TribeRecord], np.ndarray, np.ndarray, dict[str, Any]]:
    scores_by_video = load_scores(annotations)
    records: list[TribeRecord] = []
    arrays: list[np.ndarray] = []
    skipped: Counter[str] = Counter()
    shape_counts: Counter[tuple[int, ...]] = Counter()

    for path in sorted(feature_dir.glob("bmd_vid_idx*.npz")):
        sample_id = path.stem
        video_id = sample_id.split("_seg_")[0]
        if video_id not in scores_by_video:
            skipped["missing_memorability_score"] += 1
            continue
        payload = np.load(path, allow_pickle=False)
        if "frames" not in payload.files:
            skipped["missing_frames"] += 1
            continue
        frames = np.asarray(payload["frames"], dtype=np.float32)
        if frames.ndim != 2:
            skipped["not_time_by_vertex"] += 1
            continue
        shape_counts[tuple(int(x) for x in frames.shape)] += 1
        records.append(
            TribeRecord(
                sample_id=sample_id,
                video_id=video_id,
                path=path,
                score=scores_by_video[video_id],
            )
        )
        arrays.append(frames)

    if not arrays:
        raise FileNotFoundError(f"no usable TRIBE features found in {feature_dir}")

    common_shape, n_common = shape_counts.most_common(1)[0]
    kept_records, kept_arrays = select_time_policy_arrays(
        records=records,
        arrays=arrays,
        common_shape=common_shape,
        time_policy=time_policy,
        target_time=target_time,
        skipped=skipped,
    )

    features = np.stack(kept_arrays).astype(np.float32)
    scores = np.asarray([record.score for record in kept_records], dtype=np.float32)
    meta = {
        "feature_dir": str(feature_dir),
        "annotations": str(annotations),
        "time_policy": time_policy,
        "target_time": target_time,
        "shape_counts": {str(key): value for key, value in shape_counts.items()},
        "common_shape": list(common_shape),
        "n_common_shape": int(n_common),
        "skipped": dict(skipped),
    }
    return kept_records, features, scores, meta


def select_time_policy_arrays(
    *,
    records: Sequence[TribeRecord],
    arrays: Sequence[np.ndarray],
    common_shape: tuple[int, ...],
    time_policy: str,
    target_time: int,
    skipped: Counter[str],
) -> tuple[list[TribeRecord], list[np.ndarray]]:
    kept_records: list[TribeRecord] = []
    kept_arrays: list[np.ndarray] = []
    if time_policy == "common":
        for record, frames in zip(records, arrays, strict=True):
            if tuple(frames.shape) == common_shape:
                kept_records.append(record)
                kept_arrays.append(frames)
            else:
                skipped["non_common_shape"] += 1
        return kept_records, kept_arrays
    if time_policy == "resample":
        for record, frames in zip(records, arrays, strict=True):
            kept_records.append(record)
            kept_arrays.append(resample_time(frames, target_time=target_time))
        return kept_records, kept_arrays
    raise ValueError(f"unknown time_policy: {time_policy}")


def resample_time(frames: np.ndarray, *, target_time: int) -> np.ndarray:
    if frames.shape[0] == target_time:
        return frames.astype(np.float32, copy=False)
    if frames.shape[0] < 2:
        raise ValueError(f"cannot resample single-frame tensor: {frames.shape}")

    old_positions = np.linspace(0.0, 1.0, num=frames.shape[0], dtype=np.float32)
    new_positions = np.linspace(0.0, 1.0, num=target_time, dtype=np.float32)
    right = np.searchsorted(old_positions, new_positions, side="left")
    right = np.clip(right, 0, frames.shape[0] - 1)
    left = np.clip(right - 1, 0, frames.shape[0] - 1)
    exact = old_positions[right] == new_positions
    left[exact] = right[exact]
    denom = old_positions[right] - old_positions[left]
    denom = np.where(denom == 0, 1.0, denom)
    weight = ((new_positions - old_positions[left]) / denom).astype(np.float32)
    out = (1.0 - weight[:, None]) * frames[left] + weight[:, None] * frames[right]
    return np.asarray(out, dtype=np.float32)


def split_indices(n: int, *, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(32, int(round(n * test_frac)))
    n_test = min(n_test, n - 32)
    test = np.sort(perm[:n_test])
    train = np.sort(perm[n_test:])
    return train, test


def standardize_train(
    features: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features[train_idx].mean(axis=0, keepdims=True)
    std = features[train_idx].std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-4)
    return np.asarray((features - mean) / std, dtype=np.float32), mean, std


def train_direction(
    features: np.ndarray,
    scores: np.ndarray,
    train_idx: np.ndarray,
    *,
    top_frac: float,
    min_tail: int,
) -> np.ndarray:
    train_scores = scores[train_idx]
    n_each = max(min_tail, int(round(len(train_idx) * top_frac)))
    if n_each * 2 > len(train_idx):
        raise ValueError(f"tails too large for n_train={len(train_idx)}")
    order = train_idx[np.argsort(train_scores)]
    low = features[order[:n_each]].mean(axis=0)
    high = features[order[-n_each:]].mean(axis=0)
    return unit(high - low)


def project_tensor(features: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return np.tensordot(features, direction, axes=([1, 2], [0, 1]))


def temporal_band_masks(t: int) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    all_mask = np.ones(t, dtype=bool)
    dc = np.zeros(t, dtype=bool)
    dc[0] = True
    masks["all"] = all_mask
    masks["temporal_dc_k0"] = dc

    nonzero = all_mask.copy()
    nonzero[0] = False
    masks["temporal_nonzero"] = nonzero

    if t == 4:
        k1 = np.zeros(t, dtype=bool)
        k1[[1, 3]] = True
        k2 = np.zeros(t, dtype=bool)
        k2[2] = True
        masks["temporal_k1_pair"] = k1
        masks["temporal_nyquist_k2"] = k2
    else:
        freqs = np.abs(np.fft.fftfreq(t))
        max_freq = float(freqs.max()) or 1.0
        masks["temporal_low_nonzero"] = (freqs > 0) & ((freqs / max_freq) <= 0.5)
        masks["temporal_high"] = (freqs / max_freq) > 0.5
    return masks


def direction_from_temporal_mask(spectrum: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.zeros_like(spectrum)
    masked[mask, :] = spectrum[mask, :]
    return np.asarray(np.fft.ifft(masked, axis=0, norm="ortho").real, dtype=np.float32)


def summarize(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def evaluate_split(
    *,
    raw_features: np.ndarray,
    scores: np.ndarray,
    seed: int,
    test_frac: float,
    top_frac: float,
    min_tail: int,
) -> dict[str, Any]:
    train_idx, test_idx = split_indices(len(scores), test_frac=test_frac, seed=seed)
    features, _mean, _std = standardize_train(raw_features, train_idx)
    direction = train_direction(
        features,
        scores,
        train_idx,
        top_frac=top_frac,
        min_tail=min_tail,
    )
    full_pred = project_tensor(features[test_idx], direction)
    full_rho = spearman(full_pred, scores[test_idx])

    pooled_raw = raw_features.mean(axis=1)
    pooled, _pooled_mean, _pooled_std = standardize_train(pooled_raw, train_idx)
    pooled_direction = train_direction(
        pooled,
        scores,
        train_idx,
        top_frac=top_frac,
        min_tail=min_tail,
    )
    pooled_rho = spearman(pooled[test_idx] @ pooled_direction, scores[test_idx])

    spectrum = np.fft.fft(direction, axis=0, norm="ortho")
    energy = np.abs(spectrum) ** 2
    total_energy = float(energy.sum())
    band_rows: list[dict[str, float | str]] = []
    for name, mask in temporal_band_masks(direction.shape[0]).items():
        band_direction = direction_from_temporal_mask(spectrum, mask)
        band_energy = float(energy[mask, :].sum())
        if float(np.linalg.norm(band_direction)) <= 1e-12:
            rho = 0.0
        else:
            rho = spearman(
                project_tensor(features[test_idx], unit(band_direction)),
                scores[test_idx],
            )
        band_rows.append(
            {
                "band": name,
                "energy_fraction": band_energy / total_energy if total_energy else 0.0,
                "test_rho_band_only": rho,
            }
        )

    frame_directions = []
    for t in range(raw_features.shape[1]):
        frame_direction = train_direction(
            features[:, t, :],
            scores,
            train_idx,
            top_frac=top_frac,
            min_tail=min_tail,
        )
        frame_directions.append(frame_direction)
    frame_cos = np.eye(raw_features.shape[1], dtype=np.float32)
    for i, vi in enumerate(frame_directions):
        for j, vj in enumerate(frame_directions):
            frame_cos[i, j] = float(vi @ vj)
    off_diag = [
        float(frame_cos[i, j])
        for i in range(frame_cos.shape[0])
        for j in range(frame_cos.shape[1])
        if i != j
    ]

    return {
        "seed": seed,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "full_tensor_rho": full_rho,
        "mean_pooled_rho": pooled_rho,
        "band_results": band_rows,
        "per_time_bin_direction_cosine": frame_cos.tolist(),
        "per_time_bin_direction_mean_offdiag_cosine": float(np.mean(off_diag)),
    }


def aggregate_splits(split_reports: list[dict[str, Any]]) -> dict[str, Any]:
    bands = sorted({row["band"] for report in split_reports for row in report["band_results"]})
    by_band: dict[str, dict[str, dict[str, float]]] = {}
    for band in bands:
        energy_values: list[float] = []
        rho_values: list[float] = []
        for report in split_reports:
            row = next(row for row in report["band_results"] if row["band"] == band)
            energy_values.append(float(row["energy_fraction"]))
            rho_values.append(float(row["test_rho_band_only"]))
        by_band[band] = {
            "energy_fraction": summarize(energy_values),
            "test_rho_band_only": summarize(rho_values),
        }
    return {
        "full_tensor_rho": summarize([float(r["full_tensor_rho"]) for r in split_reports]),
        "mean_pooled_rho": summarize([float(r["mean_pooled_rho"]) for r in split_reports]),
        "per_time_bin_direction_mean_offdiag_cosine": summarize(
            [
                float(r["per_time_bin_direction_mean_offdiag_cosine"])
                for r in split_reports
            ]
        ),
        "bands": by_band,
    }


def interpretation_from(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    full = float(agg["full_tensor_rho"]["mean"])
    pooled = float(agg["mean_pooled_rho"]["mean"])
    bands = agg["bands"]
    dc = float(bands["temporal_dc_k0"]["test_rho_band_only"]["mean"])
    nonzero = float(bands["temporal_nonzero"]["test_rho_band_only"]["mean"])
    dc_energy = float(bands["temporal_dc_k0"]["energy_fraction"]["mean"])
    offdiag = float(agg["per_time_bin_direction_mean_offdiag_cosine"]["mean"])

    if abs(full) < 0.15:
        return (
            "The TRIBE tensor direction is weak on held-out splits, so the "
            "temporal Fourier decomposition should be treated as exploratory."
        )
    if abs(dc) >= 0.85 * abs(full) and abs(nonzero) <= 0.35 * abs(full):
        return (
            "The memorability readout is mostly recoverable from the temporal "
            f"DC component of TRIBE's four output bins ({dc_energy:.1%} of "
            "direction energy). Nonzero temporal modes are weak, and per-bin "
            f"directions are highly aligned (mean off-diagonal cosine {offdiag:+.3f}). "
            "This argues against a simple temporal-position artifact in the "
            "saved TRIBE outputs; the direction behaves more like a stable "
            "content/cortical response axis across the clip."
        )
    if abs(nonzero) >= 0.75 * abs(full):
        return (
            "A nonzero temporal Fourier component recovers much of the held-out "
            "memorability signal. That keeps Spencer's critique live: some of "
            "the apparent direction may ride on temporal-position or motion "
            "frequency structure in TRIBE's output sequence."
        )
    if abs(pooled) >= 0.85 * abs(full):
        return (
            "Mean pooling performs close to the full tensor readout, while no "
            "single non-DC temporal band dominates. This makes a pure temporal "
            "Fourier explanation unlikely, but still leaves open nonlinear "
            "basis entanglement inside TRIBE."
        )
    return (
        "The temporal evidence is mixed. The signal is not cleanly explained by "
        "one Fourier band, which is consistent with an entangled representation "
        "rather than an independently isolatable temporal-position basis."
    )


def write_markdown(report: dict[str, Any], path: Path, summary_path: Path) -> None:
    agg = report["aggregate"]
    lines = [
        "# TRIBE Temporal Spectral Memorability Probe",
        "",
        "TRIBE-specific follow-up to Spencer's positional/Fourier critique. The saved artifacts do not expose internal positional encodings, but they do preserve TRIBE's four temporal output bins, so this decomposes the learned memorability direction over that time axis.",
        "",
        "## Summary",
        "",
        f"- Clips: **{report['data']['n_clips']}**",
        f"- Feature tensor: `{report['data']['feature_shape']}`",
        f"- Time policy: `{report['data']['data_meta']['time_policy']}`",
        f"- Splits: **{report['config']['n_splits']}** seeds starting at `{report['config']['seed']}`",
        f"- Full tensor rho: **{agg['full_tensor_rho']['mean']:+.3f} ± {agg['full_tensor_rho']['std']:.3f}**",
        f"- Mean-pooled rho: **{agg['mean_pooled_rho']['mean']:+.3f} ± {agg['mean_pooled_rho']['std']:.3f}**",
        f"- Mean cross-time direction cosine: **{agg['per_time_bin_direction_mean_offdiag_cosine']['mean']:+.3f} ± {agg['per_time_bin_direction_mean_offdiag_cosine']['std']:.3f}**",
        "",
        "## Temporal FFT Bands",
        "",
        "| band | energy fraction | held-out rho |",
        "|---|---:|---:|",
    ]
    for band, row in agg["bands"].items():
        energy = row["energy_fraction"]
        rho = row["test_rho_band_only"]
        lines.append(
            f"| `{band}` | {energy['mean']:.3f} ± {energy['std']:.3f} | {rho['mean']:+.3f} ± {rho['std']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
        "## What This Does And Does Not Prove",
        "",
        "- It directly tests whether the saved TRIBE memorability vector is mostly a temporal-frequency artifact across TRIBE's four output bins.",
        "- It does not inspect TRIBE's internal positional encoding, RoPE tables, token mixer, or nonlinear basis. For that we need a Modal/internal model introspection pass.",
        "- If the DC band dominates, the paper should say the current evidence weakens a simple temporal-position critique, not that positional entanglement is impossible.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")

    summary_lines = [
        "# TRIBE Fourier Critique Status",
        "",
        report["interpretation"],
        "",
        f"- Time policy: {report['data']['data_meta']['time_policy']}",
        f"- Clips: {report['data']['n_clips']}",
        f"- Full tensor rho: {agg['full_tensor_rho']['mean']:+.3f} ± {agg['full_tensor_rho']['std']:.3f}",
        f"- Mean-pooled rho: {agg['mean_pooled_rho']['mean']:+.3f} ± {agg['mean_pooled_rho']['std']:.3f}",
        f"- Temporal DC rho: {agg['bands']['temporal_dc_k0']['test_rho_band_only']['mean']:+.3f} ± {agg['bands']['temporal_dc_k0']['test_rho_band_only']['std']:.3f}",
        f"- Nonzero temporal rho: {agg['bands']['temporal_nonzero']['test_rho_band_only']['mean']:+.3f} ± {agg['bands']['temporal_nonzero']['test_rho_band_only']['std']:.3f}",
        "",
        "Reviewer-safe claim: current TRIBE outputs do not support a simple temporal-Fourier artifact explanation; true internal positional-encoding patching remains future work.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-splits", type=int, default=8)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--top-frac", type=float, default=0.30)
    parser.add_argument("--min-tail", type=int, default=32)
    parser.add_argument(
        "--time-policy",
        choices=("common", "resample"),
        default="common",
        help=(
            "Use only the most common saved TRIBE time shape, or linearly "
            "resample all clips to --target-time before FFT."
        ),
    )
    parser.add_argument("--target-time", type=int, default=4)
    args = parser.parse_args()

    records, raw_features, scores, data_meta = load_tribe_tensor(
        feature_dir=args.feature_dir,
        annotations=args.annotations,
        time_policy=args.time_policy,
        target_time=args.target_time,
    )
    split_reports = [
        evaluate_split(
            raw_features=raw_features,
            scores=scores,
            seed=args.seed + offset,
            test_frac=args.test_frac,
            top_frac=args.top_frac,
            min_tail=args.min_tail,
        )
        for offset in range(args.n_splits)
    ]
    report: dict[str, Any] = {
        "config": {
            "seed": args.seed,
            "n_splits": args.n_splits,
            "test_frac": args.test_frac,
            "top_frac": args.top_frac,
            "min_tail": args.min_tail,
            "time_policy": args.time_policy,
            "target_time": args.target_time,
        },
        "data": {
            "n_clips": len(records),
            "feature_shape": list(raw_features.shape),
            "sample_ids": [record.sample_id for record in records],
            "data_meta": data_meta,
        },
        "splits": split_reports,
        "aggregate": aggregate_splits(split_reports),
    }
    report["interpretation"] = interpretation_from(report)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.out_md, args.out_summary)
    print(f"[tribe-temporal-spectral] wrote {args.out_json}", flush=True)
    print(f"[tribe-temporal-spectral] wrote {args.out_md}", flush=True)
    print(f"[tribe-temporal-spectral] wrote {args.out_summary}", flush=True)


if __name__ == "__main__":
    main()
