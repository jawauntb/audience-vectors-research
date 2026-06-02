"""Causal probe for TRIBE's learned temporal positional embedding.

This reruns TRIBE on a balanced BMD subset while temporarily scaling
`_model.time_pos_embed` inside the Modal TRIBE predictor. The baseline scale is
1.0; scale 0.0 is a direct learned-time-position ablation.

The readout is the existing BMD/TRIBE memorability vector trained from cached
unpatched TRIBE outputs. If the projection and high-vs-low memorability gap
survive scale=0.0, Spencer's internal-position critique is weakened. If they
collapse, the learned position table is load-bearing for the memorability axis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audience_vectors.services.tribe_service import TribeService, TribeValidationError

DEFAULT_ANNOTATIONS = Path("data/raw/bold_moments/annotations.json")
DEFAULT_FEATURE_DIR = Path("data/features/tribe")
DEFAULT_OUTPUT_DIR = Path("data/features/tribe_timepos_patch")
DEFAULT_JSON = Path("data/reports/tribe_timepos_patch_probe.json")
DEFAULT_MD = Path("data/reports/tribe_timepos_patch_probe.md")


@dataclass(frozen=True)
class BmdRecord:
    key: str
    sample_id: str
    video_id: str
    score: float
    url: str
    volume_path: str
    cached_feature_path: Path


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
    if len(a) < 2:
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
        raise ValueError("near-zero direction")
    return np.asarray(x / norm, dtype=np.float32)


def scale_label(scale: float) -> str:
    text = f"{scale:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return text.rstrip("0").rstrip("p") if "p" in text else text


def load_records(
    *,
    annotations: Path,
    feature_dir: Path,
) -> tuple[list[BmdRecord], np.ndarray, np.ndarray]:
    payload = json.loads(annotations.read_text())
    records: list[BmdRecord] = []
    features: list[np.ndarray] = []
    scores: list[float] = []
    for key, row in sorted(payload.items()):
        if "memorability_score" not in row:
            continue
        video_id = f"bmd_vid_idx{key}"
        sample_id = f"{video_id}_seg_0000"
        feature_path = feature_dir / f"{sample_id}.npz"
        if not feature_path.exists():
            continue
        frames = np.asarray(np.load(feature_path, allow_pickle=False)["frames"], dtype=np.float32)
        if frames.ndim != 2:
            continue
        score = float(row["memorability_score"])
        records.append(
            BmdRecord(
                key=key,
                sample_id=sample_id,
                video_id=video_id,
                score=score,
                url=str(row.get("MiT_url", "")),
                volume_path=f"/bmd-videos/videos/vid_idx{key}.mp4",
                cached_feature_path=feature_path,
            )
        )
        features.append(frames.mean(axis=0))
        scores.append(score)
    if not records:
        raise FileNotFoundError(f"no cached BMD/TRIBE feature rows under {feature_dir}")
    return records, np.stack(features).astype(np.float32), np.asarray(scores, dtype=np.float32)


def train_v_mem(features: np.ndarray, scores: np.ndarray, top_frac: float) -> np.ndarray:
    order = np.argsort(scores)
    n_each = max(8, int(round(len(scores) * top_frac)))
    low = features[order[:n_each]].mean(axis=0)
    high = features[order[-n_each:]].mean(axis=0)
    return unit(high - low)


def select_balanced(records: list[BmdRecord], n_each: int) -> list[BmdRecord]:
    ordered = sorted(records, key=lambda record: record.score)
    if n_each * 2 > len(ordered):
        raise ValueError(f"n_each={n_each} too large for n={len(ordered)}")
    selected = ordered[:n_each] + ordered[-n_each:]
    return sorted(selected, key=lambda record: record.key)


def output_path(output_dir: Path, record: BmdRecord, scale: float) -> Path:
    return output_dir / f"scale_{scale_label(scale)}" / f"{record.sample_id}.npz"


def result_frames(result: Any) -> tuple[np.ndarray, float]:
    if hasattr(result, "frames"):
        frames = np.asarray(result.frames, dtype=np.float32)
        duration = float(result.duration_seconds)
    else:
        frames = np.asarray(result["frames"], dtype=np.float32)
        duration = float(result["duration_seconds"])
    return frames, duration


async def fetch_one(
    *,
    service: TribeService,
    record: BmdRecord,
    scale: float,
    output_dir: Path,
    timeout: float,
    prefer_url: bool,
) -> dict[str, Any]:
    path = output_path(output_dir, record, scale)
    if path.exists() and path.stat().st_size > 0:
        return {
            "sample_id": record.sample_id,
            "scale": scale,
            "ok": True,
            "cached": True,
            "path": str(path),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    sources = [record.url] if prefer_url else [record.volume_path, record.url]
    errors: list[str] = []
    for source in [src for src in sources if src]:
        try:
            result = await asyncio.wait_for(
                service.predict_video_time_pos_scale(source, scale),
                timeout=timeout,
            )
        except TribeValidationError as exc:
            errors.append(f"{source}: validation {exc}")
            continue
        except TimeoutError:
            errors.append(f"{source}: timeout")
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
            continue
        if result is None:
            errors.append(f"{source}: empty result")
            continue
        frames, duration = result_frames(result)
        np.savez_compressed(
            path,
            frames=frames,
            duration_seconds=np.array(duration, dtype=np.float32),
            sample_id=np.array(record.sample_id),
            video_id=np.array(record.video_id),
            memorability_score=np.array(record.score, dtype=np.float32),
            time_pos_scale=np.array(scale, dtype=np.float32),
            source=np.array(source),
        )
        return {
            "sample_id": record.sample_id,
            "scale": scale,
            "ok": True,
            "cached": False,
            "path": str(path),
            "source": source,
            "shape": list(frames.shape),
        }
    return {
        "sample_id": record.sample_id,
        "scale": scale,
        "ok": False,
        "errors": errors,
    }


async def run_modal_predictions(
    *,
    records: list[BmdRecord],
    scales: list[float],
    output_dir: Path,
    concurrency: int,
    timeout: float,
    prefer_url: bool,
) -> list[dict[str, Any]]:
    service = TribeService()
    sem = asyncio.Semaphore(concurrency)

    async def guarded(record: BmdRecord, scale: float) -> dict[str, Any]:
        async with sem:
            print(f"[time-pos] {record.sample_id} scale={scale:+.3f}", flush=True)
            return await fetch_one(
                service=service,
                record=record,
                scale=scale,
                output_dir=output_dir,
                timeout=timeout,
                prefer_url=prefer_url,
            )

    return await asyncio.gather(
        *[guarded(record, scale) for scale in scales for record in records]
    )


def load_patch_features(
    *,
    records: list[BmdRecord],
    scales: list[float],
    output_dir: Path,
) -> dict[float, np.ndarray]:
    by_scale: dict[float, list[np.ndarray]] = {scale: [] for scale in scales}
    for scale in scales:
        for record in records:
            path = output_path(output_dir, record, scale)
            if not path.exists():
                raise FileNotFoundError(path)
            frames = np.asarray(np.load(path, allow_pickle=False)["frames"], dtype=np.float32)
            by_scale[scale].append(frames.mean(axis=0))
    return {
        scale: np.stack(rows).astype(np.float32)
        for scale, rows in by_scale.items()
    }


def summarize_probe(
    *,
    records: list[BmdRecord],
    cached_features: np.ndarray,
    patch_features: dict[float, np.ndarray],
    v_mem: np.ndarray,
    baseline_scale: float,
) -> dict[str, Any]:
    scores = np.asarray([record.score for record in records], dtype=np.float32)
    cached_projection = cached_features @ v_mem
    baseline_projection = patch_features[baseline_scale] @ v_mem
    order = np.argsort(scores)
    n_each = len(records) // 2
    low_idx = order[:n_each]
    high_idx = order[-n_each:]

    scale_rows: dict[str, Any] = {}
    for scale, features in sorted(patch_features.items()):
        projection = features @ v_mem
        delta = projection - baseline_projection
        high_low_gap = float(projection[high_idx].mean() - projection[low_idx].mean())
        baseline_gap = float(
            baseline_projection[high_idx].mean() - baseline_projection[low_idx].mean()
        )
        scale_rows[str(scale)] = {
            "spearman_vs_memorability": spearman(projection, scores),
            "spearman_vs_baseline_projection": spearman(projection, baseline_projection),
            "pearson_vs_baseline_projection": float(
                np.corrcoef(projection, baseline_projection)[0, 1]
            ),
            "mean_projection_delta_vs_baseline": float(delta.mean()),
            "mean_abs_projection_delta_vs_baseline": float(np.abs(delta).mean()),
            "std_projection_delta_vs_baseline": float(delta.std()),
            "projection_delta_in_baseline_std": float(
                np.abs(delta).mean() / max(float(baseline_projection.std()), 1e-12)
            ),
            "high_minus_low_gap": high_low_gap,
            "high_minus_low_gap_ratio_vs_baseline": high_low_gap
            / baseline_gap
            if abs(baseline_gap) > 1e-12
            else None,
        }

    return {
        "n_clips": len(records),
        "n_each_tail": n_each,
        "baseline_scale": baseline_scale,
        "cached_original_vs_rerun_baseline": {
            "spearman": spearman(cached_projection, baseline_projection),
            "pearson": float(np.corrcoef(cached_projection, baseline_projection)[0, 1]),
            "mean_abs_delta": float(np.abs(cached_projection - baseline_projection).mean()),
        },
        "scales": scale_rows,
        "records": [
            {
                "sample_id": record.sample_id,
                "score": record.score,
                "cached_projection": float(cached_projection[i]),
                "baseline_projection": float(baseline_projection[i]),
            }
            for i, record in enumerate(records)
        ],
    }


def interpretation_from(summary: dict[str, Any]) -> str:
    baseline_key = str(summary["baseline_scale"])
    baseline = summary["scales"][baseline_key]
    ablated = summary["scales"].get("0.0")
    if ablated is None:
        return "No time-position ablation scale=0.0 was run, so this is a scaling sensitivity report only."
    baseline_rho = float(baseline["spearman_vs_memorability"])
    ablated_rho = float(ablated["spearman_vs_memorability"])
    gap_ratio = ablated["high_minus_low_gap_ratio_vs_baseline"]
    delta_std = float(ablated["projection_delta_in_baseline_std"])
    if abs(ablated_rho) >= 0.8 * abs(baseline_rho) and (
        gap_ratio is not None and abs(float(gap_ratio)) >= 0.8
    ):
        return (
            "Ablating TRIBE's learned time-position table leaves the memorability "
            "readout largely intact on this balanced subset. This weakens the "
            "hypothesis that v_mem is mainly a learned temporal-position artifact."
        )
    if abs(ablated_rho) <= 0.5 * abs(baseline_rho) or (
        gap_ratio is not None and abs(float(gap_ratio)) <= 0.5
    ):
        return (
            "Ablating TRIBE's learned time-position table substantially weakens "
            "the memorability readout on this subset. Spencer's internal-position "
            "critique is live and should be promoted from caveat to main limitation."
        )
    return (
        "The time-position ablation changes the readout but does not cleanly "
        f"collapse it (mean absolute shift {delta_std:.2f} baseline standard deviations). "
        "This is mixed evidence: positional structure matters, but does not by itself "
        "explain v_mem."
    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# TRIBE Time-Position Patch Probe",
        "",
        report["interpretation"],
        "",
        "## Setup",
        "",
        f"- Clips: **{summary['n_clips']}** balanced top/bottom BMD memorability clips.",
        f"- Baseline scale: `{summary['baseline_scale']}`.",
        "- Patch: temporarily multiply `_model.time_pos_embed` during TRIBE inference.",
        "- Readout: projection onto the cached BMD/TRIBE memorability direction.",
        "",
        "## Sanity Check",
        "",
        f"- Cached original vs rerun baseline Spearman: **{summary['cached_original_vs_rerun_baseline']['spearman']:+.3f}**",
        f"- Cached original vs rerun baseline Pearson: **{summary['cached_original_vs_rerun_baseline']['pearson']:+.3f}**",
        f"- Mean absolute projection delta: **{summary['cached_original_vs_rerun_baseline']['mean_abs_delta']:.3f}**",
        "",
        "## Scale Results",
        "",
        "| time-pos scale | ρ vs BMD mem | ρ vs baseline projection | mean |Δproj| / baseline std | high-low gap ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for scale, row in summary["scales"].items():
        ratio = row["high_minus_low_gap_ratio_vs_baseline"]
        ratio_text = "n/a" if ratio is None else f"{float(ratio):+.3f}"
        lines.append(
            f"| {float(scale):+.3f} | {row['spearman_vs_memorability']:+.3f} | "
            f"{row['spearman_vs_baseline_projection']:+.3f} | "
            f"{row['projection_delta_in_baseline_std']:.3f} | {ratio_text} |"
        )
    lines += [
        "",
        "## Caveat",
        "",
        "This is a direct parameter patch, but still a subset experiment. A full paper-grade causal claim should repeat this over more clips and add hidden-state logging around the transformer.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_scales(text: str) -> list[float]:
    scales = [float(part.strip()) for part in text.split(",") if part.strip()]
    if 1.0 not in scales:
        scales.insert(0, 1.0)
    return scales


async def main_async(args: argparse.Namespace) -> None:
    all_records, all_features, all_scores = load_records(
        annotations=args.annotations,
        feature_dir=args.feature_dir,
    )
    v_mem = train_v_mem(all_features, all_scores, top_frac=args.top_frac)
    selected = select_balanced(all_records, n_each=args.n_each)
    selected_index = {record.sample_id: idx for idx, record in enumerate(all_records)}
    cached_selected = np.stack(
        [all_features[selected_index[record.sample_id]] for record in selected]
    ).astype(np.float32)
    scales = parse_scales(args.scales)
    prediction_results = await run_modal_predictions(
        records=selected,
        scales=scales,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        timeout=args.timeout,
        prefer_url=args.use_urls,
    )
    failed = [row for row in prediction_results if not row.get("ok")]
    if failed:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        failure_path = args.out_json.with_suffix(".failures.json")
        failure_path.write_text(json.dumps(failed, indent=2) + "\n")
        raise RuntimeError(f"{len(failed)} TRIBE calls failed; wrote {failure_path}")

    patch_features = load_patch_features(
        records=selected,
        scales=scales,
        output_dir=args.output_dir,
    )
    summary = summarize_probe(
        records=selected,
        cached_features=cached_selected,
        patch_features=patch_features,
        v_mem=v_mem,
        baseline_scale=1.0,
    )
    report: dict[str, Any] = {
        "config": {
            "n_each": args.n_each,
            "scales": scales,
            "top_frac": args.top_frac,
            "concurrency": args.concurrency,
            "timeout": args.timeout,
            "use_urls": args.use_urls,
            "output_dir": str(args.output_dir),
        },
        "summary": summary,
    }
    report["interpretation"] = interpretation_from(summary)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.out_md)
    print(f"[time-pos] wrote {args.out_json}", flush=True)
    print(f"[time-pos] wrote {args.out_md}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--n-each", type=int, default=12)
    parser.add_argument("--scales", default="1.0,0.0,0.5,2.0")
    parser.add_argument("--top-frac", type=float, default=0.30)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument(
        "--use-urls",
        action="store_true",
        help="Use MiT URLs directly instead of the Modal bmd-videos volume path.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
