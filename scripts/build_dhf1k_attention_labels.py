"""Build DHF1K attention-label CSVs for Phase 1 capture validation.

The output is not a TRIBE result. It derives external gaze/saliency labels from
DHF1K annotation maps and records enough audit metadata to decide whether a
chosen label is usable before spending GPU time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SPLIT_RANGES: dict[str, tuple[int, int]] = {
    "train": (1, 600),
    "val": (601, 700),
    "annotated": (1, 700),
    "test": (701, 1000),
    "all": (1, 1000),
}

CSV_COLUMNS = (
    "sample_id",
    "video_id",
    "split",
    "video_path",
    "n_map_frames",
    "n_fixation_frames",
    "mean_map_intensity",
    "peak_map_intensity",
    "peak_to_mean_map_ratio",
    "mean_map_concentration",
    "mean_fixation_density",
    "selected_tail",
)
RANK_COLUMNS = (
    "mean_map_intensity",
    "peak_map_intensity",
    "peak_to_mean_map_ratio",
    "mean_map_concentration",
    "mean_fixation_density",
)
DEFAULT_MIN_ROWS = 30
DEFAULT_MIN_DISTINCT_RANK_VALUES = 3


@dataclass(frozen=True)
class DHF1KRow:
    sample_id: str
    video_id: str
    split: str
    video_path: str
    n_map_frames: int
    n_fixation_frames: int
    mean_map_intensity: float | None
    peak_map_intensity: float | None
    peak_to_mean_map_ratio: float | None
    mean_map_concentration: float | None
    mean_fixation_density: float | None
    selected_tail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dhf1k-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_RANGES),
        default="annotated",
        help="DHF1K split to scan. Only train/val/annotated have released labels.",
    )
    parser.add_argument(
        "--rank-column",
        choices=RANK_COLUMNS,
        default="mean_map_intensity",
    )
    parser.add_argument(
        "--extreme-count-per-tail",
        type=int,
        default=None,
        help=(
            "Optional high/low tail selection after computing all rows. "
            "Use 175 for the proposal's DHF1K 350-video extreme-quartile run."
        ),
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=DEFAULT_MIN_ROWS,
        help="Minimum emitted rows required before the label audit is handoff-ready.",
    )
    parser.add_argument(
        "--min-distinct-rank-values",
        type=int,
        default=DEFAULT_MIN_DISTINCT_RANK_VALUES,
        help="Minimum distinct finite values required in the selected rank column.",
    )
    parser.add_argument(
        "--metric-scope",
        choices=("all", "rank"),
        default="all",
        help=(
            "Compute all audit metrics, or only the selected rank-column metric "
            "for faster manifest handoff."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(
        dhf1k_root=args.dhf1k_root,
        split=args.split,
        limit=args.limit,
        metric_columns=metric_columns_for_scope(
            scope=args.metric_scope,
            rank_column=args.rank_column,
        ),
    )
    rows = select_extreme_tails(
        rows,
        rank_column=args.rank_column,
        count_per_tail=args.extreme_count_per_tail,
    )
    audit = summarize_rows(
        rows,
        dhf1k_root=args.dhf1k_root,
        labels_csv=args.output_csv,
        split=args.split,
        rank_column=args.rank_column,
        metric_scope=args.metric_scope,
        extreme_count_per_tail=args.extreme_count_per_tail,
        min_rows=args.min_rows,
        min_distinct_rank_values=args.min_distinct_rank_values,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_rows_csv(rows, args.output_csv)
    args.output_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {args.output_csv}")
    print(f"wrote audit -> {args.output_json}")


def build_rows(
    *,
    dhf1k_root: Path,
    split: str = "annotated",
    limit: int | None = None,
    metric_columns: set[str] | None = None,
) -> list[DHF1KRow]:
    start, end = SPLIT_RANGES[split]
    rows: list[DHF1KRow] = []
    for video_index in range(start, end + 1):
        video_id = f"{video_index:03d}"
        row = build_row(
            dhf1k_root=dhf1k_root,
            video_id=video_id,
            split=split,
            metric_columns=metric_columns,
        )
        if row is not None:
            rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def build_row(
    *,
    dhf1k_root: Path,
    video_id: str,
    split: str,
    metric_columns: set[str] | None = None,
) -> DHF1KRow | None:
    metrics = normalize_metric_columns(metric_columns)
    video_path = find_video_path(dhf1k_root / "video", video_id)
    annotation_dir = find_annotation_video_dir(dhf1k_root / "annotation", video_id)
    map_paths = annotation_map_paths(annotation_dir)
    fixation_dir = annotation_dir / "fixation" if annotation_dir is not None else None
    fixation_paths = sorted(fixation_dir.glob("*.png")) if fixation_dir else []
    if video_path is None or not map_paths:
        return None

    map_stats = map_sequence_stats(map_paths, metric_columns=metrics)
    fixation_density = (
        mean_image_sequence_intensity(fixation_paths)
        if fixation_paths and "mean_fixation_density" in metrics
        else None
    )
    return DHF1KRow(
        sample_id=f"dhf1k_{video_id}",
        video_id=video_id,
        split=dhf1k_split_for_video_id(video_id),
        video_path=portable_video_path(video_path),
        n_map_frames=len(map_paths),
        n_fixation_frames=len(fixation_paths),
        mean_map_intensity=map_stats["mean_map_intensity"],
        peak_map_intensity=map_stats["peak_map_intensity"],
        peak_to_mean_map_ratio=map_stats["peak_to_mean_map_ratio"],
        mean_map_concentration=map_stats["mean_map_concentration"],
        mean_fixation_density=fixation_density,
    )


def metric_columns_for_scope(*, scope: str, rank_column: str) -> set[str] | None:
    if scope == "all":
        return None
    if scope == "rank":
        return {rank_column}
    raise ValueError(f"unsupported metric scope: {scope}")


def normalize_metric_columns(metric_columns: set[str] | None) -> set[str]:
    if metric_columns is None:
        return set(RANK_COLUMNS)
    unknown = metric_columns - set(RANK_COLUMNS)
    if unknown:
        raise ValueError(f"unknown metric columns: {sorted(unknown)}")
    return set(metric_columns)


def portable_video_path(video_path: Path) -> str:
    resolved = video_path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def video_id_variants(video_id: str) -> list[str]:
    value = int(video_id)
    return [f"{value:03d}", f"{value:04d}"]


def find_video_path(video_dir: Path, video_id: str) -> Path | None:
    for candidate_id in video_id_variants(video_id):
        for suffix in (".AVI", ".avi", ".mp4", ".MP4", ".mov", ".MOV"):
            candidate = video_dir / f"{candidate_id}{suffix}"
            if candidate.exists():
                return candidate
    return None


def find_annotation_video_dir(annotation_dir: Path, video_id: str) -> Path | None:
    for candidate_id in video_id_variants(video_id):
        candidate = annotation_dir / candidate_id
        if candidate.is_dir():
            return candidate
    return None


def annotation_map_paths(annotation_video_dir: Path | None) -> list[Path]:
    if annotation_video_dir is None:
        return []
    nested_maps = sorted((annotation_video_dir / "maps").glob("*.png"))
    if nested_maps:
        return nested_maps
    return sorted(annotation_video_dir.glob("*.png"))


def map_sequence_stats(
    image_paths: list[Path],
    *,
    metric_columns: set[str],
) -> dict[str, float | None]:
    means: list[float] = []
    peaks: list[float] = []
    ratios: list[float] = []
    concentrations: list[float] = []
    needs_mean = (
        "mean_map_intensity" in metric_columns
        or "peak_to_mean_map_ratio" in metric_columns
    )
    needs_peak = (
        "peak_map_intensity" in metric_columns
        or "peak_to_mean_map_ratio" in metric_columns
    )
    for path in image_paths:
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        mean_value = float(image.mean()) if needs_mean else None
        peak_value = float(image.max()) if needs_peak else None
        if "mean_map_intensity" in metric_columns:
            assert mean_value is not None
            means.append(mean_value)
        if "peak_map_intensity" in metric_columns:
            assert peak_value is not None
            peaks.append(peak_value)
        if "peak_to_mean_map_ratio" in metric_columns:
            assert mean_value is not None
            assert peak_value is not None
            ratios.append(float(peak_value / max(mean_value, 1e-12)))
        if "mean_map_concentration" in metric_columns:
            concentrations.append(spatial_concentration(image))
    return {
        "mean_map_intensity": mean_or_none(means),
        "peak_map_intensity": mean_or_none(peaks),
        "peak_to_mean_map_ratio": mean_or_none(ratios),
        "mean_map_concentration": mean_or_none(concentrations),
    }


def mean_image_sequence_intensity(image_paths: list[Path]) -> float:
    means: list[float] = []
    for path in image_paths:
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        means.append(float(image.mean()))
    return float(np.mean(means))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def spatial_concentration(image: np.ndarray) -> float:
    total = float(image.sum())
    if total <= 0:
        return 0.0
    probabilities = np.ravel(image / total)
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    max_entropy = math.log(float(probabilities.size))
    if max_entropy <= 0:
        return 0.0
    return float(1.0 - entropy / max_entropy)


def select_extreme_tails(
    rows: list[DHF1KRow],
    *,
    rank_column: str,
    count_per_tail: int | None,
) -> list[DHF1KRow]:
    if count_per_tail is None:
        return rows
    if count_per_tail <= 0:
        raise ValueError("count_per_tail must be positive")
    usable = [row for row in rows if getattr(row, rank_column) is not None]
    if len(usable) < count_per_tail * 2:
        raise ValueError(
            f"not enough finite {rank_column} rows for disjoint extreme tails: "
            f"{len(usable)} available, {count_per_tail * 2} required"
        )
    ordered = sorted(usable, key=lambda row: float(getattr(row, rank_column)))
    low = [replace_tail(row, "low") for row in ordered[:count_per_tail]]
    high = [replace_tail(row, "high") for row in ordered[-count_per_tail:]]
    return sorted([*low, *high], key=lambda row: row.sample_id)


def replace_tail(row: DHF1KRow, selected_tail: str) -> DHF1KRow:
    return DHF1KRow(
        sample_id=row.sample_id,
        video_id=row.video_id,
        split=row.split,
        video_path=row.video_path,
        n_map_frames=row.n_map_frames,
        n_fixation_frames=row.n_fixation_frames,
        mean_map_intensity=row.mean_map_intensity,
        peak_map_intensity=row.peak_map_intensity,
        peak_to_mean_map_ratio=row.peak_to_mean_map_ratio,
        mean_map_concentration=row.mean_map_concentration,
        mean_fixation_density=row.mean_fixation_density,
        selected_tail=selected_tail,
    )


def summarize_rows(
    rows: list[DHF1KRow],
    *,
    dhf1k_root: Path,
    split: str,
    rank_column: str,
    extreme_count_per_tail: int | None,
    metric_scope: str = "all",
    labels_csv: Path | None = None,
    min_rows: int = DEFAULT_MIN_ROWS,
    min_distinct_rank_values: int = DEFAULT_MIN_DISTINCT_RANK_VALUES,
) -> dict[str, Any]:
    metric_summaries = {
        column: summarize_metric(
            [getattr(row, column) for row in rows],
        )
        for column in RANK_COLUMNS
    }
    candidate_columns = ground_truth_column_candidates(
        metric_summaries,
        min_rows=min_rows,
        min_distinct_rank_values=min_distinct_rank_values,
    )
    blocking_reasons = dhf1k_label_blocking_reasons(
        rows=rows,
        rank_column=rank_column,
        metric_summaries=metric_summaries,
        min_rows=min_rows,
        min_distinct_rank_values=min_distinct_rank_values,
        extreme_count_per_tail=extreme_count_per_tail,
    )
    return {
        "schema_version": 1,
        "experiment": "dhf1k_attention_label_audit",
        "dataset": "DHF1K",
        "source_root": str(dhf1k_root),
        "labels_csv": str(labels_csv) if labels_csv is not None else None,
        "split": split,
        "rank_column": rank_column,
        "metric_scope": metric_scope,
        "computed_metric_columns": [
            column
            for column in RANK_COLUMNS
            if int(metric_summaries[column]["n"] or 0) > 0
        ],
        "extreme_count_per_tail": extreme_count_per_tail,
        "min_rows": min_rows,
        "min_distinct_rank_values": min_distinct_rank_values,
        "n_rows": len(rows),
        "metrics": metric_summaries,
        "rank_column_summary": metric_summaries[rank_column],
        "rank_column_ready": not blocking_reasons,
        "ready_for_manifest_alignment": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "candidate_ground_truth_columns": candidate_columns,
        "recommended_ground_truth_column": (
            candidate_columns[0]["column"] if candidate_columns else None
        ),
        "claim_boundary": (
            "DHF1K labels are external gaze/saliency labels, not retention, "
            "dopamine, or executive-control measurements."
        ),
    }


def summarize_metric(values: list[float | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {
            "n": 0,
            "n_distinct": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "n": int(arr.size),
        "n_distinct": len(set(finite)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def ground_truth_column_candidates(
    metric_summaries: dict[str, dict[str, float | int | None]],
    *,
    min_rows: int,
    min_distinct_rank_values: int,
) -> list[dict[str, float | int | str]]:
    candidates: list[dict[str, float | int | str]] = []
    for column in RANK_COLUMNS:
        summary = metric_summaries[column]
        n = int(summary["n"] or 0)
        n_distinct = int(summary["n_distinct"] or 0)
        std = float(summary["std"] or 0.0)
        if n < min_rows or n_distinct < min_distinct_rank_values or std <= 0.0:
            continue
        candidates.append(
            {
                "column": column,
                "n": n,
                "n_distinct": n_distinct,
                "std": std,
            }
        )
    return candidates


def dhf1k_label_blocking_reasons(
    *,
    rows: list[DHF1KRow],
    rank_column: str,
    metric_summaries: dict[str, dict[str, float | int | None]],
    min_rows: int,
    min_distinct_rank_values: int,
    extreme_count_per_tail: int | None,
) -> list[str]:
    reasons: list[str] = []
    summary = metric_summaries[rank_column]
    n_rows = len(rows)
    n_finite = int(summary["n"] or 0)
    n_distinct = int(summary["n_distinct"] or 0)
    std = float(summary["std"] or 0.0)
    if n_rows < min_rows:
        reasons.append(f"row count {n_rows} is below minimum {min_rows}")
    if n_finite < n_rows:
        reasons.append(
            f"{rank_column} has {n_rows - n_finite} non-finite rows"
        )
    if n_distinct < min_distinct_rank_values:
        reasons.append(
            f"{rank_column} distinct finite value count {n_distinct} "
            f"is below minimum {min_distinct_rank_values}"
        )
    if n_finite and std <= 0.0:
        reasons.append(f"{rank_column} has zero variance")
    if extreme_count_per_tail is not None:
        expected = extreme_count_per_tail * 2
        tail_counts = Counter(row.selected_tail for row in rows)
        if n_rows != expected:
            reasons.append(
                f"selected row count {n_rows} does not equal expected "
                f"extreme-tail count {expected}"
            )
        if tail_counts["low"] != extreme_count_per_tail:
            reasons.append(
                f"low-tail row count {tail_counts['low']} does not equal "
                f"{extreme_count_per_tail}"
            )
        if tail_counts["high"] != extreme_count_per_tail:
            reasons.append(
                f"high-tail row count {tail_counts['high']} does not equal "
                f"{extreme_count_per_tail}"
            )
    return reasons


def write_rows_csv(rows: list[DHF1KRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(CSV_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(getattr(row, column)) for column in CSV_COLUMNS})


def csv_value(value: object) -> object:
    if value is None:
        return ""
    return value


def dhf1k_split_for_video_id(video_id: str) -> str:
    idx = int(video_id)
    if 1 <= idx <= 600:
        return "train"
    if 601 <= idx <= 700:
        return "val"
    if 701 <= idx <= 1000:
        return "test"
    return "unknown"


if __name__ == "__main__":
    main()
