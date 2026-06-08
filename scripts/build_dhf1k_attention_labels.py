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
        choices=(
            "mean_map_intensity",
            "peak_map_intensity",
            "peak_to_mean_map_ratio",
            "mean_map_concentration",
            "mean_fixation_density",
        ),
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
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(
        dhf1k_root=args.dhf1k_root,
        split=args.split,
        limit=args.limit,
    )
    rows = select_extreme_tails(
        rows,
        rank_column=args.rank_column,
        count_per_tail=args.extreme_count_per_tail,
    )
    audit = summarize_rows(
        rows,
        dhf1k_root=args.dhf1k_root,
        split=args.split,
        rank_column=args.rank_column,
        extreme_count_per_tail=args.extreme_count_per_tail,
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
) -> list[DHF1KRow]:
    start, end = SPLIT_RANGES[split]
    rows: list[DHF1KRow] = []
    for video_index in range(start, end + 1):
        video_id = f"{video_index:03d}"
        row = build_row(dhf1k_root=dhf1k_root, video_id=video_id, split=split)
        if row is not None:
            rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def build_row(*, dhf1k_root: Path, video_id: str, split: str) -> DHF1KRow | None:
    video_path = find_video_path(dhf1k_root / "video", video_id)
    maps_dir = dhf1k_root / "annotation" / video_id / "maps"
    fixation_dir = dhf1k_root / "annotation" / video_id / "fixation"
    map_paths = sorted(maps_dir.glob("*.png"))
    fixation_paths = sorted(fixation_dir.glob("*.png"))
    if video_path is None or not map_paths:
        return None

    map_stats = image_sequence_stats(map_paths)
    fixation_density = (
        image_sequence_stats(fixation_paths)["mean_intensity"]
        if fixation_paths
        else None
    )
    return DHF1KRow(
        sample_id=f"dhf1k_{video_id}",
        video_id=video_id,
        split=dhf1k_split_for_video_id(video_id),
        video_path=str(video_path.resolve()),
        n_map_frames=len(map_paths),
        n_fixation_frames=len(fixation_paths),
        mean_map_intensity=map_stats["mean_intensity"],
        peak_map_intensity=map_stats["peak_intensity"],
        peak_to_mean_map_ratio=map_stats["peak_to_mean_ratio"],
        mean_map_concentration=map_stats["mean_concentration"],
        mean_fixation_density=fixation_density,
    )


def find_video_path(video_dir: Path, video_id: str) -> Path | None:
    for suffix in (".AVI", ".avi", ".mp4", ".MP4", ".mov", ".MOV"):
        candidate = video_dir / f"{video_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def image_sequence_stats(image_paths: list[Path]) -> dict[str, float]:
    means: list[float] = []
    peaks: list[float] = []
    ratios: list[float] = []
    concentrations: list[float] = []
    for path in image_paths:
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        mean_value = float(image.mean())
        peak_value = float(image.max())
        means.append(mean_value)
        peaks.append(peak_value)
        ratios.append(float(peak_value / max(mean_value, 1e-12)))
        concentrations.append(spatial_concentration(image))
    return {
        "mean_intensity": float(np.mean(means)),
        "peak_intensity": float(np.mean(peaks)),
        "peak_to_mean_ratio": float(np.mean(ratios)),
        "mean_concentration": float(np.mean(concentrations)),
    }


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
    usable = [row for row in rows if getattr(row, rank_column) is not None]
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
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": "DHF1K",
        "source_root": str(dhf1k_root),
        "split": split,
        "rank_column": rank_column,
        "extreme_count_per_tail": extreme_count_per_tail,
        "n_rows": len(rows),
        "metrics": {
            column: summarize_metric(
                [getattr(row, column) for row in rows],
            )
            for column in CSV_COLUMNS
            if column.startswith("mean_")
            or column.startswith("peak_")
            or column == "mean_fixation_density"
        },
        "claim_boundary": (
            "DHF1K labels are external gaze/saliency labels, not retention, "
            "dopamine, or executive-control measurements."
        ),
    }


def summarize_metric(values: list[float | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def write_rows_csv(rows: list[DHF1KRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
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
