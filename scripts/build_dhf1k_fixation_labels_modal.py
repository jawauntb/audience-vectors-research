"""Build DHF1K fixation-density labels with Modal CPU parallelism.

This is a targeted fast path for the proposal's DHF1K ocular ground truth:
mean fixation-map density.  Each Modal task receives one video's fixation PNGs
and returns one scalar row; the local driver keeps provenance and tail selection
compatible with build_dhf1k_attention_labels.py.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import modal

APP_NAME = "dhf1k-fixation-labels"
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

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy",
    "pillow",
)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=1.0,
    memory=1024,
    timeout=10 * 60,
    max_containers=64,
)
def fixation_density_for_video(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    means: list[float] = []
    for image_bytes in payload["fixation_pngs"]:
        image = np.asarray(
            Image.open(io.BytesIO(image_bytes)).convert("L"),
            dtype=np.float32,
        ) / 255.0
        means.append(float(image.mean()))
    return {
        "video_id": payload["video_id"],
        "mean_fixation_density": float(np.mean(means)) if means else None,
    }


@app.local_entrypoint()
def main(
    dhf1k_root: str,
    output_csv: str,
    output_json: str,
    extreme_count_per_tail: int = 175,
) -> None:
    root = Path(dhf1k_root)
    output_csv_path = Path(output_csv)
    output_json_path = Path(output_json)
    local_rows = build_local_rows(root)
    print(f"[dhf1k-modal] dispatching {len(local_rows)} fixation-density CPU jobs")
    results = list(
        fixation_density_for_video.map(
            payloads_for_rows(local_rows, root),
            order_outputs=False,
        )
    )
    by_video_id = {str(row["video_id"]): row for row in results}
    rows: list[dict[str, Any]] = []
    for row in local_rows:
        result = by_video_id.get(row["video_id"])
        row["mean_fixation_density"] = (
            result["mean_fixation_density"] if result is not None else None
        )
        rows.append(row)

    rows = select_extreme_tails(
        rows,
        rank_column="mean_fixation_density",
        count_per_tail=extreme_count_per_tail,
    )
    audit = summarize_rows(
        rows,
        dhf1k_root=root,
        labels_csv=output_csv_path,
        extreme_count_per_tail=extreme_count_per_tail,
    )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    write_rows_csv(rows, output_csv_path)
    output_json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {output_csv_path}")
    print(f"wrote audit -> {output_json_path}")


def build_local_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(1, 701):
        video_id = f"{idx:03d}"
        annotation_dir = find_annotation_video_dir(root / "annotation", video_id)
        fixation_paths = fixation_image_paths(annotation_dir)
        if not fixation_paths:
            continue
        video_path = find_video_path(root / "video", video_id)
        if video_path is None:
            continue
        rows.append(
            {
                "sample_id": f"dhf1k_{video_id}",
                "video_id": video_id,
                "split": dhf1k_split_for_video_id(video_id),
                "video_path": portable_video_path(video_path),
                "n_map_frames": len(annotation_map_paths(annotation_dir)),
                "n_fixation_frames": len(fixation_paths),
                "mean_map_intensity": None,
                "peak_map_intensity": None,
                "peak_to_mean_map_ratio": None,
                "mean_map_concentration": None,
                "mean_fixation_density": None,
                "selected_tail": "",
            }
        )
    return rows


def payloads_for_rows(
    rows: list[dict[str, Any]],
    root: Path,
) -> Iterable[dict[str, Any]]:
    for row in rows:
        annotation_dir = find_annotation_video_dir(
            root / "annotation",
            str(row["video_id"]),
        )
        fixation_paths = fixation_image_paths(annotation_dir)
        yield {
            "video_id": row["video_id"],
            "fixation_pngs": [path.read_bytes() for path in fixation_paths],
        }


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


def fixation_image_paths(annotation_video_dir: Path | None) -> list[Path]:
    if annotation_video_dir is None:
        return []
    return sorted((annotation_video_dir / "fixation").glob("*.png"))


def portable_video_path(video_path: Path) -> str:
    if not video_path.is_absolute():
        return str(video_path)
    resolved = video_path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def dhf1k_split_for_video_id(video_id: str) -> str:
    idx = int(video_id)
    if 1 <= idx <= 600:
        return "train"
    if 601 <= idx <= 700:
        return "val"
    return "unknown"


def select_extreme_tails(
    rows: list[dict[str, Any]],
    *,
    rank_column: str,
    count_per_tail: int,
) -> list[dict[str, Any]]:
    usable = [row for row in rows if row[rank_column] is not None]
    if len(usable) < count_per_tail * 2:
        raise ValueError(
            f"not enough finite {rank_column} rows for disjoint extreme tails: "
            f"{len(usable)} available, {count_per_tail * 2} required"
        )
    ordered = sorted(usable, key=lambda row: float(row[rank_column]))
    selected = []
    for row in ordered[:count_per_tail]:
        selected.append({**row, "selected_tail": "low"})
    for row in ordered[-count_per_tail:]:
        selected.append({**row, "selected_tail": "high"})
    return sorted(selected, key=lambda row: row["sample_id"])


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    dhf1k_root: Path,
    labels_csv: Path,
    extreme_count_per_tail: int,
) -> dict[str, Any]:
    metric_summaries = {
        column: summarize_metric([row[column] for row in rows])
        for column in RANK_COLUMNS
    }
    blocking_reasons = blocking_reasons_for_rows(
        rows,
        rank_column="mean_fixation_density",
        metric_summaries=metric_summaries,
        extreme_count_per_tail=extreme_count_per_tail,
    )
    return {
        "schema_version": 1,
        "experiment": "dhf1k_attention_label_audit",
        "dataset": "DHF1K",
        "source_root": str(dhf1k_root),
        "labels_csv": str(labels_csv),
        "split": "annotated",
        "rank_column": "mean_fixation_density",
        "metric_scope": "rank",
        "computed_metric_columns": ["mean_fixation_density"],
        "extreme_count_per_tail": extreme_count_per_tail,
        "min_rows": extreme_count_per_tail * 2,
        "min_distinct_rank_values": 3,
        "n_rows": len(rows),
        "metrics": metric_summaries,
        "rank_column_summary": metric_summaries["mean_fixation_density"],
        "rank_column_ready": not blocking_reasons,
        "ready_for_manifest_alignment": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "candidate_ground_truth_columns": [
            {
                "column": "mean_fixation_density",
                "n": metric_summaries["mean_fixation_density"]["n"],
                "n_distinct": metric_summaries["mean_fixation_density"][
                    "n_distinct"
                ],
                "std": metric_summaries["mean_fixation_density"]["std"],
            }
        ]
        if not blocking_reasons
        else [],
        "recommended_ground_truth_column": (
            "mean_fixation_density" if not blocking_reasons else None
        ),
        "claim_boundary": (
            "DHF1K labels are external gaze/fixation labels, not retention, "
            "dopamine, or executive-control measurements."
        ),
    }


def summarize_metric(values: list[object]) -> dict[str, float | int | None]:
    import numpy as np

    finite: list[float] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, str | int | float):
            finite.append(float(value))
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


def blocking_reasons_for_rows(
    rows: list[dict[str, Any]],
    *,
    rank_column: str,
    metric_summaries: dict[str, dict[str, float | int | None]],
    extreme_count_per_tail: int,
) -> list[str]:
    reasons: list[str] = []
    expected = extreme_count_per_tail * 2
    summary = metric_summaries[rank_column]
    if len(rows) != expected:
        reasons.append(f"row count {len(rows)} does not equal expected {expected}")
    if int(summary["n"] or 0) < len(rows):
        reasons.append(f"{rank_column} has non-finite rows")
    if int(summary["n_distinct"] or 0) < 3:
        reasons.append(f"{rank_column} has fewer than 3 distinct values")
    if float(summary["std"] or 0.0) <= 0.0:
        reasons.append(f"{rank_column} has zero variance")
    tail_counts = Counter(row["selected_tail"] for row in rows)
    if tail_counts["low"] != extreme_count_per_tail:
        reasons.append(f"low-tail row count {tail_counts['low']} is not {extreme_count_per_tail}")
    if tail_counts["high"] != extreme_count_per_tail:
        reasons.append(f"high-tail row count {tail_counts['high']} is not {extreme_count_per_tail}")
    return reasons


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row[column]) for column in CSV_COLUMNS})


def csv_value(value: object) -> object:
    if value is None:
        return ""
    return value
