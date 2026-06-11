"""Audit SnapUGC/VQualA-style retention labels before Phase 1 use."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

ID_COLUMN_HINTS = ("sample_id", "video_id", "video", "id")
GROUND_TRUTH_HINTS = ("ecr", "completion", "retention", "engagement")
MEDIA_PATH_HINTS = ("media_path", "video_path", "path", "url", "video_url", "filepath")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--dataset", default="SnapUGC")
    parser.add_argument("--sample-id-column", default=None)
    parser.add_argument("--ground-truth-column", default=None)
    parser.add_argument("--media-path-column", default=None)
    parser.add_argument("--ground-truth-name", default=None)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-distinct-ground-truth", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_retention_labels(
        labels_csv=args.labels_csv,
        dataset=args.dataset,
        sample_id_column=args.sample_id_column,
        ground_truth_column=args.ground_truth_column,
        media_path_column=args.media_path_column,
        ground_truth_name=args.ground_truth_name,
        min_samples=args.min_samples,
        min_distinct_ground_truth=args.min_distinct_ground_truth,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_retention_label_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not report["ready_for_manifest_alignment"]:
        raise SystemExit(1)


def audit_retention_labels(
    *,
    labels_csv: Path,
    dataset: str = "SnapUGC",
    sample_id_column: str | None = None,
    ground_truth_column: str | None = None,
    media_path_column: str | None = None,
    ground_truth_name: str | None = None,
    min_samples: int = 30,
    min_distinct_ground_truth: int = 3,
) -> dict[str, Any]:
    rows, header = read_csv_rows(labels_csv)
    resolved_sample_id = sample_id_column or choose_column(header, ID_COLUMN_HINTS)
    resolved_ground_truth = ground_truth_column or choose_column(
        header,
        GROUND_TRUTH_HINTS,
    )
    resolved_media_path = media_path_column or choose_column(header, MEDIA_PATH_HINTS)

    parsed_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sample_id = cell(row, resolved_sample_id)
        ground_truth = finite_float(cell(row, resolved_ground_truth))
        media_path = cell(row, resolved_media_path)
        parsed_rows.append(
            {
                "row_index": idx,
                "sample_id": sample_id,
                "ground_truth": ground_truth,
                "media_path": media_path,
            }
        )

    sample_ids = [str(row["sample_id"]) for row in parsed_rows if row["sample_id"]]
    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    finite_values = [
        float(row["ground_truth"])
        for row in parsed_rows
        if row["ground_truth"] is not None
    ]
    missing_ids = [row for row in parsed_rows if not row["sample_id"]]
    invalid_ground_truth = [
        row for row in parsed_rows if row["ground_truth"] is None
    ]
    missing_media_paths = [
        row for row in parsed_rows if resolved_media_path and not row["media_path"]
    ]
    blocking_reasons = retention_label_blocking_reasons(
        n_rows=len(rows),
        sample_id_column=resolved_sample_id,
        ground_truth_column=resolved_ground_truth,
        duplicate_ids=duplicate_ids,
        missing_ids=missing_ids,
        invalid_ground_truth=invalid_ground_truth,
        finite_values=finite_values,
        min_samples=min_samples,
        min_distinct_ground_truth=min_distinct_ground_truth,
    )
    ready = not blocking_reasons
    return {
        "schema_version": 1,
        "experiment": "attention_capture_retention_label_audit",
        "dataset": dataset,
        "labels_csv": str(labels_csv),
        "labels_csv_sha256": sha256(labels_csv.read_bytes()).hexdigest(),
        "columns": header,
        "sample_id_column": resolved_sample_id,
        "ground_truth_column": resolved_ground_truth,
        "ground_truth_name": ground_truth_name or resolved_ground_truth,
        "media_path_column": resolved_media_path,
        "candidate_ground_truth_columns": matching_columns(header, GROUND_TRUTH_HINTS),
        "candidate_media_path_columns": matching_columns(header, MEDIA_PATH_HINTS),
        "n_rows": len(rows),
        "n_missing_sample_ids": len(missing_ids),
        "n_duplicate_sample_ids": len(duplicate_ids),
        "duplicate_sample_ids": duplicate_ids[:20],
        "n_finite_ground_truth": len(finite_values),
        "n_invalid_ground_truth": len(invalid_ground_truth),
        "n_missing_media_paths": len(missing_media_paths),
        "ground_truth_summary": summarize_values(finite_values),
        "min_samples": min_samples,
        "min_distinct_ground_truth": min_distinct_ground_truth,
        "ready_for_manifest_alignment": ready,
        "ready_for_modal_feature_extraction": ready
        and bool(resolved_media_path)
        and not missing_media_paths,
        "blocking_reasons": blocking_reasons,
        "claim_boundary": (
            "This audit verifies external retention-label mechanics only. It "
            "does not score TRIBE features or validate attentional capture."
        ),
    }


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        return [dict(row) for row in reader], list(reader.fieldnames)


def choose_column(header: list[str], hints: tuple[str, ...]) -> str | None:
    matches = matching_columns(header, hints)
    return matches[0] if matches else None


def matching_columns(header: list[str], hints: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    for column in header:
        normalized = normalize(column)
        if normalized in hints or any(hint in normalized for hint in hints):
            matches.append(column)
    return matches


def cell(row: dict[str, str], column: str | None) -> str | None:
    if column is None:
        return None
    value = row.get(column)
    if value is None:
        return None
    value = value.strip()
    return value or None


def retention_label_blocking_reasons(
    *,
    n_rows: int,
    sample_id_column: str | None,
    ground_truth_column: str | None,
    duplicate_ids: list[str],
    missing_ids: list[dict[str, Any]],
    invalid_ground_truth: list[dict[str, Any]],
    finite_values: list[float],
    min_samples: int,
    min_distinct_ground_truth: int,
) -> list[str]:
    reasons: list[str] = []
    if sample_id_column is None:
        reasons.append("no sample-id column found")
    if ground_truth_column is None:
        reasons.append("no retention ground-truth column found")
    if n_rows < min_samples:
        reasons.append(f"row count {n_rows} is below minimum {min_samples}")
    if missing_ids:
        reasons.append(f"{len(missing_ids)} rows are missing sample ids")
    if duplicate_ids:
        reasons.append(f"{len(duplicate_ids)} duplicate sample ids found")
    if invalid_ground_truth:
        reasons.append(
            f"{len(invalid_ground_truth)} rows have non-finite ground truth"
        )
    if len(finite_values) < min_samples:
        reasons.append(
            f"finite ground-truth count {len(finite_values)} is below minimum {min_samples}"
        )
    if len(set(finite_values)) < min_distinct_ground_truth:
        reasons.append("ground truth has too few distinct finite values")
    return reasons


def render_retention_label_markdown(report: dict[str, Any]) -> str:
    summary = report["ground_truth_summary"]
    lines = [
        "# Retention Label Audit",
        "",
        "## Verdict",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Labels CSV: `{report['labels_csv']}`",
        f"- Ready for manifest alignment: {report['ready_for_manifest_alignment']}",
        (
            "- Ready for Modal feature extraction: "
            f"{report['ready_for_modal_feature_extraction']}"
        ),
        f"- Rows: {report['n_rows']}",
        f"- Sample-id column: `{report['sample_id_column'] or 'n/a'}`",
        f"- Ground-truth column: `{report['ground_truth_column'] or 'n/a'}`",
        f"- Media-path column: `{report['media_path_column'] or 'n/a'}`",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Blocking Reasons",
        "",
    ]
    blockers = report["blocking_reasons"]
    lines.extend(f"- {reason}" for reason in blockers) if blockers else lines.append(
        "- none"
    )
    lines.extend(
        [
            "",
            "## Ground Truth",
            "",
            f"- Finite rows: {report['n_finite_ground_truth']}",
            f"- Distinct values: {summary['n_distinct']}",
            f"- Mean: {fmt_float(summary['mean'])}",
            f"- Std: {fmt_float(summary['std'])}",
            f"- Min: {fmt_float(summary['min'])}",
            f"- Max: {fmt_float(summary['max'])}",
            "",
            "## Candidate Columns",
            "",
            (
                "- Ground truth: "
                f"{', '.join(report['candidate_ground_truth_columns']) or 'none'}"
            ),
            (
                "- Media path: "
                f"{', '.join(report['candidate_media_path_columns']) or 'none'}"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def finite_float(value: object) -> float | None:
    if not isinstance(value, int | float | str):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "n_distinct": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "n_distinct": len(set(values)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def fmt_float(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.6g}"


def normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
