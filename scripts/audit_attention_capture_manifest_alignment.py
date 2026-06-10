"""Audit label-to-feature alignment before building a Phase 1 manifest."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--label-audit",
        type=Path,
        default=None,
        help=(
            "Optional upstream label audit, such as the JSON output from "
            "build_dhf1k_attention_labels.py."
        ),
    )
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--ground-truth-column", default="ground_truth")
    parser.add_argument("--feature-template", default="{sample_id}.npz")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-distinct-ground-truth", type=int, default=3)
    parser.add_argument("--preview-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_manifest_alignment(
        labels_csv=args.labels_csv,
        feature_dir=args.feature_dir,
        label_audit=args.label_audit,
        sample_id_column=args.sample_id_column,
        ground_truth_column=args.ground_truth_column,
        feature_template=args.feature_template,
        dataset=args.dataset,
        min_samples=args.min_samples,
        min_distinct_ground_truth=args.min_distinct_ground_truth,
        preview_limit=args.preview_limit,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_alignment_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not report["ready_for_manifest_build"]:
        raise SystemExit(1)


def audit_manifest_alignment(
    *,
    labels_csv: Path,
    feature_dir: Path,
    label_audit: Path | None = None,
    sample_id_column: str = "sample_id",
    ground_truth_column: str = "ground_truth",
    feature_template: str = "{sample_id}.npz",
    dataset: str = "unknown",
    min_samples: int = 30,
    min_distinct_ground_truth: int = 3,
    preview_limit: int = 20,
) -> dict[str, Any]:
    rows = read_label_rows(labels_csv)
    samples = [
        parse_label_row(
            row,
            labels_csv=labels_csv,
            sample_id_column=sample_id_column,
            ground_truth_column=ground_truth_column,
            feature_dir=feature_dir,
            feature_template=feature_template,
        )
        for row in rows
    ]
    sample_ids = [sample["sample_id"] for sample in samples if sample["sample_id"]]
    duplicate_ids = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    finite_ground_truth = [
        sample["ground_truth"]
        for sample in samples
        if sample["ground_truth"] is not None and math.isfinite(sample["ground_truth"])
    ]
    existing = [sample for sample in samples if sample["feature_exists"]]
    missing = [sample for sample in samples if not sample["feature_exists"]]
    invalid_ground_truth = [sample for sample in samples if sample["ground_truth"] is None]
    label_audit_metadata = validate_label_audit(
        label_audit=label_audit,
        labels_csv=labels_csv,
        dataset=dataset,
        sample_id_column=sample_id_column,
        ground_truth_column=ground_truth_column,
    )
    blocking_reasons = alignment_blocking_reasons(
        n_aligned=len(existing),
        n_duplicate_ids=len(duplicate_ids),
        n_invalid_ground_truth=len(invalid_ground_truth),
        finite_ground_truth=finite_ground_truth,
        min_samples=min_samples,
        min_distinct_ground_truth=min_distinct_ground_truth,
    )
    blocking_reasons.extend(label_audit_metadata["blocking_reasons"])

    return {
        "schema_version": 1,
        "experiment": "phase1_manifest_alignment_audit",
        "dataset": dataset,
        "labels_csv": str(labels_csv),
        "feature_dir": str(feature_dir),
        "label_audit": label_audit_metadata,
        "sample_id_column": sample_id_column,
        "ground_truth_column": ground_truth_column,
        "feature_template": feature_template,
        "n_label_rows": len(rows),
        "n_unique_sample_ids": len(set(sample_ids)),
        "n_duplicate_sample_ids": len(duplicate_ids),
        "duplicate_sample_ids": duplicate_ids[:preview_limit],
        "n_aligned_features": len(existing),
        "n_missing_features": len(missing),
        "n_invalid_ground_truth": len(invalid_ground_truth),
        "ground_truth_summary": summarize_values(finite_ground_truth),
        "min_samples": min_samples,
        "min_distinct_ground_truth": min_distinct_ground_truth,
        "ready_for_manifest_build": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "missing_features": [
            preview_sample(sample) for sample in missing[:preview_limit]
        ],
        "invalid_ground_truth_rows": [
            preview_sample(sample) for sample in invalid_ground_truth[:preview_limit]
        ],
        "claim_boundary": (
            "This audit checks CSV/feature alignment only. It does not score "
            "TRIBE features or validate attentional capture."
        ),
    }


def validate_label_audit(
    *,
    label_audit: Path | None,
    labels_csv: Path,
    dataset: str,
    sample_id_column: str,
    ground_truth_column: str,
) -> dict[str, Any]:
    if label_audit is None:
        return {
            "path": None,
            "sha256": None,
            "experiment": None,
            "ready_for_manifest_alignment": None,
            "labels_csv_relation": None,
            "rank_column": None,
            "recommended_ground_truth_column": None,
            "n_rows": None,
            "blocking_reasons": [],
        }

    payload = json.loads(label_audit.read_text(encoding="utf-8"))
    reasons: list[str] = []
    experiment = payload.get("experiment")
    if experiment != "dhf1k_attention_label_audit":
        reasons.append(f"label audit experiment {experiment!r} is not supported")
    if not payload.get("ready_for_manifest_alignment"):
        upstream_reasons = payload.get("blocking_reasons") or []
        reason_text = "; ".join(str(reason) for reason in upstream_reasons)
        if reason_text:
            reasons.append(f"label audit is not ready: {reason_text}")
        else:
            reasons.append("label audit is not ready")

    audit_labels_csv = payload.get("labels_csv")
    labels_csv_relation = label_csv_relation(
        audit_labels_csv,
        labels_csv,
        sample_id_column=sample_id_column,
    )
    if audit_labels_csv and labels_csv_relation == "mismatch":
        reasons.append(
            "label audit labels_csv is neither the alignment labels_csv nor "
            "an exact row superset"
        )

    audit_dataset = str(payload.get("dataset") or "unknown")
    if dataset != "unknown" and audit_dataset not in ("unknown", dataset):
        reasons.append("label audit dataset differs from alignment dataset")

    rank_column = payload.get("rank_column")
    if rank_column and rank_column != ground_truth_column:
        reasons.append(
            "label audit rank_column differs from alignment ground_truth_column"
        )

    return {
        "path": str(label_audit),
        "sha256": sha256(label_audit.read_bytes()).hexdigest(),
        "experiment": experiment,
        "ready_for_manifest_alignment": payload.get("ready_for_manifest_alignment"),
        "labels_csv_relation": labels_csv_relation,
        "rank_column": rank_column,
        "recommended_ground_truth_column": payload.get(
            "recommended_ground_truth_column"
        ),
        "n_rows": payload.get("n_rows"),
        "blocking_reasons": reasons,
    }


def read_label_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        return [dict(row) for row in reader]


def parse_label_row(
    row: dict[str, str],
    *,
    labels_csv: Path,
    sample_id_column: str,
    ground_truth_column: str,
    feature_dir: Path,
    feature_template: str,
) -> dict[str, Any]:
    sample_id = required_cell(row, sample_id_column, labels_csv)
    raw_ground_truth = row.get(ground_truth_column)
    ground_truth = finite_float(raw_ground_truth)
    feature_path = feature_dir / feature_template.format(sample_id=sample_id)
    return {
        "sample_id": sample_id,
        "raw_ground_truth": raw_ground_truth,
        "ground_truth": ground_truth,
        "feature_path": str(feature_path),
        "feature_exists": feature_path.exists(),
    }


def alignment_blocking_reasons(
    *,
    n_aligned: int,
    n_duplicate_ids: int,
    n_invalid_ground_truth: int,
    finite_ground_truth: list[float],
    min_samples: int,
    min_distinct_ground_truth: int,
) -> list[str]:
    reasons: list[str] = []
    if n_aligned < min_samples:
        reasons.append(f"aligned feature count {n_aligned} is below minimum {min_samples}")
    if n_duplicate_ids:
        reasons.append(f"{n_duplicate_ids} duplicate sample ids found")
    if n_invalid_ground_truth:
        reasons.append(f"{n_invalid_ground_truth} rows have non-finite ground truth")
    if len(set(finite_ground_truth)) < min_distinct_ground_truth:
        reasons.append(
            "distinct finite ground-truth count "
            f"{len(set(finite_ground_truth))} is below minimum "
            f"{min_distinct_ground_truth}"
        )
    if finite_ground_truth and float(np.std(finite_ground_truth)) <= 0.0:
        reasons.append("ground truth has zero variance")
    return reasons


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "n_distinct": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "n_distinct": len(set(values)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def render_alignment_markdown(report: dict[str, Any]) -> str:
    summary = report["ground_truth_summary"]
    lines = [
        "# Phase 1 Manifest Alignment Audit",
        "",
        "## Verdict",
        "",
        f"- Dataset: {report['dataset']}",
        f"- Ready for manifest build: {report['ready_for_manifest_build']}",
        f"- Label audit ready: {report['label_audit']['ready_for_manifest_alignment']}",
        f"- Label audit labels relation: {report['label_audit']['labels_csv_relation'] or 'n/a'}",
        f"- Label audit rank column: {report['label_audit']['rank_column'] or 'n/a'}",
        f"- Label rows: {report['n_label_rows']}",
        f"- Unique sample ids: {report['n_unique_sample_ids']}",
        f"- Aligned feature files: {report['n_aligned_features']}",
        f"- Missing feature files: {report['n_missing_features']}",
        f"- Invalid ground truth rows: {report['n_invalid_ground_truth']}",
        f"- Distinct ground-truth values: {summary['n_distinct']}",
        f"- Ground-truth std: {format_optional_float(summary['std'])}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Blocking Reasons",
        "",
    ]
    reasons = report["blocking_reasons"]
    if reasons:
        lines.extend(f"- {reason}" for reason in reasons)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Missing Feature Preview",
            "",
            "| sample_id | expected feature |",
            "|---|---|",
        ]
    )
    missing = report["missing_features"]
    if not missing:
        lines.append("| none | n/a |")
    for item in missing:
        lines.append(f"| {item['sample_id']} | {item['feature_path']} |")

    lines.extend(
        [
            "",
            "## Invalid Ground Truth Preview",
            "",
            "| sample_id | raw ground truth |",
            "|---|---|",
        ]
    )
    invalid = report["invalid_ground_truth_rows"]
    if not invalid:
        lines.append("| none | n/a |")
    for item in invalid:
        lines.append(f"| {item['sample_id']} | {item['raw_ground_truth']} |")

    return "\n".join(lines) + "\n"


def preview_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "raw_ground_truth": sample["raw_ground_truth"],
        "feature_path": sample["feature_path"],
    }


def required_cell(row: dict[str, str], column: str, path: Path) -> str:
    value = row.get(column)
    if value is None or value == "":
        raise ValueError(f"{path} has a row missing required column {column!r}")
    return value


def same_path(value: object, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return Path(value).expanduser().resolve() == expected.expanduser().resolve()


def label_csv_relation(
    audit_labels_csv: object,
    labels_csv: Path,
    *,
    sample_id_column: str,
) -> str | None:
    if not isinstance(audit_labels_csv, str) or not audit_labels_csv:
        return None
    audit_path = Path(audit_labels_csv).expanduser().resolve()
    labels_path = labels_csv.expanduser().resolve()
    if audit_path == labels_path:
        return "same"
    if not audit_path.exists() or not labels_path.exists():
        return "mismatch"

    audit_rows = read_label_rows(audit_path)
    subset_rows = read_label_rows(labels_path)
    by_sample_id = {
        required_cell(row, sample_id_column, audit_path): row for row in audit_rows
    }
    for row in subset_rows:
        sample_id = required_cell(row, sample_id_column, labels_path)
        audit_row = by_sample_id.get(sample_id)
        if audit_row is None:
            return "mismatch"
        for key, value in row.items():
            if audit_row.get(key) != value:
                return "mismatch"
    return "subset"


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
