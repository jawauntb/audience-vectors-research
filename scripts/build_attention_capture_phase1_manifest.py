"""Build a Phase 1 attention-capture manifest from labels and TRIBE features."""

from __future__ import annotations

import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a capture-score Phase 1 manifest from a label CSV and a "
            "directory of cached TRIBE feature NPZ files."
        ),
    )
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ground-truth-name", required=True)
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--ground-truth-column", default="ground_truth")
    parser.add_argument(
        "--feature-template",
        default="{sample_id}.npz",
        help=(
            "Filename template relative to --feature-dir. The token "
            "{sample_id} is replaced with the CSV sample id."
        ),
    )
    parser.add_argument(
        "--status",
        default="real_external_attention_labels",
        help="Manifest status used by the Phase 1 claim gate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for quick dry runs.",
    )
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="Skip rows whose expected feature file is absent.",
    )
    parser.add_argument(
        "--alignment-audit",
        type=Path,
        default=None,
        help=(
            "Optional output from audit_attention_capture_manifest_alignment.py. "
            "When supplied, it must be ready and match these manifest inputs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        labels_csv=args.labels_csv,
        feature_dir=args.feature_dir,
        dataset=args.dataset,
        ground_truth_name=args.ground_truth_name,
        sample_id_column=args.sample_id_column,
        ground_truth_column=args.ground_truth_column,
        feature_template=args.feature_template,
        status=args.status,
        limit=args.limit,
        allow_missing_features=args.allow_missing_features,
        alignment_audit=args.alignment_audit,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"wrote {len(manifest['samples'])} samples -> {args.output} "
        f"({manifest['metadata']['n_missing_features']} missing features)"
    )


def build_manifest(
    *,
    labels_csv: Path,
    feature_dir: Path,
    dataset: str,
    ground_truth_name: str,
    sample_id_column: str = "sample_id",
    ground_truth_column: str = "ground_truth",
    feature_template: str = "{sample_id}.npz",
    status: str = "real_external_attention_labels",
    limit: int | None = None,
    allow_missing_features: bool = False,
    alignment_audit: Path | None = None,
) -> dict[str, Any]:
    alignment_metadata = validate_alignment_audit(
        alignment_audit=alignment_audit,
        labels_csv=labels_csv,
        feature_dir=feature_dir,
        dataset=dataset,
        sample_id_column=sample_id_column,
        ground_truth_column=ground_truth_column,
        feature_template=feature_template,
    )
    rows = _read_label_rows(labels_csv)
    samples: list[dict[str, Any]] = []
    missing_features: list[dict[str, str]] = []

    for row in rows:
        sample_id = _required_cell(row, sample_id_column, labels_csv)
        ground_truth = float(_required_cell(row, ground_truth_column, labels_csv))
        feature_path = feature_dir / feature_template.format(sample_id=sample_id)
        if not feature_path.exists():
            missing_features.append(
                {
                    "sample_id": sample_id,
                    "expected_feature_path": str(feature_path),
                }
            )
            if not allow_missing_features:
                continue

        if feature_path.exists():
            samples.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "ground_truth": ground_truth,
                    "ground_truth_name": ground_truth_name,
                    "tribe_feature_path": str(feature_path.resolve()),
                }
            )
        if limit is not None and len(samples) >= limit:
            break

    if missing_features and not allow_missing_features:
        preview = ", ".join(item["sample_id"] for item in missing_features[:5])
        raise FileNotFoundError(
            f"{len(missing_features)} feature files were missing; first ids: {preview}"
        )

    return {
        "schema_version": 1,
        "experiment": "phase1_capture_score_validation",
        "status": status,
        "dataset": dataset,
        "ground_truth_name": ground_truth_name,
        "feature_source": {
            "feature_dir": str(feature_dir),
            "feature_template": feature_template,
        },
        "roi_mask_recommendation": (
            "Use results/destrieux_roi_masks_disjoint_20260608.npz for the "
            "primary Phase 1 run; keep overlapping masks as sensitivity only."
        ),
        "samples": samples,
        "metadata": {
            "labels_csv": str(labels_csv),
            "sample_id_column": sample_id_column,
            "ground_truth_column": ground_truth_column,
            "n_label_rows": len(rows),
            "n_samples": len(samples),
            "n_missing_features": len(missing_features),
            "missing_features": missing_features,
            "alignment_audit": alignment_metadata,
        },
    }


def validate_alignment_audit(
    *,
    alignment_audit: Path | None,
    labels_csv: Path,
    feature_dir: Path,
    dataset: str,
    sample_id_column: str,
    ground_truth_column: str,
    feature_template: str,
) -> dict[str, Any] | None:
    if alignment_audit is None:
        return None

    payload = json.loads(alignment_audit.read_text(encoding="utf-8"))
    if payload.get("experiment") != "phase1_manifest_alignment_audit":
        raise ValueError(f"{alignment_audit} is not a Phase 1 alignment audit")
    if not payload.get("ready_for_manifest_build"):
        reasons = payload.get("blocking_reasons") or []
        reason_text = "; ".join(str(reason) for reason in reasons) or "unknown reason"
        raise ValueError(f"{alignment_audit} is not ready: {reason_text}")

    mismatches = alignment_audit_mismatches(
        payload,
        labels_csv=labels_csv,
        feature_dir=feature_dir,
        dataset=dataset,
        sample_id_column=sample_id_column,
        ground_truth_column=ground_truth_column,
        feature_template=feature_template,
    )
    if mismatches:
        raise ValueError(
            f"{alignment_audit} does not match manifest inputs: "
            + "; ".join(mismatches)
        )

    return {
        "path": str(alignment_audit),
        "sha256": sha256(alignment_audit.read_bytes()).hexdigest(),
        "ready_for_manifest_build": True,
        "n_aligned_features": int(payload.get("n_aligned_features") or 0),
        "n_missing_features": int(payload.get("n_missing_features") or 0),
        "ground_truth_summary": payload.get("ground_truth_summary"),
        "label_audit": payload.get("label_audit"),
    }


def alignment_audit_mismatches(
    payload: dict[str, Any],
    *,
    labels_csv: Path,
    feature_dir: Path,
    dataset: str,
    sample_id_column: str,
    ground_truth_column: str,
    feature_template: str,
) -> list[str]:
    mismatches: list[str] = []
    if not same_path(payload.get("labels_csv"), labels_csv):
        mismatches.append("labels_csv differs")
    if not same_path(payload.get("feature_dir"), feature_dir):
        mismatches.append("feature_dir differs")
    if payload.get("sample_id_column") != sample_id_column:
        mismatches.append("sample_id_column differs")
    if payload.get("ground_truth_column") != ground_truth_column:
        mismatches.append("ground_truth_column differs")
    if payload.get("feature_template") != feature_template:
        mismatches.append("feature_template differs")
    audit_dataset = str(payload.get("dataset") or "unknown")
    if audit_dataset not in ("unknown", dataset):
        mismatches.append("dataset differs")
    return mismatches


def same_path(value: object, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return Path(value).expanduser().resolve() == expected.expanduser().resolve()


def _read_label_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        return [dict(row) for row in reader]


def _required_cell(row: dict[str, str], column: str, path: Path) -> str:
    value = row.get(column)
    if value is None or value == "":
        raise ValueError(f"{path} has a row missing required column {column!r}")
    return value


if __name__ == "__main__":
    main()
