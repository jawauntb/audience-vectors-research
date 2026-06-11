"""Run Phase 1 capture-score validation from Modal-volume TRIBE features."""

from __future__ import annotations

import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from audience_vectors.attention_capture_modal_volume import (
    DEFAULT_DHF1K_FULL_FEATURE_PREFIX,
    render_modal_phase1_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--roi-masks", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--ground-truth-column", default="mean_fixation_density")
    parser.add_argument("--dataset", default="DHF1K")
    parser.add_argument("--ground-truth-name", default="mean_fixation_density")
    parser.add_argument("--manifest-status", default="real_external_attention_labels")
    parser.add_argument("--label-audit", type=Path, default=None)
    parser.add_argument("--app-name", default=None)
    parser.add_argument("--output-prefix", default=DEFAULT_DHF1K_FULL_FEATURE_PREFIX)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-distinct-ground-truth", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gate-rho", type=float, default=0.40)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--omit-rows", action="store_true")
    parser.add_argument("--allow-missing-label-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    import modal  # type: ignore[import-not-found]

    from audience_vectors.modal_app.app import get_app_name

    args = parse_args()
    app_name = args.app_name or get_app_name()
    label_records = load_label_records_from_csv(
        labels_csv=args.labels_csv,
        sample_id_column=args.sample_id_column,
        ground_truth_column=args.ground_truth_column,
        dataset=args.dataset,
        limit=args.limit,
    )
    label_audit = load_label_audit_metadata(
        args.label_audit,
        labels_csv=args.labels_csv,
        ground_truth_column=args.ground_truth_column,
    )
    function = modal.Function.from_name(
        app_name,
        "score_attention_capture_phase1_modal_volume",
    )
    report = function.remote(
        label_records,
        args.roi_masks.read_bytes(),
        args.output_prefix,
        manifest_status=args.manifest_status,
        dataset=args.dataset,
        ground_truth_name=args.ground_truth_name,
        label_audit=label_audit,
        require_label_audit=not args.allow_missing_label_audit,
        min_samples=args.min_samples,
        min_distinct_ground_truth=args.min_distinct_ground_truth,
        permutations=args.permutations,
        seed=args.seed,
        gate_rho=args.gate_rho,
        epsilon=args.epsilon,
        include_rows=not args.omit_rows,
    )
    report["local_inputs"] = {
        "labels_csv": str(args.labels_csv),
        "labels_csv_sha256": sha256(args.labels_csv.read_bytes()).hexdigest(),
        "roi_masks": str(args.roi_masks),
        "roi_masks_sha256": sha256(args.roi_masks.read_bytes()).hexdigest(),
        "label_audit": str(args.label_audit) if args.label_audit else None,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_modal_phase1_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not report["score_decision"]["scoring_executed"]:
        raise SystemExit(1)


def load_label_records_from_csv(
    *,
    labels_csv: Path,
    sample_id_column: str,
    ground_truth_column: str,
    dataset: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "sample_id": required_cell(row, sample_id_column, labels_csv),
                "ground_truth": required_cell(row, ground_truth_column, labels_csv),
                "dataset": dataset,
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def load_label_audit_metadata(
    label_audit: Path | None,
    *,
    labels_csv: Path,
    ground_truth_column: str,
) -> dict[str, Any] | None:
    if label_audit is None:
        return None

    payload = json.loads(label_audit.read_text(encoding="utf-8"))
    reasons: list[str] = []
    experiment = payload.get("experiment")
    if experiment == "dhf1k_attention_label_audit":
        if payload.get("ready_for_manifest_alignment") is not True:
            reasons.append("label audit is not ready for manifest alignment")
        if payload.get("rank_column") != ground_truth_column:
            reasons.append("label audit rank_column differs from ground_truth_column")
    elif experiment == "attention_capture_retention_label_audit":
        if payload.get("ready_for_manifest_alignment") is not True:
            reasons.append("retention label audit is not ready for manifest alignment")
        audited_ground_truth = (
            payload.get("ground_truth_name") or payload.get("ground_truth_column")
        )
        if (
            isinstance(audited_ground_truth, str)
            and audited_ground_truth
            and audited_ground_truth.lower() != ground_truth_column.lower()
        ):
            reasons.append("retention label audit ground truth differs from scoring")
    else:
        reasons.append(
            "label audit experiment must be dhf1k_attention_label_audit or "
            "attention_capture_retention_label_audit"
        )

    audit_labels_csv = payload.get("labels_csv")
    if isinstance(audit_labels_csv, str) and audit_labels_csv:
        if Path(audit_labels_csv).resolve() != labels_csv.resolve():
            reasons.append("label audit labels_csv differs from scoring labels_csv")

    return {
        "path": str(label_audit),
        "sha256": sha256(label_audit.read_bytes()).hexdigest(),
        "experiment": experiment,
        "dataset": payload.get("dataset"),
        "labels_csv": audit_labels_csv,
        "ready_for_manifest_alignment": payload.get("ready_for_manifest_alignment"),
        "rank_column": payload.get("rank_column"),
        "ground_truth_column": payload.get("ground_truth_column"),
        "ground_truth_name": payload.get("ground_truth_name"),
        "recommended_ground_truth_column": payload.get(
            "recommended_ground_truth_column"
        ),
        "n_rows": payload.get("n_rows"),
        "blocking_reasons": reasons,
    }


def required_cell(row: dict[str, str], column: str, path: Path) -> str:
    value = row.get(column)
    if value is None or value == "":
        raise ValueError(f"{path} has a row missing required column {column!r}")
    return value


if __name__ == "__main__":
    main()
