"""Phase 1 scoring helpers for TRIBE features stored in a Modal Volume."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from audience_vectors.attention_capture import (
    DEFAULT_GATE_RHO,
    CaptureRow,
    capture_scores_from_roi_values,
    load_tribe_feature_mean,
    render_phase1_markdown,
    roi_values_from_feature_vector,
    run_capture_rows,
)

DEFAULT_MODAL_FEATURES_VOLUME_NAME = "attention-capture-features-v1"
DEFAULT_MODAL_FEATURES_MOUNT = "/attention-capture-features"
DEFAULT_DHF1K_FULL_FEATURE_PREFIX = (
    "attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610"
)


@dataclass(frozen=True)
class ModalVolumeLabelRecord:
    sample_id: str
    ground_truth: float
    dataset: str


def load_roi_masks_npz_bytes(payload: bytes) -> dict[str, np.ndarray]:
    """Load ROI masks from serialized NPZ bytes."""

    with np.load(BytesIO(payload), allow_pickle=False) as masks:
        return {name: np.asarray(masks[name], dtype=bool) for name in masks.files}


def run_phase1_modal_volume_features(
    *,
    label_records: list[dict[str, Any]],
    roi_masks: dict[str, np.ndarray],
    feature_root: Path,
    output_prefix: str,
    modal_volume_name: str = DEFAULT_MODAL_FEATURES_VOLUME_NAME,
    modal_mount: str = DEFAULT_MODAL_FEATURES_MOUNT,
    manifest_status: str = "real_external_attention_labels",
    dataset: str = "DHF1K",
    ground_truth_name: str = "mean_fixation_density",
    label_audit: dict[str, Any] | None = None,
    require_label_audit: bool = True,
    min_samples: int = 30,
    min_distinct_ground_truth: int = 3,
    permutations: int = 999,
    seed: int = 17,
    gate_rho: float = DEFAULT_GATE_RHO,
    epsilon: float = 1e-6,
    include_rows: bool = True,
) -> dict[str, Any]:
    """Score a Phase 1 dataset directly from a Modal-mounted feature prefix."""

    labels, invalid_labels = _normalize_label_records(
        label_records,
        default_dataset=dataset,
    )
    duplicate_ids = sorted(
        sample_id
        for sample_id, count in Counter(record.sample_id for record in labels).items()
        if count > 1
    )
    rows: list[CaptureRow] = []
    missing_features: list[dict[str, str]] = []
    feature_errors: list[dict[str, str]] = []
    shape_counts: dict[str, int] = {}

    feature_dir = feature_root / output_prefix.strip("/")
    for record in labels:
        feature_path = feature_dir / f"{record.sample_id}.npz"
        if not feature_path.exists():
            missing_features.append(
                {"sample_id": record.sample_id, "path": str(feature_path)}
            )
            continue
        try:
            feature_vector = load_tribe_feature_mean(feature_path)
            shape_key = "x".join(str(dim) for dim in feature_vector.shape)
            shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
            roi_values = roi_values_from_feature_vector(feature_vector, roi_masks)
            scores = capture_scores_from_roi_values(roi_values, epsilon=epsilon)
        except (OSError, ValueError) as exc:
            feature_errors.append(
                {
                    "sample_id": record.sample_id,
                    "path": str(feature_path),
                    "error": str(exc),
                }
            )
            continue

        rows.append(
            CaptureRow(
                sample_id=record.sample_id,
                dataset=record.dataset,
                ground_truth=record.ground_truth,
                roi_values=roi_values,
                sensory_mean=float(scores["sensory_mean"]),
                capture_score=scores["capture_score"],
                capture_delta=float(scores["capture_delta"]),
                frontoparietal=float(scores["frontoparietal"]),
                denominator_valid=bool(scores["denominator_valid"]),
            )
        )

    finite_ground_truth = [record.ground_truth for record in labels]
    valid_denominators = sum(1 for row in rows if row.denominator_valid)
    label_audit_reasons = _label_audit_blocking_reasons(
        label_audit=label_audit,
        require_label_audit=require_label_audit,
        ground_truth_name=ground_truth_name,
    )
    blocking_reasons = _modal_phase1_blocking_reasons(
        n_label_records=len(label_records),
        n_labels=len(labels),
        n_rows=len(rows),
        duplicate_ids=duplicate_ids,
        invalid_labels=invalid_labels,
        missing_features=missing_features,
        feature_errors=feature_errors,
        finite_ground_truth=finite_ground_truth,
        valid_denominators=valid_denominators,
        min_samples=min_samples,
        min_distinct_ground_truth=min_distinct_ground_truth,
        label_audit_reasons=label_audit_reasons,
    )

    primary_report = None
    if not invalid_labels and not duplicate_ids and not missing_features and not feature_errors:
        primary_report = run_capture_rows(
            rows,
            manifest_path=f"modal-volume://{modal_volume_name}/{output_prefix.strip('/')}",
            manifest_status=manifest_status,
            permutations=permutations,
            seed=seed,
            gate_rho=gate_rho,
            include_rows=include_rows,
        )

    mechanical_ready = not blocking_reasons
    claim_update_allowed = (
        bool(primary_report["claim_update_allowed"])
        if isinstance(primary_report, dict)
        else False
    )
    claim_validated = bool(
        mechanical_ready
        and claim_update_allowed
        and isinstance(primary_report, dict)
        and primary_report["gate"]["claim_validated"]
    )
    return {
        "schema_version": 1,
        "experiment": "phase1_capture_score_modal_volume_workflow",
        "modal_volume_name": modal_volume_name,
        "modal_mount": modal_mount,
        "feature_root": str(feature_root),
        "output_prefix": output_prefix,
        "manifest_status": manifest_status,
        "dataset": dataset,
        "ground_truth_name": ground_truth_name,
        "n_label_records": len(label_records),
        "n_valid_labels": len(labels),
        "n_scored_rows": len(rows),
        "min_samples": min_samples,
        "min_distinct_ground_truth": min_distinct_ground_truth,
        "gate_rho": gate_rho,
        "permutations": permutations,
        "seed": seed,
        "mechanical_ready": mechanical_ready,
        "claim_update_allowed": claim_update_allowed,
        "claim_validated": claim_validated,
        "score_decision": _score_decision(
            primary_report=primary_report,
            mechanical_ready=mechanical_ready,
            valid_denominators=valid_denominators,
            min_samples=min_samples,
        ),
        "blocking_reasons": blocking_reasons,
        "warnings": [],
        "label_audit": label_audit
        or {
            "path": None,
            "ready_for_manifest_alignment": None,
            "blocking_reasons": label_audit_reasons,
        },
        "feature_audit": {
            "n_feature_path_samples": len(labels),
            "n_existing": len(rows),
            "n_missing": len(missing_features),
            "n_feature_errors": len(feature_errors),
            "n_valid_capture_denominators": valid_denominators,
            "n_invalid_capture_denominators": len(rows) - valid_denominators,
            "feature_shape_counts": shape_counts,
            "missing_features": missing_features[:20],
            "feature_errors": feature_errors[:20],
        },
        "label_audit_summary": {
            "n_invalid_labels": len(invalid_labels),
            "n_duplicate_sample_ids": len(duplicate_ids),
            "duplicate_sample_ids": duplicate_ids[:20],
            "invalid_labels": invalid_labels[:20],
            "ground_truth_summary": _summarize_values(finite_ground_truth),
        },
        "primary_report": primary_report,
        "claim_boundary": (
            "This workflow scores ROI aggregates from full-mode TRIBE features "
            "already stored in a Modal Volume. It tests the preregistered H2 "
            "gate for this dataset only; publication-strength claims still "
            "require an additional claim-ready real dataset."
        ),
    }


def render_modal_phase1_markdown(report: dict[str, Any]) -> str:
    """Render the Modal-volume Phase 1 report."""

    lines = [
        "# Phase 1 Modal-Volume Capture-Score Workflow",
        "",
        "## Verdict",
        "",
        f"- Mechanical ready: {report['mechanical_ready']}",
        f"- Claim update allowed: {report['claim_update_allowed']}",
        f"- Claim validated: {report['claim_validated']}",
        f"- Scoring executed: {report['score_decision']['scoring_executed']}",
        f"- Decision reason: {report['score_decision']['reason']}",
        f"- Dataset: `{report['dataset']}`",
        f"- Ground truth: `{report['ground_truth_name']}`",
        f"- Modal volume: `{report['modal_volume_name']}`",
        f"- Output prefix: `{report['output_prefix']}`",
        f"- Scored rows: {report['n_scored_rows']}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Blocking Reasons",
        "",
    ]
    blockers = report.get("blocking_reasons") or []
    lines.extend(f"- {reason}" for reason in blockers) if blockers else lines.append(
        "- none"
    )

    feature_audit = report["feature_audit"]
    lines.extend(
        [
            "",
            "## Feature Audit",
            "",
            f"- Existing features: {feature_audit['n_existing']}",
            f"- Missing features: {feature_audit['n_missing']}",
            f"- Feature errors: {feature_audit['n_feature_errors']}",
            (
                "- Valid capture denominators: "
                f"{feature_audit['n_valid_capture_denominators']}"
            ),
            (
                "- Invalid capture denominators: "
                f"{feature_audit['n_invalid_capture_denominators']}"
            ),
        ]
    )

    primary = report.get("primary_report")
    if isinstance(primary, dict):
        lines.extend(["", *render_phase1_markdown(primary).splitlines()])
    return "\n".join(lines) + "\n"


def _normalize_label_records(
    label_records: list[dict[str, Any]],
    *,
    default_dataset: str,
) -> tuple[list[ModalVolumeLabelRecord], list[dict[str, str]]]:
    labels: list[ModalVolumeLabelRecord] = []
    invalid: list[dict[str, str]] = []
    for idx, record in enumerate(label_records):
        sample_id = record.get("sample_id")
        ground_truth = _finite_float(record.get("ground_truth"))
        if not isinstance(sample_id, str) or not sample_id:
            invalid.append({"row": str(idx), "error": "missing sample_id"})
            continue
        if ground_truth is None:
            invalid.append({"sample_id": sample_id, "error": "invalid ground_truth"})
            continue
        labels.append(
            ModalVolumeLabelRecord(
                sample_id=sample_id,
                ground_truth=ground_truth,
                dataset=str(record.get("dataset") or default_dataset),
            )
        )
    return labels, invalid


def _modal_phase1_blocking_reasons(
    *,
    n_label_records: int,
    n_labels: int,
    n_rows: int,
    duplicate_ids: list[str],
    invalid_labels: list[dict[str, str]],
    missing_features: list[dict[str, str]],
    feature_errors: list[dict[str, str]],
    finite_ground_truth: list[float],
    valid_denominators: int,
    min_samples: int,
    min_distinct_ground_truth: int,
    label_audit_reasons: list[str],
) -> list[str]:
    reasons: list[str] = []
    if n_label_records < min_samples:
        reasons.append(f"label record count {n_label_records} is below minimum {min_samples}")
    if n_labels < min_samples:
        reasons.append(f"valid label count {n_labels} is below minimum {min_samples}")
    if n_rows < min_samples:
        reasons.append(f"scored row count {n_rows} is below minimum {min_samples}")
    if duplicate_ids:
        reasons.append(f"{len(duplicate_ids)} duplicate sample ids")
    if invalid_labels:
        reasons.append(f"{len(invalid_labels)} labels are invalid")
    if missing_features:
        reasons.append(f"{len(missing_features)} feature files are missing")
    if feature_errors:
        reasons.append(f"{len(feature_errors)} feature files could not be scored")
    if len(set(finite_ground_truth)) < min_distinct_ground_truth:
        reasons.append("ground truth has too few distinct finite values")
    if valid_denominators < min_samples:
        reasons.append(
            f"valid capture denominator count {valid_denominators} is below minimum {min_samples}"
        )
    reasons.extend(label_audit_reasons)
    return reasons


def _label_audit_blocking_reasons(
    *,
    label_audit: dict[str, Any] | None,
    require_label_audit: bool,
    ground_truth_name: str,
) -> list[str]:
    if not require_label_audit:
        return []
    if not isinstance(label_audit, dict):
        return ["claim-ready Modal-volume scoring requires a ready external-label audit"]
    experiment = label_audit.get("experiment")
    reasons = [str(reason) for reason in label_audit.get("blocking_reasons") or []]
    if experiment == "dhf1k_attention_label_audit":
        if label_audit.get("ready_for_manifest_alignment") is not True:
            reasons.append("label audit is not ready for manifest alignment")
        rank_column = label_audit.get("rank_column")
        if rank_column and rank_column != ground_truth_name:
            reasons.append("label audit rank column differs from ground truth")
        return reasons
    if experiment == "attention_capture_retention_label_audit":
        if label_audit.get("ready_for_manifest_alignment") is not True:
            reasons.append("retention label audit is not ready for manifest alignment")
        audited_ground_truth = (
            label_audit.get("ground_truth_name")
            or label_audit.get("ground_truth_column")
        )
        if (
            isinstance(audited_ground_truth, str)
            and audited_ground_truth
            and audited_ground_truth.lower() != ground_truth_name.lower()
        ):
            reasons.append("retention label audit ground truth differs from scoring")
        return reasons
    reasons.append(
        "label audit experiment must be dhf1k_attention_label_audit or "
        "attention_capture_retention_label_audit"
    )
    if label_audit.get("ready_for_manifest_alignment") is not True:
        reasons.append("label audit is not ready for manifest alignment")
    return reasons


def _score_decision(
    *,
    primary_report: dict[str, Any] | None,
    mechanical_ready: bool,
    valid_denominators: int,
    min_samples: int,
) -> dict[str, Any]:
    if primary_report is None:
        return {"scoring_executed": False, "reason": "preflight_failed"}
    if not mechanical_ready and valid_denominators < min_samples:
        return {
            "scoring_executed": True,
            "reason": "diagnostic_only_denominator_gate_failed",
        }
    if not mechanical_ready:
        return {"scoring_executed": True, "reason": "diagnostic_only_preflight_failed"}
    if primary_report["gate"]["claim_validated"]:
        return {"scoring_executed": True, "reason": "claim_validated"}
    return {"scoring_executed": True, "reason": "claim_ready_gate_failed"}


def _summarize_values(values: list[float]) -> dict[str, float | int | None]:
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


def _finite_float(value: object) -> float | None:
    if not isinstance(value, int | float | str):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None
