"""Attention-capture scoring helpers for TRIBE feature dry runs.

The module intentionally treats TRIBE as a black-box cortical feature source.
It does not claim that a computed score is measured attention, dopamine, or
executive control. The first valid gate is whether a predeclared ROI-derived
proxy correlates with external attention labels.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np

SENSORY_ROIS = ("V1", "PPA", "language")
CONTROL_ROI = "frontoparietal"
REQUIRED_ROIS = (*SENSORY_ROIS, CONTROL_ROI)
DEFAULT_GATE_RHO = 0.40

Alternative = Literal["greater", "two-sided"]
OverlapPolicy = Literal["allow", "drop_shared"]


@dataclass(frozen=True)
class ROIGroupSpec:
    """Substring definition for an exploratory atlas-derived ROI group."""

    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()


class CaptureScoreValues(TypedDict):
    sensory_mean: float
    frontoparietal: float
    capture_score: float | None
    capture_delta: float
    denominator_valid: bool


@dataclass(frozen=True)
class ROISelection:
    """ROI masks plus the atlas labels that produced them."""

    masks: dict[str, np.ndarray]
    selected_labels: dict[str, tuple[str, ...]]
    n_vertices: int
    n_labels: int
    overlap_policy: OverlapPolicy = "allow"


DEFAULT_DESTRIEUX_ROI_GROUPS: dict[str, ROIGroupSpec] = {
    # Exploratory defaults only. These are deliberately broad parcel families,
    # not a substitute for a preregistered neuroscientific ROI definition.
    "V1": ROIGroupSpec(include=("calcar", "occipital", "lingual")),
    "PPA": ROIGroupSpec(include=("parahip", "oc-temp_med")),
    "language": ROIGroupSpec(
        include=(
            "temporal",
            "front_inf",
            "front_middle",
            "s_circular_insula",
        ),
    ),
    "frontoparietal": ROIGroupSpec(
        include=(
            "front_sup",
            "front_middle",
            "intrapariet",
            "parietal_sup",
            "precentral",
        ),
        exclude=("insula",),
    ),
}


@dataclass(frozen=True)
class CaptureRow:
    """One scored sample from a Phase 1 manifest."""

    sample_id: str
    dataset: str
    ground_truth: float
    roi_values: dict[str, float]
    sensory_mean: float
    capture_score: float | None
    capture_delta: float
    frontoparietal: float
    denominator_valid: bool


def capture_scores_from_roi_values(
    roi_values: dict[str, float],
    *,
    epsilon: float = 1e-6,
) -> CaptureScoreValues:
    """Compute proposal-v2 scores from named ROI values.

    The primary proposal score is
    mean(V1, PPA, language) / (frontoparietal + epsilon).

    TRIBE outputs may be signed depending on preprocessing and ROI aggregation,
    so ratios with non-positive denominators are flagged and withheld from the
    primary correlation. The additive contrast remains available as a secondary
    stress-test metric.
    """

    missing = [roi for roi in REQUIRED_ROIS if roi not in roi_values]
    if missing:
        raise ValueError(f"missing ROI values: {missing}")

    sensory_mean = float(np.mean([float(roi_values[roi]) for roi in SENSORY_ROIS]))
    frontoparietal = float(roi_values[CONTROL_ROI])
    denominator = frontoparietal + epsilon
    denominator_valid = frontoparietal > 0.0 and abs(denominator) > epsilon
    capture_score = sensory_mean / denominator if denominator_valid else None

    return {
        "sensory_mean": sensory_mean,
        "frontoparietal": frontoparietal,
        "capture_score": capture_score,
        "capture_delta": sensory_mean - frontoparietal,
        "denominator_valid": denominator_valid,
    }


def load_tribe_feature_mean(path: Path) -> np.ndarray:
    """Load a TRIBE feature NPZ and return a mean 20,484-vertex vector."""

    payload = np.load(path, allow_pickle=False)
    if "frames" not in payload:
        available = ", ".join(payload.files)
        raise ValueError(f"{path} missing 'frames' array; available: {available}")

    frames = np.asarray(payload["frames"], dtype=np.float32)
    if frames.ndim == 1:
        return frames
    if frames.ndim == 2:
        return frames.mean(axis=0)
    raise ValueError(f"expected 1D or 2D TRIBE frames in {path}, got {frames.shape}")


def roi_values_from_feature_vector(
    feature_vector: np.ndarray,
    roi_masks: dict[str, np.ndarray],
) -> dict[str, float]:
    """Average a vertex feature vector within each ROI mask."""

    values: dict[str, float] = {}
    for roi in REQUIRED_ROIS:
        if roi not in roi_masks:
            raise ValueError(f"missing ROI mask: {roi}")
        mask = np.asarray(roi_masks[roi], dtype=bool)
        if mask.shape != feature_vector.shape:
            raise ValueError(
                f"ROI mask {roi!r} shape {mask.shape} does not match "
                f"feature shape {feature_vector.shape}"
            )
        if not mask.any():
            raise ValueError(f"ROI mask {roi!r} is empty")
        values[roi] = float(np.asarray(feature_vector, dtype=np.float32)[mask].mean())
    return values


def load_roi_masks_npz(path: Path) -> dict[str, np.ndarray]:
    """Load named ROI masks from an NPZ file."""

    payload = np.load(path, allow_pickle=False)
    return {name: np.asarray(payload[name], dtype=bool) for name in payload.files}


def load_destrieux_roi_masks(
    *,
    overlap_policy: OverlapPolicy = "allow",
) -> dict[str, np.ndarray]:
    """Load exploratory ROI masks from Nilearn's fsaverage5 Destrieux atlas."""

    return load_destrieux_roi_selection(overlap_policy=overlap_policy).masks


def load_destrieux_roi_selection(
    *,
    overlap_policy: OverlapPolicy = "allow",
) -> ROISelection:
    """Load exploratory ROI masks and label metadata from Destrieux."""

    from nilearn.datasets import fetch_atlas_surf_destrieux  # noqa: PLC0415

    atlas = fetch_atlas_surf_destrieux()
    lh = np.asarray(atlas["map_left"], dtype=int)
    rh = np.asarray(atlas["map_right"], dtype=int)
    labels = [str(label) for label in atlas["labels"]]
    n_regions = len(labels)
    parcels = np.concatenate([lh, rh])
    parcels[10242:] = parcels[10242:] + n_regions
    all_labels = [f"L_{label}" for label in labels] + [
        f"R_{label}" for label in labels
    ]
    return build_roi_selection_from_parcels(
        parcels,
        all_labels,
        overlap_policy=overlap_policy,
    )


def build_roi_masks_from_parcels(
    parcels: np.ndarray,
    labels: list[str],
    *,
    group_specs: dict[str, ROIGroupSpec] | None = None,
    overlap_policy: OverlapPolicy = "allow",
) -> dict[str, np.ndarray]:
    """Build ROI masks from a surface atlas parcel vector and labels."""

    return build_roi_selection_from_parcels(
        parcels,
        labels,
        group_specs=group_specs,
        overlap_policy=overlap_policy,
    ).masks


def build_roi_selection_from_parcels(
    parcels: np.ndarray,
    labels: list[str],
    *,
    group_specs: dict[str, ROIGroupSpec] | None = None,
    overlap_policy: OverlapPolicy = "allow",
) -> ROISelection:
    """Build ROI masks and selected-label metadata from a parcel vector."""

    if overlap_policy not in ("allow", "drop_shared"):
        raise ValueError(f"unknown ROI overlap policy: {overlap_policy}")

    specs = group_specs or DEFAULT_DESTRIEUX_ROI_GROUPS
    parcel_ids = np.asarray(parcels, dtype=int)
    selected = _select_roi_label_indices(labels, specs)
    masks: dict[str, np.ndarray] = {}
    selected_labels: dict[str, tuple[str, ...]] = {}

    for roi, selected_ids in selected.items():
        masks[roi] = np.isin(parcel_ids, list(selected_ids))
        selected_labels[roi] = tuple(labels[idx] for idx in sorted(selected_ids))

    if overlap_policy == "drop_shared":
        masks = _drop_shared_roi_vertices(masks)

    return ROISelection(
        masks=masks,
        selected_labels=selected_labels,
        n_vertices=int(parcel_ids.shape[0]),
        n_labels=len(labels),
        overlap_policy=overlap_policy,
    )


def roi_mask_audit(selection: ROISelection) -> dict[str, Any]:
    """Return a JSON-serializable audit for a frozen ROI selection."""

    roi_items: dict[str, Any] = {}
    roi_names = tuple(selection.masks)
    for roi in roi_names:
        mask = np.asarray(selection.masks[roi], dtype=bool)
        spec = DEFAULT_DESTRIEUX_ROI_GROUPS.get(roi, ROIGroupSpec(include=()))
        roi_items[roi] = {
            "include": list(spec.include),
            "exclude": list(spec.exclude),
            "n_vertices": int(mask.sum()),
            "vertex_fraction": float(mask.mean()),
            "mask_sha256": sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
            "selected_labels": list(selection.selected_labels[roi]),
        }

    overlaps: dict[str, dict[str, int]] = {}
    for roi in roi_names:
        left = np.asarray(selection.masks[roi], dtype=bool)
        overlaps[roi] = {}
        for other_roi in roi_names:
            right = np.asarray(selection.masks[other_roi], dtype=bool)
            overlaps[roi][other_roi] = int(np.logical_and(left, right).sum())

    return {
        "schema_version": 1,
        "atlas": "nilearn.fetch_atlas_surf_destrieux fsaverage5",
        "overlap_policy": selection.overlap_policy,
        "n_vertices": selection.n_vertices,
        "n_labels": selection.n_labels,
        "roi_groups": roi_items,
        "vertex_overlaps": overlaps,
        "claim_boundary": (
            "Exploratory ROI masks for proxy dry runs. These masks are not "
            "validated attention, dopamine, or executive-control measurements."
        ),
    }


def render_roi_mask_audit_markdown(audit: dict[str, Any]) -> str:
    """Render a concise Markdown view of `roi_mask_audit`."""

    lines = [
        "# Destrieux ROI Mask Audit",
        "",
        f"- Atlas: {audit['atlas']}",
        f"- Overlap policy: {audit.get('overlap_policy', 'allow')}",
        f"- Vertices: {audit['n_vertices']}",
        f"- Labels: {audit['n_labels']}",
        f"- Claim boundary: {audit['claim_boundary']}",
        "",
        "## ROI Coverage",
        "",
        "| ROI | vertices | fraction | selected labels | mask sha256 |",
        "|---|---:|---:|---:|---|",
    ]

    roi_names = tuple(audit["roi_groups"])
    for roi in roi_names:
        item = audit["roi_groups"][roi]
        lines.append(
            "| "
            f"{roi} | {item['n_vertices']} | "
            f"{float(item['vertex_fraction']):.4f} | "
            f"{len(item['selected_labels'])} | "
            f"`{str(item['mask_sha256'])[:12]}` |"
        )

    lines.extend(["", "## Selected Labels", ""])
    for roi in roi_names:
        labels = audit["roi_groups"][roi]["selected_labels"]
        lines.append(f"### {roi}")
        lines.append("")
        for label in labels:
            lines.append(f"- `{label}`")
        lines.append("")

    lines.extend(["## Vertex Overlaps", ""])
    header = " | ".join(["ROI", *roi_names])
    divider = "|".join(["---", *["---:" for _ in roi_names]])
    lines.append(f"| {header} |")
    lines.append(f"|{divider}|")
    for roi in roi_names:
        values = " | ".join(
            str(audit["vertex_overlaps"][roi][other]) for other in roi_names
        )
        lines.append(f"| {roi} | {values} |")

    return "\n".join(lines) + "\n"


def load_manifest_rows(
    manifest_path: Path,
    *,
    roi_masks: dict[str, np.ndarray] | None = None,
    epsilon: float = 1e-6,
) -> list[CaptureRow]:
    """Load and score every sample in a Phase 1 manifest."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{manifest_path} missing list field 'samples'")

    rows: list[CaptureRow] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError(f"manifest sample is not an object: {sample!r}")
        roi_values = _sample_roi_values(sample, manifest_path, roi_masks=roi_masks)
        scores = capture_scores_from_roi_values(roi_values, epsilon=epsilon)
        rows.append(
            CaptureRow(
                sample_id=_required_str(sample, "sample_id"),
                dataset=str(sample.get("dataset") or "unknown"),
                ground_truth=_required_float(sample, "ground_truth"),
                roi_values=roi_values,
                sensory_mean=float(scores["sensory_mean"]),
                capture_score=_optional_float(scores["capture_score"]),
                capture_delta=float(scores["capture_delta"]),
                frontoparietal=float(scores["frontoparietal"]),
                denominator_valid=bool(scores["denominator_valid"]),
            )
        )
    return rows


def run_phase1_manifest(
    manifest_path: Path,
    *,
    roi_masks_path: Path | None = None,
    permutations: int = 999,
    seed: int = 17,
    gate_rho: float = DEFAULT_GATE_RHO,
    epsilon: float = 1e-6,
    include_rows: bool = True,
) -> dict[str, Any]:
    """Run the Phase 1 capture-score validation gate for a manifest."""

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_status = str(manifest_payload.get("status") or "unspecified")
    roi_masks = load_roi_masks_npz(roi_masks_path) if roi_masks_path else None
    rows = load_manifest_rows(manifest_path, roi_masks=roi_masks, epsilon=epsilon)
    return run_capture_rows(
        rows,
        manifest_path=str(manifest_path),
        manifest_status=manifest_status,
        permutations=permutations,
        seed=seed,
        gate_rho=gate_rho,
        include_rows=include_rows,
    )


def preflight_phase1_manifest(
    manifest_path: Path,
    *,
    roi_masks_path: Path | None = None,
    min_samples: int = 30,
    min_distinct_ground_truth: int = 3,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Audit a Phase 1 manifest before claim-relevant scoring.

    This is a verifier, not a result. It checks external-label variance,
    feature coverage, ROI-mask compatibility, and claim-blocking status so a
    real-data run has an auditable go/no-go gate before GPU or correlation
    work.
    """

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{manifest_path} missing list field 'samples'")

    sample_dicts = _manifest_sample_dicts(samples)
    roi_masks = load_roi_masks_npz(roi_masks_path) if roi_masks_path else None
    feature_audit = _preflight_feature_paths(
        sample_dicts,
        manifest_path=manifest_path,
        roi_masks=roi_masks,
    )
    label_groups = _preflight_label_groups(
        sample_dicts,
        min_samples=min_samples,
        min_distinct_ground_truth=min_distinct_ground_truth,
    )
    kind_counts = _preflight_sample_kind_counts(sample_dicts)
    reasons, warnings = _preflight_readiness_reasons(
        n_samples=len(sample_dicts),
        min_samples=min_samples,
        kind_counts=kind_counts,
        feature_audit=feature_audit,
        label_groups=label_groups,
        roi_masks_path=roi_masks_path,
    )

    manifest_status = str(manifest.get("status") or "unspecified")
    claim_update_allowed = not _is_claim_blocked_run(
        manifest_status,
        _synthetic_rows_for_claim_block_check(sample_dicts),
    )
    if not claim_update_allowed:
        warnings.append("manifest status or dataset names block claim updates")

    scoring_audit = _preflight_scoring(
        manifest_path,
        roi_masks=roi_masks,
        epsilon=epsilon,
        enabled=not reasons,
    )
    if scoring_audit["attempted"]:
        valid = int(scoring_audit["n_valid_capture_denominators"])
        if valid < min_samples:
            reasons.append(
                f"valid capture denominator count {valid} is below minimum {min_samples}"
            )
    elif not scoring_audit["attempted_reason"]:
        warnings.append("ROI scoring preflight was skipped")

    mechanical_ready = not reasons
    return {
        "schema_version": 1,
        "experiment": "phase1_capture_score_preflight",
        "manifest_path": str(manifest_path),
        "manifest_status": manifest_status,
        "roi_masks_path": str(roi_masks_path) if roi_masks_path else None,
        "n_samples": len(sample_dicts),
        "sample_kind_counts": kind_counts,
        "label_groups": label_groups,
        "feature_audit": feature_audit,
        "scoring_audit": scoring_audit,
        "min_samples": min_samples,
        "min_distinct_ground_truth": min_distinct_ground_truth,
        "mechanical_ready": mechanical_ready,
        "claim_update_allowed": claim_update_allowed,
        "claim_ready": mechanical_ready and claim_update_allowed,
        "blocking_reasons": reasons,
        "warnings": warnings,
        "claim_boundary": (
            "Preflight verifies manifest mechanics and label variance only. "
            "It does not validate attentional capture."
        ),
    }


def render_preflight_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report for `preflight_phase1_manifest`."""

    lines = [
        "# Phase 1 Capture-Score Preflight",
        "",
        "## Verdict",
        "",
        f"- Mechanical ready: {report['mechanical_ready']}",
        f"- Claim update allowed: {report['claim_update_allowed']}",
        f"- Claim ready: {report['claim_ready']}",
        f"- Samples: {report['n_samples']}",
        f"- ROI masks: {report.get('roi_masks_path') or 'none'}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Blocking Reasons",
        "",
    ]
    reasons = report.get("blocking_reasons") or []
    if reasons:
        lines.extend(f"- {reason}" for reason in reasons)
    else:
        lines.append("- none")

    warnings = report.get("warnings") or []
    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Label Groups",
            "",
            "| dataset | n | finite | distinct | std | ready |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for group in report["label_groups"]:
        lines.append(
            "| "
            f"{group['dataset']} | {group['n']} | {group['n_finite']} | "
            f"{group['n_distinct']} | {_fmt_optional_float(group['std'])} | "
            f"{group['ready']} |"
        )

    feature_audit = report["feature_audit"]
    scoring_audit = report["scoring_audit"]
    lines.extend(
        [
            "",
            "## Feature Audit",
            "",
            f"- Feature-path samples: {feature_audit['n_feature_path_samples']}",
            f"- Existing features: {feature_audit['n_existing']}",
            f"- Missing features: {feature_audit['n_missing']}",
            f"- Shape mismatches: {feature_audit['n_shape_mismatch']}",
            "",
            "## Scoring Audit",
            "",
            f"- Attempted: {scoring_audit['attempted']}",
            f"- Reason skipped: {scoring_audit['attempted_reason'] or 'n/a'}",
            (
                "- Invalid capture denominators: "
                f"{scoring_audit['n_invalid_capture_denominators']}"
            ),
            (
                "- Valid capture denominators: "
                f"{scoring_audit['n_valid_capture_denominators']}"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run_capture_rows(
    rows: list[CaptureRow],
    *,
    manifest_path: str,
    manifest_status: str,
    permutations: int = 999,
    seed: int = 17,
    gate_rho: float = DEFAULT_GATE_RHO,
    include_rows: bool = True,
) -> dict[str, Any]:
    """Build a Phase 1 report from already-scored capture rows."""

    datasets = sorted({row.dataset for row in rows})
    group_summaries = [
        summarize_capture_rows(
            [row for row in rows if row.dataset == dataset],
            group_name=dataset,
            permutations=permutations,
            seed=seed + idx,
            gate_rho=gate_rho,
        )
        for idx, dataset in enumerate(datasets)
    ]
    pooled = summarize_capture_rows(
        rows,
        group_name="pooled",
        permutations=permutations,
        seed=seed + len(datasets),
        gate_rho=gate_rho,
    )

    gate_passed_groups = [
        item["group"]
        for item in group_summaries
        if item["metrics"]["capture_score"]["gate_passed"]
    ]
    claim_update_allowed = not _is_claim_blocked_run(manifest_status, rows)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "phase1_capture_score_validation",
        "manifest_path": manifest_path,
        "manifest_status": manifest_status,
        "claim_update_allowed": claim_update_allowed,
        "n_samples": len(rows),
        "n_invalid_capture_denominators": sum(
            1 for row in rows if not row.denominator_valid
        ),
        "gate_rho": gate_rho,
        "permutations": permutations,
        "seed": seed,
        "groups": group_summaries,
        "pooled": pooled,
        "gate": {
            "passed": bool(gate_passed_groups),
            "claim_validated": bool(gate_passed_groups) and claim_update_allowed,
            "passed_groups": gate_passed_groups,
            "rule": (
                "capture_score Spearman rho >= "
                f"{gate_rho:.2f} in at least one dataset"
            ),
        },
    }
    if include_rows:
        report["rows"] = [_row_to_json(row) for row in rows]
    return report


def run_phase1_sensitivity(
    manifest_path: Path,
    *,
    primary_label: str,
    primary_roi_masks_path: Path | None,
    sensitivity_roi_masks: dict[str, Path],
    permutations: int = 999,
    seed: int = 17,
    gate_rho: float = DEFAULT_GATE_RHO,
    epsilon: float = 1e-6,
    include_rows: bool = False,
) -> dict[str, Any]:
    """Run one Phase 1 manifest through primary and sensitivity ROI masks."""

    runs: list[dict[str, Any]] = []
    all_specs: list[tuple[str, str, Path | None]] = [
        ("primary", primary_label, primary_roi_masks_path),
        *[
            ("sensitivity", label, path)
            for label, path in sorted(sensitivity_roi_masks.items())
        ],
    ]
    for idx, (role, label, roi_masks_path) in enumerate(all_specs):
        report = run_phase1_manifest(
            manifest_path,
            roi_masks_path=roi_masks_path,
            permutations=permutations,
            seed=seed + idx,
            gate_rho=gate_rho,
            epsilon=epsilon,
            include_rows=include_rows,
        )
        runs.append(
            {
                "role": role,
                "label": label,
                "roi_masks_path": str(roi_masks_path) if roi_masks_path else None,
                "report": report,
            }
        )

    return {
        "schema_version": 1,
        "experiment": "phase1_capture_score_sensitivity",
        "manifest_path": str(manifest_path),
        "primary_label": primary_label,
        "gate_rho": gate_rho,
        "permutations": permutations,
        "seed": seed,
        "runs": runs,
        "comparison": _compare_sensitivity_runs(runs),
        "claim_boundary": (
            "Sensitivity compares ROI scoring policies for the same manifest. "
            "It does not validate attention capture without real external labels."
        ),
    }


def render_sensitivity_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report for `run_phase1_sensitivity`."""

    lines = [
        "# Phase 1 Capture-Score Sensitivity",
        "",
        "## Setup",
        "",
        f"- Manifest: {report['manifest_path']}",
        f"- Primary: {report['primary_label']}",
        f"- Gate rho: {float(report['gate_rho']):.2f}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Runs",
        "",
        (
            "| role | label | group | n valid | invalid denominators | "
            "capture rho | gate | claim validated |"
        ),
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for run in report["runs"]:
        pooled = run["report"]["pooled"]
        capture = pooled["metrics"]["capture_score"]
        lines.append(
            "| "
            f"{run['role']} | {run['label']} | pooled | "
            f"{capture['n']} | {pooled['n_invalid_capture_denominators']} | "
            f"{_fmt_optional_float(capture['rho'])} | "
            f"{capture['gate_passed']} | "
            f"{run['report']['gate']['claim_validated']} |"
        )

    lines.extend(
        [
            "",
            "## Sensitivity Delta",
            "",
            "| group | sensitivity | primary rho | sensitivity rho | delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in report["comparison"]["capture_score_deltas"]:
        lines.append(
            "| "
            f"{item['group']} | {item['sensitivity_label']} | "
            f"{_fmt_optional_float(item['primary_rho'])} | "
            f"{_fmt_optional_float(item['sensitivity_rho'])} | "
            f"{_fmt_optional_float(item['rho_delta'])} |"
        )
    return "\n".join(lines) + "\n"


def run_phase1_workflow(
    manifest_path: Path,
    *,
    primary_label: str = "primary",
    primary_roi_masks_path: Path | None = None,
    sensitivity_roi_masks: dict[str, Path] | None = None,
    min_samples: int = 30,
    min_distinct_ground_truth: int = 3,
    permutations: int = 999,
    seed: int = 17,
    gate_rho: float = DEFAULT_GATE_RHO,
    epsilon: float = 1e-6,
    include_rows: bool = False,
    score_claim_blocked: bool = False,
) -> dict[str, Any]:
    """Run the guarded Phase 1 sequence: preflight, score, sensitivity."""

    preflight = preflight_phase1_manifest(
        manifest_path,
        roi_masks_path=primary_roi_masks_path,
        min_samples=min_samples,
        min_distinct_ground_truth=min_distinct_ground_truth,
        epsilon=epsilon,
    )
    score_decision = _phase1_workflow_score_decision(
        preflight,
        score_claim_blocked=score_claim_blocked,
    )

    primary_report = None
    sensitivity_report = None
    if score_decision["scoring_executed"]:
        primary_report = run_phase1_manifest(
            manifest_path,
            roi_masks_path=primary_roi_masks_path,
            permutations=permutations,
            seed=seed,
            gate_rho=gate_rho,
            epsilon=epsilon,
            include_rows=include_rows,
        )
        if sensitivity_roi_masks:
            sensitivity_report = run_phase1_sensitivity(
                manifest_path,
                primary_label=primary_label,
                primary_roi_masks_path=primary_roi_masks_path,
                sensitivity_roi_masks=sensitivity_roi_masks,
                permutations=permutations,
                seed=seed,
                gate_rho=gate_rho,
                epsilon=epsilon,
                include_rows=include_rows,
            )

    return {
        "schema_version": 1,
        "experiment": "phase1_capture_score_workflow",
        "manifest_path": str(manifest_path),
        "primary_label": primary_label,
        "primary_roi_masks_path": (
            str(primary_roi_masks_path) if primary_roi_masks_path else None
        ),
        "sensitivity_roi_masks": {
            label: str(path) for label, path in (sensitivity_roi_masks or {}).items()
        },
        "min_samples": min_samples,
        "min_distinct_ground_truth": min_distinct_ground_truth,
        "gate_rho": gate_rho,
        "permutations": permutations,
        "seed": seed,
        "preflight": preflight,
        "score_decision": score_decision,
        "primary_report": primary_report,
        "sensitivity_report": sensitivity_report,
        "claim_boundary": (
            "This workflow only allows claim-relevant scoring after preflight "
            "passes. Smoke or control manifests may be scored for diagnostics "
            "only when explicitly requested, and remain claim-blocked."
        ),
    }


def render_phase1_workflow_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report for `run_phase1_workflow`."""

    preflight = report["preflight"]
    decision = report["score_decision"]
    primary = report.get("primary_report")
    lines = [
        "# Phase 1 Capture-Score Workflow",
        "",
        "## Verdict",
        "",
        f"- Manifest: {report['manifest_path']}",
        f"- Mechanical ready: {preflight['mechanical_ready']}",
        f"- Claim update allowed: {preflight['claim_update_allowed']}",
        f"- Claim ready: {preflight['claim_ready']}",
        f"- Scoring executed: {decision['scoring_executed']}",
        f"- Decision reason: {decision['reason']}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Preflight Blocks",
        "",
    ]
    reasons = preflight.get("blocking_reasons") or []
    if reasons:
        lines.extend(f"- {reason}" for reason in reasons)
    else:
        lines.append("- none")

    warnings = preflight.get("warnings") or []
    lines.extend(["", "## Preflight Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")

    if isinstance(primary, dict):
        lines.extend(
            [
                "",
                "## Primary Score",
                "",
                f"- Claim validated: {primary['gate']['claim_validated']}",
                f"- Gate passed groups: {_joined_or_none(primary['gate']['passed_groups'])}",
                (
                    "- Invalid capture denominators: "
                    f"{primary['n_invalid_capture_denominators']}"
                ),
                "",
                "| group | n valid | capture rho | permutation p | gate |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for summary in [*primary["groups"], primary["pooled"]]:
            capture = summary["metrics"]["capture_score"]
            lines.append(
                "| "
                f"{summary['group']} | {capture['n']} | "
                f"{_fmt_optional_float(capture['rho'])} | "
                f"{_fmt_optional_float(capture['permutation_p_greater'])} | "
                f"{capture['gate_passed']} |"
            )

    sensitivity = report.get("sensitivity_report")
    if isinstance(sensitivity, dict):
        lines.extend(
            [
                "",
                "## Sensitivity Delta",
                "",
                "| group | sensitivity | primary rho | sensitivity rho | delta |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for item in sensitivity["comparison"]["capture_score_deltas"]:
            lines.append(
                "| "
                f"{item['group']} | {item['sensitivity_label']} | "
                f"{_fmt_optional_float(item['primary_rho'])} | "
                f"{_fmt_optional_float(item['sensitivity_rho'])} | "
                f"{_fmt_optional_float(item['rho_delta'])} |"
            )

    return "\n".join(lines) + "\n"


def summarize_capture_rows(
    rows: list[CaptureRow],
    *,
    group_name: str,
    permutations: int,
    seed: int,
    gate_rho: float,
) -> dict[str, Any]:
    """Summarize correlations for one dataset or pooled group."""

    metric_values: dict[str, list[float | None]] = {
        "capture_score": [row.capture_score for row in rows],
        "capture_delta": [row.capture_delta for row in rows],
        "sensory_mean": [row.sensory_mean for row in rows],
        "frontoparietal": [row.frontoparietal for row in rows],
    }
    ground_truth = [row.ground_truth for row in rows]
    metrics: dict[str, Any] = {}
    for offset, (name, values) in enumerate(metric_values.items()):
        stats = correlation_summary(
            values,
            ground_truth,
            permutations=permutations,
            seed=seed + offset,
        )
        stats["gate_passed"] = bool(
            name == "capture_score"
            and stats["rho"] is not None
            and float(stats["rho"]) >= gate_rho
        )
        metrics[name] = stats

    return {
        "group": group_name,
        "n_samples": len(rows),
        "n_invalid_capture_denominators": sum(
            1 for row in rows if not row.denominator_valid
        ),
        "metrics": metrics,
    }


def correlation_summary(
    values: list[float | None],
    ground_truth: list[float],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Compute Spearman rho plus an optional permutation p-value."""

    clean_x: list[float] = []
    clean_y: list[float] = []
    for x, y in zip(values, ground_truth, strict=True):
        if x is None:
            continue
        fx = float(x)
        fy = float(y)
        if math.isfinite(fx) and math.isfinite(fy):
            clean_x.append(fx)
            clean_y.append(fy)

    rho = spearman_rho(clean_x, clean_y)
    p_value = None
    if rho is not None and permutations > 0:
        p_value = permutation_p_value(
            clean_x,
            clean_y,
            observed_rho=rho,
            permutations=permutations,
            seed=seed,
            alternative="greater",
        )

    return {
        "n": len(clean_x),
        "rho": rho,
        "permutation_p_greater": p_value,
    }


def spearman_rho(x_values: list[float], y_values: list[float]) -> float | None:
    """Spearman rank correlation with average ranks for ties."""

    if len(x_values) != len(y_values):
        raise ValueError("x and y must have the same length")
    if len(x_values) < 3:
        return None

    x = _rankdata_average(np.asarray(x_values, dtype=np.float64))
    y = _rankdata_average(np.asarray(y_values, dtype=np.float64))
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denom <= 1e-12:
        return None
    return float((x_centered @ y_centered) / denom)


def permutation_p_value(
    x_values: list[float],
    y_values: list[float],
    *,
    observed_rho: float,
    permutations: int,
    seed: int,
    alternative: Alternative = "greater",
) -> float:
    """Permutation p-value for Spearman rho."""

    if permutations <= 0:
        raise ValueError("permutations must be positive")
    rng = np.random.default_rng(seed)
    y = np.asarray(y_values, dtype=np.float64)
    count = 0
    for _ in range(permutations):
        permuted = rng.permutation(y).tolist()
        rho = spearman_rho(x_values, permuted)
        if rho is None:
            continue
        if alternative == "greater":
            count += int(rho >= observed_rho)
        else:
            count += int(abs(rho) >= abs(observed_rho))
    return float((count + 1) / (permutations + 1))


def render_phase1_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report from `run_phase1_manifest` output."""

    lines = [
        "# Phase 1 Capture-Score Dry Run",
        "",
        "## Gate",
        "",
        f"- Manifest status: {report.get('manifest_status', 'unspecified')}",
        f"- Claim update allowed: {report.get('claim_update_allowed', False)}",
        f"- Rule: {report['gate']['rule']}",
        f"- Mechanical gate passed: {report['gate']['passed']}",
        f"- Claim validated: {report['gate'].get('claim_validated', False)}",
        f"- Passed groups: {', '.join(report['gate']['passed_groups']) or 'none'}",
        f"- Samples: {report['n_samples']}",
        (
            "- Invalid capture denominators: "
            f"{report['n_invalid_capture_denominators']}"
        ),
        "",
        "## Correlations",
        "",
        "| group | metric | n | Spearman rho | permutation p (greater) | gate |",
        "|---|---|---:|---:|---:|---|",
    ]

    for summary in [*report["groups"], report["pooled"]]:
        for metric_name, metric in summary["metrics"].items():
            rho = _fmt_optional_float(metric["rho"])
            p_value = _fmt_optional_float(metric["permutation_p_greater"])
            gate = "pass" if metric["gate_passed"] else ""
            lines.append(
                "| "
                f"{summary['group']} | {metric_name} | {metric['n']} | "
                f"{rho} | {p_value} | {gate} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "This is a pipeline and metric dry run. It validates that the "
                "manifest, ROI score, correlation, and gate machinery are "
                "working; it does not validate attention capture unless the "
                "manifest contains real external ground-truth labels."
            ),
        ]
    )
    control = report.get("control")
    if isinstance(control, dict):
        lines.extend(
            [
                "",
                "## Control Note",
                "",
                f"- Control: {control.get('name', 'unspecified')}",
                f"- Ground truth: {control.get('ground_truth_name', 'unspecified')}",
                f"- ROI source: {control.get('roi_source', 'unspecified')}",
                f"- Interpretation: {control.get('interpretation', 'unspecified')}",
            ]
        )
    return "\n".join(lines) + "\n"


def _manifest_sample_dicts(samples: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError(f"manifest sample is not an object: {sample!r}")
        out.append(sample)
    return out


def _preflight_readiness_reasons(
    *,
    n_samples: int,
    min_samples: int,
    kind_counts: dict[str, int],
    feature_audit: dict[str, Any],
    label_groups: list[dict[str, Any]],
    roi_masks_path: Path | None,
) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    if n_samples < min_samples:
        reasons.append(f"sample count {n_samples} is below minimum {min_samples}")
    if kind_counts["neither"]:
        reasons.append(f"{kind_counts['neither']} samples have no ROI values or feature path")
    if kind_counts["both"]:
        warnings.append(f"{kind_counts['both']} samples include both ROI values and features")
    if kind_counts["tribe_feature_path"] and roi_masks_path is None:
        reasons.append("feature-path samples require --roi-masks")
    if feature_audit["n_missing"]:
        reasons.append(f"{feature_audit['n_missing']} feature files are missing")
    if feature_audit["n_shape_mismatch"]:
        reasons.append(
            f"{feature_audit['n_shape_mismatch']} feature files do not match ROI masks"
        )
    if not any(group["ready"] for group in label_groups):
        reasons.append("no dataset group has enough non-degenerate ground truth labels")
    return reasons, warnings


def _preflight_sample_kind_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "roi_values": 0,
        "tribe_feature_path": 0,
        "both": 0,
        "neither": 0,
    }
    for sample in samples:
        has_roi_values = isinstance(sample.get("roi_values"), dict)
        has_feature_path = isinstance(sample.get("tribe_feature_path"), str)
        counts["roi_values"] += int(has_roi_values)
        counts["tribe_feature_path"] += int(has_feature_path)
        counts["both"] += int(has_roi_values and has_feature_path)
        counts["neither"] += int(not has_roi_values and not has_feature_path)
    return counts


def _preflight_label_groups(
    samples: list[dict[str, Any]],
    *,
    min_samples: int,
    min_distinct_ground_truth: int,
) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[float | None]] = {}
    for sample in samples:
        dataset = str(sample.get("dataset") or "unknown")
        by_dataset.setdefault(dataset, []).append(_finite_float(sample.get("ground_truth")))

    groups: list[dict[str, Any]] = []
    for dataset in sorted(by_dataset):
        values = by_dataset[dataset]
        finite = [float(value) for value in values if value is not None]
        std = float(np.std(finite)) if finite else None
        n_distinct = len(set(finite))
        ready = bool(
            len(values) >= min_samples
            and len(finite) == len(values)
            and n_distinct >= min_distinct_ground_truth
            and std is not None
            and std > 0.0
        )
        groups.append(
            {
                "dataset": dataset,
                "n": len(values),
                "n_finite": len(finite),
                "n_distinct": n_distinct,
                "mean": float(np.mean(finite)) if finite else None,
                "std": std,
                "min": float(np.min(finite)) if finite else None,
                "max": float(np.max(finite)) if finite else None,
                "ready": ready,
            }
        )
    return groups


def _preflight_feature_paths(
    samples: list[dict[str, Any]],
    *,
    manifest_path: Path,
    roi_masks: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    roi_shape = _common_roi_shape(roi_masks)
    missing: list[dict[str, str]] = []
    mismatches: list[dict[str, str]] = []
    shape_counts: dict[str, int] = {}
    existing = 0
    feature_samples = 0

    for sample in samples:
        raw_feature_path = sample.get("tribe_feature_path")
        if not isinstance(raw_feature_path, str) or not raw_feature_path:
            continue
        feature_samples += 1
        feature_path = Path(raw_feature_path)
        if not feature_path.is_absolute():
            feature_path = manifest_path.parent / feature_path
        sample_id = str(sample.get("sample_id") or "unknown")
        if not feature_path.exists():
            missing.append({"sample_id": sample_id, "path": str(feature_path)})
            continue
        existing += 1
        try:
            feature_shape = load_tribe_feature_mean(feature_path).shape
        except ValueError as exc:
            mismatches.append(
                {
                    "sample_id": sample_id,
                    "path": str(feature_path),
                    "error": str(exc),
                }
            )
            continue
        shape_key = "x".join(str(dim) for dim in feature_shape)
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
        if roi_shape is not None and tuple(feature_shape) != roi_shape:
            mismatches.append(
                {
                    "sample_id": sample_id,
                    "path": str(feature_path),
                    "feature_shape": shape_key,
                    "roi_shape": "x".join(str(dim) for dim in roi_shape),
                }
            )

    return {
        "n_feature_path_samples": feature_samples,
        "n_existing": existing,
        "n_missing": len(missing),
        "n_shape_mismatch": len(mismatches),
        "feature_shape_counts": shape_counts,
        "missing_features": missing[:20],
        "shape_mismatches": mismatches[:20],
        "roi_shape": "x".join(str(dim) for dim in roi_shape) if roi_shape else None,
    }


def _preflight_scoring(
    manifest_path: Path,
    *,
    roi_masks: dict[str, np.ndarray] | None,
    epsilon: float,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "attempted": False,
            "attempted_reason": "skipped because blocking preflight issues exist",
            "n_invalid_capture_denominators": None,
            "n_valid_capture_denominators": None,
        }
    try:
        rows = load_manifest_rows(manifest_path, roi_masks=roi_masks, epsilon=epsilon)
    except ValueError as exc:
        return {
            "attempted": False,
            "attempted_reason": str(exc),
            "n_invalid_capture_denominators": None,
            "n_valid_capture_denominators": None,
        }
    invalid = sum(1 for row in rows if not row.denominator_valid)
    return {
        "attempted": True,
        "attempted_reason": None,
        "n_invalid_capture_denominators": invalid,
        "n_valid_capture_denominators": len(rows) - invalid,
    }


def _compare_sensitivity_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    primary = next(run for run in runs if run["role"] == "primary")
    primary_by_group = _capture_metric_by_group(primary["report"])
    deltas: list[dict[str, Any]] = []
    for run in runs:
        if run["role"] == "primary":
            continue
        sensitivity_by_group = _capture_metric_by_group(run["report"])
        for group, primary_metric in primary_by_group.items():
            sensitivity_metric = sensitivity_by_group.get(group)
            if sensitivity_metric is None:
                continue
            primary_rho = primary_metric["rho"]
            sensitivity_rho = sensitivity_metric["rho"]
            deltas.append(
                {
                    "group": group,
                    "sensitivity_label": run["label"],
                    "primary_rho": primary_rho,
                    "sensitivity_rho": sensitivity_rho,
                    "rho_delta": _optional_delta(sensitivity_rho, primary_rho),
                    "primary_n": primary_metric["n"],
                    "sensitivity_n": sensitivity_metric["n"],
                    "primary_gate_passed": primary_metric["gate_passed"],
                    "sensitivity_gate_passed": sensitivity_metric["gate_passed"],
                }
            )
    return {"capture_score_deltas": deltas}


def _capture_metric_by_group(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summaries = [*report["groups"], report["pooled"]]
    return {
        summary["group"]: summary["metrics"]["capture_score"]
        for summary in summaries
    }


def _optional_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _phase1_workflow_score_decision(
    preflight: dict[str, Any],
    *,
    score_claim_blocked: bool,
) -> dict[str, Any]:
    if not preflight["mechanical_ready"]:
        return {
            "scoring_executed": False,
            "reason": "preflight_failed",
        }
    if preflight["claim_ready"]:
        return {
            "scoring_executed": True,
            "reason": "claim_ready",
        }
    if score_claim_blocked:
        return {
            "scoring_executed": True,
            "reason": "claim_blocked_scored_for_diagnostic_only",
        }
    return {
        "scoring_executed": False,
        "reason": "claim_blocked",
    }


def _joined_or_none(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _synthetic_rows_for_claim_block_check(
    samples: list[dict[str, Any]],
) -> list[CaptureRow]:
    rows: list[CaptureRow] = []
    for sample in samples:
        rows.append(
            CaptureRow(
                sample_id=str(sample.get("sample_id") or "unknown"),
                dataset=str(sample.get("dataset") or "unknown"),
                ground_truth=0.0,
                roi_values={},
                sensory_mean=0.0,
                capture_score=None,
                capture_delta=0.0,
                frontoparietal=0.0,
                denominator_valid=False,
            )
        )
    return rows


def _common_roi_shape(roi_masks: dict[str, np.ndarray] | None) -> tuple[int, ...] | None:
    if roi_masks is None:
        return None
    shapes = {tuple(np.asarray(mask, dtype=bool).shape) for mask in roi_masks.values()}
    if len(shapes) != 1:
        return None
    return next(iter(shapes))


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _sample_roi_values(
    sample: dict[str, Any],
    manifest_path: Path,
    *,
    roi_masks: dict[str, np.ndarray] | None,
) -> dict[str, float]:
    roi_values = sample.get("roi_values")
    if isinstance(roi_values, dict):
        return {str(key): float(value) for key, value in roi_values.items()}

    feature_path_value = sample.get("tribe_feature_path")
    if not isinstance(feature_path_value, str) or not feature_path_value:
        raise ValueError(
            "sample must include either 'roi_values' or 'tribe_feature_path'"
        )
    if roi_masks is None:
        raise ValueError("tribe_feature_path samples require --roi-masks")

    feature_path = Path(feature_path_value)
    if not feature_path.is_absolute():
        feature_path = manifest_path.parent / feature_path
    feature_vector = load_tribe_feature_mean(feature_path)
    return roi_values_from_feature_vector(feature_vector, roi_masks)


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    sorter = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[sorter]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[sorter[start:end]] = average_rank
        start = end
    return ranks


def _normalize_label(label: str) -> str:
    return str(label).lower().replace("_", "-")


def _select_roi_label_indices(
    labels: list[str],
    group_specs: dict[str, ROIGroupSpec],
) -> dict[str, set[int]]:
    normalized_labels = [_normalize_label(label) for label in labels]
    selected: dict[str, set[int]] = {}
    for roi, spec in group_specs.items():
        selected_ids: set[int] = set()
        include = tuple(_normalize_label(token) for token in spec.include)
        exclude = tuple(_normalize_label(token) for token in spec.exclude)
        for idx, label in enumerate(normalized_labels):
            if any(token in label for token in include) and not any(
                token in label for token in exclude
            ):
                selected_ids.add(idx)
        selected[roi] = selected_ids
    return selected


def _drop_shared_roi_vertices(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not masks:
        return {}

    membership_count = np.zeros(next(iter(masks.values())).shape, dtype=np.int16)
    for mask in masks.values():
        membership_count += np.asarray(mask, dtype=bool).astype(np.int16)

    return {
        roi: np.logical_and(np.asarray(mask, dtype=bool), membership_count == 1)
        for roi, mask in masks.items()
    }


def _required_str(sample: dict[str, Any], field: str) -> str:
    value = sample.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"sample missing string field {field!r}")
    return value


def _required_float(sample: dict[str, Any], field: str) -> float:
    if field not in sample:
        raise ValueError(f"sample missing numeric field {field!r}")
    return float(sample[field])


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _row_to_json(row: CaptureRow) -> dict[str, Any]:
    return {
        "sample_id": row.sample_id,
        "dataset": row.dataset,
        "ground_truth": row.ground_truth,
        "roi_values": row.roi_values,
        "sensory_mean": row.sensory_mean,
        "frontoparietal": row.frontoparietal,
        "capture_score": row.capture_score,
        "capture_delta": row.capture_delta,
        "denominator_valid": row.denominator_valid,
    }


def _is_claim_blocked_run(manifest_status: str, rows: list[CaptureRow]) -> bool:
    status = manifest_status.lower()
    blocked_terms = ("synthetic", "fixture", "smoke", "control", "not_attention")
    if any(term in status for term in blocked_terms):
        return True
    return any("fixture" in row.dataset.lower() for row in rows)


def _fmt_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
