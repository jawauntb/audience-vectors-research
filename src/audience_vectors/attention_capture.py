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


def load_destrieux_roi_masks() -> dict[str, np.ndarray]:
    """Load exploratory ROI masks from Nilearn's fsaverage5 Destrieux atlas."""

    return load_destrieux_roi_selection().masks


def load_destrieux_roi_selection() -> ROISelection:
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
    return build_roi_selection_from_parcels(parcels, all_labels)


def build_roi_masks_from_parcels(
    parcels: np.ndarray,
    labels: list[str],
    *,
    group_specs: dict[str, ROIGroupSpec] | None = None,
) -> dict[str, np.ndarray]:
    """Build ROI masks from a surface atlas parcel vector and labels."""

    return build_roi_selection_from_parcels(
        parcels,
        labels,
        group_specs=group_specs,
    ).masks


def build_roi_selection_from_parcels(
    parcels: np.ndarray,
    labels: list[str],
    *,
    group_specs: dict[str, ROIGroupSpec] | None = None,
) -> ROISelection:
    """Build ROI masks and selected-label metadata from a parcel vector."""

    specs = group_specs or DEFAULT_DESTRIEUX_ROI_GROUPS
    parcel_ids = np.asarray(parcels, dtype=int)
    selected = _select_roi_label_indices(labels, specs)
    masks: dict[str, np.ndarray] = {}
    selected_labels: dict[str, tuple[str, ...]] = {}

    for roi, selected_ids in selected.items():
        masks[roi] = np.isin(parcel_ids, list(selected_ids))
        selected_labels[roi] = tuple(labels[idx] for idx in sorted(selected_ids))

    return ROISelection(
        masks=masks,
        selected_labels=selected_labels,
        n_vertices=int(parcel_ids.shape[0]),
        n_labels=len(labels),
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
