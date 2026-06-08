from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audience_vectors.attention_capture import (
    CaptureRow,
    ROIGroupSpec,
    build_roi_masks_from_parcels,
    build_roi_selection_from_parcels,
    capture_scores_from_roi_values,
    load_manifest_rows,
    preflight_phase1_manifest,
    render_phase1_markdown,
    render_phase1_workflow_markdown,
    render_preflight_markdown,
    render_roi_mask_audit_markdown,
    render_sensitivity_markdown,
    roi_mask_audit,
    run_capture_rows,
    run_phase1_manifest,
    run_phase1_sensitivity,
    run_phase1_workflow,
    spearman_rho,
)


def test_capture_score_uses_sensory_over_frontoparietal() -> None:
    scores = capture_scores_from_roi_values(
        {
            "V1": 0.8,
            "PPA": 0.7,
            "language": 0.5,
            "frontoparietal": 0.25,
        },
        epsilon=0.0,
    )

    assert scores["sensory_mean"] == pytest.approx(2.0 / 3.0)
    assert scores["capture_score"] == pytest.approx((2.0 / 3.0) / 0.25)
    assert scores["capture_delta"] == pytest.approx((2.0 / 3.0) - 0.25)
    assert scores["denominator_valid"] is True


def test_capture_score_withholds_non_positive_denominator() -> None:
    scores = capture_scores_from_roi_values(
        {
            "V1": 0.3,
            "PPA": 0.2,
            "language": 0.1,
            "frontoparietal": -0.01,
        },
    )

    assert scores["capture_score"] is None
    assert scores["denominator_valid"] is False
    assert scores["capture_delta"] == pytest.approx(0.21)


def test_spearman_rho_handles_ties_with_average_ranks() -> None:
    assert spearman_rho([1, 2, 2, 4], [1, 2, 3, 4]) == pytest.approx(
        0.948683298,
    )


def test_build_roi_masks_from_parcels_uses_include_and_exclude() -> None:
    parcels = np.array([0, 1, 2, 3, 1, 2])
    labels = [
        "unknown",
        "G_front_middle",
        "S_intrapariet_and_P_trans",
        "G_front_middle_insula",
    ]
    masks = build_roi_masks_from_parcels(
        parcels,
        labels,
        group_specs={
            "frontoparietal": ROIGroupSpec(
                include=("front_middle", "intrapariet"),
                exclude=("insula",),
            ),
        },
    )

    assert masks["frontoparietal"].tolist() == [False, True, True, False, True, True]


def test_roi_selection_audit_reports_labels_counts_and_overlaps() -> None:
    parcels = np.array([0, 1, 2, 3, 1, 2])
    labels = [
        "unknown",
        "G_front_middle",
        "S_intrapariet_and_P_trans",
        "G_front_middle_insula",
    ]
    selection = build_roi_selection_from_parcels(
        parcels,
        labels,
        group_specs={
            "frontoparietal": ROIGroupSpec(
                include=("front_middle", "intrapariet"),
                exclude=("insula",),
            ),
        },
    )

    assert selection.selected_labels["frontoparietal"] == (
        "G_front_middle",
        "S_intrapariet_and_P_trans",
    )

    audit = roi_mask_audit(selection)
    assert audit["roi_groups"]["frontoparietal"]["n_vertices"] == 4
    assert audit["vertex_overlaps"]["frontoparietal"]["frontoparietal"] == 4
    assert "Destrieux ROI Mask Audit" in render_roi_mask_audit_markdown(audit)


def test_drop_shared_roi_policy_removes_overlapping_vertices() -> None:
    parcels = np.array([0, 1, 2, 3, 4, 5])
    labels = [
        "unknown",
        "G_occipital",
        "G_occipital_parahip",
        "G_parahip",
        "G_front_middle",
        "G_front_middle_temporal",
    ]
    selection = build_roi_selection_from_parcels(
        parcels,
        labels,
        group_specs={
            "V1": ROIGroupSpec(include=("occipital",)),
            "PPA": ROIGroupSpec(include=("parahip",)),
            "language": ROIGroupSpec(include=("temporal",)),
            "frontoparietal": ROIGroupSpec(include=("front_middle",)),
        },
        overlap_policy="drop_shared",
    )

    assert selection.overlap_policy == "drop_shared"
    assert selection.masks["V1"].tolist() == [False, True, False, False, False, False]
    assert selection.masks["PPA"].tolist() == [False, False, False, True, False, False]
    assert selection.masks["language"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert selection.masks["frontoparietal"].tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
    ]

    audit = roi_mask_audit(selection)
    assert audit["overlap_policy"] == "drop_shared"
    assert audit["vertex_overlaps"]["V1"]["PPA"] == 0
    assert audit["vertex_overlaps"]["language"]["frontoparietal"] == 0


def test_run_phase1_manifest_passes_synthetic_gate(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "samples": [
                    {
                        "sample_id": f"s{i}",
                        "dataset": "fixture",
                        "ground_truth": float(i),
                        "roi_values": {
                            "V1": 0.2 + i * 0.03,
                            "PPA": 0.2 + i * 0.02,
                            "language": 0.2 + i * 0.01,
                            "frontoparietal": 0.8 - i * 0.04,
                        },
                    }
                    for i in range(10)
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = load_manifest_rows(manifest)
    assert len(rows) == 10

    report = run_phase1_manifest(manifest, permutations=99, seed=1)
    assert report["gate"]["passed"] is True
    assert report["gate"]["claim_validated"] is False
    assert report["groups"][0]["metrics"]["capture_score"]["rho"] > 0.9
    assert "Phase 1 Capture-Score Dry Run" in render_phase1_markdown(report)


def test_control_status_blocks_claim_validation() -> None:
    rows = [
        CaptureRow(
            sample_id=f"s{i}",
            dataset="BOLD_Moments_control",
            ground_truth=float(i),
            roi_values={
                "V1": 0.2 + i,
                "PPA": 0.2 + i,
                "language": 0.2 + i,
                "frontoparietal": 1.0,
            },
            sensory_mean=0.2 + i,
            capture_score=0.2 + i,
            capture_delta=i - 0.8,
            frontoparietal=1.0,
            denominator_valid=True,
        )
        for i in range(5)
    ]

    report = run_capture_rows(
        rows,
        manifest_path="control",
        manifest_status="real_control_not_attention_capture",
        permutations=0,
        include_rows=False,
    )

    assert report["gate"]["passed"] is True
    assert report["claim_update_allowed"] is False
    assert report["gate"]["claim_validated"] is False
    assert "rows" not in report


def ready_alignment_metadata(*, n: int, dhf1k: bool = True) -> dict[str, object]:
    label_audit = (
        {
            "path": "results/dhf1k_attention_label_audit_20260608.json",
            "sha256": "b" * 64,
            "ready_for_manifest_alignment": True,
            "rank_column": "mean_map_intensity",
        }
        if dhf1k
        else None
    )
    return {
        "alignment_audit": {
            "path": "results/phase1_alignment_20260608.json",
            "sha256": "a" * 64,
            "ready_for_manifest_build": True,
            "n_aligned_features": n,
            "n_missing_features": 0,
            "label_audit": label_audit,
        },
    }


def test_preflight_phase1_manifest_accepts_ready_explicit_roi_manifest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "real_external_attention_labels",
                "metadata": ready_alignment_metadata(n=3),
                "samples": [
                    {
                        "sample_id": f"s{i}",
                        "dataset": "DHF1K",
                        "ground_truth": float(i),
                        "roi_values": {
                            "V1": 0.2 + i,
                            "PPA": 0.3 + i,
                            "language": 0.4 + i,
                            "frontoparietal": 1.0,
                        },
                    }
                    for i in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )

    report = preflight_phase1_manifest(manifest, min_samples=3)

    assert report["mechanical_ready"] is True
    assert report["claim_update_allowed"] is True
    assert report["claim_ready"] is True
    assert report["provenance_audit"]["ready"] is True
    assert report["provenance_audit"]["alignment_audit"]["label_audit_ready"] is True
    assert report["scoring_audit"]["attempted"] is True
    assert report["scoring_audit"]["n_valid_capture_denominators"] == 3
    assert "Phase 1 Capture-Score Preflight" in render_preflight_markdown(report)


def test_preflight_phase1_manifest_blocks_missing_alignment_provenance(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "real_external_attention_labels",
                "samples": [
                    {
                        "sample_id": f"s{i}",
                        "dataset": "SnapUGC",
                        "ground_truth": float(i),
                        "roi_values": {
                            "V1": 0.2 + i,
                            "PPA": 0.3 + i,
                            "language": 0.4 + i,
                            "frontoparietal": 1.0,
                        },
                    }
                    for i in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )

    report = preflight_phase1_manifest(manifest, min_samples=3)

    assert report["mechanical_ready"] is False
    assert report["claim_ready"] is False
    assert (
        "claim-ready manifests require metadata.alignment_audit provenance"
        in report["blocking_reasons"]
    )
    assert report["provenance_audit"]["required"] is True


def test_preflight_phase1_manifest_blocks_unready_dhf1k_label_provenance(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    metadata = ready_alignment_metadata(n=3)
    alignment = metadata["alignment_audit"]
    assert isinstance(alignment, dict)
    label_audit = alignment["label_audit"]
    assert isinstance(label_audit, dict)
    label_audit["ready_for_manifest_alignment"] = False
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "real_external_attention_labels",
                "metadata": metadata,
                "samples": [
                    {
                        "sample_id": f"s{i}",
                        "dataset": "DHF1K",
                        "ground_truth": float(i),
                        "roi_values": {
                            "V1": 0.2 + i,
                            "PPA": 0.3 + i,
                            "language": 0.4 + i,
                            "frontoparietal": 1.0,
                        },
                    }
                    for i in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )

    report = preflight_phase1_manifest(manifest, min_samples=3)

    assert report["mechanical_ready"] is False
    assert "alignment_audit.label_audit is not ready" in report["blocking_reasons"]
    assert report["provenance_audit"]["alignment_audit"]["label_audit_ready"] is False


def test_preflight_phase1_manifest_blocks_feature_rows_without_roi_masks(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for i in range(3):
        np.savez_compressed(
            feature_dir / f"s{i}.npz",
            frames=np.ones((2, 4), dtype=np.float32),
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "real_external_attention_labels",
                "metadata": ready_alignment_metadata(n=3),
                "samples": [
                    {
                        "sample_id": f"s{i}",
                        "dataset": "DHF1K",
                        "ground_truth": float(i),
                        "tribe_feature_path": str(feature_dir / f"s{i}.npz"),
                    }
                    for i in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )

    report = preflight_phase1_manifest(manifest, min_samples=3)

    assert report["mechanical_ready"] is False
    assert report["claim_ready"] is False
    assert "feature-path samples require --roi-masks" in report["blocking_reasons"]
    assert report["feature_audit"]["n_existing"] == 3


def test_run_phase1_sensitivity_compares_primary_and_sensitivity_masks(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    samples = []
    for i in range(5):
        frames = np.array(
            [
                [0.1 + i, 0.2 + i, 1.0, 0.6 - i * 0.05],
                [0.2 + i, 0.3 + i, 1.0, 0.6 - i * 0.05],
            ],
            dtype=np.float32,
        )
        feature_path = feature_dir / f"s{i}.npz"
        np.savez_compressed(feature_path, frames=frames)
        samples.append(
            {
                "sample_id": f"s{i}",
                "dataset": "DHF1K",
                "ground_truth": float(i),
                "tribe_feature_path": str(feature_path),
            }
        )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "real_external_attention_labels",
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    primary_masks = tmp_path / "primary_masks.npz"
    sensitivity_masks = tmp_path / "sensitivity_masks.npz"
    np.savez_compressed(
        primary_masks,
        V1=np.array([True, False, False, False]),
        PPA=np.array([False, True, False, False]),
        language=np.array([False, False, True, False]),
        frontoparietal=np.array([False, False, False, True]),
    )
    np.savez_compressed(
        sensitivity_masks,
        V1=np.array([False, False, False, True]),
        PPA=np.array([False, False, True, False]),
        language=np.array([False, True, False, False]),
        frontoparietal=np.array([True, False, False, False]),
    )

    report = run_phase1_sensitivity(
        manifest,
        primary_label="disjoint",
        primary_roi_masks_path=primary_masks,
        sensitivity_roi_masks={"overlap": sensitivity_masks},
        permutations=0,
        seed=1,
    )

    assert report["runs"][0]["label"] == "disjoint"
    assert report["runs"][1]["label"] == "overlap"
    assert report["runs"][0]["report"]["pooled"]["metrics"]["capture_score"]["rho"] > 0
    assert report["runs"][1]["report"]["pooled"]["metrics"]["capture_score"]["rho"] < 0
    [delta] = [
        item
        for item in report["comparison"]["capture_score_deltas"]
        if item["group"] == "pooled"
    ]
    assert delta["sensitivity_label"] == "overlap"
    assert delta["rho_delta"] < 0
    assert "Phase 1 Capture-Score Sensitivity" in render_sensitivity_markdown(report)


def test_run_phase1_workflow_scores_claim_ready_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "real_external_attention_labels",
                "metadata": ready_alignment_metadata(n=5),
                "samples": [
                    {
                        "sample_id": f"s{i}",
                        "dataset": "DHF1K",
                        "ground_truth": float(i),
                        "roi_values": {
                            "V1": 0.2 + i * 0.03,
                            "PPA": 0.2 + i * 0.02,
                            "language": 0.2 + i * 0.01,
                            "frontoparietal": 0.8 - i * 0.04,
                        },
                    }
                    for i in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_phase1_workflow(
        manifest,
        min_samples=5,
        permutations=0,
    )

    assert report["preflight"]["claim_ready"] is True
    assert report["score_decision"] == {
        "scoring_executed": True,
        "reason": "claim_ready",
    }
    assert report["primary_report"]["gate"]["claim_validated"] is True
    assert "Phase 1 Capture-Score Workflow" in render_phase1_workflow_markdown(report)


def test_run_phase1_workflow_withholds_claim_blocked_manifest_by_default(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "synthetic_smoke_only",
                "samples": [
                    {
                        "sample_id": f"s{i}",
                        "dataset": "DHF1K_fixture",
                        "ground_truth": float(i),
                        "roi_values": {
                            "V1": 0.2 + i * 0.03,
                            "PPA": 0.2 + i * 0.02,
                            "language": 0.2 + i * 0.01,
                            "frontoparietal": 0.8 - i * 0.04,
                        },
                    }
                    for i in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )

    withheld = run_phase1_workflow(
        manifest,
        min_samples=5,
        permutations=0,
    )
    diagnostic = run_phase1_workflow(
        manifest,
        min_samples=5,
        permutations=0,
        score_claim_blocked=True,
    )

    assert withheld["preflight"]["mechanical_ready"] is True
    assert withheld["preflight"]["claim_ready"] is False
    assert withheld["score_decision"] == {
        "scoring_executed": False,
        "reason": "claim_blocked",
    }
    assert withheld["primary_report"] is None
    assert diagnostic["score_decision"]["scoring_executed"] is True
    assert diagnostic["primary_report"]["gate"]["claim_validated"] is False
