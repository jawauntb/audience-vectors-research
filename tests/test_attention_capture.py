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
    render_phase1_markdown,
    render_roi_mask_audit_markdown,
    roi_mask_audit,
    run_capture_rows,
    run_phase1_manifest,
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
