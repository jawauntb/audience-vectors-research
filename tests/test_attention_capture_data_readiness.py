from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_attention_capture_data_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_attention_capture_data_readiness",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_readiness_report_detects_phase1_inputs(tmp_path: Path) -> None:
    module = load_module()
    dhf1k = tmp_path / "DHF1K"
    (dhf1k / "video").mkdir(parents=True)
    (dhf1k / "video" / "001.AVI").write_bytes(b"fake")
    maps = dhf1k / "annotation" / "001" / "maps"
    maps.mkdir(parents=True)
    (maps / "0001.png").write_bytes(b"fake")

    labels = tmp_path / "snapugc_ecr_labels.csv"
    labels.write_text(
        "sample_id,ecr\nvideo_a,0.8\nvideo_b,0.2\n",
        encoding="utf-8",
    )
    dhf1k_labels = tmp_path / "dhf1k_attention_labels_extremes.csv"
    dhf1k_labels.write_text(
        "sample_id,mean_map_intensity\n"
        "dhf1k_001,0.1\n"
        "dhf1k_002,0.9\n",
        encoding="utf-8",
    )
    dhf1k_audit = tmp_path / "dhf1k_attention_label_audit.json"
    dhf1k_audit.write_text(
        json.dumps(
            {
                "experiment": "dhf1k_attention_label_audit",
                "labels_csv": str(dhf1k_labels),
                "rank_column": "mean_map_intensity",
                "recommended_ground_truth_column": "mean_map_intensity",
                "n_rows": 2,
                "ready_for_manifest_alignment": True,
                "blocking_reasons": [],
            },
        ),
        encoding="utf-8",
    )
    feature_dir = tmp_path / "tribe_features"
    feature_dir.mkdir()
    np.savez_compressed(
        feature_dir / "video_a.npz",
        frames=np.zeros((2, 4), dtype=np.float32),
    )
    mask_path = (
        tmp_path
        / "research_program"
        / "dopamine_detox_attention_capture"
        / "results"
        / "destrieux_roi_masks_disjoint_20260608.npz"
    )
    mask_path.parent.mkdir(parents=True)
    np.savez_compressed(mask_path, V1=np.array([True]))

    report = module.build_readiness_report(
        search_roots=[tmp_path],
        repo_root=tmp_path,
        feature_sample_limit=4,
    )

    assert report["readiness"]["dhf1k_labels_ready"] is True
    assert report["readiness"]["dhf1k_root_ready_for_label_build"] is True
    assert report["readiness"]["dhf1k_label_audit_ready"] is True
    assert report["readiness"]["snapugc_labels_ready"] is True
    assert report["readiness"]["tribe_features_ready"] is True
    assert report["readiness"]["roi_masks_ready"] is True
    assert report["readiness"]["blocking_reasons"] == []
    assert report["dhf1k_candidates"][0]["ready_for_label_build"] is True
    assert report["dhf1k_label_audits"][0]["ready_for_handoff"] is True
    assert report["snapugc_label_candidates"][0]["n_rows"] == 2
    assert report["tribe_feature_dirs"][0]["ready_as_feature_cache"] is True
    assert report["roi_masks"]["disjoint"]["path"] == (
        "research_program/dopamine_detox_attention_capture/results/"
        "destrieux_roi_masks_disjoint_20260608.npz"
    )
    assert "Phase 1 Data Readiness Audit" in module.render_readiness_markdown(report)
    assert "DHF1K Label Audits" in module.render_readiness_markdown(report)


def test_dhf1k_root_without_ready_label_audit_blocks_label_handoff(
    tmp_path: Path,
) -> None:
    module = load_module()
    dhf1k = tmp_path / "DHF1K"
    (dhf1k / "video").mkdir(parents=True)
    (dhf1k / "video" / "001.AVI").write_bytes(b"fake")
    maps = dhf1k / "annotation" / "001" / "maps"
    maps.mkdir(parents=True)
    (maps / "0001.png").write_bytes(b"fake")
    mask_path = (
        tmp_path
        / "research_program"
        / "dopamine_detox_attention_capture"
        / "results"
        / "destrieux_roi_masks_disjoint_20260608.npz"
    )
    mask_path.parent.mkdir(parents=True)
    np.savez_compressed(mask_path, V1=np.array([True]))

    report = module.build_readiness_report(
        search_roots=[tmp_path],
        repo_root=tmp_path,
    )

    assert report["readiness"]["dhf1k_root_ready_for_label_build"] is True
    assert report["readiness"]["dhf1k_label_audit_ready"] is False
    assert report["readiness"]["dhf1k_labels_ready"] is False
    assert "DHF1K root found but no ready DHF1K label audit found" in report[
        "readiness"
    ]["blocking_reasons"]
    assert report["readiness"][
        "recommended_next_action"
    ] == "build DHF1K labels and confirm ready_for_manifest_alignment=true"


def test_synthetic_ecr_csv_is_not_real_snapugc_candidate(tmp_path: Path) -> None:
    module = load_module()
    synthetic_labels = tmp_path / "phase1_synthetic_alignment_labels.csv"
    synthetic_labels.write_text(
        "sample_id,ecr\nfixture_a,0.1\nfixture_b,0.9\n",
        encoding="utf-8",
    )
    synthetic_features = tmp_path / "fixtures" / "phase1_synthetic_features"
    synthetic_features.mkdir(parents=True)
    np.savez_compressed(
        synthetic_features / "fixture_a.npz",
        frames=np.zeros((2, 4), dtype=np.float32),
    )

    report = module.build_readiness_report(
        search_roots=[tmp_path],
        repo_root=tmp_path,
    )

    assert report["snapugc_label_candidates"] == []
    assert report["readiness"]["snapugc_labels_ready"] is False
    assert report["readiness"]["tribe_features_ready"] is False
    assert report["tribe_feature_dirs"][0]["claim_blocked"] is True
    assert report["tribe_feature_dirs"][0]["ready_as_feature_cache"] is False
    assert "no external attention-label source found" in report["readiness"][
        "blocking_reasons"
    ]


def test_claim_updatable_manifest_requires_alignment_provenance(
    tmp_path: Path,
) -> None:
    module = load_module()
    manifest = tmp_path / "phase1_snapugc_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "ready_for_phase1",
                "samples": [
                    {
                        "sample_id": "snapugc_001",
                        "dataset": "SnapUGC",
                        "ground_truth": 0.2,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    mask_path = (
        tmp_path
        / "research_program"
        / "dopamine_detox_attention_capture"
        / "results"
        / "destrieux_roi_masks_disjoint_20260608.npz"
    )
    mask_path.parent.mkdir(parents=True)
    np.savez_compressed(mask_path, V1=np.array([True]))

    report = module.build_readiness_report(
        search_roots=[tmp_path],
        repo_root=tmp_path,
    )

    manifest_audit = report["phase1_manifests"][0]
    assert manifest_audit["claim_blocked"] is False
    assert manifest_audit["provenance_required"] is True
    assert manifest_audit["provenance_ready"] is False
    assert manifest_audit["ready_for_workflow"] is False
    assert "claim-updatable manifest requires metadata.alignment_audit" in (
        manifest_audit["provenance_blocking_reasons"]
    )
    assert report["readiness"]["real_manifest_ready"] is False
    assert report["readiness"]["phase1_can_run_now"] is False
    assert any(
        reason.startswith("manifest not workflow-ready:")
        and "claim-updatable manifest requires metadata.alignment_audit" in reason
        for reason in report["readiness"]["blocking_reasons"]
    )
    assert "no external attention-label source found" not in report["readiness"][
        "blocking_reasons"
    ]
    assert report["readiness"][
        "recommended_next_action"
    ] == "fix Phase 1 manifest provenance, then rerun the guarded workflow"


def test_ready_manifest_provenance_unblocks_workflow_readiness(
    tmp_path: Path,
) -> None:
    module = load_module()
    manifest = tmp_path / "phase1_snapugc_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "ready_for_phase1",
                "metadata": {
                    "alignment_audit": {
                        "path": str(tmp_path / "phase1_snapugc_alignment.json"),
                        "sha256": "a" * 64,
                        "ready_for_manifest_build": True,
                        "n_aligned_features": 1,
                        "n_missing_features": 0,
                    },
                },
                "samples": [
                    {
                        "sample_id": "snapugc_001",
                        "dataset": "SnapUGC",
                        "ground_truth": 0.2,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    mask_path = (
        tmp_path
        / "research_program"
        / "dopamine_detox_attention_capture"
        / "results"
        / "destrieux_roi_masks_disjoint_20260608.npz"
    )
    mask_path.parent.mkdir(parents=True)
    np.savez_compressed(mask_path, V1=np.array([True]))

    report = module.build_readiness_report(
        search_roots=[tmp_path],
        repo_root=tmp_path,
    )

    manifest_audit = report["phase1_manifests"][0]
    assert manifest_audit["provenance_ready"] is True
    assert manifest_audit["ready_for_workflow"] is True
    assert report["readiness"]["real_manifest_ready"] is True
    assert report["readiness"]["phase1_can_run_now"] is True
    assert report["readiness"][
        "recommended_next_action"
    ] == "run scripts/run_attention_capture_phase1_workflow.py"


def test_synthetic_dataset_manifest_is_claim_blocked_even_with_real_status(
    tmp_path: Path,
) -> None:
    module = load_module()
    manifest = tmp_path / "phase1_synthetic_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "ready_for_phase1",
                "samples": [
                    {
                        "sample_id": "synthetic_001",
                        "dataset": "synthetic_alignment_fixture",
                        "ground_truth": 0.2,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    report = module.build_readiness_report(
        search_roots=[tmp_path],
        repo_root=tmp_path,
    )

    manifest_audit = report["phase1_manifests"][0]
    assert manifest_audit["claim_blocked"] is True
    assert manifest_audit["provenance_required"] is False
    assert manifest_audit["provenance_ready"] is True
    assert manifest_audit["ready_for_workflow"] is False


def test_build_readiness_report_blocks_when_external_assets_absent(
    tmp_path: Path,
) -> None:
    module = load_module()
    report = module.build_readiness_report(
        search_roots=[tmp_path],
        repo_root=tmp_path,
    )

    assert report["readiness"]["phase1_can_run_now"] is False
    assert "no external attention-label source found" in report["readiness"][
        "blocking_reasons"
    ]
    assert report["readiness"]["recommended_next_action"].startswith("acquire")
