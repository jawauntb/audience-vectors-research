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
        / "audit_attention_capture_manifest_alignment.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_attention_capture_manifest_alignment",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_alignment_audit_accepts_ready_label_feature_join(tmp_path: Path) -> None:
    module = load_module()
    labels = tmp_path / "labels.csv"
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    labels.write_text(
        "sample_id,ecr\nvideo_a,0.2\nvideo_b,0.5\nvideo_c,0.8\n",
        encoding="utf-8",
    )
    for sample_id in ("video_a", "video_b", "video_c"):
        np.savez_compressed(feature_dir / f"{sample_id}.npz", frames=np.zeros(4))

    report = module.audit_manifest_alignment(
        labels_csv=labels,
        feature_dir=feature_dir,
        dataset="SnapUGC",
        ground_truth_column="ecr",
        min_samples=3,
        min_distinct_ground_truth=3,
    )

    assert report["ready_for_manifest_build"] is True
    assert report["n_aligned_features"] == 3
    assert report["n_missing_features"] == 0
    assert report["label_audit"]["ready_for_manifest_alignment"] is None
    assert report["blocking_reasons"] == []
    assert "Phase 1 Manifest Alignment Audit" in module.render_alignment_markdown(
        report,
    )


def test_alignment_audit_records_ready_dhf1k_label_audit(tmp_path: Path) -> None:
    module = load_module()
    labels = tmp_path / "labels.csv"
    label_audit = tmp_path / "label_audit.json"
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    labels.write_text(
        (
            "sample_id,mean_map_intensity\n"
            "dhf1k_001,0.2\n"
            "dhf1k_002,0.5\n"
            "dhf1k_003,0.8\n"
        ),
        encoding="utf-8",
    )
    label_audit.write_text(
        json.dumps(
            {
                "experiment": "dhf1k_attention_label_audit",
                "dataset": "DHF1K",
                "labels_csv": str(labels),
                "rank_column": "mean_map_intensity",
                "ready_for_manifest_alignment": True,
                "recommended_ground_truth_column": "mean_map_intensity",
                "n_rows": 3,
                "blocking_reasons": [],
            },
        ),
        encoding="utf-8",
    )
    for sample_id in ("dhf1k_001", "dhf1k_002", "dhf1k_003"):
        np.savez_compressed(feature_dir / f"{sample_id}.npz", frames=np.zeros(4))

    report = module.audit_manifest_alignment(
        labels_csv=labels,
        feature_dir=feature_dir,
        label_audit=label_audit,
        dataset="DHF1K",
        ground_truth_column="mean_map_intensity",
        min_samples=3,
        min_distinct_ground_truth=3,
    )

    assert report["ready_for_manifest_build"] is True
    assert report["label_audit"]["path"] == str(label_audit)
    assert len(report["label_audit"]["sha256"]) == 64
    assert report["label_audit"]["rank_column"] == "mean_map_intensity"
    assert report["label_audit"]["blocking_reasons"] == []
    assert "- Label audit ready: True" in module.render_alignment_markdown(report)


def test_alignment_audit_accepts_exact_subset_of_dhf1k_label_audit(
    tmp_path: Path,
) -> None:
    module = load_module()
    audited_labels = tmp_path / "audited_labels.csv"
    subset_labels = tmp_path / "subset_labels.csv"
    label_audit = tmp_path / "label_audit.json"
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    audited_labels.write_text(
        (
            "sample_id,mean_map_intensity,selected_tail\n"
            "dhf1k_001,0.2,low\n"
            "dhf1k_002,0.5,high\n"
            "dhf1k_003,0.8,high\n"
        ),
        encoding="utf-8",
    )
    subset_labels.write_text(
        (
            "sample_id,mean_map_intensity,selected_tail\n"
            "dhf1k_001,0.2,low\n"
            "dhf1k_003,0.8,high\n"
        ),
        encoding="utf-8",
    )
    label_audit.write_text(
        json.dumps(
            {
                "experiment": "dhf1k_attention_label_audit",
                "dataset": "DHF1K",
                "labels_csv": str(audited_labels),
                "rank_column": "mean_map_intensity",
                "ready_for_manifest_alignment": True,
                "recommended_ground_truth_column": "mean_map_intensity",
                "n_rows": 3,
                "blocking_reasons": [],
            },
        ),
        encoding="utf-8",
    )
    for sample_id in ("dhf1k_001", "dhf1k_003"):
        np.savez_compressed(feature_dir / f"{sample_id}.npz", frames=np.zeros(4))

    report = module.audit_manifest_alignment(
        labels_csv=subset_labels,
        feature_dir=feature_dir,
        label_audit=label_audit,
        dataset="DHF1K",
        ground_truth_column="mean_map_intensity",
        min_samples=2,
        min_distinct_ground_truth=2,
    )

    assert report["ready_for_manifest_build"] is True
    assert report["label_audit"]["labels_csv_relation"] == "subset"
    assert report["label_audit"]["blocking_reasons"] == []


def test_alignment_audit_blocks_unready_dhf1k_label_audit(tmp_path: Path) -> None:
    module = load_module()
    labels = tmp_path / "labels.csv"
    label_audit = tmp_path / "label_audit.json"
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    labels.write_text(
        (
            "sample_id,mean_map_intensity\n"
            "dhf1k_001,0.2\n"
            "dhf1k_002,0.5\n"
            "dhf1k_003,0.8\n"
        ),
        encoding="utf-8",
    )
    label_audit.write_text(
        json.dumps(
            {
                "experiment": "dhf1k_attention_label_audit",
                "dataset": "DHF1K",
                "labels_csv": str(labels),
                "rank_column": "mean_map_intensity",
                "ready_for_manifest_alignment": False,
                "recommended_ground_truth_column": None,
                "n_rows": 3,
                "blocking_reasons": ["mean_map_intensity has zero variance"],
            },
        ),
        encoding="utf-8",
    )
    for sample_id in ("dhf1k_001", "dhf1k_002", "dhf1k_003"):
        np.savez_compressed(feature_dir / f"{sample_id}.npz", frames=np.zeros(4))

    report = module.audit_manifest_alignment(
        labels_csv=labels,
        feature_dir=feature_dir,
        label_audit=label_audit,
        dataset="DHF1K",
        ground_truth_column="mean_map_intensity",
        min_samples=3,
        min_distinct_ground_truth=3,
    )

    assert report["ready_for_manifest_build"] is False
    assert (
        "label audit is not ready: mean_map_intensity has zero variance"
        in report["blocking_reasons"]
    )


def test_alignment_audit_blocks_missing_duplicate_and_bad_labels(
    tmp_path: Path,
) -> None:
    module = load_module()
    labels = tmp_path / "labels.csv"
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    labels.write_text(
        "sample_id,ecr\nvideo_a,0.2\nvideo_a,not_a_number\nvideo_c,0.8\n",
        encoding="utf-8",
    )
    np.savez_compressed(feature_dir / "video_a.npz", frames=np.zeros(4))

    report = module.audit_manifest_alignment(
        labels_csv=labels,
        feature_dir=feature_dir,
        dataset="SnapUGC",
        ground_truth_column="ecr",
        min_samples=3,
        min_distinct_ground_truth=3,
    )

    assert report["ready_for_manifest_build"] is False
    assert report["n_duplicate_sample_ids"] == 1
    assert report["n_invalid_ground_truth"] == 1
    assert report["n_missing_features"] == 1
    assert "aligned feature count 2 is below minimum 3" in report["blocking_reasons"]
    assert "1 duplicate sample ids found" in report["blocking_reasons"]
    assert "1 rows have non-finite ground truth" in report["blocking_reasons"]
