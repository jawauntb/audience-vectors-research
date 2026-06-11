from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_attention_capture_retention_labels.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_attention_capture_retention_labels",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_retention_label_audit_accepts_ready_ecr_csv(tmp_path: Path) -> None:
    module = load_module()
    labels = tmp_path / "snapugc_vquala_labels.csv"
    labels.write_text(
        "sample_id,video_path,ecr\n"
        "snap_001,/bmd-videos/snap_001.mp4,0.1\n"
        "snap_002,/bmd-videos/snap_002.mp4,0.4\n"
        "snap_003,/bmd-videos/snap_003.mp4,0.8\n",
        encoding="utf-8",
    )

    report = module.audit_retention_labels(
        labels_csv=labels,
        dataset="SnapUGC",
        min_samples=3,
        min_distinct_ground_truth=3,
    )

    assert report["experiment"] == "attention_capture_retention_label_audit"
    assert report["sample_id_column"] == "sample_id"
    assert report["ground_truth_column"] == "ecr"
    assert report["media_path_column"] == "video_path"
    assert report["ready_for_manifest_alignment"] is True
    assert report["ready_for_modal_feature_extraction"] is True
    assert report["blocking_reasons"] == []
    assert "Retention Label Audit" in module.render_retention_label_markdown(report)


def test_retention_label_audit_blocks_bad_or_duplicate_labels(tmp_path: Path) -> None:
    module = load_module()
    labels = tmp_path / "snapugc_vquala_labels.csv"
    labels.write_text(
        "sample_id,video_path,ecr\n"
        "snap_001,/bmd-videos/snap_001.mp4,0.1\n"
        "snap_001,/bmd-videos/snap_001b.mp4,not-a-number\n"
        "snap_003,,0.1\n",
        encoding="utf-8",
    )

    report = module.audit_retention_labels(
        labels_csv=labels,
        dataset="SnapUGC",
        min_samples=3,
        min_distinct_ground_truth=3,
    )

    assert report["ready_for_manifest_alignment"] is False
    assert report["ready_for_modal_feature_extraction"] is False
    assert "1 duplicate sample ids found" in report["blocking_reasons"]
    assert "1 rows have non-finite ground truth" in report["blocking_reasons"]
    assert "ground truth has too few distinct finite values" in report["blocking_reasons"]


def test_retention_label_audit_blocks_missing_required_columns(tmp_path: Path) -> None:
    module = load_module()
    labels = tmp_path / "labels.csv"
    labels.write_text("title,score\nhello,0.1\nworld,0.2\n", encoding="utf-8")

    report = module.audit_retention_labels(
        labels_csv=labels,
        dataset="SnapUGC",
        min_samples=2,
        min_distinct_ground_truth=2,
    )

    assert "no sample-id column found" in report["blocking_reasons"]
    assert "no retention ground-truth column found" in report["blocking_reasons"]
