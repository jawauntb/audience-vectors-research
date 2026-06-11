from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_attention_capture_retention_baselines.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_attention_capture_retention_baselines",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_retention_baseline_detects_cheap_metadata_signal(tmp_path: Path) -> None:
    module = load_module()
    labels = tmp_path / "snapugc_labels.csv"
    label_audit = tmp_path / "retention_label_audit.json"
    labels.write_text(
        (
            "sample_id,video_path,ecr,duration_seconds,title\n"
            "snap_a,/videos/a.mp4,0.1,5,a\n"
            "snap_b,/videos/b.mp4,0.3,10,medium title\n"
            "snap_c,/videos/c.mp4,0.6,20,a longer title here\n"
            "snap_d,/videos/d.mp4,0.9,40,a much longer title goes here\n"
        ),
        encoding="utf-8",
    )
    label_audit.write_text(
        json.dumps(
            {
                "experiment": "attention_capture_retention_label_audit",
                "dataset": "SnapUGC",
                "labels_csv": str(labels),
                "sample_id_column": "sample_id",
                "ground_truth_column": "ecr",
                "ground_truth_name": "ecr",
                "ready_for_manifest_alignment": True,
                "n_rows": 4,
                "blocking_reasons": [],
            },
        ),
        encoding="utf-8",
    )

    report = module.audit_retention_baselines(
        labels_csv=labels,
        label_audit=label_audit,
        dataset="SnapUGC",
        min_samples=4,
        min_distinct_ground_truth=4,
        min_baseline_abs_rho=0.5,
        max_control_abs_rho=1.1,
    )
    markdown = module.render_retention_baseline_markdown(report)

    assert report["ready_for_modal_slice"] is True
    assert report["baseline_signal_detected"] is True
    assert report["best_feature"]["feature"] == "duration_seconds"
    assert report["label_audit"]["labels_csv_relation"] == "same"
    assert "Retention Baseline Audit" in markdown
    assert "duration_seconds" in markdown


def test_retention_baseline_blocks_broken_labels(tmp_path: Path) -> None:
    module = load_module()
    labels = tmp_path / "snapugc_labels.csv"
    labels.write_text(
        (
            "sample_id,video_path,ecr,duration_seconds\n"
            "snap_a,/videos/a.mp4,0.1,5\n"
            "snap_a,/videos/b.mp4,not-a-number,10\n"
            "snap_c,/videos/c.mp4,0.1,20\n"
        ),
        encoding="utf-8",
    )

    report = module.audit_retention_baselines(
        labels_csv=labels,
        dataset="SnapUGC",
        min_samples=3,
        min_distinct_ground_truth=3,
    )

    assert report["ready_for_modal_slice"] is False
    assert "1 duplicate sample ids found" in report["blocking_reasons"]
    assert (
        "finite ground-truth count 2 is below minimum 3"
        in report["blocking_reasons"]
    )
    assert (
        "distinct finite ground-truth count 1 is below minimum 3"
        in report["blocking_reasons"]
    )


def test_retention_baseline_flags_row_order_control(tmp_path: Path) -> None:
    module = load_module()
    labels = tmp_path / "snapugc_labels.csv"
    labels.write_text(
        (
            "sample_id,video_path,ecr,duration_seconds\n"
            "snap_a,/videos/a.mp4,0.1,10\n"
            "snap_b,/videos/b.mp4,0.2,10\n"
            "snap_c,/videos/c.mp4,0.3,10\n"
            "snap_d,/videos/d.mp4,0.4,10\n"
        ),
        encoding="utf-8",
    )

    report = module.audit_retention_baselines(
        labels_csv=labels,
        dataset="SnapUGC",
        min_samples=4,
        min_distinct_ground_truth=4,
        max_control_abs_rho=0.3,
    )

    assert report["ready_for_modal_slice"] is False
    assert any("negative control row_index" in warning for warning in report["warnings"])
    assert any("label leakage" in action for action in report["next_actions"])
