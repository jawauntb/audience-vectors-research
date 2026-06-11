from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from audience_vectors.attention_capture_modal_volume import (
    run_phase1_modal_volume_features,
)


def load_script_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_attention_capture_phase1_modal_volume.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_attention_capture_phase1_modal_volume",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_modal_volume_phase1_scores_local_feature_fixture(tmp_path: Path) -> None:
    feature_root = tmp_path / "features"
    output_prefix = "attention_capture/DHF1K/full"
    feature_dir = feature_root / output_prefix
    feature_dir.mkdir(parents=True)
    label_records = []
    for i in range(5):
        frames = np.array(
            [
                [0.10 + i * 0.10, 0.20 + i * 0.10, 0.30 + i * 0.10, 1.0],
                [0.20 + i * 0.10, 0.30 + i * 0.10, 0.40 + i * 0.10, 1.0],
            ],
            dtype=np.float32,
        )
        np.savez_compressed(feature_dir / f"s{i}.npz", frames=frames)
        label_records.append(
            {"sample_id": f"s{i}", "ground_truth": float(i), "dataset": "DHF1K"}
        )

    report = run_phase1_modal_volume_features(
        label_records=label_records,
        roi_masks={
            "V1": np.array([True, False, False, False]),
            "PPA": np.array([False, True, False, False]),
            "language": np.array([False, False, True, False]),
            "frontoparietal": np.array([False, False, False, True]),
        },
        feature_root=feature_root,
        output_prefix=output_prefix,
        label_audit={
            "experiment": "dhf1k_attention_label_audit",
            "ready_for_manifest_alignment": True,
            "rank_column": "mean_fixation_density",
            "blocking_reasons": [],
        },
        min_samples=5,
        permutations=0,
    )

    assert report["mechanical_ready"] is True
    assert report["claim_validated"] is True
    assert report["feature_audit"]["n_existing"] == 5
    assert report["primary_report"]["pooled"]["metrics"]["capture_score"]["rho"] > 0.9


def test_modal_volume_phase1_blocks_missing_features(tmp_path: Path) -> None:
    report = run_phase1_modal_volume_features(
        label_records=[
            {"sample_id": "s0", "ground_truth": 0.0, "dataset": "DHF1K"},
            {"sample_id": "s1", "ground_truth": 1.0, "dataset": "DHF1K"},
            {"sample_id": "s2", "ground_truth": 2.0, "dataset": "DHF1K"},
        ],
        roi_masks={
            "V1": np.array([True, False, False, False]),
            "PPA": np.array([False, True, False, False]),
            "language": np.array([False, False, True, False]),
            "frontoparietal": np.array([False, False, False, True]),
        },
        feature_root=tmp_path,
        output_prefix="missing",
        label_audit={
            "experiment": "dhf1k_attention_label_audit",
            "ready_for_manifest_alignment": True,
            "rank_column": "mean_fixation_density",
            "blocking_reasons": [],
        },
        min_samples=3,
        permutations=0,
    )

    assert report["mechanical_ready"] is False
    assert report["score_decision"] == {
        "scoring_executed": False,
        "reason": "preflight_failed",
    }
    assert "3 feature files are missing" in report["blocking_reasons"]


def test_load_label_records_from_csv_uses_named_ground_truth(tmp_path: Path) -> None:
    module = load_script_module()
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "sample_id,mean_fixation_density\ns0,0.1\ns1,0.2\n",
        encoding="utf-8",
    )

    records = module.load_label_records_from_csv(
        labels_csv=labels_csv,
        sample_id_column="sample_id",
        ground_truth_column="mean_fixation_density",
        dataset="DHF1K",
    )

    assert records == [
        {"sample_id": "s0", "ground_truth": "0.1", "dataset": "DHF1K"},
        {"sample_id": "s1", "ground_truth": "0.2", "dataset": "DHF1K"},
    ]


def test_label_audit_metadata_blocks_mismatched_rank(tmp_path: Path) -> None:
    module = load_script_module()
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("sample_id,mean_fixation_density\ns0,0.1\n", encoding="utf-8")
    label_audit = tmp_path / "label_audit.json"
    label_audit.write_text(
        """
        {
          "experiment": "dhf1k_attention_label_audit",
          "labels_csv": "__LABELS__",
          "ready_for_manifest_alignment": true,
          "rank_column": "mean_map_intensity"
        }
        """.replace("__LABELS__", str(labels_csv)),
        encoding="utf-8",
    )

    metadata = module.load_label_audit_metadata(
        label_audit,
        labels_csv=labels_csv,
        ground_truth_column="mean_fixation_density",
    )

    assert metadata is not None
    assert (
        "label audit rank_column differs from ground_truth_column"
        in metadata["blocking_reasons"]
    )
