from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_build_attention_capture_phase1_manifest_cli(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    feature_dir = tmp_path / "features"
    output = tmp_path / "manifest.json"
    feature_dir.mkdir()
    labels.write_text(
        "sample_id,ecr\nvideo_a,0.8\nvideo_b,0.2\n",
        encoding="utf-8",
    )
    for sample_id in ("video_a", "video_b"):
        np.savez_compressed(feature_dir / f"{sample_id}.npz", frames=np.zeros(4))

    subprocess.run(
        [
            sys.executable,
            "scripts/build_attention_capture_phase1_manifest.py",
            "--labels-csv",
            str(labels),
            "--feature-dir",
            str(feature_dir),
            "--output",
            str(output),
            "--dataset",
            "SnapUGC",
            "--ground-truth-name",
            "ECR",
            "--ground-truth-column",
            "ecr",
        ],
        check=True,
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["status"] == "real_external_attention_labels"
    assert manifest["dataset"] == "SnapUGC"
    assert manifest["ground_truth_name"] == "ECR"
    assert manifest["metadata"]["n_samples"] == 2
    assert manifest["metadata"]["n_missing_features"] == 0
    assert manifest["samples"][0] == {
        "sample_id": "video_a",
        "dataset": "SnapUGC",
        "ground_truth": 0.8,
        "ground_truth_name": "ECR",
        "tribe_feature_path": str((feature_dir / "video_a.npz").resolve()),
    }


def test_build_manifest_records_ready_alignment_audit(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    feature_dir = tmp_path / "features"
    alignment = tmp_path / "alignment.json"
    alignment_md = tmp_path / "alignment.md"
    output = tmp_path / "manifest.json"
    feature_dir.mkdir()
    labels.write_text(
        "sample_id,ecr\nvideo_a,0.2\nvideo_b,0.5\nvideo_c,0.8\n",
        encoding="utf-8",
    )
    for sample_id in ("video_a", "video_b", "video_c"):
        np.savez_compressed(feature_dir / f"{sample_id}.npz", frames=np.zeros(4))

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_attention_capture_manifest_alignment.py",
            "--labels-csv",
            str(labels),
            "--feature-dir",
            str(feature_dir),
            "--output-json",
            str(alignment),
            "--output-md",
            str(alignment_md),
            "--dataset",
            "SnapUGC",
            "--ground-truth-column",
            "ecr",
            "--min-samples",
            "3",
            "--min-distinct-ground-truth",
            "3",
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/build_attention_capture_phase1_manifest.py",
            "--labels-csv",
            str(labels),
            "--feature-dir",
            str(feature_dir),
            "--output",
            str(output),
            "--dataset",
            "SnapUGC",
            "--ground-truth-name",
            "ECR",
            "--ground-truth-column",
            "ecr",
            "--alignment-audit",
            str(alignment),
        ],
        check=True,
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    audit_metadata = manifest["metadata"]["alignment_audit"]
    assert audit_metadata["path"] == str(alignment)
    assert audit_metadata["ready_for_manifest_build"] is True
    assert audit_metadata["n_aligned_features"] == 3
    assert len(audit_metadata["sha256"]) == 64


def test_build_manifest_rejects_blocked_alignment_audit(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    feature_dir = tmp_path / "features"
    alignment = tmp_path / "alignment.json"
    output = tmp_path / "manifest.json"
    feature_dir.mkdir()
    labels.write_text("sample_id,ecr\nvideo_a,0.2\n", encoding="utf-8")
    np.savez_compressed(feature_dir / "video_a.npz", frames=np.zeros(4))
    alignment.write_text(
        json.dumps(
            {
                "experiment": "phase1_manifest_alignment_audit",
                "ready_for_manifest_build": False,
                "blocking_reasons": ["aligned feature count 1 is below minimum 30"],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_attention_capture_phase1_manifest.py",
            "--labels-csv",
            str(labels),
            "--feature-dir",
            str(feature_dir),
            "--output",
            str(output),
            "--dataset",
            "SnapUGC",
            "--ground-truth-name",
            "ECR",
            "--ground-truth-column",
            "ecr",
            "--alignment-audit",
            str(alignment),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "is not ready" in result.stderr
    assert not output.exists()
