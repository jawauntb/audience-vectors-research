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
