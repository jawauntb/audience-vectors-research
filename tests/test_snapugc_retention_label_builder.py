from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_snapugc_retention_labels.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_snapugc_retention_labels",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_snapugc_retention_labels_joins_official_columns(
    tmp_path: Path,
) -> None:
    module = load_module()
    metadata = tmp_path / "val_data.csv"
    scores = tmp_path / "labels.csv"
    output = tmp_path / "canonical.csv"
    metadata.write_text(
        (
            "Id,Title,Description,Download_link\n"
            "snap_a,A title,A description,https://example.test/a.mp4\n"
            "snap_b,B title,B description,https://example.test/b.mp4\n"
            "snap_c,C title,C description,https://example.test/c.mp4\n"
        ),
        encoding="utf-8",
    )
    scores.write_text("Id,ECR\nsnap_a,0.1\nsnap_b,0.4\nsnap_c,0.8\n", encoding="utf-8")

    report = module.build_snapugc_retention_labels(
        metadata_csv=metadata,
        scores_csv=scores,
        output_csv=output,
        min_samples=3,
        min_distinct_ecr=3,
    )

    assert report["ready_for_retention_audit"] is True
    assert report["n_output_rows"] == 3
    assert report["blocking_reasons"] == []
    rows = output.read_text(encoding="utf-8")
    assert "sample_id,video_path,ecr,title,description,download_link" in rows
    assert "snap_a,https://example.test/a.mp4,0.1,A title" in rows
    assert "SnapUGC Retention Label Builder" in module.render_snapugc_builder_markdown(
        report
    )


def test_build_snapugc_retention_labels_uses_media_path_template(
    tmp_path: Path,
) -> None:
    module = load_module()
    metadata = tmp_path / "val_data.csv"
    scores = tmp_path / "labels.csv"
    output = tmp_path / "canonical.csv"
    metadata.write_text(
        (
            "Id,Title,Description,Download_link\n"
            "snap_a,A title,A description,https://example.test/a.mp4\n"
            "snap_b,B title,B description,https://example.test/b.mp4\n"
            "snap_c,C title,C description,https://example.test/c.mp4\n"
        ),
        encoding="utf-8",
    )
    scores.write_text("Id,ECR\nsnap_a,0.1\nsnap_b,0.4\nsnap_c,0.8\n", encoding="utf-8")

    report = module.build_snapugc_retention_labels(
        metadata_csv=metadata,
        scores_csv=scores,
        output_csv=output,
        media_path_template="/bmd-videos/attention_capture/SnapUGC/{sample_id}.mp4",
        min_samples=3,
        min_distinct_ecr=3,
    )

    assert report["ready_for_retention_audit"] is True
    assert "/bmd-videos/attention_capture/SnapUGC/snap_a.mp4" in output.read_text(
        encoding="utf-8"
    )


def test_build_snapugc_retention_labels_blocks_duplicates_and_missing_media(
    tmp_path: Path,
) -> None:
    module = load_module()
    metadata = tmp_path / "val_data.csv"
    scores = tmp_path / "labels.csv"
    output = tmp_path / "canonical.csv"
    metadata.write_text(
        (
            "Id,Title,Description,Download_link\n"
            "snap_a,A title,A description,\n"
            "snap_a,A title again,A description again,\n"
            "snap_c,C title,C description,\n"
        ),
        encoding="utf-8",
    )
    scores.write_text("Id,ECR\nsnap_a,0.1\nsnap_c,0.8\nsnap_c,0.9\n", encoding="utf-8")

    report = module.build_snapugc_retention_labels(
        metadata_csv=metadata,
        scores_csv=scores,
        output_csv=output,
        require_local_video=True,
        min_samples=3,
        min_distinct_ecr=3,
    )

    assert report["ready_for_retention_audit"] is False
    assert "3 rows are missing usable media paths" in report["blocking_reasons"]
    assert "1 duplicate metadata ids found" in report["blocking_reasons"]
    assert "1 duplicate score ids found" in report["blocking_reasons"]


def test_build_snapugc_retention_labels_blocks_baseline_prediction_file(
    tmp_path: Path,
) -> None:
    module = load_module()
    metadata = tmp_path / "val_data.csv"
    scores = tmp_path / "submission_baseline.csv"
    output = tmp_path / "canonical.csv"
    metadata.write_text(
        (
            "Id,Title,Description,Download_link\n"
            "snap_a,A title,A description,https://example.test/a.mp4\n"
            "snap_b,B title,B description,https://example.test/b.mp4\n"
            "snap_c,C title,C description,https://example.test/c.mp4\n"
        ),
        encoding="utf-8",
    )
    scores.write_text("Id,ECR\nsnap_a,0.1\nsnap_b,0.4\nsnap_c,0.8\n", encoding="utf-8")

    blocked = module.build_snapugc_retention_labels(
        metadata_csv=metadata,
        scores_csv=scores,
        output_csv=output,
        min_samples=3,
        min_distinct_ecr=3,
    )
    allowed = module.build_snapugc_retention_labels(
        metadata_csv=metadata,
        scores_csv=scores,
        output_csv=output,
        allow_prediction_score_file=True,
        min_samples=3,
        min_distinct_ecr=3,
    )

    assert blocked["ready_for_retention_audit"] is False
    assert any("prediction output" in reason for reason in blocked["blocking_reasons"])
    assert allowed["ready_for_retention_audit"] is True


def test_build_snapugc_retention_labels_cli(tmp_path: Path) -> None:
    import subprocess

    metadata = tmp_path / "val_data.csv"
    scores = tmp_path / "labels.csv"
    output = tmp_path / "canonical.csv"
    output_json = tmp_path / "builder.json"
    output_md = tmp_path / "builder.md"
    metadata.write_text(
        (
            "Id,Title,Description,Download_link\n"
            "snap_a,A title,A description,https://example.test/a.mp4\n"
            "snap_b,B title,B description,https://example.test/b.mp4\n"
            "snap_c,C title,C description,https://example.test/c.mp4\n"
        ),
        encoding="utf-8",
    )
    scores.write_text("Id,ECR\nsnap_a,0.1\nsnap_b,0.4\nsnap_c,0.8\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/build_snapugc_retention_labels.py",
            "--metadata-csv",
            str(metadata),
            "--scores-csv",
            str(scores),
            "--output-csv",
            str(output),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--min-samples",
            "3",
            "--min-distinct-ecr",
            "3",
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["ready_for_retention_audit"] is True
    assert output.exists()
    assert output_md.exists()
