from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_attention_capture_dhf1k_modal_media.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_attention_capture_dhf1k_modal_media",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_modal_video_path_rewrites_committed_dhf1k_path() -> None:
    module = load_module()

    path = module.modal_video_path_from_source(
        "data/attention_capture/DHF1K/video/003.AVI",
        modal_root="/bmd-videos",
        modal_prefix="attention_capture/DHF1K",
        video_id="003",
    )

    assert path == "/bmd-videos/attention_capture/DHF1K/video/003.AVI"


def test_candidate_modal_video_paths_include_common_id_and_suffix_variants() -> None:
    module = load_module()

    candidates = module.candidate_modal_video_paths(
        "/bmd-videos/attention_capture/DHF1K/video/003.AVI",
        video_id="003",
    )

    assert candidates[0] == "/bmd-videos/attention_capture/DHF1K/video/003.AVI"
    assert "/bmd-videos/attention_capture/DHF1K/video/0003.AVI" in candidates
    assert "/bmd-videos/attention_capture/DHF1K/video/003.mp4" in candidates


def test_report_blocks_when_modal_videos_are_missing() -> None:
    module = load_module()

    report = module.build_report(
        labels_csv=Path("labels.csv"),
        output_modal_csv=Path("labels_modal.csv"),
        sample_id_column="sample_id",
        video_id_column="video_id",
        media_path_column="video_path",
        modal_root="/bmd-videos",
        modal_prefix="attention_capture/DHF1K",
        remote_audit={
            "modal_root": "/bmd-videos",
            "modal_prefix": "attention_capture/DHF1K",
            "volume_mount_exists": True,
            "prefix_exists": False,
            "video_dir_exists": False,
            "n_expected": 2,
            "n_found": 0,
            "n_missing": 2,
            "n_zero_byte_found": 0,
            "missing_preview": [
                {
                    "sample_id": "dhf1k_003",
                    "expected_modal_video_path": (
                        "/bmd-videos/attention_capture/DHF1K/video/003.AVI"
                    ),
                }
            ],
            "found_preview": [],
            "checked_rows": [],
        },
    )

    markdown = module.render_dhf1k_modal_media_markdown(report)

    assert report["ready_for_full_feature_extraction"] is False
    assert "2 expected DHF1K videos are missing" in report["blocking_reasons"][-1]
    assert "dhf1k_003" in markdown


def test_write_modal_path_csv_preserves_source_and_uses_found_path(
    tmp_path: Path,
) -> None:
    module = load_module()
    labels = tmp_path / "labels.csv"
    out = tmp_path / "labels_modal.csv"
    labels.write_text(
        "sample_id,video_id,video_path,mean_fixation_density\n"
        "dhf1k_003,003,data/attention_capture/DHF1K/video/003.AVI,0.1\n",
        encoding="utf-8",
    )

    module.write_modal_path_csv(
        labels_csv=labels,
        output_csv=out,
        checked_rows=[
            {
                "sample_id": "dhf1k_003",
                "expected_modal_video_path": (
                    "/bmd-videos/attention_capture/DHF1K/video/003.AVI"
                ),
                "found": True,
                "found_path": "/bmd-videos/attention_capture/DHF1K/video/003.mp4",
                "size_bytes": 123,
            }
        ],
        sample_id_column="sample_id",
        media_path_column="video_path",
    )

    with out.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["video_path"] == "/bmd-videos/attention_capture/DHF1K/video/003.mp4"
    assert row["source_video_path"] == "data/attention_capture/DHF1K/video/003.AVI"
    assert row["modal_media_found"] == "True"
    assert row["modal_media_size_bytes"] == "123"
