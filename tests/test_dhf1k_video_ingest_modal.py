from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "ingest_dhf1k_videos_modal.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ingest_dhf1k_videos_modal",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_expected_video_ids_from_labels_dedupes_and_normalizes(tmp_path: Path) -> None:
    module = load_module()
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "sample_id,video_id\n"
        "dhf1k_003,3\n"
        "dhf1k_003,003\n"
        "dhf1k_701,\n",
        encoding="utf-8",
    )

    ids = module.expected_video_ids_from_labels(
        labels,
        sample_id_column="sample_id",
        video_id_column="video_id",
    )

    assert ids == ["003", "701"]


def test_find_extracted_video_dir_selects_largest_video_directory(
    tmp_path: Path,
) -> None:
    module = load_module()
    small = tmp_path / "other" / "video"
    large = tmp_path / "DHF1K" / "video"
    small.mkdir(parents=True)
    large.mkdir(parents=True)
    (small / "001.AVI").write_bytes(b"1")
    (large / "001.AVI").write_bytes(b"1")
    (large / "002.AVI").write_bytes(b"2")

    assert module.find_extracted_video_dir(tmp_path) == large


def test_copy_videos_to_volume_copies_expected_subset(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "003.AVI").write_bytes(b"video-003")
    (source / "004.AVI").write_bytes(b"video-004")

    report = module.copy_videos_to_volume(
        source_video_dir=source,
        target_video_dir=target,
        expected_video_ids=["003", "004"],
        copy_all_videos=False,
    )

    assert report["ok"] is True
    assert report["n_copied"] == 2
    assert report["n_missing_expected"] == 0
    assert (target / "003.AVI").read_bytes() == b"video-003"
    assert (target / "004.AVI").read_bytes() == b"video-004"


def test_copy_videos_to_volume_reports_missing_expected_ids(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "003.AVI").write_bytes(b"video-003")

    report = module.copy_videos_to_volume(
        source_video_dir=source,
        target_video_dir=target,
        expected_video_ids=["003", "004"],
        copy_all_videos=False,
    )

    assert report["ok"] is False
    assert report["missing_expected_video_ids"] == ["004"]
