from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "ingest_snapugc_videos_modal.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ingest_snapugc_videos_modal",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_split_zip_parts_detects_complete_sequence(tmp_path: Path) -> None:
    module = load_module()
    folder = tmp_path / "drive"
    folder.mkdir()
    for suffix in ("z01", "z02", "z03", "zip"):
        (folder / f"train_videos_split.{suffix}").write_bytes(b"part")

    parts = module.find_split_zip_parts(tmp_path, stem="train_videos_split")

    assert parts.main_zip == folder / "train_videos_split.zip"
    assert [path.name for path in parts.numbered_parts] == [
        "train_videos_split.z01",
        "train_videos_split.z02",
        "train_videos_split.z03",
    ]
    assert parts.missing_part_suffixes == []


def test_official_archive_part_ids_cover_complete_split_archive() -> None:
    module = load_module()
    expected_names = [
        *(f"train_videos_split.z{index:02d}" for index in range(1, 15)),
        "train_videos_split.zip",
    ]

    assert list(module.OFFICIAL_SNAPUGC_TRAIN_VIDEO_PART_IDS) == expected_names
    assert all(module.OFFICIAL_SNAPUGC_TRAIN_VIDEO_PART_IDS.values())


def test_find_split_zip_parts_reports_missing_sequence_gap(tmp_path: Path) -> None:
    module = load_module()
    folder = tmp_path / "drive"
    folder.mkdir()
    for suffix in ("z01", "z03", "zip"):
        (folder / f"train_videos_split.{suffix}").write_bytes(b"part")

    parts = module.find_split_zip_parts(tmp_path, stem="train_videos_split")

    assert parts.missing_part_suffixes == ["z02"]


def test_collect_video_paths_is_recursive_and_case_insensitive(tmp_path: Path) -> None:
    module = load_module()
    (tmp_path / "a" / "nested").mkdir(parents=True)
    (tmp_path / "a" / "one.MP4").write_bytes(b"1")
    (tmp_path / "a" / "nested" / "two.webm").write_bytes(b"2")
    (tmp_path / "a" / "notes.txt").write_text("ignore", encoding="utf-8")

    paths = module.collect_video_paths(tmp_path)

    assert {path.name for path in paths} == {"one.MP4", "two.webm"}


def test_archive_part_ready_requires_nonempty_file(tmp_path: Path) -> None:
    module = load_module()
    archive_part = tmp_path / "train_videos_split.z01"

    assert module.archive_part_ready(archive_part) is False
    archive_part.write_bytes(b"")
    assert module.archive_part_ready(archive_part) is False
    archive_part.write_bytes(b"part")
    assert module.archive_part_ready(archive_part) is True


def test_archive_download_report_counts_part_statuses(tmp_path: Path) -> None:
    module = load_module()
    files = [
        {"status": "existing", "path": str(tmp_path / "train_videos_split.z01")},
        {"status": "downloaded", "path": str(tmp_path / "train_videos_split.z02")},
        {"status": "failed", "path": str(tmp_path / "train_videos_split.z03")},
    ]

    report = module.build_archive_download_report(
        archive_dir=tmp_path,
        files=files,
        paths=[file["path"] for file in files[:2]],
        mode="official_file_ids",
        preview_limit=1,
    )

    assert report["ok"] is False
    assert report["n_existing"] == 1
    assert report["n_downloaded"] == 1
    assert report["n_failed"] == 1
    assert report["paths_preview"] == [str(tmp_path / "train_videos_split.z01")]


def test_copy_videos_to_volume_handles_duplicate_names(tmp_path: Path) -> None:
    module = load_module()
    source_a = tmp_path / "source" / "a"
    source_b = tmp_path / "source" / "b"
    target = tmp_path / "target"
    source_a.mkdir(parents=True)
    source_b.mkdir(parents=True)
    target.mkdir()
    video_a = source_a / "clip.mp4"
    video_b = source_b / "clip.mp4"
    video_a.write_bytes(b"a")
    video_b.write_bytes(b"bb")

    report = module.copy_videos_to_volume(
        source_video_paths=[video_a, video_b],
        target_video_dir=target,
        preview_limit=10,
    )

    assert report["ok"] is True
    assert report["n_copied"] == 2
    assert (target / "a__clip.mp4").read_bytes() == b"a"
    assert (target / "b__clip.mp4").read_bytes() == b"bb"
