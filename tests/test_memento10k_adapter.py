"""Tests for the Memento10k adapter.

Uses a synthetic mini-dataset on disk so the test doesn't require the real
~10k-video download. If you later download Memento10k and the JSON schema
differs from what this adapter expects, this fixture will go stale and the
test will catch the mismatch when you point the adapter at real data —
before any downstream segmentation/labeling work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audience_vectors.datasets import Memento10kAdapter
from audience_vectors.datasets.memento10k import ANNOTATION_FILES


def _build_fake_memento(root: Path) -> None:
    """Construct a tiny Memento10k-shaped directory under `root`."""
    (root / "videos").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)

    # Two splits, two videos each. video files don't need real bytes — the
    # adapter only emits CanonicalVideo rows pointing at paths.
    splits = {
        "train": [
            {
                "filename": "video_00001.webm",
                "mem_score": 0.87,
                "alpha": -0.012,
                "captions": ["a dog runs across a beach", "a brown dog runs"],
                "actions": ["running", "dog"],
            },
            {
                "filename": "video_00002.webm",
                "mem_score": 0.62,
                "alpha": -0.018,
                "captions": ["a child kicks a ball"],
                "actions": ["kicking"],
            },
        ],
        "val": [
            {
                "filename": "video_07001.webm",
                "mem_score": 0.74,
                "alpha": -0.009,
                "captions": ["sunset over a city skyline"],
                "actions": [],
            },
        ],
    }
    for split, entries in splits.items():
        (root / "annotations" / ANNOTATION_FILES[split]).write_text(
            json.dumps(entries), encoding="utf-8",
        )
        for entry in entries:
            (root / "videos" / entry["filename"]).touch()


def test_adapter_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Memento10kAdapter(tmp_path / "does_not_exist")


def test_adapter_rejects_unknown_split(tmp_path: Path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "annotations").mkdir()
    with pytest.raises(ValueError, match="unknown Memento10k splits"):
        Memento10kAdapter(tmp_path, splits=("train", "wat"))


def test_adapter_yields_canonical_videos_across_splits(tmp_path: Path):
    _build_fake_memento(tmp_path)
    adapter = Memento10kAdapter(tmp_path)
    videos = list(adapter)

    # Two train + one val
    assert len(videos) == 3
    assert {v.source_dataset for v in videos} == {"Memento10k"}
    assert all(v.domain == "everyday" for v in videos)


def test_adapter_skips_missing_split_files(tmp_path: Path):
    # Only create train, request all three.
    _build_fake_memento(tmp_path)
    (tmp_path / "annotations" / ANNOTATION_FILES["val"]).unlink()
    adapter = Memento10kAdapter(tmp_path, splits=("train", "val", "test"))
    videos = list(adapter)
    assert len(videos) == 2  # only train survived


def test_raw_labels_and_metadata_populate(tmp_path: Path):
    _build_fake_memento(tmp_path)
    adapter = Memento10kAdapter(tmp_path, splits=("train",))
    videos = list(adapter)

    v = next(v for v in videos if v.video_id == "memento_video_00001")
    assert v.raw_labels["mem_score"] == pytest.approx(0.87)
    assert v.raw_labels["alpha"] == pytest.approx(-0.012)
    assert v.metadata["split"] == "train"
    assert v.metadata["n_captions"] == 2
    assert v.metadata["n_actions"] == 2
    assert "a dog runs" in str(v.metadata["caption_joined"])
    assert v.metadata["first_action"] == "running"
    # Sanity: media_uri points under the fixture root.
    assert v.media_uri.endswith("video_00001.webm")


def test_entry_without_filename_raises(tmp_path: Path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / ANNOTATION_FILES["train"]).write_text(
        json.dumps([{"mem_score": 0.5}]), encoding="utf-8",
    )
    adapter = Memento10kAdapter(tmp_path, splits=("train",))
    with pytest.raises(ValueError, match="missing 'filename'"):
        list(adapter)
