"""Tests for the BOLD Moments adapter.

Most tests use a synthetic mini-dataset on disk that mirrors the real
schema. The last test, `test_real_annotations_if_present`, runs against
the actual downloaded `annotations.json` if it's on disk — that one
catches schema drift the synthetic fixture can't.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from audience_vectors.datasets import BoldMomentsAdapter

REAL_ANNOTATIONS = Path(
    os.environ.get("BOLD_MOMENTS_ROOT", "./data/raw/bold_moments")
).resolve() / "annotations.json"


def _build_fake_bmd(root: Path, with_local_video: bool = False) -> None:
    entries = {
        "0001": {
            "bmd_matrixfilename": "vid_idx0001",
            "MiT_url": "https://example.com/clip0001.mp4",
            "MiT_filename": "wetting/clip0001.mp4",
            "set": "train",
            "objects": [
                ["duck", "bird", "feather"],
                ["duck", "--", "--"],
            ],
            "scenes": ["pond", "lake/natural"],
            "actions": ["swimming", "paddling"],
            "text_descriptions": [
                "A duck is swimming in a lake.",
                "A mallard floats on water.",
            ],
            "spoken_transcription": "a duck swims in a lake",
            "memorability_score": 0.8147719988084737,
            "memorability_decay": -0.000405,
        },
        "1001": {
            "bmd_matrixfilename": "vid_idx1001",
            "MiT_url": "",
            "MiT_filename": "test/clip1001.mp4",
            "set": "test",
            "objects": [["--", "--", "--"]],
            "scenes": [],
            "actions": ["dancing"],
            "text_descriptions": ["A person dances."],
            "spoken_transcription": "a person is dancing",
            "memorability_score": 0.62,
            "memorability_decay": -0.018,
        },
    }
    (root / "annotations.json").write_text(json.dumps(entries), encoding="utf-8")
    if with_local_video:
        (root / "videos").mkdir(parents=True, exist_ok=True)
        (root / "videos" / "vid_idx0001.mp4").touch()


def test_missing_annotations_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="annotations.json"):
        BoldMomentsAdapter(tmp_path)


def test_unknown_set_rejected(tmp_path: Path):
    _build_fake_bmd(tmp_path)
    with pytest.raises(ValueError, match="unknown BOLD Moments sets"):
        BoldMomentsAdapter(tmp_path, sets=("train", "wat"))


def test_emits_one_per_entry_across_sets(tmp_path: Path):
    _build_fake_bmd(tmp_path)
    videos = list(BoldMomentsAdapter(tmp_path))
    assert len(videos) == 2
    assert {v.source_dataset for v in videos} == {"BOLDMoments"}
    assert all(v.duration_s == 3.0 for v in videos)


def test_split_filtering(tmp_path: Path):
    _build_fake_bmd(tmp_path)
    train_only = list(BoldMomentsAdapter(tmp_path, sets=("train",)))
    test_only = list(BoldMomentsAdapter(tmp_path, sets=("test",)))
    assert len(train_only) == 1
    assert len(test_only) == 1
    assert train_only[0].metadata["split"] == "train"
    assert test_only[0].metadata["split"] == "test"


def test_labels_and_metadata_populate(tmp_path: Path):
    _build_fake_bmd(tmp_path)
    videos = list(BoldMomentsAdapter(tmp_path))
    v = next(v for v in videos if v.video_id == "bmd_vid_idx0001")
    assert v.raw_labels["memorability_score"] == pytest.approx(0.8147719988084737)
    assert v.raw_labels["memorability_decay"] == pytest.approx(-0.000405)
    # objects flatten + drop '--'
    assert v.metadata["n_objects"] == 4
    assert v.metadata["n_scenes"] == 2
    assert v.metadata["n_actions"] == 2
    assert v.metadata["n_captions"] == 2
    assert v.metadata["top_scene"] == "pond"
    assert v.metadata["top_action"] == "swimming"
    assert "duck" in str(v.metadata["caption_joined"])
    assert v.metadata["spoken_transcription"] == "a duck swims in a lake"
    assert v.metadata["mit_url"].endswith("clip0001.mp4")


def test_falls_back_to_mit_url_when_no_local(tmp_path: Path):
    _build_fake_bmd(tmp_path, with_local_video=False)
    videos = list(BoldMomentsAdapter(tmp_path))
    v = next(v for v in videos if v.video_id == "bmd_vid_idx0001")
    assert v.media_uri == "https://example.com/clip0001.mp4"
    assert v.metadata["local_video_present"] is False


def test_prefers_local_video_when_present(tmp_path: Path):
    _build_fake_bmd(tmp_path, with_local_video=True)
    videos = list(BoldMomentsAdapter(tmp_path))
    v = next(v for v in videos if v.video_id == "bmd_vid_idx0001")
    assert v.media_uri.endswith("videos/vid_idx0001.mp4")
    assert v.metadata["local_video_present"] is True


@pytest.mark.skipif(
    not REAL_ANNOTATIONS.exists(),
    reason="real BOLD Moments annotations.json not downloaded",
)
def test_real_annotations_if_present():
    """Sanity-check against the actual published annotations.json.

    Catches the case where the upstream schema drifts and our synthetic
    fixture stays in sync with itself but not with reality.
    """
    adapter = BoldMomentsAdapter(REAL_ANNOTATIONS.parent)
    videos = list(adapter)
    # Known: 1,000 train + 102 test = 1,102 entries.
    assert len(videos) == 1102
    train = [v for v in videos if v.metadata["split"] == "train"]
    test = [v for v in videos if v.metadata["split"] == "test"]
    assert len(train) == 1000
    assert len(test) == 102

    sample = videos[0]
    assert sample.source_dataset == "BOLDMoments"
    assert sample.duration_s == 3.0
    # Memorability scores are in [0, 1].
    score = sample.raw_labels.get("memorability_score")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # Every entry should carry a MiT_url for the stimulus fallback.
    assert isinstance(sample.metadata["mit_url"], str)
