from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_screen_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "screen_content_pocket_validation_stimuli.py"
    )
    spec = importlib.util.spec_from_file_location(
        "screen_content_pocket_validation_stimuli",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def frame(value: int) -> np.ndarray:
    image = np.full((32, 32, 3), value, dtype=np.uint8)
    image[8:24, 8:24] = 255 - value
    return image


def stimulus(path: Path, sha: str | None = None) -> dict:
    return {
        "analysis_tier": "primary",
        "role": "candidate",
        "pocket": "fresh24_orange_flowers",
        "task_id": "sobol_prompt_search_519_slot10",
        "recipe_index": 519,
        "replicate_index": 0,
        "label": "bo_replay_00",
        "local_video_path": "data/generated/fake/bo_replay_00.mp4",
        "source_absolute_path": str(path),
        "sha256": sha,
        "replay_tribe_score": 4.2,
    }


def write_manifest(path: Path, source_path: Path, sha: str) -> Path:
    manifest = {
        "task_pool": {"task_payload_sha256": "abc123"},
        "tasks": [{"comparison": "primary_content_pocket_vs_hard_negative"}],
        "stimuli": [stimulus(source_path, sha=sha)],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_screen_stimulus_records_clean_sampled_frame_gate(monkeypatch, tmp_path):
    module = load_screen_module()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    sha = hashlib.sha256(b"fake-video").hexdigest()

    monkeypatch.setattr(module, "sample_video_frames", lambda _path, samples: [frame(40), frame(40), frame(40)])
    monkeypatch.setattr(module, "frame_count", lambda _path: 12)

    row, frames = module.screen_stimulus(stimulus(video, sha=sha), samples=3)

    assert frames is not None
    assert row["exists"] is True
    assert row["sha256_matches_manifest"] is True
    assert row["frame_sample_ok"] is True
    assert row["frame_count"] == 12
    assert row["screening_flags"] == []
    assert row["visual_gate"]["passes_visual_gate"] is True


def test_screen_stimulus_flags_missing_file(tmp_path):
    module = load_screen_module()
    row, frames = module.screen_stimulus(
        stimulus(tmp_path / "missing.mp4", sha="nope"),
        samples=3,
    )

    assert frames is None
    assert row["exists"] is False
    assert row["screening_flags"] == ["missing_file"]


def test_build_screening_writes_contact_sheet_metadata(monkeypatch, tmp_path):
    module = load_screen_module()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    sha = hashlib.sha256(b"fake-video").hexdigest()
    manifest = write_manifest(tmp_path / "manifest.json", video, sha)

    monkeypatch.setattr(module, "sample_video_frames", lambda _path, samples: [frame(40), frame(40), frame(40)])
    monkeypatch.setattr(module, "frame_count", lambda _path: 12)

    report, markdown = module.build_screening(
        manifest_path=manifest,
        sheet_dir=tmp_path / "sheets",
        samples=3,
        agent_review_note="synthetic review note",
    )

    assert report["n_stimuli"] == 1
    assert report["n_screening_failures"] == 0
    assert report["n_visual_gate_failures"] == 0
    assert report["contact_sheets"][0]["n_stimuli"] == 1
    assert report["agent_contact_sheet_review"] == "synthetic review note"
    assert Path(report["contact_sheets"][0]["path"]).exists()
    assert "No automated screening failures" in markdown

    url_map = module.build_url_map_template(report)
    assert len(url_map["videos"]) == 1
    assert url_map["videos"][0]["agent_prescreened"] is True
    assert url_map["videos"][0]["screened"] is False
