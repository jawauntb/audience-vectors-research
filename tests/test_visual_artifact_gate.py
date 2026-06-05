from __future__ import annotations

import numpy as np
import pytest

from audience_vectors.visual_artifact_gate import (
    ArtifactThresholds,
    frame_contrast,
    frame_sharpness,
    summarize_frames,
    summarize_video_dir,
)


def checkerboard(size: int = 64) -> np.ndarray:
    grid = np.indices((size, size)).sum(axis=0) % 2
    image = (grid * 255).astype(np.uint8)
    return np.stack([image, image, image], axis=2)


def smooth_gradient(size: int = 64) -> np.ndarray:
    line = np.linspace(80, 110, size, dtype=np.uint8)
    image = np.tile(line[None, :], (size, 1))
    return np.stack([image, image, image], axis=2)


def test_frame_sharpness_distinguishes_edges_from_smooth_gradient():
    assert frame_sharpness(checkerboard()) > frame_sharpness(smooth_gradient())
    assert frame_contrast(checkerboard()) > frame_contrast(smooth_gradient())


def test_summarize_frames_flags_tail_collapse():
    result = summarize_frames(
        [checkerboard(), smooth_gradient(), smooth_gradient()],
        thresholds=ArtifactThresholds(
            min_tail_sharpness_ratio=0.35,
            min_tail_contrast_ratio=0.55,
            min_tail_contrast=0.04,
        ),
    )

    assert not result["passes_visual_gate"]
    assert "tail_sharpness_collapse" in result["artifact_flags"]
    assert "tail_contrast_collapse" in result["artifact_flags"]


def test_summarize_frames_passes_stable_detail():
    result = summarize_frames([checkerboard(), checkerboard(), checkerboard()])

    assert result["passes_visual_gate"]
    assert result["artifact_flags"] == []
    assert result["tail_sharpness_ratio"] == pytest.approx(1.0)
    assert result["tail_contrast_ratio"] == pytest.approx(1.0)


def test_summarize_video_dir_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no videos matched"):
        summarize_video_dir(tmp_path)
