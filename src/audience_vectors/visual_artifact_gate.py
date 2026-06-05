"""Lightweight visual artifact metrics for generated video panels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

RGBFrame = NDArray[np.uint8]


@dataclass(frozen=True)
class ArtifactThresholds:
    """Thresholds for detecting clips that lose usable visual content."""

    min_tail_sharpness_ratio: float = 0.35
    min_tail_contrast_ratio: float = 0.55
    min_tail_contrast: float = 0.04


def _rgb_array(frame: NDArray[Any]) -> RGBFrame:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        msg = f"expected grayscale or RGB/RGBA frame, got shape {array.shape}"
        raise ValueError(msg)
    array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def luminance(frame: NDArray[Any]) -> NDArray[np.float32]:
    """Return normalized luminance for a frame."""
    rgb = _rgb_array(frame).astype(np.float32) / 255.0
    return (
        0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    ).astype(np.float32)


def frame_sharpness(frame: NDArray[Any]) -> float:
    """Approximate sharpness using mean squared luminance gradients."""
    y = luminance(frame)
    if min(y.shape) < 2:
        return 0.0
    dx = np.diff(y, axis=1)
    dy = np.diff(y, axis=0)
    return float(0.5 * (np.mean(dx * dx) + np.mean(dy * dy)))


def frame_contrast(frame: NDArray[Any]) -> float:
    """Return normalized luminance contrast."""
    return float(np.std(luminance(frame)))


def summarize_frames(
    frames: list[NDArray[Any]],
    *,
    thresholds: ArtifactThresholds | None = None,
) -> dict[str, Any]:
    """Summarize start/mid/end frames and flag tail-frame collapse."""
    if len(frames) < 3:
        msg = f"expected at least 3 sampled frames, got {len(frames)}"
        raise ValueError(msg)
    thresholds = thresholds or ArtifactThresholds()
    start, mid, end = frames[0], frames[len(frames) // 2], frames[-1]
    sharpness = {
        "start": frame_sharpness(start),
        "mid": frame_sharpness(mid),
        "end": frame_sharpness(end),
    }
    contrast = {
        "start": frame_contrast(start),
        "mid": frame_contrast(mid),
        "end": frame_contrast(end),
    }
    eps = 1e-12
    min_tail_sharpness = min(sharpness["mid"], sharpness["end"])
    min_tail_contrast = min(contrast["mid"], contrast["end"])
    tail_sharpness_ratio = min_tail_sharpness / (sharpness["start"] + eps)
    tail_contrast_ratio = min_tail_contrast / (contrast["start"] + eps)
    failures = []
    if tail_sharpness_ratio < thresholds.min_tail_sharpness_ratio:
        failures.append("tail_sharpness_collapse")
    if tail_contrast_ratio < thresholds.min_tail_contrast_ratio:
        failures.append("tail_contrast_collapse")
    if min_tail_contrast < thresholds.min_tail_contrast:
        failures.append("tail_low_contrast")
    collapse_score = max(0.0, 1.0 - 0.5 * (tail_sharpness_ratio + tail_contrast_ratio))
    return {
        "sharpness": sharpness,
        "contrast": contrast,
        "tail_sharpness_ratio": float(tail_sharpness_ratio),
        "tail_contrast_ratio": float(tail_contrast_ratio),
        "min_tail_contrast": float(min_tail_contrast),
        "collapse_score": float(collapse_score),
        "artifact_flags": failures,
        "passes_visual_gate": not failures,
    }


def sample_video_frames(path: Path, *, samples: int = 3) -> list[RGBFrame]:
    """Sample evenly spaced frames from an MP4 using imageio/ffmpeg."""
    if samples < 3:
        msg = "samples must be at least 3"
        raise ValueError(msg)
    import imageio.v3 as iio  # noqa: PLC0415

    raw_frames = [_rgb_array(frame) for frame in iio.imiter(path)]
    if len(raw_frames) < samples:
        msg = f"{path} has only {len(raw_frames)} readable frames"
        raise ValueError(msg)
    indices = np.linspace(0, len(raw_frames) - 1, samples, dtype=int)
    return [raw_frames[int(index)] for index in indices]


def summarize_video(
    path: Path,
    *,
    samples: int = 3,
    thresholds: ArtifactThresholds | None = None,
) -> dict[str, Any]:
    """Sample a video and return artifact-gate metrics."""
    frames = sample_video_frames(path, samples=samples)
    summary = summarize_frames(frames, thresholds=thresholds)
    summary.update(
        {
            "video_path": str(path),
            "sample_count": samples,
        }
    )
    return summary


def summarize_video_dir(
    video_dir: Path,
    *,
    pattern: str = "*.mp4",
    samples: int = 3,
    thresholds: ArtifactThresholds | None = None,
) -> dict[str, Any]:
    """Summarize all matching videos in a directory."""
    videos = sorted(video_dir.glob(pattern))
    if not videos:
        msg = f"no videos matched {pattern!r} in {video_dir}"
        raise ValueError(msg)
    rows = [
        summarize_video(video, samples=samples, thresholds=thresholds)
        for video in videos
    ]
    failures = [row for row in rows if not row["passes_visual_gate"]]
    return {
        "schema_version": 1,
        "video_dir": str(video_dir),
        "pattern": pattern,
        "samples": samples,
        "thresholds": (thresholds or ArtifactThresholds()).__dict__,
        "n_videos": len(rows),
        "n_failed": len(failures),
        "passes_visual_gate": len(failures) == 0,
        "rows": rows,
    }
