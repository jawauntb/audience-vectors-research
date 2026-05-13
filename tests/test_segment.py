"""Tests for the segmentation module.

A handful run against real BMD videos if they exist on disk
(`data/raw/bold_moments/videos/`). The rest use synthetic fixtures
generated with ffmpeg's testsrc filter — no external files needed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from audience_vectors.media.segment import (
    Segmenter,
    SegmentationError,
    probe_duration,
)
from audience_vectors.schemas import CanonicalVideo

REAL_BMD_VIDEOS = Path("./data/raw/bold_moments/videos").resolve()


def _make_test_video(path: Path, duration_s: float) -> None:
    """Synthesize a tiny `duration_s`-long mp4 using ffmpeg's testsrc filter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi",
            "-i", f"testsrc=duration={duration_s}:size=160x120:rate=10",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(path),
        ],
        check=True, capture_output=True,
    )


@pytest.fixture
def ffmpeg_available() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not on PATH")


def test_probe_duration_raises_on_missing(tmp_path: Path):
    with pytest.raises(SegmentationError):
        probe_duration(tmp_path / "does_not_exist.mp4")


def test_probe_duration_on_synth_video(tmp_path: Path, ffmpeg_available):
    src = tmp_path / "synth.mp4"
    _make_test_video(src, duration_s=2.5)
    duration = probe_duration(src)
    assert 2.3 <= duration <= 2.7  # ffmpeg testsrc isn't perfectly precise


def test_rejects_bad_segment_lengths(tmp_path: Path):
    with pytest.raises(ValueError):
        Segmenter(segment_length_s=0, segment_stride_s=1.0, output_dir=tmp_path)
    with pytest.raises(ValueError):
        Segmenter(segment_length_s=3.0, segment_stride_s=-1, output_dir=tmp_path)


def test_passthrough_when_exact(tmp_path: Path, ffmpeg_available):
    src = tmp_path / "exact.mp4"
    _make_test_video(src, duration_s=3.0)
    seg = Segmenter(
        segment_length_s=3.0,
        segment_stride_s=3.0,
        output_dir=tmp_path / "clips",
    )
    video = CanonicalVideo(
        video_id="t1",
        source_dataset="Synth",
        media_uri=str(src),
        duration_s=3.0,
    )
    jobs = seg.plan_jobs(video)
    assert len(jobs) == 1
    assert jobs[0].is_passthrough
    assert jobs[0].output_path == src  # no copy written


def test_long_video_yields_multiple_windows(tmp_path: Path, ffmpeg_available):
    src = tmp_path / "long.mp4"
    _make_test_video(src, duration_s=9.0)
    seg = Segmenter(
        segment_length_s=3.0,
        segment_stride_s=3.0,
        output_dir=tmp_path / "clips",
    )
    video = CanonicalVideo(
        video_id="t2",
        source_dataset="Synth",
        media_uri=str(src),
        duration_s=9.0,
    )
    jobs = seg.plan_jobs(video)
    # 9s / 3s stride = 3 non-overlapping windows
    assert len(jobs) == 3
    assert [j.start_time for j in jobs] == [0.0, 3.0, 6.0]
    assert all(not j.is_passthrough for j in jobs)
    assert all(j.sample_id.endswith(f"_seg_{i:04d}") for i, j in enumerate(jobs))


def test_overlapping_stride(tmp_path: Path, ffmpeg_available):
    src = tmp_path / "long.mp4"
    _make_test_video(src, duration_s=6.0)
    seg = Segmenter(
        segment_length_s=3.0,
        segment_stride_s=1.5,
        output_dir=tmp_path / "clips",
    )
    video = CanonicalVideo(
        video_id="t3",
        source_dataset="Synth",
        media_uri=str(src),
        duration_s=6.0,
    )
    jobs = seg.plan_jobs(video)
    # windows: [0,3], [1.5,4.5], [3,6] = 3 windows
    assert len(jobs) == 3
    assert [round(j.start_time, 2) for j in jobs] == [0.0, 1.5, 3.0]


def test_remote_uri_passes_through_when_duration_matches(tmp_path: Path):
    """URL with declared duration matching segment_length should passthrough."""
    seg = Segmenter(segment_length_s=3.0, segment_stride_s=3.0, output_dir=tmp_path)
    video = CanonicalVideo(
        video_id="remote",
        source_dataset="Synth",
        media_uri="https://example.com/clip.mp4",
        duration_s=3.0,
    )
    jobs = seg.plan_jobs(video)
    assert len(jobs) == 1
    assert jobs[0].is_passthrough
    assert jobs[0].duration == 3.0


def test_remote_uri_skipped_when_no_duration(tmp_path: Path):
    """URL without a declared duration can't be probed remotely; skip it."""
    seg = Segmenter(segment_length_s=3.0, segment_stride_s=3.0, output_dir=tmp_path)
    video = CanonicalVideo(
        video_id="remote",
        source_dataset="Synth",
        media_uri="https://example.com/clip.mp4",
        duration_s=None,
    )
    assert seg.plan_jobs(video) == []


def test_remote_uri_skipped_when_duration_mismatch(tmp_path: Path):
    """URL with declared duration far from segment_length: also skip."""
    seg = Segmenter(segment_length_s=3.0, segment_stride_s=3.0, output_dir=tmp_path)
    video = CanonicalVideo(
        video_id="remote",
        source_dataset="Synth",
        media_uri="https://example.com/clip.mp4",
        duration_s=10.0,
    )
    assert seg.plan_jobs(video) == []


def test_run_job_extracts_and_is_idempotent(tmp_path: Path, ffmpeg_available):
    src = tmp_path / "long.mp4"
    _make_test_video(src, duration_s=6.0)
    out_dir = tmp_path / "clips"
    seg = Segmenter(
        segment_length_s=3.0,
        segment_stride_s=3.0,
        output_dir=out_dir,
        passthrough_when_exact=False,  # force extraction even if 3s
    )
    video = CanonicalVideo(
        video_id="t4",
        source_dataset="Synth",
        media_uri=str(src),
        duration_s=6.0,
    )
    jobs = seg.plan_jobs(video)
    for job in jobs:
        seg.run_job(job)
        assert job.output_path.exists()
        assert job.output_path.stat().st_size > 0
    # Re-run is a no-op (idempotent — file still exists, same mtime).
    mtimes_before = [j.output_path.stat().st_mtime for j in jobs]
    for job in jobs:
        seg.run_job(job)
    mtimes_after = [j.output_path.stat().st_mtime for j in jobs]
    assert mtimes_before == mtimes_after


@pytest.mark.skipif(
    not (REAL_BMD_VIDEOS.exists() and any(REAL_BMD_VIDEOS.glob("*.mp4"))),
    reason="no real BOLD Moments videos on disk",
)
def test_real_bmd_videos_are_passthrough(tmp_path: Path):
    """BMD's ~3s stimuli should hit the passthrough fast path."""
    seg = Segmenter(
        segment_length_s=3.0,
        segment_stride_s=3.0,
        output_dir=tmp_path / "clips",
    )
    samples = sorted(REAL_BMD_VIDEOS.glob("*.mp4"))[:5]
    for src in samples:
        video = CanonicalVideo(
            video_id=f"bmd_{src.stem}",
            source_dataset="BOLDMoments",
            media_uri=str(src),
            duration_s=None,  # exercise probe path
        )
        jobs = seg.plan_jobs(video)
        assert len(jobs) == 1
        assert jobs[0].is_passthrough, f"expected passthrough for {src.name}"
