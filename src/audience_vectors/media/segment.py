"""Cut canonical videos into segment-length windows.

A `Segmenter` walks a stream of `CanonicalVideo` rows and emits
`Segment` rows plus on-disk clip files. Three optimizations matter:

  1. **No-op short-circuit.** When a source video's duration matches
     `segment_length_s` (within tolerance), point the segment at the
     original file instead of writing an identical copy. BMD's 1,102
     three-second clips hit this path.

  2. **Stream copy.** When we do need to cut, use `ffmpeg -c copy` —
     no re-encoding, ~realtime per clip on a laptop.

  3. **Idempotent.** Skip segments whose output file already exists
     with non-zero size. Re-runs after a partial crash do the right
     thing.

Long videos use `start = k * stride` windows up to `duration - segment_length`.
For BMD-style 3s clips with stride==length, this yields one segment per video.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from audience_vectors.schemas import CanonicalVideo, Segment


_DURATION_TOLERANCE_S = 0.15  # ffprobe + ffmpeg disagree at frame-boundary level


class SegmentationError(RuntimeError):
    """Raised when a video can't be probed or cut. Caller decides whether to skip."""


@dataclass(frozen=True)
class SegmentJob:
    """One window to extract from one source video."""

    sample_id: str
    source_path: Path
    start_time: float
    end_time: float
    duration: float
    output_path: Path
    is_passthrough: bool


def probe_duration(path: Path | str, *, timeout: float = 30.0) -> float:
    """Return media duration in seconds via ffprobe.

    Raises SegmentationError on any failure — callers wrap with try/skip
    in batch loops.
    """
    if not shutil.which("ffprobe"):
        raise SegmentationError("ffprobe not on PATH; install ffmpeg")
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-print_format", "json",
                "-show_format",
                str(path),
            ],
            capture_output=True, check=True, text=True, timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        raise SegmentationError(
            f"ffprobe failed (exit {exc.returncode}) for {path}: {exc.stderr.strip()[:200]}"
        ) from exc
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise SegmentationError(f"ffprobe error for {path}: {exc}") from exc

    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise SegmentationError(f"ffprobe gave no duration for {path}") from exc
    if duration <= 0:
        raise SegmentationError(f"non-positive duration {duration} for {path}")
    return duration


def _is_local_path(uri: str) -> bool:
    parsed = urlparse(uri)
    return parsed.scheme in ("", "file")


def _cut_with_ffmpeg(
    src: Path,
    dst: Path,
    *,
    start: float,
    duration: float,
    timeout: float = 120.0,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Keep `.mp4` last so ffmpeg picks the right muxer from the extension.
    tmp = dst.parent / f"{dst.stem}.part{dst.suffix}"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        "-movflags", "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise SegmentationError(
            f"ffmpeg failed for {src} -> {dst}: {exc.stderr.decode()[:200]}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SegmentationError(f"ffmpeg timeout for {src}") from exc
    tmp.replace(dst)


class Segmenter:
    """Cuts CanonicalVideos into Segments.

    Construct once, call `.segment_many(videos)` for a stream of (Segment, job)
    pairs. The job is None when the source was passthrough'd (no clip extraction
    needed) — useful for downstream code that wants to know whether a fresh
    clip file was written.
    """

    def __init__(
        self,
        *,
        segment_length_s: float = 3.0,
        segment_stride_s: float = 3.0,
        output_dir: Path,
        passthrough_when_exact: bool = True,
    ) -> None:
        if segment_length_s <= 0:
            raise ValueError("segment_length_s must be > 0")
        if segment_stride_s <= 0:
            raise ValueError("segment_stride_s must be > 0")
        self.segment_length_s = float(segment_length_s)
        self.segment_stride_s = float(segment_stride_s)
        self.output_dir = Path(output_dir)
        self.passthrough_when_exact = passthrough_when_exact

    # -- planning ---------------------------------------------------------

    def plan_jobs(self, video: CanonicalVideo) -> list[SegmentJob]:
        """Return the segment jobs for one video. May be empty (skipped).

        URL-based media is supported as long as `video.duration_s` is set
        AND the duration matches `segment_length_s` (so we can passthrough
        without probing or downloading). For longer URL sources you'll
        need to download locally first — ffmpeg can't cut a remote stream
        in `-c copy` mode reliably.
        """
        is_local = _is_local_path(video.media_uri)
        if is_local:
            src = Path(video.media_uri)
            if not src.exists():
                return []
            duration = video.duration_s
            if duration is None or duration <= 0:
                try:
                    duration = probe_duration(src)
                except SegmentationError:
                    return []
        else:
            # Remote URL — we don't probe/cut remotely. Only viable path
            # is passthrough: emit one Segment pointing at the URL.
            if (
                not self.passthrough_when_exact
                or video.duration_s is None
                or video.duration_s <= 0
                or abs(video.duration_s - self.segment_length_s) > _DURATION_TOLERANCE_S
            ):
                return []
            sample_id = self._sample_id(video, 0)
            return [SegmentJob(
                sample_id=sample_id,
                source_path=Path(video.media_uri),
                start_time=0.0,
                end_time=video.duration_s,
                duration=video.duration_s,
                output_path=Path(video.media_uri),
                is_passthrough=True,
            )]

        jobs: list[SegmentJob] = []
        seg_len = self.segment_length_s
        stride = self.segment_stride_s

        # Passthrough: clip already matches segment length within tolerance.
        if (
            self.passthrough_when_exact
            and abs(duration - seg_len) <= _DURATION_TOLERANCE_S
        ):
            sample_id = self._sample_id(video, 0)
            jobs.append(SegmentJob(
                sample_id=sample_id,
                source_path=src,
                start_time=0.0,
                end_time=duration,
                duration=duration,
                output_path=src,  # point at the original
                is_passthrough=True,
            ))
            return jobs

        # Sliding windows. Last full window may end before `duration`; we
        # don't emit a short tail to keep all segments comparable.
        idx = 0
        start = 0.0
        while start + seg_len <= duration + _DURATION_TOLERANCE_S:
            end = min(start + seg_len, duration)
            sample_id = self._sample_id(video, idx)
            out_path = self.output_dir / video.source_dataset / f"{sample_id}.mp4"
            jobs.append(SegmentJob(
                sample_id=sample_id,
                source_path=src,
                start_time=start,
                end_time=end,
                duration=end - start,
                output_path=out_path,
                is_passthrough=False,
            ))
            idx += 1
            start += stride
        return jobs

    # -- execution --------------------------------------------------------

    def run_job(self, job: SegmentJob) -> None:
        """Idempotent: skip if `job.output_path` already exists with size > 0."""
        if job.is_passthrough:
            return
        if job.output_path.exists() and job.output_path.stat().st_size > 0:
            return
        _cut_with_ffmpeg(
            job.source_path,
            job.output_path,
            start=job.start_time,
            duration=job.duration,
        )

    def segment_many(
        self,
        videos: Iterable[CanonicalVideo],
    ) -> Iterator[tuple[Segment, SegmentJob]]:
        """Yield (Segment, job) for every window across the input stream.

        Skips videos with no local file, no probable duration, or no valid
        windows — caller can count those externally via len() on the result.
        """
        for video in videos:
            for job in self.plan_jobs(video):
                self.run_job(job)
                yield self._segment_from_job(video, job), job

    # -- helpers ----------------------------------------------------------

    def _sample_id(self, video: CanonicalVideo, idx: int) -> str:
        return f"{video.video_id}_seg_{idx:04d}"

    def _segment_from_job(
        self,
        video: CanonicalVideo,
        job: SegmentJob,
    ) -> Segment:
        return Segment(
            sample_id=job.sample_id,
            source_dataset=video.source_dataset,
            video_id=video.video_id,
            domain=video.domain,
            start_time=job.start_time,
            end_time=job.end_time,
            duration=job.duration,
            media_path=str(job.output_path),
        )
