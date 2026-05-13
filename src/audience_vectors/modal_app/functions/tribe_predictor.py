"""TRIBE v2 predictor on Modal — research-grade, simplified.

This is a stripped-down version of the production pattern in
superoptimizers (gic_modal/functions/neural_engagement_scoring.py).
Dropped relative to the production version:

  - SSRF guards / pinned-IP HTTPS connection handlers
  - Prod/dev environment splits + HF_HUB_OFFLINE enforcement
  - Logfire span instrumentation
  - whisperx uvx shim (model path patching) — re-add if text input matters
  - ffprobe duration validation guards
  - Modal observability decorators

Kept because they genuinely matter even in research:

  - Pinned HF revisions (drift = silently wrong activations)
  - Shared weights volume (cold-start without 10 GB of downloads)
  - @modal.enter(snap=True) + GPU memory snapshots
  - Suffix normalization for signed URLs ending in `.bin`
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import urllib.parse
import urllib.request

import modal
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from audience_vectors.modal_app.app import MODAL_REGION, app, env_secrets
from audience_vectors.modal_app.image_factory import (
    HF_CACHE_DIR,
    TRIBE_FEATURE_MODEL_PINS,
    TRIBE_HF_REPO_ID,
    TRIBE_HF_REVISION,
    WHISPERX_MODEL_REPO_ID,
    WHISPERX_MODEL_REVISION,
    tribe_image,
)


# Shared weights volume — populate once via `populate_tribe_weights`.
# Set `create_if_missing=True` for first run; flip to False once you've
# validated it and want loud failures on missing volumes.
tribe_weights_volume = modal.Volume.from_name(
    "tribe-v2-weights-v1", create_if_missing=True
)
bmd_videos_volume = modal.Volume.from_name(
    "bmd-videos-v1", create_if_missing=True,
)
BMD_VIDEOS_MOUNT = "/bmd-videos"


_TRIBE_VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mkv", ".mov", ".webm"})
_MAX_VIDEO_DURATION_SECONDS = 30.0
_MAX_REMOTE_DOWNLOAD_BYTES = 512 * 1024 * 1024  # 512 MiB


class VideoPredictionResult(BaseModel):
    """TRIBE v2 video output: per-frame (time, vertices) activation tensor.

    Service layer means across frames for a scalar score; keeps both
    tensors + duration so callers can window/aggregate as needed.
    """

    frames: list[list[float]]
    duration_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_remote(url: str) -> str:
    """Stream a remote URL to /tmp, enforcing a size cap. Returns local path."""
    suffix = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suffix, dir="/tmp")
    os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": "audience-vectors/0.1"})
    written = 0
    with urllib.request.urlopen(req, timeout=60) as response:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > _MAX_REMOTE_DOWNLOAD_BYTES:
            raise RuntimeError(f"remote media too large: {declared} bytes")
        with open(tmp, "wb") as out:
            while chunk := response.read(1 << 20):
                written += len(chunk)
                if written > _MAX_REMOTE_DOWNLOAD_BYTES:
                    raise RuntimeError("remote media exceeded size cap")
                out.write(chunk)
    return tmp


def _resolve_local_path(path_or_url: str) -> tuple[str, bool]:
    parsed = urllib.parse.urlparse(path_or_url)
    if parsed.scheme == "":
        return path_or_url, False
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URI scheme: {parsed.scheme}")
    return _download_remote(path_or_url), True


def _ensure_tribe_suffix(local_path: str) -> tuple[str, str | None]:
    """Symlink to `.mp4` if suffix doesn't match TRIBE's allowlist.

    Returns (path_to_use, cleanup_dir_or_None).
    """
    suffix = os.path.splitext(local_path)[1].lower()
    if suffix in _TRIBE_VIDEO_SUFFIXES:
        return local_path, None
    link_dir = tempfile.mkdtemp(prefix="tribe_video_", dir="/tmp")
    target = os.path.join(link_dir, "video.mp4")
    os.symlink(os.path.abspath(local_path), target)
    return target, link_dir


def _probe_duration(local_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", local_path],
        capture_output=True, check=True, text=True, timeout=30,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise RuntimeError(f"non-positive video duration: {duration}")
    return duration


# ---------------------------------------------------------------------------
# One-shot weight populator
# ---------------------------------------------------------------------------


@app.function(
    image=tribe_image,
    volumes={HF_CACHE_DIR: tribe_weights_volume},
    secrets=env_secrets,
    timeout=60 * 60,
    cpu=4.0,
    memory=16 * 1024,
)
def populate_tribe_weights() -> None:
    """Populate the shared HF cache volume.

    Run once after deploy:

        modal run audience_vectors.modal_app.functions.tribe_predictor::populate_tribe_weights

    `meta-llama/Llama-3.2-3B` is gated — your HF token must have accepted
    Meta's license. The token flows in via env_secrets.
    """
    for repo_id, revision in (
        (TRIBE_HF_REPO_ID, TRIBE_HF_REVISION),
        *TRIBE_FEATURE_MODEL_PINS,
        (WHISPERX_MODEL_REPO_ID, WHISPERX_MODEL_REVISION),
    ):
        snapshot_download(repo_id, revision=revision)
    tribe_weights_volume.commit()


# ---------------------------------------------------------------------------
# Predictor — class-based so weights load once per warm container.
# ---------------------------------------------------------------------------


@app.cls(
    region=MODAL_REGION,
    image=tribe_image,
    # B200 (sm_100) needs torch>=2.7; TRIBE pins torch<2.7 so we use H100 (sm_90).
    # Strict `H100!` prevents Modal from auto-upgrading to H200 (kernel mismatch).
    gpu="H100!",
    volumes={
        HF_CACHE_DIR: tribe_weights_volume,
        BMD_VIDEOS_MOUNT: bmd_videos_volume,
    },
    timeout=20 * 60,
    min_containers=0,
    scaledown_window=300,
    # Snapshots cached a stale view of the bmd-videos volume; disable for now.
    enable_memory_snapshot=False,
    secrets=env_secrets,
)
class TribeV2Predictor:
    """TRIBE v2 wrapped for Modal. Load once, predict many times."""

    @modal.enter()
    def load_model(self) -> None:
        # Inline import — keeps module-import cheap for orchestration paths.
        from tribev2 import TribeModel  # type: ignore[import-not-found]

        # Refresh BMD videos volume so containers see the latest uploaded files.
        bmd_videos_volume.reload()

        tribe_path = snapshot_download(TRIBE_HF_REPO_ID, revision=TRIBE_HF_REVISION)
        self.model = TribeModel.from_pretrained(tribe_path, device="cuda")

    @modal.method()
    def predict_video(self, video_path_or_url: str) -> VideoPredictionResult:
        """Predict per-vertex brain activations for a video stimulus."""
        import numpy as np  # inline import to keep module-load light

        local_path, is_temp = _resolve_local_path(video_path_or_url)
        cleanup_dir: str | None = None
        try:
            duration = _probe_duration(local_path)
            if duration > _MAX_VIDEO_DURATION_SECONDS:
                raise ValueError(
                    f"video too long: {duration:.1f}s > {_MAX_VIDEO_DURATION_SECONDS:.0f}s"
                )
            tribe_path, cleanup_dir = _ensure_tribe_suffix(local_path)
            events = self.model.get_events_dataframe(video_path=tribe_path)
            preds, _segments = self.model.predict(events, verbose=False)
            return VideoPredictionResult(
                frames=np.asarray(preds).tolist(),
                duration_seconds=duration,
            )
        finally:
            if cleanup_dir is not None:
                try:
                    os.unlink(os.path.join(cleanup_dir, "video.mp4"))
                    os.rmdir(cleanup_dir)
                except OSError:
                    pass
            if is_temp:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass
