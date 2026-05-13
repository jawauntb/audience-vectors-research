"""V-JEPA 2 video feature extractor on Modal.

Plays the same role as `tribe_predictor` but uses Meta's open V-JEPA 2
(MIT-licensed) so we're not blocked on Llama-3.2-3B access approval.

Output is a single mean-pooled embedding per clip — the natural unit for
segment-level contrastive vector extraction. If you need per-frame
features (for time-resolved analysis), reshape on the caller side.

Run once after deploy:
    modal deploy -m audience_vectors.modal_app.app
    modal run -m audience_vectors.modal_app.functions.vjepa_predictor::populate_vjepa_weights
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
from typing import Any

import modal
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from audience_vectors.modal_app.app import MODAL_REGION, app, env_secrets
from audience_vectors.modal_app.image_factory import HF_CACHE_DIR, base_image

# image-rebuild-tag: 2026-05-13-v2
VJEPA_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
VJEPA_REVISION = ""  # latest; pin once we know it stabilizes

vjepa_weights_volume = modal.Volume.from_name(
    "vjepa-weights-v1", create_if_missing=True,
)
bmd_videos_volume = modal.Volume.from_name(
    "bmd-videos-v1", create_if_missing=True,
)
BMD_VIDEOS_MOUNT = "/bmd-videos"


vjepa_image = (
    base_image.apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.5.1,<2.7",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.46,<5.0",
        "decord>=0.6,<1.0",
        "einops>=0.8,<1.0",
        "safetensors>=0.4,<1.0",
        "huggingface_hub[hf_transfer]",
        "av>=12.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": HF_CACHE_DIR,
    })
)


_MAX_VIDEO_DURATION_S = 30.0
_MAX_REMOTE_DOWNLOAD_BYTES = 512 * 1024 * 1024


class VjepaPredictionResult(BaseModel):
    """One mean-pooled V-JEPA embedding per clip, plus the source duration."""

    embedding: list[float]
    duration_seconds: float
    n_frames: int


def _download_remote(url: str) -> str:
    suffix = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suffix, dir="/tmp")
    os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": "audience-vectors/0.1"})
    written = 0
    with urllib.request.urlopen(req, timeout=60) as response:
        with open(tmp, "wb") as out:
            while chunk := response.read(1 << 20):
                written += len(chunk)
                if written > _MAX_REMOTE_DOWNLOAD_BYTES:
                    raise RuntimeError("remote media too large")
                out.write(chunk)
    return tmp


def _resolve_local_path(path_or_url: str) -> tuple[str, bool]:
    parsed = urllib.parse.urlparse(path_or_url)
    if parsed.scheme == "":
        return path_or_url, False
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported scheme: {parsed.scheme}")
    return _download_remote(path_or_url), True


def _probe_duration(local_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", local_path],
        capture_output=True, check=True, text=True, timeout=30,
    )
    return float(result.stdout.strip())


@app.function(
    image=vjepa_image,
    volumes={HF_CACHE_DIR: vjepa_weights_volume},
    secrets=env_secrets,
    timeout=30 * 60,
    cpu=4.0,
    memory=16 * 1024,
)
def populate_vjepa_weights() -> None:
    """Pre-download V-JEPA 2 to the shared volume so cold-start is fast."""
    kwargs: dict[str, Any] = {}
    if VJEPA_REVISION:
        kwargs["revision"] = VJEPA_REVISION
    snapshot_download(VJEPA_MODEL_ID, **kwargs)
    vjepa_weights_volume.commit()


@app.cls(
    region=MODAL_REGION,
    image=vjepa_image,
    gpu="H100!",  # strict — no auto-upgrade to H200/B200 (sm_100 not in torch 2.6 wheel)
    volumes={
        HF_CACHE_DIR: vjepa_weights_volume,
        BMD_VIDEOS_MOUNT: bmd_videos_volume,
    },
    timeout=20 * 60,
    min_containers=0,
    scaledown_window=300,
    # Snapshots cached a stale view of the bmd-videos volume; disable until
    # the volume contents are stable across runs.
    enable_memory_snapshot=False,
    secrets=env_secrets,
)
class VjepaPredictor:
    """V-JEPA 2 wrapped for Modal. Load once, predict many times."""

    @modal.enter()
    def load_model(self) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoVideoProcessor  # noqa: PLC0415

        # Refresh the BMD videos volume so any files uploaded after the
        # container started become visible.
        bmd_videos_volume.reload()

        kwargs: dict[str, Any] = {"torch_dtype": torch.float32}
        if VJEPA_REVISION:
            kwargs["revision"] = VJEPA_REVISION
        self.processor = AutoVideoProcessor.from_pretrained(VJEPA_MODEL_ID, **kwargs)
        self.model = AutoModel.from_pretrained(VJEPA_MODEL_ID, **kwargs).to("cuda").eval()

    @modal.method()
    def predict_video(self, video_path_or_url: str) -> VjepaPredictionResult:
        """Mean-pooled V-JEPA embedding for one clip."""
        import torch  # noqa: PLC0415

        local_path, is_temp = _resolve_local_path(video_path_or_url)
        try:
            duration = _probe_duration(local_path)
            if duration > _MAX_VIDEO_DURATION_S:
                raise ValueError(
                    f"video too long: {duration:.1f}s > {_MAX_VIDEO_DURATION_S:.0f}s"
                )

            # decord-backed loader keeps memory tight and FFmpeg-free for the model.
            inputs = self.processor(videos=local_path, return_tensors="pt")
            pixel_values = inputs["pixel_values_videos"].to("cuda")

            with torch.inference_mode():
                outputs = self.model(pixel_values_videos=pixel_values)
            # V-JEPA encoder output: (B, T*P, D). Mean over time*patches.
            hidden = outputs.last_hidden_state
            pooled = hidden.mean(dim=1).squeeze(0).detach().cpu().to(torch.float32).numpy()

            return VjepaPredictionResult(
                embedding=pooled.tolist(),
                duration_seconds=float(duration),
                n_frames=int(hidden.shape[1]),
            )
        finally:
            if is_temp:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass
