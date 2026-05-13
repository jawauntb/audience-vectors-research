"""Modal image builders. Two images:

- `base_image`: light Python env for orchestration / cron-style jobs.
- `tribe_image`: heavy CUDA image for TRIBE v2 inference, with the right
  whisperx + cuDNN8 stack pinned to versions known to work.

Pins mirror the working production setup in superoptimizers. When you
bump any of these, re-run `populate_tribe_weights` and re-validate
predictions before relying on them.
"""

from __future__ import annotations

import modal

PYTHON_VERSION = "3.12"

# ---------------------------------------------------------------------------
# Base — just enough to run orchestration code, no GPU deps.
# ---------------------------------------------------------------------------

base_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "ffmpeg")
    .pip_install(
        "pydantic[email]==2.12.5",
        "huggingface-hub>=0.36",
        "anthropic==0.80.0",
        "google-genai==1.56.0",
        "numpy>=1.26,<3.0",
        "pyarrow==23.0.1",
        "polars>=1.20",
        "httpx==0.28.1",
        "logfire==4.16.0",
    )
)


# ---------------------------------------------------------------------------
# TRIBE v2 — pinned stack. CC-BY-NC-4.0; pipeline is non-commercial.
# ---------------------------------------------------------------------------

# HuggingFace repo + git revision pins. Same SHAs the production
# superoptimizers code uses (captured 2026-04-23 / 2026-03-30).
TRIBE_HF_REPO_ID = "facebook/tribev2"
TRIBE_HF_REVISION = "f894e783020944dcd96e5568550afe2aa9743f9f"
TRIBE_GIT_REF = "72399081ed3f1040c4d996cefb2864a4c46f5b8e"

# whisperx + cuDNN dance: whisperx 3.4.3 pins ctranslate2<4.5, whose CUDA
# 12 wheels still dlopen cuDNN 8. PyTorch ships cuDNN 9, so cuDNN 8 must
# live in a separate prefix that we LD_LIBRARY_PATH ahead of it.
WHISPERX_VERSION = "3.4.3"
PYANNOTE_AUDIO_VERSION = "3.4.0"
LIGHTNING_VERSION = "2.6.1"  # last clean release before 2.6.2/2.6.3 compromise
CUDNN8_PACKAGE = "nvidia-cudnn-cu12==8.9.7.29"
CUDNN8_TARGET = "/opt/tribe-whisperx-cudnn8"
CUDNN8_LIB_DIR = f"{CUDNN8_TARGET}/nvidia/cudnn/lib"

# Shared HuggingFace cache. Populated once via `populate_tribe_weights`
# so cold starts don't re-download ~10 GB of foundation-encoder weights.
HF_CACHE_DIR = "/hf-cache"
HF_IMAGE_CACHE_DIR = "/hf-image-cache"

# Feature-encoder pins (referenced from TRIBE v2's config.yaml).
TRIBE_FEATURE_MODEL_PINS: tuple[tuple[str, str], ...] = (
    ("meta-llama/Llama-3.2-3B", "13afe5124825b4f3751f836b40dafda64c1ed062"),
    ("facebook/w2v-bert-2.0", "da985ba0987f70aaeb84a80f2851cfac8c697a7b"),
    ("facebook/vjepa2-vitg-fpc64-256", "875c192b7b704b87d1e1d99345769632dd5f739a"),
    ("facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"),
)
WHISPERX_MODEL_REPO_ID = "Systran/faster-whisper-large-v3"
WHISPERX_MODEL_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"


# Image-rebuild-tag: 2026-05-13-v3-h100
tribe_image = (
    base_image.apt_install("git")
    .pip_install(
        # cu124 wheel includes sm_100 (Blackwell / B200) support. cu121 was
        # the older default but is sm_90-max and rejects B200 at runtime.
        # cu124 stays forward-compatible with H100/A10G so we don't need
        # to fork the image per-GPU.
        "torch>=2.5.1,<2.7",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("uv", "huggingface_hub[hf_transfer]")
    .pip_install(
        f"lightning=={LIGHTNING_VERSION}",
        f"whisperx=={WHISPERX_VERSION}",
        f"pyannote-audio=={PYANNOTE_AUDIO_VERSION}",
    )
    # Explicitly DISABLE hf_transfer at the image level. TRIBE invokes
    # `uvx whisperx` which creates an isolated env without `hf_transfer`
    # installed, but inherits parent env vars — so an enabled HF_TRANSFER
    # makes whisperx's faster-whisper download bail. The runtime fall
    # back to standard HTTP is fine since the weights volume is already
    # populated.
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "HF_HOME": HF_IMAGE_CACHE_DIR,
    })
    # uv (not pip) for the TRIBE git install — pip can try to uninstall
    # uv-managed base packages and fail on Modal's `/.uv/.venv` paths.
    .uv_pip_install(
        f"git+https://github.com/facebookresearch/tribev2.git@{TRIBE_GIT_REF}"
    )
    .run_commands(
        f"python -m pip install --no-deps --target {CUDNN8_TARGET} {CUDNN8_PACKAGE}",
        f"test -f {CUDNN8_LIB_DIR}/libcudnn_ops_infer.so.8",
        "python -m spacy download en_core_web_sm",
    )
    .env({"HF_HOME": HF_CACHE_DIR})
)


__all__ = [
    "base_image",
    "tribe_image",
    "PYTHON_VERSION",
    "TRIBE_HF_REPO_ID",
    "TRIBE_HF_REVISION",
    "TRIBE_GIT_REF",
    "TRIBE_FEATURE_MODEL_PINS",
    "WHISPERX_MODEL_REPO_ID",
    "WHISPERX_MODEL_REVISION",
    "HF_CACHE_DIR",
    "CUDNN8_LIB_DIR",
]
