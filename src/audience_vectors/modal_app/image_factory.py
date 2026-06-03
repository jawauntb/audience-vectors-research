"""Modal image builders. Two images:

- `base_image`: light Python env for orchestration / cron-style jobs.
- `tribe_image`: heavy CUDA image for TRIBE v2 inference, with the right
  whisperx + cuDNN8 stack pinned to versions known to work.

Pins mirror the working production setup in superoptimizers. When you
bump any of these, re-run `populate_tribe_weights` and re-validate
predictions before relying on them.
"""

from __future__ import annotations

import modal  # type: ignore[import-not-found]

PYTHON_VERSION = "3.12"

# ---------------------------------------------------------------------------
# Base — just enough to run orchestration code, no GPU deps.
# ---------------------------------------------------------------------------

base_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "ffmpeg", "nodejs")
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
        "fastapi>=0.115,<1.0",
        "python-multipart>=0.0.20,<1.0",
        "pillow==12.1.0",
        "scipy>=1.14,<2.0",
        "scikit-learn>=1.8.0",
        "yt-dlp",
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
_TRIBE_EXCA_VERSION = "0.5.25"
TRIBE_UV_PIP_INSTALL_REQUIREMENTS = (
    f"git+https://github.com/facebookresearch/tribev2.git@{TRIBE_GIT_REF}",
    f"exca=={_TRIBE_EXCA_VERSION}",
)
_TRIBE_IMPORT_RUNTIME_PREFLIGHT_COMMAND = (
    'python -c "import exca.steps.base as exca_base; '
    "exca_base.NoValue(); "
    "from tribev2 import TribeModel; "
    'print(TribeModel.__name__)"'
)

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
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "HF_HOME": HF_IMAGE_CACHE_DIR,
        }
    )
    # uv (not pip) for the TRIBE git install — pip can try to uninstall
    # uv-managed base packages and fail on Modal's `/.uv/.venv` paths.
    .uv_pip_install(*TRIBE_UV_PIP_INSTALL_REQUIREMENTS)
    .run_commands(
        _TRIBE_IMPORT_RUNTIME_PREFLIGHT_COMMAND,
        f"python -m pip install --no-deps --target {CUDNN8_TARGET} {CUDNN8_PACKAGE}",
        f"test -f {CUDNN8_LIB_DIR}/libcudnn_ops_infer.so.8",
        "python -m spacy download en_core_web_sm",
        # TRIBE's native text path asks for the large English model at
        # request time if it is absent. Bake it in so text scoring does not
        # spend several minutes pip-installing during a live HTTP request.
        "python -m spacy download en_core_web_lg",
    )
    .env({"HF_HOME": HF_CACHE_DIR})
)


# ---------------------------------------------------------------------------
# CogVideoX-5B — text-to-video for brain-direction-conditioned generation.
# Uses torch 2.7+ so we can target B200 (sm_100). CogVideoX has no torch pin,
# unlike TRIBE, so this is safe.
# ---------------------------------------------------------------------------

COGVIDEOX_REPO_ID = "THUDM/CogVideoX-5b"
COGVIDEOX_CACHE_DIR = "/cogvideox-cache"

# Image-rebuild-tag: 2026-05-13-v1-cogvideox-b200
cogvideox_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "ffmpeg")
    .pip_install(
        # torch 2.7+ for sm_100 (B200). torch 2.7 ships in cu126 wheels.
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "diffusers>=0.32",
        "transformers>=4.45",
        "accelerate>=1.0",
        "sentencepiece",
        "protobuf",
        "huggingface_hub[hf_transfer]",
        "imageio==2.36.0",
        "imageio-ffmpeg==0.5.1",
        "numpy>=1.26,<3.0",
        # base deps needed because app.py imports every function module
        "pydantic[email]==2.12.5",
        "polars>=1.20",
        "google-genai==1.56.0",
        "anthropic==0.80.0",
        "fastapi>=0.115,<1.0",
        "python-multipart>=0.0.20,<1.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": COGVIDEOX_CACHE_DIR,
        }
    )
)


# ---------------------------------------------------------------------------
# Stable Video Diffusion (SVD-XT). img2vid → 25 frames @ 1024x576.
# Conditions on CLIP-ViT-H-14 image embeddings + VAE-encoded image.
# ---------------------------------------------------------------------------

SVD_REPO_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
SVD_CACHE_DIR = "/svd-cache"

svd_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "diffusers>=0.32",
        "transformers>=4.45",
        "accelerate>=1.0",
        "huggingface_hub[hf_transfer]",
        "imageio==2.36.0",
        "imageio-ffmpeg==0.5.1",
        "numpy>=1.26,<3.0",
        "pillow>=10",
        # base deps needed because app.py imports every function module
        "pydantic[email]==2.12.5",
        "polars>=1.20",
        "google-genai==1.56.0",
        "anthropic==0.80.0",
        "fastapi>=0.115,<1.0",
        "python-multipart>=0.0.20,<1.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": SVD_CACHE_DIR,
        }
    )
)


# ---------------------------------------------------------------------------
# Wan2.2 — high-quality open video generation target.
#
# We use strict B200 at the function/class level for this model: 180 GB VRAM is
# enough to run the 14B I2V/T2V models without the slow CPU-offload defaults,
# while avoiding B200+ / B300 because B300 currently requires a CUDA 13 stack.
# ---------------------------------------------------------------------------

WAN22_REPO_URL = "https://github.com/Wan-Video/Wan2.2.git"
WAN22_GIT_REF = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
WAN22_REPO_DIR = "/opt/Wan2.2"
WAN22_CACHE_DIR = "/wan22-cache"
WAN22_MODEL_REPOS = {
    "t2v-A14B": "Wan-AI/Wan2.2-T2V-A14B",
    "i2v-A14B": "Wan-AI/Wan2.2-I2V-A14B",
    "ti2v-5B": "Wan-AI/Wan2.2-TI2V-5B",
}

wan22_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        # PyTorch 2.7+ adds Blackwell support via the CUDA 12.8 wheels. The
        # CUDA 12.6 wheel can import on B200 but fails at runtime with
        # "no kernel image is available for execution on the device".
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "accelerate>=1.1.1",
        "dashscope",
        "decord",
        "diffusers>=0.31.0",
        "easydict",
        "einops",
        "ftfy",
        "huggingface_hub[hf_transfer]>=0.36",
        "imageio-ffmpeg>=0.5.1",
        "imageio[ffmpeg]>=2.36",
        "numpy>=1.23.5,<2",
        "opencv-python-headless>=4.9.0.80",
        "peft",
        "pillow>=10",
        "protobuf",
        "regex",
        "scipy",
        "sentencepiece",
        "tokenizers>=0.20.3",
        "tqdm",
        "transformers>=4.49.0,<=4.51.3",
        # base deps needed because app.py imports every function module
        "anthropic==0.80.0",
        "google-genai==1.56.0",
        "polars>=1.20",
        "pydantic[email]==2.12.5",
        "fastapi>=0.115,<1.0",
        "python-multipart>=0.0.20,<1.0",
    )
    .run_commands(
        f"git clone {WAN22_REPO_URL} {WAN22_REPO_DIR}",
        f"cd {WAN22_REPO_DIR} && git checkout {WAN22_GIT_REF}",
        (
            "python - <<'PY'\n"
            "from pathlib import Path\n"
            f"p = Path({WAN22_REPO_DIR!r}) / 'wan' / '__init__.py'\n"
            'p.write_text("""from . import configs, distributed, modules\\n'
            "from .image2video import WanI2V\\n"
            "from .text2video import WanT2V\\n"
            "from .textimage2video import WanTI2V\\n"
            '""")\n'
            f"model = Path({WAN22_REPO_DIR!r}) / 'wan' / 'modules' / 'model.py'\n"
            "text = model.read_text()\n"
            "text = text.replace(\n"
            "    'from .attention import flash_attention',\n"
            "    'from .attention import attention as flash_attention',\n"
            ")\n"
            "model.write_text(text)\n"
            "PY"
        ),
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": WAN22_CACHE_DIR,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)


# ---------------------------------------------------------------------------
# Musubi Tuner — Wan2.2 LoRA training.
#
# This is separate from the official Wan generation image because Musubi pins a
# newer Transformers stack and different runtime dependencies. The training
# target is Wan2.2 14B I2V/T2V; Musubi does not currently train TI2V-5B.
# ---------------------------------------------------------------------------

MUSUBI_TUNER_REPO_URL = "https://github.com/kohya-ss/musubi-tuner.git"
MUSUBI_TUNER_GIT_REF = "6306e839608cd6525def3e6b0d8ec8cd17ff459e"
WAN22_LORA_CACHE_DIR = "/wan22-lora-cache"

wan22_lora_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        f"git+{MUSUBI_TUNER_REPO_URL}@{MUSUBI_TUNER_GIT_REF}",
        "huggingface_hub[hf_transfer]==0.34.3",
        # base deps needed because app.py imports every function module
        "anthropic==0.80.0",
        "google-genai==1.56.0",
        "polars>=1.20",
        "pydantic[email]==2.12.5",
        "fastapi>=0.115,<1.0",
        "python-multipart>=0.0.20,<1.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": WAN22_LORA_CACHE_DIR,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)


__all__ = [
    "base_image",
    "tribe_image",
    "cogvideox_image",
    "svd_image",
    "wan22_image",
    "wan22_lora_image",
    "PYTHON_VERSION",
    "TRIBE_HF_REPO_ID",
    "TRIBE_HF_REVISION",
    "TRIBE_GIT_REF",
    "TRIBE_UV_PIP_INSTALL_REQUIREMENTS",
    "TRIBE_FEATURE_MODEL_PINS",
    "WHISPERX_MODEL_REPO_ID",
    "WHISPERX_MODEL_REVISION",
    "HF_CACHE_DIR",
    "CUDNN8_LIB_DIR",
    "COGVIDEOX_REPO_ID",
    "COGVIDEOX_CACHE_DIR",
    "SVD_REPO_ID",
    "SVD_CACHE_DIR",
    "WAN22_REPO_URL",
    "WAN22_GIT_REF",
    "WAN22_REPO_DIR",
    "WAN22_CACHE_DIR",
    "WAN22_MODEL_REPOS",
    "MUSUBI_TUNER_REPO_URL",
    "MUSUBI_TUNER_GIT_REF",
    "WAN22_LORA_CACHE_DIR",
]
