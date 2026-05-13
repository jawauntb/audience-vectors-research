"""Environment-backed config. Loads `.env` once, exposes typed accessors."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_path(name: str, default: str) -> Path:
    raw = os.environ.get(name, default)
    return (_REPO_ROOT / raw).resolve() if raw.startswith("./") else Path(raw).resolve()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


class Paths(BaseModel):
    repo_root: Path = _REPO_ROOT
    data_root: Path = _env_path("DATA_ROOT", "./data")
    raw: Path = _env_path("RAW_DIR", "./data/raw")
    processed: Path = _env_path("PROCESSED_DIR", "./data/processed")
    clips: Path = _env_path("CLIPS_DIR", "./data/processed/clips_3s")
    frames: Path = _env_path("FRAMES_DIR", "./data/processed/frames")
    audio: Path = _env_path("AUDIO_DIR", "./data/processed/audio")
    transcripts: Path = _env_path("TRANSCRIPTS_DIR", "./data/processed/transcripts")
    features: Path = _env_path("FEATURES_DIR", "./data/features")
    labels: Path = _env_path("LABELS_DIR", "./data/labels")
    training: Path = _env_path("TRAINING_DIR", "./data/training")
    models: Path = _env_path("MODELS_DIR", "./data/models")
    vectors: Path = _env_path("VECTORS_DIR", "./data/models/vectors")
    reports: Path = _env_path("REPORTS_DIR", "./data/reports")
    manifest_db_url: str = _env("DATABASE_URL", "sqlite:///./data/manifest.db")

    def ensure(self) -> None:
        for p in [
            self.data_root, self.raw, self.processed, self.clips, self.frames,
            self.audio, self.transcripts, self.features, self.labels,
            self.training, self.models, self.vectors, self.reports,
        ]:
            p.mkdir(parents=True, exist_ok=True)


class ModelIds(BaseModel):
    tribe: str = _env("TRIBE_MODEL_ID", "facebook/tribev2")
    tribe_revision: str = _env("TRIBE_MODEL_REVISION", "")
    tribe_git_ref: str = _env("TRIBE_GIT_REF", "")
    vjepa: str = _env("VJEPA_MODEL_ID", "facebook/vjepa2-vitl-fpc64-256")
    vjepa_large: str = _env("VJEPA_LARGE_MODEL_ID", "facebook/vjepa2-vitg-fpc64-256")
    internvideo: str = _env("INTERNVIDEO_MODEL_ID", "OpenGVLab/InternVideo2-Stage2_1B-224p-f4")
    qwen_vl: str = _env("QWEN_VL_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
    whisper: str = _env("WHISPER_MODEL_ID", "openai/whisper-large-v3")
    faster_whisper: str = _env("FASTER_WHISPER_MODEL_ID", "Systran/faster-whisper-large-v3")


class ApiKeys(BaseModel):
    anthropic: str = _env("ANTHROPIC_API_KEY")
    openai: str = _env("OPENAI_API_KEY")
    google: str = _env("GOOGLE_API_KEY")
    huggingface: str = _env("HUGGINGFACE_TOKEN")
    wandb: str = _env("WANDB_API_KEY")


class Pipeline(BaseModel):
    segment_length_s: float = _env_float("SEGMENT_LENGTH_SECONDS", 3.0)
    segment_stride_s: float = _env_float("SEGMENT_STRIDE_SECONDS", 3.0)
    sample_fps: int = _env_int("SAMPLE_FPS", 2)
    audio_sample_rate: int = _env_int("AUDIO_SAMPLE_RATE", 16000)
    dev_video_limit: int = _env_int("DEV_VIDEO_LIMIT", 20)
    random_seed: int = _env_int("RANDOM_SEED", 42)


class Models(BaseModel):
    anthropic_judge: str = _env("ANTHROPIC_MODEL_JUDGE", "claude-sonnet-4-6")
    anthropic_bulk: str = _env("ANTHROPIC_MODEL_BULK", "claude-haiku-4-5-20251001")
    gemini_video: str = _env("GEMINI_MODEL_VIDEO", "gemini-2.0-flash")
    gemini_judge: str = _env("GEMINI_MODEL_JUDGE", "gemini-2.5-pro")


class Config(BaseModel):
    project_name: str = Field(default_factory=lambda: _env("PROJECT_NAME", "audience_vectors"))
    env: str = Field(default_factory=lambda: _env("ENV", "dev"))
    log_level: str = Field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    paths: Paths = Field(default_factory=Paths)
    model_ids: ModelIds = Field(default_factory=ModelIds)
    api_keys: ApiKeys = Field(default_factory=ApiKeys)
    models: Models = Field(default_factory=Models)
    pipeline: Pipeline = Field(default_factory=Pipeline)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()
