from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Paths:
    vmem: Path = Path(os.getenv("VMEM_PATH", "artifacts/v_mem.npy"))
    persona_dirs: Path = Path(os.getenv("PERSONA_DIRS_PATH", "artifacts/persona_directions.npy"))
    tribe_clip_adapter: Path = Path(
        os.getenv("TRIBE_CLIP_ADAPTER_PATH", "artifacts/tribe_clip_adapter.pt")
    )
    tribe_features_cache: Path | None = field(
        default_factory=lambda: Path(p) if (p := os.getenv("TRIBE_FEATURES_CACHE")) else None
    )
    persona_scores_csv: Path = Path(
        os.getenv("PERSONA_SCORES_CSV", "../audience_vectors_share/reports/datasets/persona_scores.csv")
    )
    generated_svd_dir: Path = Path(
        os.getenv("GENERATED_SVD_DIR", "../audience_vectors_share/generated/svd_best_of_n")
    )


@dataclass
class BOConfig:
    # Search space
    k_directions: int = int(os.getenv("BO_K_DIRECTIONS", "5"))
    alpha_min: float = float(os.getenv("BO_ALPHA_MIN", "-10.0"))
    alpha_max: float = float(os.getenv("BO_ALPHA_MAX", "10.0"))
    guidance_min: float = float(os.getenv("BO_GUIDANCE_SCALE_MIN", "1.0"))
    guidance_max: float = float(os.getenv("BO_GUIDANCE_SCALE_MAX", "15.0"))

    # Loop
    batch_size: int = int(os.getenv("BO_BATCH_SIZE", "4"))
    n_initial: int = int(os.getenv("BO_N_INITIAL", "12"))
    n_iterations: int = int(os.getenv("BO_N_ITERATIONS", "10"))

    # Constraint
    fvd_threshold: float = float(os.getenv("FVD_THRESHOLD", "300.0"))

    # Seed pool (discrete categorical)
    n_seeds: int = 16

    @property
    def dim_continuous(self) -> int:
        return self.k_directions + 1  # α_1..α_k + guidance_scale

    @property
    def dim_total(self) -> int:
        return self.dim_continuous + 1  # + seed_id (categorical)


@dataclass
class SVDConfig:
    model_id: str = os.getenv("SVD_MODEL_ID", "stabilityai/stable-video-diffusion-img2vid-xt")
    num_frames: int = int(os.getenv("SVD_NUM_FRAMES", "25"))
    steps_low: int = int(os.getenv("SVD_NUM_INFERENCE_STEPS_LOW", "10"))
    steps_med: int = int(os.getenv("SVD_NUM_INFERENCE_STEPS_MED", "25"))
    steps_high: int = int(os.getenv("SVD_NUM_INFERENCE_STEPS_HIGH", "50"))
    fps: int = 6


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    bo: BOConfig = field(default_factory=BOConfig)
    svd: SVDConfig = field(default_factory=SVDConfig)
    device: str = field(default_factory=lambda: "cuda" if __import__("torch").cuda.is_available() else "mps" if __import__("torch").backends.mps.is_available() else "cpu")
    wandb_project: str = os.getenv("WANDB_PROJECT", "bo-memorability")
    hf_token: str | None = os.getenv("HUGGINGFACE_TOKEN")
