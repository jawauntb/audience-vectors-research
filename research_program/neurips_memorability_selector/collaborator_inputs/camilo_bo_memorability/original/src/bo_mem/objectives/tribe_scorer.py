"""TRIBE v2 memorability scorer.

Computes proj(TRIBE(video), v_mem) — the projection of the video's cortical
activation onto the memorability direction.

Supports two scoring modes:
  - cached: score directly from pre-computed .npz feature files (no GPU needed)
  - live:   score via a TribeBackend (Modal inference, requires GPU)

Artifacts loaded from bo_memorability/artifacts/:
  - v_mem.npz            : keys include 'direction' (20484,) float32
  - tribe_clip_adapter.pt: keys include 'v_mem_clip_h_via_adapter' (1024,)
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

_DEFAULT_VMEM = Path(__file__).parent.parent.parent.parent / "artifacts" / "v_mem.npz"


class TribeBackend(Protocol):
    """Interface the real TRIBE inference must satisfy."""

    def predict(self, video_path: Path) -> np.ndarray:
        """Return (20484,) cortical activation vector for a video clip."""
        ...


class MockTribeBackend:
    """Placeholder that returns random activations. Replace with real backend."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(self, video_path: Path) -> np.ndarray:
        return self._rng.standard_normal(20484).astype(np.float32)


class CachedTribeBackend:
    """Returns pre-computed TRIBE activations from .npz feature files (no GPU)."""

    def __init__(self, features_dir: Path) -> None:
        self._dir = features_dir

    def predict(self, video_path: Path) -> np.ndarray:
        npz_path = self._dir / (video_path.stem + ".npz")
        if not npz_path.exists():
            raise FileNotFoundError(f"No cached features for {video_path.stem}")
        data = np.load(npz_path)
        # frames: (4, 20484) — mean across frames to get (20484,)
        frames = data["frames"].astype(np.float32)
        return frames.mean(axis=0)


def _load_vmem(vmem_path: Path) -> np.ndarray:
    """Load v_mem direction from .npz or .npy; always returns unit-normed (20484,)."""
    if vmem_path.suffix == ".npz":
        data = np.load(vmem_path)
        raw = data["direction"].astype(np.float32)
    else:
        raw = np.load(vmem_path).astype(np.float32)
    norm = np.linalg.norm(raw)
    return raw / norm if norm > 0 else raw


class TribeScorer:
    """Projects TRIBE cortical activations onto v_mem to compute memorability score."""

    def __init__(
        self,
        vmem_path: Path | None = None,
        backend: TribeBackend | None = None,
        features_dir: Path | None = None,
    ) -> None:
        # Resolve v_mem
        resolved = vmem_path or _DEFAULT_VMEM
        if resolved.exists():
            self._vmem: np.ndarray = _load_vmem(resolved)
        else:
            self._vmem = np.zeros(20484, dtype=np.float32)

        # Resolve backend
        if backend is not None:
            self._backend: TribeBackend = backend
        elif features_dir is not None:
            self._backend = CachedTribeBackend(features_dir)
        else:
            self._backend = MockTribeBackend()

    @property
    def vmem_loaded(self) -> bool:
        return bool(np.any(self._vmem != 0))

    def score(self, video_path: Path) -> float:
        """Return scalar memorability projection for a single video."""
        activation = self._backend.predict(video_path)  # (20484,)
        return float(np.dot(activation, self._vmem))

    def score_activation(self, activation: np.ndarray) -> float:
        """Score a pre-computed (20484,) activation directly."""
        return float(np.dot(activation.astype(np.float32), self._vmem))

    def score_batch(self, video_paths: list[Path]) -> Tensor:
        """Score a list of videos; returns shape (N,)."""
        scores = [self.score(p) for p in video_paths]
        return torch.tensor(scores, dtype=torch.double)
