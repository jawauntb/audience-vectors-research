"""Helpers for replaying collaborator BO memorability trials.

The collaborator intake keeps the original BO code as provenance. This module
contains the small repo-native pieces we need to safely replay saved trial
tables on our Modal SVD/TRIBE stack without importing the whole BoTorch runner.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

TrialSelection = Literal["first", "top-tribe", "top-quality", "top-clip"]


@dataclass(frozen=True)
class CollaboratorBOTrial:
    """One evaluated point from the collaborator BO run table."""

    task_id: str
    alpha: float
    guidance: float
    seed_idx: int
    noise_seed: int
    filename: str | None
    prompt: str | None
    tribe_score: float | None
    clip_score: float | None
    quality_score: float | None

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> CollaboratorBOTrial:
        return cls(
            task_id=str(row["task_id"]),
            alpha=float(row["alpha"]),
            guidance=float(row.get("guidance", row.get("guidance_scale", 3.0))),
            seed_idx=int(row["seed_idx"]),
            noise_seed=int(row.get("noise_seed", 0)),
            filename=str(row["filename"]) if row.get("filename") else None,
            prompt=str(row["prompt"]) if row.get("prompt") else None,
            tribe_score=optional_float(row.get("tribe_score")),
            clip_score=optional_float(row.get("clip_score")),
            quality_score=optional_float(row.get("quality_score")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "alpha": self.alpha,
            "guidance": self.guidance,
            "seed_idx": self.seed_idx,
            "noise_seed": self.noise_seed,
            "filename": self.filename,
            "prompt": self.prompt,
            "tribe_score": self.tribe_score,
            "clip_score": self.clip_score,
            "quality_score": self.quality_score,
        }


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def load_collaborator_trials(path: Path) -> list[CollaboratorBOTrial]:
    """Load the `all_meta` table from a collaborator BO results JSON file."""
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        rows = payload.get("all_meta")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a list or `all_meta` list")
    return [CollaboratorBOTrial.from_mapping(row) for row in rows]


def select_trials(
    trials: list[CollaboratorBOTrial],
    *,
    selection: TrialSelection = "top-tribe",
    max_evals: int = 2,
    task_ids: set[str] | None = None,
) -> list[CollaboratorBOTrial]:
    """Select trials for a smoke/replay run."""
    if max_evals <= 0:
        raise ValueError("max_evals must be positive")
    if task_ids:
        by_id = {trial.task_id: trial for trial in trials}
        missing = sorted(task_ids - set(by_id))
        if missing:
            raise ValueError(f"unknown task ids: {', '.join(missing)}")
        return [by_id[task_id] for task_id in sorted(task_ids)]

    if selection == "first":
        ordered = trials
    elif selection == "top-tribe":
        ordered = sorted(
            trials,
            key=lambda trial: trial.tribe_score if trial.tribe_score is not None else -np.inf,
            reverse=True,
        )
    elif selection == "top-clip":
        ordered = sorted(
            trials,
            key=lambda trial: trial.clip_score if trial.clip_score is not None else -np.inf,
            reverse=True,
        )
    elif selection == "top-quality":
        ordered = sorted(
            trials,
            key=lambda trial: trial.quality_score
            if trial.quality_score is not None
            else -np.inf,
            reverse=True,
        )
    else:
        raise ValueError(f"unsupported selection: {selection}")
    return ordered[:max_evals]


def load_unit_npz_vector(path: Path, *, key: str = "direction") -> np.ndarray:
    """Load and unit-normalize a vector from an npz file."""
    payload = np.load(path, allow_pickle=False)
    if key not in payload:
        available = ", ".join(payload.files)
        raise ValueError(f"{path} does not contain key {key!r}; available: {available}")
    return normalize_vector(np.asarray(payload[key], dtype=np.float32))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("near-zero vector")
    return vector.astype(np.float32) / norm


def score_projection(frames: np.ndarray, direction: np.ndarray) -> float:
    """Project a TRIBE frame tensor or mean vector onto a unit direction."""
    arr = np.asarray(frames, dtype=np.float32)
    vec = arr.mean(axis=0) if arr.ndim == 2 else arr
    if vec.shape != direction.shape:
        raise ValueError(f"feature shape {vec.shape} does not match {direction.shape}")
    return float(vec @ direction)


def safe_label(value: str) -> str:
    """Return a filesystem/Modal-volume-safe label."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "bo_trial"


def replay_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute a compact summary for a replay report."""
    scored = [row for row in rows if row.get("replay_tribe_score") is not None]
    deltas = [
        float(row["replay_tribe_score"]) - float(row["original_tribe_score"])
        for row in scored
        if row.get("original_tribe_score") is not None
    ]
    return {
        "n_requested": len(rows),
        "n_scored": len(scored),
        "mean_score_delta_vs_original": float(np.mean(deltas)) if deltas else None,
        "max_abs_score_delta_vs_original": (
            float(np.max(np.abs(deltas))) if deltas else None
        ),
    }
