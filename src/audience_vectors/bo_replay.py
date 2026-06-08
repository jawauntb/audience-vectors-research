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

TrialSelection = Literal[
    "first",
    "top-tribe",
    "top-quality",
    "top-clip",
    "top-bo-tribe",
    "top-sobol-tribe",
    "top-bo-vs-top-sobol",
    "seed-stratified-bo-vs-sobol",
    "top-bo-per-stratum",
]
TrialStratum = Literal["prompt", "seed_idx"]


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
    stratify_by: TrialStratum = "prompt",
) -> list[CollaboratorBOTrial]:
    """Select trials for a smoke/replay run.

    For `top-bo-vs-top-sobol`, `max_evals` is applied per group so the returned
    panel has up to `2 * max_evals` trials.
    """
    if max_evals <= 0:
        raise ValueError("max_evals must be positive")
    if task_ids:
        by_id = {trial.task_id: trial for trial in trials}
        missing = sorted(task_ids - set(by_id))
        if missing:
            raise ValueError(f"unknown task ids: {', '.join(missing)}")
        return [by_id[task_id] for task_id in sorted(task_ids)]

    if selection == "first":
        return trials[:max_evals]
    if selection == "top-bo-vs-top-sobol":
        return select_top_policy_trials(
            trials,
            policy_group="bo",
            max_evals=max_evals,
        ) + select_top_policy_trials(
            trials,
            policy_group="sobol",
            max_evals=max_evals,
        )
    if selection == "seed-stratified-bo-vs-sobol":
        return select_seed_stratified_policy_trials(
            trials,
            policy_groups=("bo", "sobol"),
            max_evals_per_group=max_evals,
            stratify_by=stratify_by,
        )
    if selection == "top-bo-per-stratum":
        return select_top_policy_trials_per_stratum(
            trials,
            policy_group="bo",
            max_evals_per_stratum=max_evals,
            stratify_by=stratify_by,
        )

    selection_config: dict[str, tuple[str | None, str]] = {
        "top-tribe": (None, "tribe_score"),
        "top-clip": (None, "clip_score"),
        "top-quality": (None, "quality_score"),
        "top-bo-tribe": ("bo", "tribe_score"),
        "top-sobol-tribe": ("sobol", "tribe_score"),
    }
    if selection not in selection_config:
        raise ValueError(f"unsupported selection: {selection}")
    policy_group, score_name = selection_config[selection]
    return select_top_policy_trials(
        trials,
        policy_group=policy_group,
        score_name=score_name,
        max_evals=max_evals,
    )


def select_top_policy_trials(
    trials: list[CollaboratorBOTrial],
    *,
    policy_group: str | None,
    max_evals: int,
    score_name: str = "tribe_score",
) -> list[CollaboratorBOTrial]:
    group_trials = (
        trials
        if policy_group is None
        else [
            trial
            for trial in trials
            if trial_policy_group(trial.task_id) == policy_group
        ]
    )
    return sort_by_score(group_trials, score_name=score_name)[:max_evals]


def select_seed_stratified_policy_trials(
    trials: list[CollaboratorBOTrial],
    *,
    policy_groups: tuple[str, ...],
    max_evals_per_group: int,
    score_name: str = "tribe_score",
    stratify_by: TrialStratum = "prompt",
) -> list[CollaboratorBOTrial]:
    """Select top trials per policy only inside matched strata."""
    if max_evals_per_group <= 0:
        raise ValueError("max_evals_per_group must be positive")
    if not policy_groups:
        raise ValueError("policy_groups must not be empty")

    grouped: dict[str, dict[str, list[CollaboratorBOTrial]]] = {}
    for trial in trials:
        policy_group = trial_policy_group(trial.task_id)
        if policy_group not in policy_groups:
            continue
        stratum_key = trial_stratum_key(trial, stratify_by=stratify_by)
        grouped.setdefault(stratum_key, {}).setdefault(policy_group, []).append(trial)

    selected: list[CollaboratorBOTrial] = []
    for stratum_key in sorted(grouped):
        stratum = grouped[stratum_key]
        if not all(stratum.get(policy_group) for policy_group in policy_groups):
            continue
        for policy_group in policy_groups:
            selected.extend(
                sort_by_score(
                    stratum[policy_group],
                    score_name=score_name,
                )[:max_evals_per_group]
            )
    return selected


def select_top_policy_trials_per_stratum(
    trials: list[CollaboratorBOTrial],
    *,
    policy_group: str,
    max_evals_per_stratum: int,
    score_name: str = "tribe_score",
    stratify_by: TrialStratum = "prompt",
) -> list[CollaboratorBOTrial]:
    """Select the top scored trials for one policy inside each available stratum."""
    if max_evals_per_stratum <= 0:
        raise ValueError("max_evals_per_stratum must be positive")

    grouped: dict[str, list[CollaboratorBOTrial]] = {}
    for trial in trials:
        if trial_policy_group(trial.task_id) != policy_group:
            continue
        grouped.setdefault(
            trial_stratum_key(trial, stratify_by=stratify_by),
            [],
        ).append(trial)

    selected: list[CollaboratorBOTrial] = []
    for stratum_key in sorted(grouped):
        selected.extend(
            sort_by_score(grouped[stratum_key], score_name=score_name)[
                :max_evals_per_stratum
            ]
        )
    return selected


def sort_by_score(
    trials: list[CollaboratorBOTrial],
    *,
    score_name: str,
) -> list[CollaboratorBOTrial]:
    """Sort trials by one original objective score, descending."""
    return sorted(
        trials,
        key=lambda trial: getattr(trial, score_name)
        if getattr(trial, score_name) is not None
        else -np.inf,
        reverse=True,
    )


def trial_policy_group(task_id: str) -> str:
    """Return the policy group represented by a collaborator task id."""
    return "sobol" if task_id.startswith("sobol") else "bo"


def trial_stratum_key(
    trial: CollaboratorBOTrial,
    *,
    stratify_by: TrialStratum = "prompt",
) -> str:
    """Return the stratum key used for matched tournament selection."""
    if stratify_by == "seed_idx":
        return str(trial.seed_idx)
    if trial.prompt:
        return trial.prompt
    return f"seed_idx:{trial.seed_idx}"


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


def replicate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize replay score distributions by collaborator trial."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        trial = row.get("trial")
        task_id = None
        if isinstance(trial, dict):
            task_id = trial.get("task_id")
        if task_id is None:
            task_id = row.get("task_id") or row.get("label") or "unknown"
        grouped.setdefault(str(task_id), []).append(row)

    summaries: list[dict[str, Any]] = []
    for task_id, task_rows in grouped.items():
        scores = [
            float(row["replay_tribe_score"])
            for row in task_rows
            if row.get("replay_tribe_score") is not None
        ]
        score_array = np.asarray(scores, dtype=np.float64)
        mean_score = float(np.mean(score_array)) if scores else None
        std_score = (
            float(np.std(score_array, ddof=1))
            if len(scores) > 1
            else (0.0 if scores else None)
        )

        original_score = first_original_tribe_score(task_rows)
        summaries.append(
            {
                "task_id": task_id,
                "n_requested": len(task_rows),
                "n_scored": len(scores),
                "original_tribe_score": original_score,
                "mean_replay_tribe_score": mean_score,
                "std_replay_tribe_score": std_score,
                "sem_replay_tribe_score": (
                    float(std_score / np.sqrt(len(scores)))
                    if std_score is not None and len(scores) > 1
                    else (0.0 if scores else None)
                ),
                "min_replay_tribe_score": float(np.min(score_array))
                if scores
                else None,
                "max_replay_tribe_score": float(np.max(score_array))
                if scores
                else None,
                "mean_delta_vs_original": (
                    mean_score - original_score
                    if mean_score is not None and original_score is not None
                    else None
                ),
                "replay_tribe_scores": scores,
                "noise_seeds": [
                    int(row["noise_seed"])
                    for row in task_rows
                    if row.get("noise_seed") is not None
                ],
            }
        )

    ranked = sorted(
        summaries,
        key=lambda item: item["mean_replay_tribe_score"]
        if item["mean_replay_tribe_score"] is not None
        else -np.inf,
        reverse=True,
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank_by_mean_replay_tribe_score"] = rank
    return ranked


def policy_group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare replay score distributions by search policy group."""
    candidate_summaries = replicate_summary(rows)
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    candidate_summaries_by_group: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        task_id = row_task_id(row)
        rows_by_group.setdefault(trial_policy_group(task_id), []).append(row)

    for item in candidate_summaries:
        task_id = str(item["task_id"])
        candidate_summaries_by_group.setdefault(
            trial_policy_group(task_id),
            [],
        ).append(item)

    summaries: list[dict[str, Any]] = []
    for group in sorted(rows_by_group):
        group_rows = rows_by_group[group]
        group_candidates = candidate_summaries_by_group.get(group, [])
        scores = [
            float(row["replay_tribe_score"])
            for row in group_rows
            if row.get("replay_tribe_score") is not None
        ]
        candidate_means = [
            float(item["mean_replay_tribe_score"])
            for item in group_candidates
            if item.get("mean_replay_tribe_score") is not None
        ]
        original_scores = [
            float(row["original_tribe_score"])
            for row in group_rows
            if row.get("original_tribe_score") is not None
        ]
        best_candidate = next(iter(group_candidates), None)

        score_array = np.asarray(scores, dtype=np.float64)
        candidate_mean_array = np.asarray(candidate_means, dtype=np.float64)
        original_array = np.asarray(original_scores, dtype=np.float64)
        summaries.append(
            {
                "policy_group": group,
                "n_candidates": len(group_candidates),
                "n_requested": len(group_rows),
                "n_scored": len(scores),
                "pooled_mean_replay_tribe_score": float(np.mean(score_array))
                if scores
                else None,
                "pooled_std_replay_tribe_score": sample_std(scores),
                "mean_candidate_replay_tribe_score": (
                    float(np.mean(candidate_mean_array)) if candidate_means else None
                ),
                "std_candidate_replay_tribe_score": sample_std(candidate_means),
                "mean_original_tribe_score": float(np.mean(original_array))
                if original_scores
                else None,
                "best_candidate_task_id": best_candidate["task_id"]
                if best_candidate
                else None,
                "best_candidate_mean_replay_tribe_score": (
                    best_candidate["mean_replay_tribe_score"]
                    if best_candidate
                    else None
                ),
            }
        )
    return summaries


def stratum_policy_summary(
    rows: list[dict[str, Any]],
    *,
    stratify_by: TrialStratum = "prompt",
) -> list[dict[str, Any]]:
    """Compare replay score distributions by stratum and search policy."""
    candidate_summaries = {
        str(item["task_id"]): item for item in replicate_summary(rows)
    }
    rows_by_stratum_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    groups_by_stratum: dict[str, set[str]] = {}

    for row in rows:
        task_id = row_task_id(row)
        policy_group = trial_policy_group(task_id)
        stratum_key = row_stratum_key(row, stratify_by=stratify_by)
        rows_by_stratum_group.setdefault((stratum_key, policy_group), []).append(row)
        groups_by_stratum.setdefault(stratum_key, set()).add(policy_group)

    summaries: list[dict[str, Any]] = []
    for stratum_key in sorted(groups_by_stratum):
        stratum_groups = groups_by_stratum[stratum_key]
        matched_bo_sobol = {"bo", "sobol"}.issubset(stratum_groups)
        for policy_group in sorted(stratum_groups):
            group_rows = rows_by_stratum_group[(stratum_key, policy_group)]
            task_ids = sorted({row_task_id(row) for row in group_rows})
            group_candidates = [
                candidate_summaries[task_id]
                for task_id in task_ids
                if task_id in candidate_summaries
            ]
            scores = [
                float(row["replay_tribe_score"])
                for row in group_rows
                if row.get("replay_tribe_score") is not None
            ]
            candidate_means = [
                float(item["mean_replay_tribe_score"])
                for item in group_candidates
                if item.get("mean_replay_tribe_score") is not None
            ]
            original_scores = [
                float(row["original_tribe_score"])
                for row in group_rows
                if row.get("original_tribe_score") is not None
            ]
            score_array = np.asarray(scores, dtype=np.float64)
            candidate_mean_array = np.asarray(candidate_means, dtype=np.float64)
            original_array = np.asarray(original_scores, dtype=np.float64)
            summaries.append(
                {
                    "stratum_key": stratum_key,
                    "stratify_by": stratify_by,
                    "policy_group": policy_group,
                    "matched_bo_sobol": matched_bo_sobol,
                    "available_policy_groups": sorted(stratum_groups),
                    "n_candidates": len(group_candidates),
                    "n_requested": len(group_rows),
                    "n_scored": len(scores),
                    "pooled_mean_replay_tribe_score": float(np.mean(score_array))
                    if scores
                    else None,
                    "pooled_std_replay_tribe_score": sample_std(scores),
                    "mean_candidate_replay_tribe_score": (
                        float(np.mean(candidate_mean_array))
                        if candidate_means
                        else None
                    ),
                    "std_candidate_replay_tribe_score": sample_std(candidate_means),
                    "mean_original_tribe_score": float(np.mean(original_array))
                    if original_scores
                    else None,
                    "task_ids": task_ids,
                }
            )
    return summaries


def sample_std(values: list[float]) -> float | None:
    if len(values) > 1:
        return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
    if values:
        return 0.0
    return None


def row_task_id(row: dict[str, Any]) -> str:
    trial = row.get("trial")
    if isinstance(trial, dict) and trial.get("task_id") is not None:
        return str(trial["task_id"])
    return str(row.get("task_id") or row.get("label") or "unknown")


def row_stratum_key(
    row: dict[str, Any],
    *,
    stratify_by: TrialStratum = "prompt",
) -> str:
    trial = row.get("trial")
    if isinstance(trial, dict):
        if stratify_by == "seed_idx" and trial.get("seed_idx") is not None:
            return str(trial["seed_idx"])
        if stratify_by == "prompt" and trial.get("prompt"):
            return str(trial["prompt"])
    seed = row.get("seed")
    if isinstance(seed, dict) and seed.get("bmd_name"):
        return str(seed["bmd_name"])
    return "unknown"


def first_original_tribe_score(rows: list[dict[str, Any]]) -> float | None:
    for row in rows:
        value = row.get("original_tribe_score")
        if value is None and isinstance(row.get("trial"), dict):
            value = row["trial"].get("tribe_score")
        if value is not None:
            return float(value)
    return None
