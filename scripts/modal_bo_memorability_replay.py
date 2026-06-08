"""Replay collaborator BO memorability trials through the repo Modal stack.

This is the bridge between the collaborator's local RTX run and our shared
Modal infrastructure:

1. Read a saved BO trial table (`all_meta`).
2. Select a tiny smoke batch or a larger fixed-budget replay.
3. Generate SVD-XT videos on Modal, passing alpha and guidance.
4. Upload generated MP4s to the `bmd-videos-v1` Modal volume.
5. Score them with the deployed TRIBE predictor and project onto cortical v_mem.

The script intentionally does not commit or require model weights in the repo.
Pass those via local paths or environment variables.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from audience_vectors.bo_prompt_manifests import DEFAULT_REPLAY_SEED_POOL_SIZE
from audience_vectors.bo_replay import (
    CollaboratorBOTrial,
    TrialStratum,
    load_collaborator_trials,
    load_unit_npz_vector,
    policy_group_summary,
    replay_summary,
    replicate_summary,
    safe_label,
    score_projection,
    select_trials,
    stratum_policy_summary,
    trial_policy_group,
    trial_stratum_key,
)
from audience_vectors.visual_artifact_gate import ArtifactThresholds, summarize_video

REPO_ROOT = Path(__file__).resolve().parents[1]
INTAKE_ROOT = (
    REPO_ROOT
    / "research_program"
    / "neurips_memorability_selector"
    / "collaborator_inputs"
    / "camilo_bo_memorability"
)
DEFAULT_TRIAL_TABLE = INTAKE_ROOT / "raw_results" / "gpu_run_3obj_all_results.json"
DEFAULT_SEED_ROOT = INTAKE_ROOT / "original"
ORIGINAL_SOBOL_SEED_SLOTS = 15


@dataclass(frozen=True)
class ReplayJob:
    """One Modal replay attempt for a collaborator BO trial."""

    trial: CollaboratorBOTrial
    trial_index: int
    replicate_index: int
    noise_seed: int


def log(message: str) -> None:
    print(f"[bo-replay] {message}", flush=True)


def load_seed_pool(
    seed_root: Path,
    *,
    n_pool: int = DEFAULT_REPLAY_SEED_POOL_SIZE,
) -> list[dict[str, Any]]:
    """Load the collaborator seed pool, cycling available images to n_pool."""
    if n_pool <= 0:
        raise ValueError("seed pool size must be positive")

    prompts_path = seed_root / "seeds" / "prompts.json"
    prompts = json.loads(prompts_path.read_text())
    available: list[dict[str, Any]] = []
    for row in prompts:
        seed_image = row.get("seed_image")
        if not seed_image:
            continue
        image_path = seed_root / seed_image
        if not image_path.exists():
            continue
        available.append(
            {
                "idx": int(row["idx"]),
                "bmd_name": str(row["bmd_name"]),
                "prompt": str(row["prompt"]),
                "image_path": image_path,
            }
        )
    if not available:
        raise FileNotFoundError(f"no available seed images under {seed_root}")
    return [available[idx % len(available)] for idx in range(n_pool)]


def draw_sobol_points(count: int, *, scramble_seed: int) -> np.ndarray:
    """Draw collaborator-compatible Sobol points for regenerated controls."""
    from torch.quasirandom import SobolEngine  # noqa: PLC0415

    engine = SobolEngine(dimension=3, scramble=True, seed=scramble_seed)
    return np.asarray(engine.draw(count), dtype=np.float64)


def sobol_control_trial(
    *,
    sobol_index: int,
    point: np.ndarray,
    seed_pool: list[dict[str, Any]],
) -> CollaboratorBOTrial:
    """Convert one Sobol point to a replayable, unscored control trial."""
    seed_idx = int(float(point[2]) * ORIGINAL_SOBOL_SEED_SLOTS)
    seed = seed_pool[seed_idx % len(seed_pool)]
    return CollaboratorBOTrial(
        task_id=f"sobol_regen_{sobol_index:03d}",
        alpha=float(point[0] * 20.0 - 10.0),
        guidance=float(point[1] * 9.0 + 1.0),
        seed_idx=seed_idx,
        noise_seed=sobol_index,
        filename=None,
        prompt=str(seed["prompt"]),
        tribe_score=None,
        clip_score=None,
        quality_score=None,
    )


def generate_sobol_control_trials(
    *,
    seed_pool: list[dict[str, Any]],
    existing_task_ids: set[str],
    start_index: int,
    pool_size: int,
    scramble_seed: int,
    sobol_points: np.ndarray | None = None,
) -> list[CollaboratorBOTrial]:
    """Generate fresh deterministic Sobol controls, skipping saved-table ids."""
    if start_index < 0:
        raise ValueError("regenerated Sobol start index must be non-negative")
    if pool_size <= 0:
        raise ValueError("regenerated Sobol pool size must be positive")

    if sobol_points is None:
        points = draw_sobol_points(
            start_index + pool_size,
            scramble_seed=scramble_seed,
        )[start_index:]
    else:
        points = np.asarray(sobol_points, dtype=np.float64)[:pool_size]

    controls: list[CollaboratorBOTrial] = []
    for offset, point in enumerate(points):
        sobol_index = start_index + offset
        if f"sobol_{sobol_index:03d}" in existing_task_ids:
            continue
        if f"sobol_regen_{sobol_index:03d}" in existing_task_ids:
            continue
        controls.append(
            sobol_control_trial(
                sobol_index=sobol_index,
                point=point,
                seed_pool=seed_pool,
            )
        )
    return controls


def append_regenerated_sobol_controls(
    selected: list[CollaboratorBOTrial],
    *,
    all_trials: list[CollaboratorBOTrial],
    seed_pool: list[dict[str, Any]],
    stratify_by: TrialStratum,
    controls_per_stratum: int,
    pool_size: int,
    start_index: int,
    scramble_seed: int,
    sobol_points: np.ndarray | None = None,
) -> tuple[list[CollaboratorBOTrial], dict[str, Any] | None]:
    """Append unscored regenerated Sobol controls for selected BO strata."""
    if controls_per_stratum <= 0:
        return selected, None

    target_strata = sorted(
        {
            trial_stratum_key(trial, stratify_by=stratify_by)
            for trial in selected
            if trial_policy_group(trial.task_id) == "bo"
        }
    )
    if not target_strata:
        return selected, {
            "schema_version": 1,
            "controls_per_stratum": controls_per_stratum,
            "pool_size": pool_size,
            "start_index": start_index,
            "scramble_seed": scramble_seed,
            "seed_slots": ORIGINAL_SOBOL_SEED_SLOTS,
            "stratify_by": stratify_by,
            "target_strata": [],
            "n_generated_controls": 0,
            "controls": [],
            "missing_strata": [],
        }

    existing_task_ids = {trial.task_id for trial in all_trials}
    generated_controls = generate_sobol_control_trials(
        seed_pool=seed_pool,
        existing_task_ids=existing_task_ids,
        start_index=start_index,
        pool_size=pool_size,
        scramble_seed=scramble_seed,
        sobol_points=sobol_points,
    )

    controls_by_stratum: dict[str, list[CollaboratorBOTrial]] = {
        key: [] for key in target_strata
    }
    for control in generated_controls:
        stratum_key = trial_stratum_key(control, stratify_by=stratify_by)
        if stratum_key not in controls_by_stratum:
            continue
        if len(controls_by_stratum[stratum_key]) >= controls_per_stratum:
            continue
        controls_by_stratum[stratum_key].append(control)
        if all(
            len(items) >= controls_per_stratum
            for items in controls_by_stratum.values()
        ):
            break

    controls = [
        control
        for stratum_key in target_strata
        for control in controls_by_stratum[stratum_key]
    ]
    summary = {
        "schema_version": 1,
        "controls_per_stratum": controls_per_stratum,
        "pool_size": pool_size,
        "start_index": start_index,
        "scramble_seed": scramble_seed,
        "seed_slots": ORIGINAL_SOBOL_SEED_SLOTS,
        "stratify_by": stratify_by,
        "target_strata": target_strata,
        "n_generated_controls": len(controls),
        "controls": [control.to_json() for control in controls],
        "missing_strata": [
            stratum_key
            for stratum_key in target_strata
            if len(controls_by_stratum[stratum_key]) < controls_per_stratum
        ],
    }
    return selected + controls, summary


def image_bytes(path: Path) -> bytes:
    """Load a seed image as PNG bytes for Modal SVD."""
    image = Image.open(path).convert("RGB").resize((1024, 576))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_steering_vector(path: Path, *, key: str) -> list[float]:
    """Load v_mem_CLIP from a local `.pt`, `.npz`, or `.npy` artifact."""
    if path.suffix == ".pt":
        import torch  # noqa: PLC0415

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if key not in payload:
            available = ", ".join(sorted(str(item) for item in payload))
            raise ValueError(f"{path} missing {key!r}; available: {available}")
        vector = np.asarray(payload[key], dtype=np.float32)
    elif path.suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        if key not in payload:
            available = ", ".join(payload.files)
            raise ValueError(f"{path} missing {key!r}; available: {available}")
        vector = np.asarray(payload[key], dtype=np.float32)
    elif path.suffix == ".npy":
        vector = np.load(path, allow_pickle=False).astype(np.float32)
    else:
        raise ValueError(f"unsupported steering artifact suffix: {path.suffix}")

    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("near-zero steering vector")
    return (vector / norm).astype(np.float32).tolist()


def trial_label(
    trial: CollaboratorBOTrial,
    index: int,
    *,
    replicate_index: int = 0,
    total_replicates: int = 1,
) -> str:
    suffix = f"_rep{replicate_index:02d}" if total_replicates > 1 else ""
    return safe_label(f"bo_replay_{index:02d}_{trial.task_id}{suffix}")


def expand_replay_jobs(
    trials: list[CollaboratorBOTrial],
    *,
    replicates: int,
    seed_stride: int,
    seed_offset: int,
) -> list[ReplayJob]:
    """Expand selected trials into deterministic replicate jobs."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if seed_stride <= 0:
        raise ValueError("replicate seed stride must be positive")

    jobs: list[ReplayJob] = []
    for trial_index, trial in enumerate(trials):
        for replicate_index in range(replicates):
            noise_seed = trial.noise_seed
            if replicate_index > 0:
                noise_seed = (
                    trial.noise_seed
                    + seed_offset
                    + (replicate_index * seed_stride)
                )
            jobs.append(
                ReplayJob(
                    trial=trial,
                    trial_index=trial_index,
                    replicate_index=replicate_index,
                    noise_seed=noise_seed,
                )
            )
    return jobs


def populate_svd_cache_on_modal(*, app_name: str) -> None:
    """Run the deployed SVD cache population job before replay generation."""
    import modal  # type: ignore[import-not-found]  # noqa: PLC0415

    log(f"populating SVD cache through Modal app {app_name!r}")
    started = time.monotonic()
    function = modal.Function.from_name(app_name, "populate_svd_weights")
    function.remote()
    log(f"SVD cache population finished in {time.monotonic() - started:.1f}s")


def generate_videos_on_modal(
    *,
    jobs: list[ReplayJob],
    total_replicates: int,
    seed_pool: list[dict[str, Any]],
    steering_vector: list[float],
    app_name: str,
    output_dir: Path,
    num_frames: int,
    num_inference_steps: int,
    motion_bucket_id: int,
    noise_aug_strength: float,
    fps: int,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Spawn Modal SVD jobs and write returned MP4 bytes locally."""
    import modal  # type: ignore[import-not-found]  # noqa: PLC0415

    generator_cls = modal.Cls.from_name(app_name, "SVDGenerator")
    generator = generator_cls()
    pending = []
    for job in jobs:
        trial = job.trial
        seed = seed_pool[trial.seed_idx % len(seed_pool)]
        label = trial_label(
            trial,
            job.trial_index,
            replicate_index=job.replicate_index,
            total_replicates=total_replicates,
        )
        started = time.monotonic()
        log(
            "spawning SVD job "
            f"{label} alpha={trial.alpha:.3f} guidance={trial.guidance:.3f} "
            f"seed={job.noise_seed} seed_image={seed['bmd_name']}"
        )
        call = generator.generate.spawn(
            image_bytes(seed["image_path"]),
            steering_vector=steering_vector,
            alpha=trial.alpha,
            guidance_scale=trial.guidance,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            fps=fps,
            seed=job.noise_seed,
            output_label=label,
            persist_output=False,
        )
        pending.append((job, seed, label, started, call))

    rows: list[dict[str, Any]] = []
    for job, seed, label, started, call in pending:
        row: dict[str, Any] = {
            "trial": job.trial.to_json(),
            "trial_index": job.trial_index,
            "replicate_index": job.replicate_index,
            "noise_seed": job.noise_seed,
            "source_noise_seed": job.trial.noise_seed,
            "seed": {
                "idx": seed["idx"],
                "bmd_name": seed["bmd_name"],
                "image_path": str(seed["image_path"]),
            },
            "label": label,
            "local_video_path": None,
            "generation_seconds": None,
            "generation_error": None,
        }
        try:
            log(f"waiting for SVD job {label}")
            video = call.get(timeout=timeout_seconds)
            out_path = output_dir / f"{label}.mp4"
            out_path.write_bytes(video)
            row["local_video_path"] = str(out_path)
            row["video_bytes"] = len(video)
            row["generation_seconds"] = time.monotonic() - started
            log(
                f"wrote {out_path} "
                f"({len(video)} bytes, {row['generation_seconds']:.1f}s)"
            )
        except Exception as exc:  # noqa: BLE001
            row["generation_error"] = repr(exc)
            log(f"SVD job {label} failed: {exc!r}")
        rows.append(row)
    return rows


def upload_generated_videos(rows: list[dict[str, Any]], *, volume_name: str) -> None:
    """Upload generated videos to the TRIBE-visible Modal volume."""
    import modal  # type: ignore[import-not-found]  # noqa: PLC0415

    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        for row in rows:
            local_path_raw = row.get("local_video_path")
            if not local_path_raw:
                continue
            local_path = Path(str(local_path_raw))
            remote_path = f"/generated/bo_memorability_replay/{local_path.name}"
            batch.put_file(local_path, remote_path)
            row["modal_video_path"] = f"/bmd-videos{remote_path}"
            log(f"uploaded {local_path.name} to {row['modal_video_path']}")


def result_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    return dict(result)


def row_label(row: dict[str, Any]) -> str:
    return str(
        row.get("label")
        or row.get("modal_video_path")
        or row.get("local_video_path")
        or "unknown"
    )


async def tribe_preflight_row(
    service: Any,
    row: dict[str, Any],
    *,
    input_mode: str,
) -> Any:
    local_path_raw = row.get("local_video_path")
    modal_path = row.get("modal_video_path")
    if input_mode == "bytes" and local_path_raw:
        return await service.preflight_video_bytes(Path(str(local_path_raw)).read_bytes())
    if modal_path:
        return await service.preflight_video(str(modal_path))
    return None


async def tribe_predict_row(
    service: Any,
    row: dict[str, Any],
    *,
    input_mode: str,
) -> Any:
    local_path_raw = row.get("local_video_path")
    modal_path = row.get("modal_video_path")
    if input_mode == "bytes" and local_path_raw:
        return await service.predict_video_bytes(Path(str(local_path_raw)).read_bytes())
    if modal_path:
        return await service.predict_video(str(modal_path))
    return None


async def run_tribe_preflight(
    service: Any,
    row: dict[str, Any],
    *,
    input_mode: str,
    timeout_seconds: float,
) -> None:
    started = time.monotonic()
    label = row_label(row)
    log(f"preflighting {label} with TRIBE input={input_mode}")
    result = await asyncio.wait_for(
        tribe_preflight_row(service, row, input_mode=input_mode),
        timeout=timeout_seconds,
    )
    if result is None:
        row["tribe_preflight_error"] = "empty TRIBE preflight result"
        return
    row["tribe_preflight"] = result_payload(result)
    row["tribe_wall_seconds"] = time.monotonic() - started
    log(f"TRIBE preflight passed for {label}")


async def attach_timeout_preflight(
    service: Any,
    row: dict[str, Any],
    *,
    input_mode: str,
    timeout_seconds: float,
) -> None:
    label = row_label(row)
    try:
        diagnostic = await asyncio.wait_for(
            tribe_preflight_row(service, row, input_mode=input_mode),
            timeout=min(timeout_seconds, 120.0),
        )
        if diagnostic is not None:
            row["tribe_timeout_preflight"] = result_payload(diagnostic)
            log(f"TRIBE timeout preflight passed for {label}")
    except Exception as diag_exc:  # noqa: BLE001
        row["tribe_timeout_preflight_error"] = repr(diag_exc)
        log(f"TRIBE timeout preflight failed for {label}: {diag_exc!r}")


async def run_tribe_full_score(
    service: Any,
    row: dict[str, Any],
    *,
    cortical_vmem: np.ndarray,
    input_mode: str,
    timeout_seconds: float,
    diagnose_on_timeout: bool,
) -> None:
    started = time.monotonic()
    label = row_label(row)
    try:
        log(f"scoring {label} with TRIBE input={input_mode}")
        result = await asyncio.wait_for(
            tribe_predict_row(service, row, input_mode=input_mode),
            timeout=timeout_seconds,
        )
        if result is None:
            row["tribe_error"] = "empty TRIBE result"
            return
        frames = np.asarray(result.frames, dtype=np.float32)
        row["replay_tribe_score"] = score_projection(frames, cortical_vmem)
        row["tribe_frames_shape"] = list(frames.shape)
        row["tribe_duration_seconds"] = float(result.duration_seconds)
        row["tribe_wall_seconds"] = time.monotonic() - started
        log(f"TRIBE score for {label}: {row['replay_tribe_score']:.4f}")
    except TimeoutError as exc:
        row["tribe_error"] = repr(exc)
        row["tribe_wall_seconds"] = time.monotonic() - started
        log(f"TRIBE full timed out for {label}: {exc!r}")
        if diagnose_on_timeout:
            await attach_timeout_preflight(
                service,
                row,
                input_mode=input_mode,
                timeout_seconds=timeout_seconds,
            )
    except Exception as exc:  # noqa: BLE001
        row["tribe_error"] = repr(exc)
        row["tribe_wall_seconds"] = time.monotonic() - started
        log(f"TRIBE full failed for {label}: {exc!r}")


async def score_rows_with_tribe(
    rows: list[dict[str, Any]],
    *,
    app_name: str,
    cortical_vmem: np.ndarray,
    concurrency: int,
    timeout_seconds: float,
    mode: str,
    input_mode: str,
    diagnose_on_timeout: bool,
) -> None:
    """Run TRIBE on uploaded rows and attach replay projection scores."""
    from audience_vectors.services.tribe_service import TribeService  # noqa: PLC0415

    service = TribeService(app_name)
    sem = asyncio.Semaphore(concurrency)

    async def one(row: dict[str, Any]) -> None:
        if not row.get("modal_video_path") and not row.get("local_video_path"):
            return
        async with sem:
            if mode == "skip":
                row["tribe_status"] = "skipped"
            elif mode == "preflight":
                await run_tribe_preflight(
                    service,
                    row,
                    input_mode=input_mode,
                    timeout_seconds=timeout_seconds,
                )
            else:
                await run_tribe_full_score(
                    service,
                    row,
                    cortical_vmem=cortical_vmem,
                    input_mode=input_mode,
                    timeout_seconds=timeout_seconds,
                    diagnose_on_timeout=diagnose_on_timeout,
                )

    await asyncio.gather(*(one(row) for row in rows))


def attach_original_scores(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        trial = row["trial"]
        row["trial_policy_group"] = trial_policy_group(str(trial.get("task_id")))
        row["original_tribe_score"] = trial.get("tribe_score")
        row["original_clip_score"] = trial.get("clip_score")
        row["original_quality_score"] = trial.get("quality_score")


def attach_visual_artifact_gate(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    thresholds: ArtifactThresholds,
) -> dict[str, Any]:
    """Attach visual artifact-gate metrics to generated-video rows."""
    evaluated = 0
    failures: list[dict[str, Any]] = []
    for row in rows:
        local_path_raw = row.get("local_video_path")
        if not local_path_raw:
            continue
        evaluated += 1
        try:
            gate = summarize_video(
                Path(str(local_path_raw)),
                samples=samples,
                thresholds=thresholds,
            )
        except Exception as exc:  # noqa: BLE001
            gate = {
                "video_path": str(local_path_raw),
                "sample_count": samples,
                "artifact_flags": ["visual_gate_error"],
                "passes_visual_gate": False,
                "error": repr(exc),
            }
        row["visual_artifact_gate"] = gate
        if not gate["passes_visual_gate"]:
            failures.append(
                {
                    "label": row_label(row),
                    "video_path": gate.get("video_path"),
                    "artifact_flags": gate.get("artifact_flags", []),
                    "error": gate.get("error"),
                }
            )

    return {
        "schema_version": 1,
        "samples": samples,
        "thresholds": thresholds.__dict__,
        "n_videos": evaluated,
        "n_failed": len(failures),
        "passes_visual_gate": len(failures) == 0,
        "failures": failures,
    }


def row_task_id(row: dict[str, Any]) -> str:
    trial = row.get("trial")
    if isinstance(trial, dict) and trial.get("task_id") is not None:
        return str(trial["task_id"])
    return row_label(row)


def apply_visual_first_retention(
    rows: list[dict[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    """Mark rows retained for scoring after visual gating."""
    if mode == "none":
        for row in rows:
            row["visual_first_status"] = "not_applied"
        return {
            "schema_version": 1,
            "mode": mode,
            "n_rows": len(rows),
            "n_retained_rows": len(rows),
            "n_withheld_rows": 0,
            "n_candidates": len({row_task_id(row) for row in rows}),
            "n_retained_candidates": len({row_task_id(row) for row in rows}),
            "n_withheld_candidates": 0,
            "retained_task_ids": sorted({row_task_id(row) for row in rows}),
            "withheld_task_ids": [],
        }

    if mode not in {"passing-videos", "complete-candidates"}:
        raise ValueError(f"unsupported visual-first retention mode: {mode}")

    row_passes = {
        row_label(row): bool(
            (row.get("visual_artifact_gate") or {}).get("passes_visual_gate")
        )
        for row in rows
    }
    retained_labels: set[str]
    if mode == "passing-videos":
        retained_labels = {label for label, passes in row_passes.items() if passes}
    else:
        labels_by_task: dict[str, list[str]] = {}
        for row in rows:
            labels_by_task.setdefault(row_task_id(row), []).append(row_label(row))
        retained_tasks = {
            task_id
            for task_id, labels in labels_by_task.items()
            if labels and all(row_passes[label] for label in labels)
        }
        retained_labels = {
            row_label(row) for row in rows if row_task_id(row) in retained_tasks
        }

    retained_task_ids: set[str] = set()
    withheld_task_ids: set[str] = set()
    retained_rows = 0
    for row in rows:
        retained = row_label(row) in retained_labels
        row["visual_first_retained"] = retained
        if retained:
            row["visual_first_status"] = "retained"
            retained_rows += 1
            retained_task_ids.add(row_task_id(row))
            continue

        withheld_task_ids.add(row_task_id(row))
        gate = row.get("visual_artifact_gate") or {}
        if gate.get("passes_visual_gate") is False:
            row["visual_first_status"] = "withheld_visual_failure"
        else:
            row["visual_first_status"] = "withheld_candidate_has_visual_failure"

    return {
        "schema_version": 1,
        "mode": mode,
        "n_rows": len(rows),
        "n_retained_rows": retained_rows,
        "n_withheld_rows": len(rows) - retained_rows,
        "n_candidates": len({row_task_id(row) for row in rows}),
        "n_retained_candidates": len(retained_task_ids),
        "n_withheld_candidates": len(withheld_task_ids),
        "retained_task_ids": sorted(retained_task_ids),
        "withheld_task_ids": sorted(withheld_task_ids),
        "withheld_failures": [
            {
                "label": row_label(row),
                "task_id": row_task_id(row),
                "status": row.get("visual_first_status"),
                "artifact_flags": (row.get("visual_artifact_gate") or {}).get(
                    "artifact_flags",
                    [],
                ),
            }
            for row in rows
            if not row.get("visual_first_retained")
        ],
    }


def validate_regenerated_sobol_inputs(args: argparse.Namespace) -> None:
    """Fail early on invalid regenerated-control settings."""
    controls_per_stratum = int(
        getattr(args, "regenerated_sobol_controls_per_stratum", 0)
    )
    if controls_per_stratum < 0:
        raise ValueError("--regenerated-sobol-controls-per-stratum must be >= 0")
    if controls_per_stratum > 0:
        if int(getattr(args, "regenerated_sobol_pool_size", 0)) <= 0:
            raise ValueError("--regenerated-sobol-pool-size must be positive")
        if int(getattr(args, "regenerated_sobol_start_index", 0)) < 0:
            raise ValueError("--regenerated-sobol-start-index must be >= 0")


def validate_replay_seed_pool_inputs(args: argparse.Namespace) -> None:
    """Fail early on invalid replay seed-pool settings."""
    replay_seed_pool_size = int(
        getattr(args, "replay_seed_pool_size", DEFAULT_REPLAY_SEED_POOL_SIZE)
    )
    if replay_seed_pool_size <= 0:
        raise ValueError("--replay-seed-pool-size must be positive")


def validate_run_inputs(args: argparse.Namespace, *, require_artifacts: bool) -> None:
    """Fail early if a non-dry replay is missing local run artifacts."""
    skip_visual_gate = bool(getattr(args, "skip_visual_gate", False))
    fail_on_visual_artifacts = bool(getattr(args, "fail_on_visual_artifacts", False))
    visual_first_retention = str(getattr(args, "visual_first_retention", "none"))
    if skip_visual_gate and visual_first_retention != "none":
        raise ValueError("--visual-first-retention requires the visual gate")
    if fail_on_visual_artifacts and visual_first_retention != "none":
        raise ValueError(
            "--fail-on-visual-artifacts cannot be combined with "
            "--visual-first-retention"
        )
    validate_regenerated_sobol_inputs(args)
    validate_replay_seed_pool_inputs(args)
    if args.trial_table is None or not args.trial_table.exists():
        raise FileNotFoundError(f"trial table not found: {args.trial_table}")
    if args.seed_root is None or not args.seed_root.exists():
        raise FileNotFoundError(f"seed root not found: {args.seed_root}")
    if not require_artifacts:
        return

    missing: list[str] = []
    for label, path in [
        ("--steering-artifact / BO_MEM_STEERING_ARTIFACT", args.steering_artifact),
        ("--cortical-vmem / BO_MEM_CORTICAL_VMEM", args.cortical_vmem),
    ]:
        if path is None:
            missing.append(f"{label} is not set")
        elif not path.exists():
            missing.append(f"{label} path does not exist: {path}")
    if missing:
        raise FileNotFoundError("; ".join(missing))


async def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_run_inputs(
        args,
        require_artifacts=args.require_artifacts or not args.dry_run,
    )
    trials = load_collaborator_trials(args.trial_table)
    seed_pool = load_seed_pool(
        args.seed_root,
        n_pool=args.replay_seed_pool_size,
    )
    selected = select_trials(
        trials,
        selection=args.selection,
        max_evals=args.max_evals,
        task_ids=set(args.task_id) if args.task_id else None,
        stratify_by=args.stratify_by,
    )
    selected, regenerated_sobol_controls = append_regenerated_sobol_controls(
        selected,
        all_trials=trials,
        seed_pool=seed_pool,
        stratify_by=args.stratify_by,
        controls_per_stratum=args.regenerated_sobol_controls_per_stratum,
        pool_size=args.regenerated_sobol_pool_size,
        start_index=args.regenerated_sobol_start_index,
        scramble_seed=args.regenerated_sobol_scramble_seed,
    )
    jobs = expand_replay_jobs(
        selected,
        replicates=args.replicates,
        seed_stride=args.replicate_seed_stride,
        seed_offset=args.replicate_seed_offset,
    )
    log(
        f"loaded {len(trials)} trials; selected {len(selected)} "
        f"with selection={args.selection!r}; expanded to {len(jobs)} replay jobs"
    )
    if regenerated_sobol_controls is not None:
        log(
            "appended "
            f"{regenerated_sobol_controls['n_generated_controls']} regenerated "
            "Sobol controls"
        )

    rows: list[dict[str, Any]]
    visual_artifact_gate: dict[str, Any] | None = None
    visual_first_retention: dict[str, Any] | None = None
    visual_gate_blocked_scoring = False
    if args.dry_run:
        log("dry run only; not generating or scoring videos")
        rows = [
            {
                "trial": job.trial.to_json(),
                "trial_index": job.trial_index,
                "replicate_index": job.replicate_index,
                "noise_seed": job.noise_seed,
                "source_noise_seed": job.trial.noise_seed,
                "seed": {
                    "idx": seed_pool[job.trial.seed_idx % len(seed_pool)]["idx"],
                    "bmd_name": seed_pool[job.trial.seed_idx % len(seed_pool)][
                        "bmd_name"
                    ],
                    "image_path": str(
                        seed_pool[job.trial.seed_idx % len(seed_pool)]["image_path"]
                    ),
                },
                "label": trial_label(
                    job.trial,
                    job.trial_index,
                    replicate_index=job.replicate_index,
                    total_replicates=args.replicates,
                ),
            }
            for job in jobs
        ]
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.populate_svd_cache:
            populate_svd_cache_on_modal(app_name=args.app_name)
        log(f"loading steering vector from {args.steering_artifact}")
        steering_vector = load_steering_vector(
            args.steering_artifact,
            key=args.steering_key,
        )
        rows = generate_videos_on_modal(
            jobs=jobs,
            total_replicates=args.replicates,
            seed_pool=seed_pool,
            steering_vector=steering_vector,
            app_name=args.app_name,
            output_dir=args.output_dir,
            num_frames=args.svd_num_frames,
            num_inference_steps=args.num_inference_steps,
            motion_bucket_id=args.svd_motion_bucket_id,
            noise_aug_strength=args.svd_noise_aug_strength,
            fps=args.svd_fps,
            timeout_seconds=args.generation_timeout,
        )
        if args.skip_visual_gate:
            log("visual artifact gate skipped")
        else:
            thresholds = ArtifactThresholds(
                min_tail_sharpness_ratio=args.visual_min_tail_sharpness_ratio,
                min_tail_contrast_ratio=args.visual_min_tail_contrast_ratio,
                min_tail_contrast=args.visual_min_tail_contrast,
            )
            visual_artifact_gate = attach_visual_artifact_gate(
                rows,
                samples=args.visual_gate_samples,
                thresholds=thresholds,
            )
            log(
                "visual artifact gate "
                f"{visual_artifact_gate['n_failed']}/"
                f"{visual_artifact_gate['n_videos']} failed"
            )
            visual_gate_blocked_scoring = (
                args.fail_on_visual_artifacts
                and not visual_artifact_gate["passes_visual_gate"]
            )
            if args.visual_first_retention != "none":
                visual_first_retention = apply_visual_first_retention(
                    rows,
                    mode=args.visual_first_retention,
                )
                log(
                    "visual-first retention kept "
                    f"{visual_first_retention['n_retained_rows']}/"
                    f"{visual_first_retention['n_rows']} rows and "
                    f"{visual_first_retention['n_retained_candidates']}/"
                    f"{visual_first_retention['n_candidates']} candidates"
                )

        if visual_gate_blocked_scoring:
            log("visual artifact gate failed; skipping upload and TRIBE scoring")
        else:
            scoring_rows = [
                row
                for row in rows
                if args.visual_first_retention == "none"
                or row.get("visual_first_retained")
            ]
            upload_generated_videos(scoring_rows, volume_name=args.volume)
            log(f"loading cortical v_mem from {args.cortical_vmem}")
            cortical_vmem = load_unit_npz_vector(args.cortical_vmem, key=args.vmem_key)
            await score_rows_with_tribe(
                scoring_rows,
                app_name=args.app_name,
                cortical_vmem=cortical_vmem,
                concurrency=args.tribe_concurrency,
                timeout_seconds=args.tribe_timeout,
                mode=args.tribe_mode,
                input_mode=args.tribe_input,
                diagnose_on_timeout=not args.no_diagnose_on_timeout,
            )

    attach_original_scores(rows)
    payload = {
        "schema_version": 1,
        "source_trial_table": str(args.trial_table),
        "seed_root": str(args.seed_root),
        "replay_seed_pool_size": args.replay_seed_pool_size,
        "selection": args.selection,
        "stratify_by": args.stratify_by,
        "max_evals": args.max_evals,
        "regenerated_sobol_controls_per_stratum": (
            args.regenerated_sobol_controls_per_stratum
        ),
        "regenerated_sobol_pool_size": args.regenerated_sobol_pool_size,
        "regenerated_sobol_start_index": args.regenerated_sobol_start_index,
        "regenerated_sobol_scramble_seed": args.regenerated_sobol_scramble_seed,
        "replicates": args.replicates,
        "replicate_seed_stride": args.replicate_seed_stride,
        "replicate_seed_offset": args.replicate_seed_offset,
        "app_name": args.app_name,
        "svd_num_frames": args.svd_num_frames,
        "num_inference_steps": args.num_inference_steps,
        "svd_motion_bucket_id": args.svd_motion_bucket_id,
        "svd_noise_aug_strength": args.svd_noise_aug_strength,
        "svd_fps": args.svd_fps,
        "tribe_mode": args.tribe_mode,
        "tribe_input": args.tribe_input,
        "skip_visual_gate": bool(args.skip_visual_gate),
        "fail_on_visual_artifacts": bool(args.fail_on_visual_artifacts),
        "visual_first_retention_mode": args.visual_first_retention,
        "visual_first_retention": visual_first_retention,
        "regenerated_sobol_controls": regenerated_sobol_controls,
        "visual_artifact_gate": visual_artifact_gate,
        "visual_gate_blocked_scoring": visual_gate_blocked_scoring,
        "dry_run": bool(args.dry_run),
        "preflight_only": bool(args.dry_run),
        "summary": replay_summary(rows),
        "replicate_summary": replicate_summary(rows),
        "policy_group_summary": policy_group_summary(rows),
        "stratum_policy_summary": stratum_policy_summary(
            rows,
            stratify_by=args.stratify_by,
        ),
        "rows": rows,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(payload, indent=2))
    log(f"wrote report to {args.report_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-table", type=Path, default=DEFAULT_TRIAL_TABLE)
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    parser.add_argument(
        "--replay-seed-pool-size",
        type=int,
        default=DEFAULT_REPLAY_SEED_POOL_SIZE,
        help=(
            "Number of image-backed seed slots in the replay pool. Must match "
            "trial tables built with --replay-seed-pool-size when they target "
            "seed_idx values beyond the default 16-slot collaborator pool."
        ),
    )
    parser.add_argument(
        "--selection",
        choices=[
            "first",
            "top-tribe",
            "top-quality",
            "top-clip",
            "top-bo-tribe",
            "top-sobol-tribe",
            "top-bo-vs-top-sobol",
            "seed-stratified-bo-vs-sobol",
            "top-bo-per-stratum",
        ],
        default="top-tribe",
        help=(
            "Trial selector. top-bo-vs-top-sobol uses --max-evals per group. "
            "seed-stratified-bo-vs-sobol uses --max-evals per policy inside "
            "each matched stratum. top-bo-per-stratum selects top saved BO "
            "candidates in each BO-covered stratum."
        ),
    )
    parser.add_argument(
        "--stratify-by",
        choices=["prompt", "seed_idx"],
        default="prompt",
        help=(
            "Stratum key for seed-stratified selectors and reports. Prompt is "
            "the default because it tracks repeated seed-image content better "
            "than the raw optimizer seed_idx slot."
        ),
    )
    parser.add_argument("--max-evals", type=int, default=2)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument(
        "--regenerated-sobol-controls-per-stratum",
        type=int,
        default=0,
        help=(
            "Append this many deterministic, unscored Sobol controls for each "
            "selected BO stratum. Use with top-bo-per-stratum for regenerated "
            "matched controls."
        ),
    )
    parser.add_argument(
        "--regenerated-sobol-pool-size",
        type=int,
        default=128,
        help=(
            "Number of Sobol sequence indices to scan when finding regenerated "
            "controls for selected BO strata."
        ),
    )
    parser.add_argument(
        "--regenerated-sobol-start-index",
        type=int,
        default=0,
        help="First Sobol sequence index scanned for regenerated controls.",
    )
    parser.add_argument(
        "--regenerated-sobol-scramble-seed",
        type=int,
        default=42,
        help="Scramble seed used by the collaborator-compatible Sobol sequence.",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Number of stochastic noise-seed replays per selected BO trial.",
    )
    parser.add_argument(
        "--replicate-seed-stride",
        type=int,
        default=10_000,
        help="Seed increment used for replicate 1, 2, etc.",
    )
    parser.add_argument(
        "--replicate-seed-offset",
        type=int,
        default=0,
        help="Optional extra seed offset for nonzero replicate indices.",
    )
    parser.add_argument(
        "--app-name",
        default=os.environ.get("MODAL_APP_NAME", "audience-vectors-dev"),
    )
    parser.add_argument("--volume", default="bmd-videos-v1")
    parser.add_argument(
        "--steering-artifact",
        type=Path,
        default=Path(os.environ["BO_MEM_STEERING_ARTIFACT"])
        if os.environ.get("BO_MEM_STEERING_ARTIFACT")
        else None,
    )
    parser.add_argument("--steering-key", default="v_mem_clip_h_via_adapter")
    parser.add_argument(
        "--cortical-vmem",
        type=Path,
        default=Path(os.environ["BO_MEM_CORTICAL_VMEM"])
        if os.environ.get("BO_MEM_CORTICAL_VMEM")
        else None,
    )
    parser.add_argument("--vmem-key", default="direction")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated/bo_modal_replay"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/reports/bo_modal_replay.json"),
    )
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument(
        "--svd-num-frames",
        type=int,
        default=25,
        help="Number of SVD-XT frames to generate per clip.",
    )
    parser.add_argument(
        "--svd-motion-bucket-id",
        type=int,
        default=127,
        help="SVD-XT motion bucket. Lower values usually preserve the seed image more.",
    )
    parser.add_argument(
        "--svd-noise-aug-strength",
        type=float,
        default=0.02,
        help="SVD-XT seed-image noise augmentation strength.",
    )
    parser.add_argument(
        "--svd-fps",
        type=int,
        default=7,
        help="FPS used when encoding the generated SVD frames.",
    )
    parser.add_argument("--generation-timeout", type=int, default=20 * 60)
    parser.add_argument("--tribe-timeout", type=float, default=10 * 60)
    parser.add_argument("--tribe-concurrency", type=int, default=2)
    parser.add_argument(
        "--tribe-mode",
        choices=["full", "preflight", "skip"],
        default="full",
        help="Run full TRIBE scoring, lightweight preflight, or skip TRIBE.",
    )
    parser.add_argument(
        "--tribe-input",
        choices=["bytes", "volume"],
        default="bytes",
        help="Send local MP4 bytes directly to TRIBE or score uploaded volume paths.",
    )
    parser.add_argument("--no-diagnose-on-timeout", action="store_true")
    parser.add_argument("--populate-svd-cache", action="store_true")
    parser.add_argument(
        "--skip-visual-gate",
        action="store_true",
        help="Do not attach visual artifact-gate metrics to generated videos.",
    )
    parser.add_argument(
        "--fail-on-visual-artifacts",
        action="store_true",
        help=(
            "After generation, write the replay report and exit nonzero if any "
            "generated video fails the visual artifact gate. Upload/TRIBE scoring "
            "is skipped when the gate fails."
        ),
    )
    parser.add_argument(
        "--visual-first-retention",
        choices=["none", "passing-videos", "complete-candidates"],
        default="none",
        help=(
            "After visual gating, choose which generated videos are retained for "
            "upload/TRIBE scoring. `passing-videos` scores only passing videos. "
            "`complete-candidates` scores only candidates whose full replicate "
            "set passed the visual gate."
        ),
    )
    parser.add_argument(
        "--visual-gate-samples",
        type=int,
        default=3,
        help="Number of evenly spaced frames sampled by the visual artifact gate.",
    )
    parser.add_argument(
        "--visual-min-tail-sharpness-ratio",
        type=float,
        default=0.35,
        help="Minimum mid/end sharpness ratio relative to the first sampled frame.",
    )
    parser.add_argument(
        "--visual-min-tail-contrast-ratio",
        type=float,
        default=0.55,
        help="Minimum mid/end contrast ratio relative to the first sampled frame.",
    )
    parser.add_argument(
        "--visual-min-tail-contrast",
        type=float,
        default=0.04,
        help="Minimum absolute contrast for the weaker mid/end sampled frame.",
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help=(
            "Validate --steering-artifact and --cortical-vmem even in dry-run "
            "mode. Useful before queueing an expensive Modal replay."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run(args))
    print(json.dumps(payload["summary"], indent=2))
    if payload.get("visual_gate_blocked_scoring"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
