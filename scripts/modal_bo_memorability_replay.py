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

from audience_vectors.bo_replay import (
    CollaboratorBOTrial,
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
)

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


@dataclass(frozen=True)
class ReplayJob:
    """One Modal replay attempt for a collaborator BO trial."""

    trial: CollaboratorBOTrial
    trial_index: int
    replicate_index: int
    noise_seed: int


def log(message: str) -> None:
    print(f"[bo-replay] {message}", flush=True)


def load_seed_pool(seed_root: Path, *, n_pool: int = 16) -> list[dict[str, Any]]:
    """Load the collaborator seed pool, cycling available images to n_pool."""
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
    num_inference_steps: int,
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
            num_inference_steps=num_inference_steps,
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


def validate_run_inputs(args: argparse.Namespace, *, require_artifacts: bool) -> None:
    """Fail early if a non-dry replay is missing local run artifacts."""
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
    selected = select_trials(
        trials,
        selection=args.selection,
        max_evals=args.max_evals,
        task_ids=set(args.task_id) if args.task_id else None,
        stratify_by=args.stratify_by,
    )
    jobs = expand_replay_jobs(
        selected,
        replicates=args.replicates,
        seed_stride=args.replicate_seed_stride,
        seed_offset=args.replicate_seed_offset,
    )
    seed_pool = load_seed_pool(args.seed_root)
    log(
        f"loaded {len(trials)} trials; selected {len(selected)} "
        f"with selection={args.selection!r}; expanded to {len(jobs)} replay jobs"
    )

    rows: list[dict[str, Any]]
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
            num_inference_steps=args.num_inference_steps,
            timeout_seconds=args.generation_timeout,
        )
        upload_generated_videos(rows, volume_name=args.volume)
        log(f"loading cortical v_mem from {args.cortical_vmem}")
        cortical_vmem = load_unit_npz_vector(args.cortical_vmem, key=args.vmem_key)
        await score_rows_with_tribe(
            rows,
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
        "selection": args.selection,
        "stratify_by": args.stratify_by,
        "max_evals": args.max_evals,
        "replicates": args.replicates,
        "replicate_seed_stride": args.replicate_seed_stride,
        "replicate_seed_offset": args.replicate_seed_offset,
        "app_name": args.app_name,
        "num_inference_steps": args.num_inference_steps,
        "tribe_mode": args.tribe_mode,
        "tribe_input": args.tribe_input,
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
        ],
        default="top-tribe",
        help=(
            "Trial selector. top-bo-vs-top-sobol uses --max-evals per group. "
            "seed-stratified-bo-vs-sobol uses --max-evals per policy inside "
            "each matched stratum."
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
    payload = asyncio.run(run(parse_args()))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
