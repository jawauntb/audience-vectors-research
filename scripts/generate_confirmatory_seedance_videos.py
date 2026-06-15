"""Seedance candidate generation for the confirmatory recognition study.

This script is intentionally conservative. Dry-run mode validates the frozen
candidate manifest, resolves non-secret runtime settings, estimates OpenRouter
Seedance cost from the committed generation settings, and writes review
artifacts. Live mode is guarded by an explicit flag plus a maximum-cost budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import httpx

EXPERIMENT_DIR = Path(
    "research_program/neurips_memorability_selector/experiments/"
    "content_pocket_confirmatory_recognition_20260615"
)
DEFAULT_CONFIG = EXPERIMENT_DIR / "confirmatory_study_config_20260615.json"
DEFAULT_MANIFEST = EXPERIMENT_DIR / "seedance_candidate_generation_manifest_20260615.json"
DEFAULT_OUT_JSON = EXPERIMENT_DIR / "seedance_candidate_generation_dry_run_20260615.json"
DEFAULT_OUT_MD = EXPERIMENT_DIR / "seedance_candidate_generation_dry_run_20260615.md"
DEFAULT_LIVE_OUT_JSON = Path(
    "data/generated/content_pocket_confirmatory_recognition_20260615/"
    "candidate_old_videos/openrouter_seedance_generation_result_20260615.json"
)
DEFAULT_LIVE_OUT_MD = Path(
    "data/generated/content_pocket_confirmatory_recognition_20260615/"
    "candidate_old_videos/openrouter_seedance_generation_result_20260615.md"
)
PLACEHOLDER_MODEL_ID = "resolve_from_provider_before_generation"
OPENROUTER_SEEDANCE_2_MODEL_ID = "bytedance/seedance-2.0"
OPENROUTER_SEEDANCE_DOLLARS_PER_TOKEN = Decimal("0.000007")
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}
OPENROUTER_ALLOWED_POLLING_HOST_SUFFIX = "openrouter.ai"
PHASE_TO_ROLE = {
    "candidate_old_videos": "candidate_old_video",
}
SEEDANCE_CREDENTIAL_ENV_NAMES = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "SEEDANCE_API_KEY",
    "SEEDANCE_ACCESS_KEY_ID",
    "SEEDANCE_SECRET_ACCESS_KEY",
    "ARK_API_KEY",
    "VOLCENGINE_ACCESS_KEY_ID",
    "VOLCENGINE_SECRET_ACCESS_KEY",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--phase",
        choices=sorted(PHASE_TO_ROLE),
        default="candidate_old_videos",
    )
    parser.add_argument("--family-id", action="append", default=[])
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--model-id",
        default=os.environ.get("SEEDANCE_MODEL_ID"),
        help="Provider model ID to record in the dry-run report.",
    )
    parser.add_argument(
        "--estimated-cost-per-video-usd",
        default=os.environ.get("SEEDANCE_ESTIMATED_COST_PER_VIDEO_USD"),
        help=(
            "Optional per-video cost override. If omitted, known OpenRouter "
            "Seedance 2.0 pricing is estimated from resolution, duration, and fps."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report without calling OpenRouter.",
    )
    parser.add_argument(
        "--execute-live",
        action="store_true",
        help="Actually submit selected jobs to OpenRouter and download MP4 outputs.",
    )
    parser.add_argument(
        "--max-cost-usd",
        help="Required for --execute-live. Refuse if estimated selected-job cost exceeds it.",
    )
    parser.add_argument(
        "--openrouter-base-url",
        default=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE_URL),
    )
    parser.add_argument("--poll-interval-seconds", type=int, default=10)
    parser.add_argument("--max-poll-attempts", type=int, default=90)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum simultaneous live provider jobs. Default is sequential mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate jobs even when their output MP4 already exists.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_output_paths(args: argparse.Namespace) -> None:
    if args.execute_live and args.out_json == DEFAULT_OUT_JSON:
        args.out_json = DEFAULT_LIVE_OUT_JSON
    if args.execute_live and args.out_md == DEFAULT_OUT_MD:
        args.out_md = DEFAULT_LIVE_OUT_MD


def parse_decimal(value: str | None) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal value: {value}") from exc
    if parsed < 0:
        raise ValueError("estimated cost per video must be non-negative")
    return parsed


def money(value: Decimal | None) -> str:
    if value is None:
        return "not_estimated"
    return f"${value.quantize(Decimal('0.01'))}"


def decimal_as_float(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(value)


def parse_resolution_pixels(resolution: str) -> tuple[int, int] | None:
    normalized = resolution.lower().strip()
    if "x" not in normalized:
        return None
    width_text, height_text = normalized.split("x", maxsplit=1)
    if not width_text.isdecimal() or not height_text.isdecimal():
        return None
    width = int(width_text)
    height = int(height_text)
    if width <= 0 or height <= 0:
        return None
    return width, height


def estimate_openrouter_seedance_cost_per_video(
    *,
    config: dict[str, Any],
    model_id: str,
) -> tuple[Decimal | None, str]:
    if model_id != OPENROUTER_SEEDANCE_2_MODEL_ID:
        return None, "not_available_for_model"

    generator = config["generator"]
    resolution = parse_resolution_pixels(str(generator["resolution"]))
    if resolution is None:
        return None, "resolution_not_pixel_dimensions"

    width, height = resolution
    duration_seconds = Decimal(str(generator["duration_seconds"]))
    fps = Decimal(str(generator.get("fps", 24)))
    tokens = (Decimal(width) * Decimal(height) * duration_seconds * fps) / Decimal(
        1024
    )
    return tokens * OPENROUTER_SEEDANCE_DOLLARS_PER_TOKEN, "openrouter_formula"


def resolve_cost_per_video(
    *,
    config: dict[str, Any],
    model_id: str,
    cost_override: str | None,
) -> tuple[Decimal | None, str]:
    override = parse_decimal(cost_override)
    if override is not None:
        return override, "user_or_environment_supplied"
    return estimate_openrouter_seedance_cost_per_video(
        config=config,
        model_id=model_id,
    )


def selected_jobs(
    manifest: dict[str, Any],
    *,
    phase: str,
    family_ids: set[str],
    job_ids: set[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    role = PHASE_TO_ROLE[phase]
    jobs = [job for job in manifest["jobs"] if job["role"] == role]
    if family_ids:
        jobs = [job for job in jobs if job["family_id"] in family_ids]
    if job_ids:
        jobs = [job for job in jobs if job["job_id"] in job_ids]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        jobs = jobs[:limit]
    return jobs


def selected_live_jobs(
    jobs: list[dict[str, Any]],
    *,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    skipped: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for job in jobs:
        output_path = Path(str(job["output_video"]["path"]))
        if output_path.exists() and not overwrite:
            skipped.append(
                {
                    "job_id": job["job_id"],
                    "family_id": job["family_id"],
                    "status": "skipped_existing_output",
                    "output_path": str(output_path),
                }
            )
        else:
            selected.append(job)
    return selected, skipped


def resolve_model_id(
    *,
    cli_or_env_model_id: str | None,
    manifest_jobs: list[dict[str, Any]],
) -> tuple[str, str, bool]:
    if cli_or_env_model_id:
        return cli_or_env_model_id, "cli_or_seedance_model_id_env", True
    if manifest_jobs:
        manifest_model = str(manifest_jobs[0]["generator"]["model_id"])
        resolved = manifest_model != PLACEHOLDER_MODEL_ID
        source = "manifest" if resolved else "manifest_placeholder"
        return manifest_model, source, resolved
    return PLACEHOLDER_MODEL_ID, "no_jobs_selected", False


def env_presence() -> dict[str, bool]:
    return {name: bool(os.environ.get(name)) for name in SEEDANCE_CREDENTIAL_ENV_NAMES}


def openrouter_api_key() -> str | None:
    value = os.environ.get("OPENROUTER_API_KEY")
    if value is None:
        return None
    value = value.strip()
    return value or None


def add_manifest_identity_checks(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if manifest.get("experiment_id") != config.get("experiment_id"):
        blockers.append("Manifest experiment_id does not match config experiment_id.")
    if manifest.get("status") != "manifest_only_no_api_calls":
        warnings.append("Manifest status is not manifest_only_no_api_calls.")
    if int(manifest.get("job_count", -1)) != len(manifest.get("jobs", [])):
        blockers.append("Manifest job_count does not match jobs length.")
    return blockers, warnings


def add_job_selection_checks(
    *,
    jobs: list[dict[str, Any]],
    blockers: list[str],
    warnings: list[str],
) -> tuple[list[str], list[str]]:
    job_ids = [str(job["job_id"]) for job in jobs]
    output_paths = [str(job["output_video"]["path"]) for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        blockers.append("Selected jobs contain duplicate job_id values.")
    if len(output_paths) != len(set(output_paths)):
        blockers.append("Selected jobs contain duplicate output paths.")
    if not jobs:
        blockers.append("No jobs selected for dry run.")

    not_retained = [job["job_id"] for job in jobs if not job.get("retain_if_failed")]
    if not_retained:
        blockers.append("Some selected jobs are not marked retain_if_failed.")

    statuses = Counter(str(job["output_video"]["status"]) for job in jobs)
    unexpected_statuses = sorted(status for status in statuses if status != "not_generated")
    if unexpected_statuses:
        warnings.append(
            "Some selected jobs are not in not_generated status: "
            + ", ".join(unexpected_statuses)
        )

    generator_signatures = {
        json.dumps(job["generator"], sort_keys=True) for job in jobs
    }
    if len(generator_signatures) > 1:
        warnings.append("Selected jobs do not share one generator configuration.")

    return blockers, warnings


def add_runtime_readiness_checks(
    *,
    model_resolved: bool,
    cost_per_video: Decimal | None,
    blockers: list[str],
    warnings: list[str],
) -> tuple[list[str], list[str]]:
    if not model_resolved:
        blockers.append("Seedance provider model ID is unresolved.")
    if cost_per_video is None:
        warnings.append("No per-video cost estimate supplied; total cost is not estimated.")
    if not any(env_presence().values()):
        warnings.append(
            "No recognized Seedance credential environment variables are present in this shell."
        )
    return blockers, warnings


def manifest_checks(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    jobs: list[dict[str, Any]],
    model_resolved: bool,
    cost_per_video: Decimal | None,
) -> tuple[list[str], list[str]]:
    blockers, warnings = add_manifest_identity_checks(
        config=config,
        manifest=manifest,
    )
    blockers, warnings = add_job_selection_checks(
        jobs=jobs,
        blockers=blockers,
        warnings=warnings,
    )
    blockers, warnings = add_runtime_readiness_checks(
        model_resolved=model_resolved,
        cost_per_video=cost_per_video,
        blockers=blockers,
        warnings=warnings,
    )
    return blockers, warnings


def openrouter_resolution_label(config: dict[str, Any]) -> str:
    resolution = parse_resolution_pixels(str(config["generator"]["resolution"]))
    if resolution is None:
        raise ValueError("OpenRouter live mode requires WIDTHxHEIGHT resolution")
    width, height = resolution
    short_side = min(width, height)
    supported = {
        480: "480p",
        720: "720p",
        1080: "1080p",
        1024: "1K",
        2048: "2K",
        4096: "4K",
    }
    try:
        return supported[short_side]
    except KeyError as exc:
        raise ValueError(f"Unsupported OpenRouter short-side resolution: {short_side}") from exc


def openrouter_aspect_ratio(config: dict[str, Any]) -> str:
    resolution = parse_resolution_pixels(str(config["generator"]["resolution"]))
    if resolution is None:
        raise ValueError("OpenRouter live mode requires WIDTHxHEIGHT resolution")
    width, height = resolution
    if width == 1280 and height == 720:
        return "16:9"
    if width == 720 and height == 1280:
        return "9:16"
    if width == height:
        return "1:1"
    ratio = Decimal(width) / Decimal(height)
    known = {
        "16:9": Decimal(16) / Decimal(9),
        "9:16": Decimal(9) / Decimal(16),
        "4:3": Decimal(4) / Decimal(3),
        "3:4": Decimal(3) / Decimal(4),
        "3:2": Decimal(3) / Decimal(2),
        "2:3": Decimal(2) / Decimal(3),
        "21:9": Decimal(21) / Decimal(9),
        "9:21": Decimal(9) / Decimal(21),
    }
    for label, value in known.items():
        if abs(ratio - value) < Decimal("0.01"):
            return label
    raise ValueError(f"Unsupported OpenRouter aspect ratio for {width}x{height}")


def is_openrouter_polling_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return False
    host = parsed.hostname
    if host is None:
        return False
    host = host.lower()
    return host == OPENROUTER_ALLOWED_POLLING_HOST_SUFFIX or host.endswith(
        "." + OPENROUTER_ALLOWED_POLLING_HOST_SUFFIX
    )


def unwrap_openrouter_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("OpenRouter response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("OpenRouter response was not a JSON object")
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return dict(payload)


def openrouter_error_message(response: httpx.Response, default: str) -> str:
    try:
        payload = unwrap_openrouter_payload(response)
    except RuntimeError:
        return default
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return default


def raise_for_openrouter_status(response: httpx.Response, default: str) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(openrouter_error_message(response, default)) from exc


def parse_openrouter_job(payload: dict[str, Any]) -> dict[str, Any]:
    job_id = payload.get("id")
    status = payload.get("status")
    if not isinstance(job_id, str) or not job_id.strip():
        raise RuntimeError("OpenRouter job payload missing id")
    if not isinstance(status, str) or not status.strip():
        raise RuntimeError("OpenRouter job payload missing status")
    unsigned_urls = payload.get("unsigned_urls")
    if not isinstance(unsigned_urls, list):
        unsigned_urls = []
    return {
        "id": job_id.strip(),
        "status": status.strip(),
        "polling_url": payload.get("polling_url")
        if isinstance(payload.get("polling_url"), str)
        else None,
        "unsigned_urls": [
            item.strip()
            for item in unsigned_urls
            if isinstance(item, str) and item.strip()
        ],
        "generation_id": payload.get("generation_id")
        if isinstance(payload.get("generation_id"), str)
        else None,
        "usage": payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
        "error": payload.get("error") if isinstance(payload.get("error"), str) else None,
    }


def openrouter_payload(job: dict[str, Any], config: dict[str, Any], model_id: str) -> dict[str, Any]:
    generator = config["generator"]
    return {
        "model": model_id,
        "prompt": job["prompt"],
        "duration": int(generator["duration_seconds"]),
        "aspect_ratio": openrouter_aspect_ratio(config),
        "resolution": openrouter_resolution_label(config),
        "generate_audio": not bool(generator.get("silence_audio", True)),
    }


def submit_openrouter_job(
    *,
    client: httpx.Client,
    base_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = client.post(
        f"{base_url.rstrip('/')}/videos",
        headers=headers,
        json=payload,
    )
    raise_for_openrouter_status(response, "OpenRouter video job creation failed")
    return parse_openrouter_job(unwrap_openrouter_payload(response))


def poll_openrouter_job(
    *,
    client: httpx.Client,
    headers: dict[str, str],
    initial_job: dict[str, Any],
    poll_interval_seconds: int,
    max_poll_attempts: int,
) -> dict[str, Any]:
    polling_url = initial_job.get("polling_url")
    if not isinstance(polling_url, str) or not polling_url.strip():
        return initial_job
    if not is_openrouter_polling_url(polling_url):
        raise RuntimeError("OpenRouter polling URL failed host validation")

    current = initial_job
    for attempt in range(max_poll_attempts):
        response = client.get(polling_url, headers=headers)
        raise_for_openrouter_status(response, "OpenRouter video job fetch failed")
        current = parse_openrouter_job(unwrap_openrouter_payload(response))
        if current["status"] in OPENROUTER_TERMINAL_STATUSES:
            return current
        if attempt < max_poll_attempts - 1:
            time.sleep(poll_interval_seconds)
    return current


def download_generated_video(
    *,
    client: httpx.Client,
    url: str,
    output_path: Path,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    with client.stream("GET", url, headers=headers, follow_redirects=True) as response:
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        bytes_written = 0
        with output_path.open("wb") as handle:
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                bytes_written += len(chunk)
    return {
        "path": str(output_path),
        "bytes": bytes_written,
        "sha256": digest.hexdigest(),
    }


def run_live_generation(  # noqa: C901
    *,
    config: dict[str, Any],
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    model_id: str,
) -> list[dict[str, Any]]:
    api_key = openrouter_api_key()
    if api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is required for --execute-live")
    if args.poll_interval_seconds <= 0:
        raise ValueError("--poll-interval-seconds must be positive")
    if args.max_poll_attempts <= 0:
        raise ValueError("--max-poll-attempts must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")

    headers = {"Authorization": f"Bearer {api_key}"}
    print_lock = Lock()

    def print_event(payload: dict[str, Any]) -> None:
        with print_lock:
            print(
                json.dumps(payload, sort_keys=True),
                flush=True,
            )

    def run_one_job(index: int, job: dict[str, Any]) -> dict[str, Any]:
        print_event(
            {
                "event": "submit_job",
                "index": index,
                "total": len(jobs),
                "job_id": job["job_id"],
                "family_id": job["family_id"],
            }
        )
        row: dict[str, Any] = {
            "job_id": job["job_id"],
            "family_id": job["family_id"],
            "status": "submitted",
            "output_path": job["output_video"]["path"],
            "prompt": job["prompt"],
        }
        try:
            with httpx.Client(timeout=args.request_timeout_seconds) as client:
                submitted = submit_openrouter_job(
                    client=client,
                    base_url=str(args.openrouter_base_url),
                    headers=headers,
                    payload=openrouter_payload(job, config, model_id),
                )
                row["provider_job"] = submitted
                completed = poll_openrouter_job(
                    client=client,
                    headers=headers,
                    initial_job=submitted,
                    poll_interval_seconds=args.poll_interval_seconds,
                    max_poll_attempts=args.max_poll_attempts,
                )
                row["provider_job"] = completed
                row["provider_status"] = completed["status"]
                if completed["status"] != "completed":
                    row["status"] = "provider_not_completed"
                    row["error"] = completed.get("error")
                    return row
                if not completed["unsigned_urls"]:
                    row["status"] = "provider_completed_without_video_url"
                    return row
                output = download_generated_video(
                    client=client,
                    url=completed["unsigned_urls"][0],
                    output_path=Path(str(job["output_video"]["path"])),
                    headers=headers,
                )
                row["status"] = "downloaded"
                row["output_video"] = output
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = str(exc)
        finally:
            print_event(
                {
                    "event": "finish_job",
                    "index": index,
                    "total": len(jobs),
                    "job_id": job["job_id"],
                    "family_id": job["family_id"],
                    "status": row["status"],
                }
            )
        return row

    rows: list[dict[str, Any]] = []
    max_workers = min(args.concurrency, len(jobs)) if jobs else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one_job, index, job): index
            for index, job in enumerate(jobs, start=1)
        }
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def live_generation_markdown_lines(report: dict[str, Any]) -> list[str]:
    if not report["live_generation_attempted"]:
        return []
    lines = ["", "## Live Generation Results", ""]
    if report["counts"]["live_statuses"]:
        for status, count in report["counts"]["live_statuses"].items():
            lines.append(f"- `{status}`: `{count}`")
    else:
        lines.append("- No provider jobs were run.")

    rows = report["live_rows"][:24]
    if rows:
        lines.extend(["", "| job | status | output |", "|---|---|---|"])
        for row in rows:
            output = row.get("output_video", {}).get("path") or row.get(
                "output_path", ""
            )
            lines.append(f"| `{row['job_id']}` | `{row['status']}` | `{output}` |")
    return lines


def build_report(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
    live_rows: list[dict[str, Any]] | None = None,
    skipped_rows: list[dict[str, Any]] | None = None,
    model_jobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    live_rows = live_rows or []
    skipped_rows = skipped_rows or []
    live_generation_attempted = bool(args.execute_live)
    model_id, model_source, model_resolved = resolve_model_id(
        cli_or_env_model_id=args.model_id,
        manifest_jobs=model_jobs or jobs,
    )
    cost_per_video, cost_source = resolve_cost_per_video(
        config=config,
        model_id=model_id,
        cost_override=args.estimated_cost_per_video_usd,
    )
    total_cost = cost_per_video * len(jobs) if cost_per_video is not None else None
    blockers, warnings = manifest_checks(
        config=config,
        manifest=manifest,
        jobs=jobs,
        model_resolved=model_resolved,
        cost_per_video=cost_per_video,
    )
    if live_generation_attempted and skipped_rows and not jobs:
        blockers = [
            blocker
            for blocker in blockers
            if blocker != "No jobs selected for dry run."
        ]
    family_counts = Counter(str(job["family_id"]) for job in jobs)
    role_counts = Counter(str(job["role"]) for job in jobs)
    live_status_counts = Counter(str(row["status"]) for row in live_rows)
    live_failures = [
        row
        for row in live_rows
        if row["status"] not in {"downloaded", "skipped_existing_output"}
    ]
    if live_generation_attempted:
        if blockers:
            status = "live_preflight_blocked"
        elif live_failures:
            status = "live_generation_partial_or_failed"
        elif live_rows or skipped_rows:
            status = "live_generation_complete"
        else:
            status = "live_generation_noop"
    else:
        status = "dry_run_blocked" if blockers else "dry_run_ready_for_review"

    return {
        "schema_version": "confirmatory_seedance_generation.v1",
        "created_at_utc": config["manifest_created_at_utc"],
        "experiment_id": config["experiment_id"],
        "status": status,
        "dry_run": not live_generation_attempted,
        "live_generation_attempted": live_generation_attempted,
        "phase": args.phase,
        "source_config": str(args.config),
        "source_manifest": str(args.manifest),
        "output_json": str(args.out_json),
        "output_md": str(args.out_md),
        "provider": {
            "model_family": config["generator"]["model_family"],
            "resolved_model_id": model_id,
            "model_id_source": model_source,
            "model_id_resolved": model_resolved,
        },
        "generation_settings": {
            "duration_seconds": config["generator"]["duration_seconds"],
            "fps": config["generator"]["fps"],
            "resolution": config["generator"]["resolution"],
            "silence_audio": config["generator"]["silence_audio"],
        },
        "counts": {
            "selected_jobs": len(jobs),
            "skipped_existing_outputs": len(skipped_rows),
            "live_rows": len(live_rows),
            "families": len(family_counts),
            "roles": dict(sorted(role_counts.items())),
            "by_family": dict(sorted(family_counts.items())),
            "live_statuses": dict(sorted(live_status_counts.items())),
        },
        "cost_estimate": {
            "currency": "USD",
            "per_video": str(cost_per_video) if cost_per_video is not None else None,
            "total": str(total_cost) if total_cost is not None else None,
            "per_video_float": decimal_as_float(cost_per_video),
            "total_float": decimal_as_float(total_cost),
            "display_per_video": money(cost_per_video),
            "display_total": money(total_cost),
            "source": cost_source,
            "formula": (
                "(width * height * duration_seconds * fps / 1024) * "
                "0.000007 USD"
            )
            if cost_source == "openrouter_formula"
            else None,
        },
        "runtime_env_checks": {
            "credential_env_present": env_presence(),
            "credential_values_logged": False,
        },
        "preflight_blockers": blockers,
        "warnings": warnings,
        "live_options": {
            "execute_live": args.execute_live,
            "overwrite": args.overwrite,
            "max_cost_usd": args.max_cost_usd,
            "openrouter_base_url": str(args.openrouter_base_url),
            "poll_interval_seconds": args.poll_interval_seconds,
            "max_poll_attempts": args.max_poll_attempts,
            "request_timeout_seconds": args.request_timeout_seconds,
            "concurrency": args.concurrency,
        },
        "skipped_rows": skipped_rows,
        "live_rows": live_rows,
        "review_sample_jobs": [
            {
                "job_id": job["job_id"],
                "family_id": job["family_id"],
                "output_path": job["output_video"]["path"],
                "prompt": job["prompt"],
            }
            for job in jobs[:12]
        ],
        "all_output_paths": [job["output_video"]["path"] for job in jobs],
        "claim_boundary": [
            "Generated recognition MP4s are not human-memory evidence.",
            "Video-level screening and final human-facing review are required before launch.",
            "Failed, skipped, and visually rejected candidates must be retained with reasons.",
            "No human-memory or generator-control claim is created by this report.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    provider = report["provider"]
    counts = report["counts"]
    cost = report["cost_estimate"]
    env_checks = report["runtime_env_checks"]["credential_env_present"]
    live_attempted = bool(report["live_generation_attempted"])
    title = (
        "# Seedance Candidate Generation Result"
        if live_attempted
        else "# Seedance Candidate Generation Dry Run"
    )

    lines = [
        title,
        "",
        f"Created: `{report['created_at_utc']}`",
        f"Experiment: `{report['experiment_id']}`",
        f"Status: `{report['status']}`",
        f"Phase: `{report['phase']}`",
        "",
        "## Provider Resolution",
        "",
        f"- Model family: `{provider['model_family']}`",
        f"- Resolved model ID: `{provider['resolved_model_id']}`",
        f"- Model ID source: `{provider['model_id_source']}`",
        f"- Model ID resolved: `{provider['model_id_resolved']}`",
        "",
        "## Job And Cost Summary",
        "",
        f"- Selected jobs: `{counts['selected_jobs']}`",
        f"- Skipped existing outputs: `{counts['skipped_existing_outputs']}`",
        f"- Families: `{counts['families']}`",
        f"- Estimated cost per video: `{cost['display_per_video']}`",
        f"- Estimated total cost: `{cost['display_total']}`",
        "",
        "## Credential Environment Check",
        "",
    ]
    for name, present in sorted(env_checks.items()):
        lines.append(f"- `{name}` present: `{present}`")

    lines.extend(["", "## Preflight Blockers", ""])
    if report["preflight_blockers"]:
        lines.extend(f"- {item}" for item in report["preflight_blockers"])
    else:
        lines.append("- None.")

    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {item}" for item in report["warnings"])
    else:
        lines.append("- None.")

    lines.extend(["", "## Jobs By Family", ""])
    for family_id, count in report["counts"]["by_family"].items():
        lines.append(f"- `{family_id}`: `{count}`")
    lines.extend(live_generation_markdown_lines(report))

    lines.extend(["", "## Review Sample Jobs", ""])
    for job in report["review_sample_jobs"]:
        lines.extend(
            [
                f"### {job['job_id']}",
                "",
                f"- Family: `{job['family_id']}`",
                f"- Output: `{job['output_path']}`",
                f"- Prompt: {job['prompt']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Claim Boundary",
            "",
            *[f"- {item}" for item in report["claim_boundary"]],
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    if args.dry_run == args.execute_live:
        raise SystemExit("Choose exactly one of --dry-run or --execute-live.")
    normalize_output_paths(args)

    config = load_json(args.config)
    manifest = load_json(args.manifest)
    jobs = selected_jobs(
        manifest,
        phase=args.phase,
        family_ids=set(args.family_id),
        job_ids=set(args.job_id),
        limit=args.limit,
    )
    live_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    report_jobs = jobs
    if args.execute_live:
        report_jobs, skipped_rows = selected_live_jobs(
            jobs,
            overwrite=args.overwrite,
        )

    report = build_report(
        config=config,
        manifest=manifest,
        jobs=report_jobs,
        args=args,
        live_rows=live_rows,
        skipped_rows=skipped_rows,
        model_jobs=jobs,
    )
    if args.execute_live:
        if report["preflight_blockers"]:
            raise SystemExit(
                "Live generation preflight blocked: "
                + "; ".join(report["preflight_blockers"])
            )
        max_cost = parse_decimal(args.max_cost_usd)
        if max_cost is None:
            raise SystemExit("--max-cost-usd is required for --execute-live.")
        estimated_total = parse_decimal(report["cost_estimate"]["total"])
        if estimated_total is None:
            raise SystemExit("Cannot execute live generation without cost estimate.")
        if estimated_total > max_cost:
            raise SystemExit(
                "Refusing live generation: estimated selected-job cost "
                f"{money(estimated_total)} exceeds --max-cost-usd {money(max_cost)}."
            )
        live_rows = run_live_generation(
            config=config,
            jobs=report_jobs,
            args=args,
            model_id=str(report["provider"]["resolved_model_id"]),
        )
        report = build_report(
            config=config,
            manifest=manifest,
            jobs=report_jobs,
            args=args,
            live_rows=live_rows,
            skipped_rows=skipped_rows,
            model_jobs=jobs,
        )
    write_json(args.out_json, report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_jobs": report["counts"]["selected_jobs"],
                "skipped_existing_outputs": report["counts"][
                    "skipped_existing_outputs"
                ],
                "resolved_model_id": report["provider"]["resolved_model_id"],
                "estimated_total": report["cost_estimate"]["display_total"],
                "output_json": report["output_json"],
                "output_md": report["output_md"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
