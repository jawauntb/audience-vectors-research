"""Dry-run Seedance candidate generation for the confirmatory recognition study.

This script is intentionally conservative. The first pass validates the frozen
candidate manifest, resolves non-secret runtime settings, estimates cost when a
per-video estimate is supplied, and writes review artifacts. It does not call a
provider API yet.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(
    "research_program/neurips_memorability_selector/experiments/"
    "content_pocket_confirmatory_recognition_20260615"
)
DEFAULT_CONFIG = EXPERIMENT_DIR / "confirmatory_study_config_20260615.json"
DEFAULT_MANIFEST = EXPERIMENT_DIR / "seedance_candidate_generation_manifest_20260615.json"
DEFAULT_OUT_JSON = EXPERIMENT_DIR / "seedance_candidate_generation_dry_run_20260615.json"
DEFAULT_OUT_MD = EXPERIMENT_DIR / "seedance_candidate_generation_dry_run_20260615.md"
PLACEHOLDER_MODEL_ID = "resolve_from_provider_before_generation"
PHASE_TO_ROLE = {
    "candidate_old_videos": "candidate_old_video",
}
SEEDANCE_CREDENTIAL_ENV_NAMES = (
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
        help="Optional per-video cost estimate. No provider pricing is assumed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Required. Live Seedance generation is intentionally not implemented yet.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def build_report(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    jobs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cost_per_video = parse_decimal(args.estimated_cost_per_video_usd)
    total_cost = cost_per_video * len(jobs) if cost_per_video is not None else None
    model_id, model_source, model_resolved = resolve_model_id(
        cli_or_env_model_id=args.model_id,
        manifest_jobs=jobs,
    )
    blockers, warnings = manifest_checks(
        config=config,
        manifest=manifest,
        jobs=jobs,
        model_resolved=model_resolved,
        cost_per_video=cost_per_video,
    )
    family_counts = Counter(str(job["family_id"]) for job in jobs)
    role_counts = Counter(str(job["role"]) for job in jobs)

    return {
        "schema_version": "confirmatory_seedance_generation_dry_run.v1",
        "created_at_utc": config["manifest_created_at_utc"],
        "experiment_id": config["experiment_id"],
        "status": "dry_run_blocked" if blockers else "dry_run_ready_for_review",
        "dry_run": True,
        "live_generation_attempted": False,
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
            "families": len(family_counts),
            "roles": dict(sorted(role_counts.items())),
            "by_family": dict(sorted(family_counts.items())),
        },
        "cost_estimate": {
            "currency": "USD",
            "per_video": str(cost_per_video) if cost_per_video is not None else None,
            "total": str(total_cost) if total_cost is not None else None,
            "display_per_video": money(cost_per_video),
            "display_total": money(total_cost),
            "source": "user_or_environment_supplied"
            if cost_per_video is not None
            else "not_supplied",
        },
        "runtime_env_checks": {
            "credential_env_present": env_presence(),
            "credential_values_logged": False,
        },
        "preflight_blockers": blockers,
        "warnings": warnings,
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
            "This is a dry-run manifest review artifact only.",
            "No Seedance API calls were made.",
            "No videos were generated.",
            "No human-memory or generator-control claim is created by this report.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    provider = report["provider"]
    counts = report["counts"]
    cost = report["cost_estimate"]
    env_checks = report["runtime_env_checks"]["credential_env_present"]

    lines = [
        "# Seedance Candidate Generation Dry Run",
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
    if not args.dry_run:
        raise SystemExit(
            "Refusing to run: live Seedance generation is not implemented yet. "
            "Re-run with --dry-run for the preflight report."
        )

    config = load_json(args.config)
    manifest = load_json(args.manifest)
    jobs = selected_jobs(
        manifest,
        phase=args.phase,
        family_ids=set(args.family_id),
        job_ids=set(args.job_id),
        limit=args.limit,
    )
    report = build_report(config=config, manifest=manifest, jobs=jobs, args=args)
    write_json(args.out_json, report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_jobs": report["counts"]["selected_jobs"],
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
