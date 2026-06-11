"""Extract attention-capture TRIBE features into a Modal Volume.

The local process only coordinates jobs and writes a compact report. Full TRIBE
activation tensors stay in the `attention-capture-features-v1` Modal Volume.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_PREFIX = "attention_capture/DHF1K/tribe_dhf1k_attention_full_20260610"
DEFAULT_VOLUME_NAME = "attention-capture-features-v1"
DEFAULT_MODAL_MOUNT = "/attention-capture-features"


@dataclass(frozen=True)
class ModalFeatureJob:
    sample_id: str
    media_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--media-path-column", default="video_path")
    parser.add_argument("--app-name", default=None)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--event-mode",
        choices=("full", "audio-only"),
        default="full",
    )
    return parser.parse_args()


def main() -> None:
    from audience_vectors.modal_app.app import get_app_name

    args = parse_args()
    report = asyncio.run(
        run_modal_volume_extraction(
            source_csv=args.source_csv,
            sample_id_column=args.sample_id_column,
            media_path_column=args.media_path_column,
            app_name=args.app_name or get_app_name(),
            output_prefix=args.output_prefix,
            concurrency=args.concurrency,
            limit=args.limit,
            event_mode=args.event_mode,
            overwrite=args.overwrite,
        )
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_extraction_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not report["ready"]:
        raise SystemExit(1)


async def run_modal_volume_extraction(
    *,
    source_csv: Path,
    sample_id_column: str,
    media_path_column: str,
    app_name: str,
    output_prefix: str,
    concurrency: int,
    limit: int | None,
    event_mode: str,
    overwrite: bool,
) -> dict[str, Any]:
    import modal  # type: ignore[import-not-found]

    jobs = load_jobs_from_csv(
        source_csv=source_csv,
        sample_id_column=sample_id_column,
        media_path_column=media_path_column,
        limit=limit,
    )
    cls = modal.Cls.from_name(app_name, "TribeV2Predictor")
    predictor = cls()
    audio_only = event_mode == "audio-only"
    sem = asyncio.Semaphore(max(1, concurrency))

    async def run_one(job: ModalFeatureJob) -> dict[str, Any]:
        async with sem:
            try:
                result = await predictor.predict_video_to_feature_volume.remote.aio(
                    job.media_path,
                    job.sample_id,
                    output_prefix,
                    audio_only,
                    overwrite,
                )
                payload = modal_result_to_dict(result)
            except Exception as exc:  # noqa: BLE001
                payload = modal_exception_to_error_result(
                    job=job,
                    output_prefix=output_prefix,
                    event_mode=event_mode,
                    exc=exc,
                )
            print(
                "[modal-feature] "
                f"{payload['sample_id']} {payload['status']} "
                f"frames={payload.get('frames_rows') or 0}x"
                f"{payload.get('frames_cols') or 0}",
                flush=True,
            )
            return payload

    results = await asyncio.gather(*(run_one(job) for job in jobs))
    return build_report(
        source_csv=source_csv,
        sample_id_column=sample_id_column,
        media_path_column=media_path_column,
        app_name=app_name,
        output_prefix=output_prefix,
        event_mode=event_mode,
        overwrite=overwrite,
        concurrency=concurrency,
        jobs=jobs,
        results=results,
    )


def load_jobs_from_csv(
    *,
    source_csv: Path,
    sample_id_column: str = "sample_id",
    media_path_column: str = "video_path",
    limit: int | None = None,
) -> list[ModalFeatureJob]:
    with source_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    jobs: list[ModalFeatureJob] = []
    for row in rows:
        jobs.append(
            ModalFeatureJob(
                sample_id=required_cell(row, sample_id_column, source_csv),
                media_path=required_cell(row, media_path_column, source_csv),
            )
        )
        if limit is not None and len(jobs) >= limit:
            break
    return jobs


def required_cell(row: dict[str, str], column: str, path: Path) -> str:
    value = row.get(column)
    if value is None or value == "":
        raise ValueError(f"{path} has a row missing required column {column!r}")
    return value


def modal_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return dict(result)
    payload: dict[str, Any] = {}
    for key in (
        "sample_id",
        "media_path",
        "event_mode",
        "output_path",
        "status",
        "duration_seconds",
        "frames_rows",
        "frames_cols",
        "frames_dtype",
        "size_bytes",
        "error_type",
        "error",
    ):
        payload[key] = getattr(result, key, None)
    return payload


def modal_exception_to_error_result(
    *,
    job: ModalFeatureJob,
    output_prefix: str,
    event_mode: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "sample_id": job.sample_id,
        "media_path": job.media_path,
        "event_mode": event_mode,
        "output_path": f"{DEFAULT_MODAL_MOUNT}/{output_prefix.strip('/')}/{job.sample_id}.npz",
        "status": "error",
        "duration_seconds": None,
        "frames_rows": None,
        "frames_cols": None,
        "frames_dtype": None,
        "size_bytes": None,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def build_report(
    *,
    source_csv: Path,
    sample_id_column: str,
    media_path_column: str,
    app_name: str,
    output_prefix: str,
    event_mode: str,
    overwrite: bool,
    concurrency: int,
    jobs: list[ModalFeatureJob],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    n_written = sum(1 for result in results if result.get("status") == "written")
    n_cached = sum(1 for result in results if result.get("status") == "cached")
    errors = [result for result in results if result.get("status") == "error"]
    shape_mismatches = [
        result
        for result in results
        if result.get("status") in {"written", "cached"}
        and int(result.get("frames_cols") or 0) != 20484
    ]
    ready = not errors and not shape_mismatches and len(results) == len(jobs)
    return {
        "schema_version": 1,
        "experiment": "attention_capture_tribe_modal_volume_extraction",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_csv": str(source_csv),
        "sample_id_column": sample_id_column,
        "media_path_column": media_path_column,
        "app_name": app_name,
        "modal_volume_name": DEFAULT_VOLUME_NAME,
        "modal_mount": DEFAULT_MODAL_MOUNT,
        "output_prefix": output_prefix,
        "event_mode": event_mode,
        "overwrite": overwrite,
        "concurrency": concurrency,
        "n_jobs": len(jobs),
        "n_results": len(results),
        "n_written": n_written,
        "n_cached": n_cached,
        "n_errors": len(errors),
        "n_shape_mismatches": len(shape_mismatches),
        "ready": ready,
        "blocking_reasons": modal_volume_blockers(
            n_jobs=len(jobs),
            n_results=len(results),
            errors=errors,
            shape_mismatches=shape_mismatches,
        ),
        "errors": errors[:50],
        "shape_mismatches": shape_mismatches[:50],
        "results": results,
        "claim_boundary": (
            "This report verifies Modal-side feature extraction and storage only. "
            "It does not validate the Phase 1 capture-score hypothesis."
        ),
    }


def modal_volume_blockers(
    *,
    n_jobs: int,
    n_results: int,
    errors: list[dict[str, Any]],
    shape_mismatches: list[dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if n_results != n_jobs:
        blockers.append(f"expected {n_jobs} results, received {n_results}")
    if errors:
        blockers.append(f"{len(errors)} feature extraction jobs failed")
    if shape_mismatches:
        blockers.append(
            f"{len(shape_mismatches)} feature files have unexpected vertex count"
        )
    return blockers


def render_extraction_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Attention-Capture TRIBE Modal-Volume Extraction",
        "",
        "## Verdict",
        "",
        f"- Ready: {report['ready']}",
        f"- App: `{report['app_name']}`",
        f"- Modal volume: `{report['modal_volume_name']}`",
        f"- Output prefix: `{report['output_prefix']}`",
        f"- Event mode: `{report['event_mode']}`",
        f"- Jobs: {report['n_jobs']}",
        f"- Written: {report['n_written']}",
        f"- Cached: {report['n_cached']}",
        f"- Errors: {report['n_errors']}",
        f"- Shape mismatches: {report['n_shape_mismatches']}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Blocking Reasons",
        "",
    ]
    blockers = report["blocking_reasons"]
    lines.extend(f"- {reason}" for reason in blockers) if blockers else lines.append(
        "- none"
    )
    lines.extend(["", "## Error Preview", ""])
    errors = report.get("errors") or []
    if not errors:
        lines.append("- none")
    else:
        for error in errors[:10]:
            lines.append(
                f"- `{error.get('sample_id')}` {error.get('error_type')}: "
                f"{error.get('error')}"
            )
    lines.extend(["", "## Output Preview", ""])
    for result in report["results"][:10]:
        lines.append(
            "| {sample_id} | {status} | {frames_rows} x {frames_cols} | {path} |".format(
                sample_id=result.get("sample_id"),
                status=result.get("status"),
                frames_rows=result.get("frames_rows") or 0,
                frames_cols=result.get("frames_cols") or 0,
                path=result.get("output_path"),
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
