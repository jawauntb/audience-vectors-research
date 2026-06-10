"""Audit whether Modal TRIBE can run full video event preflight.

This is a runtime verifier for the full multimodal path. It does not score a
Phase 1 claim; it only checks whether TRIBE can build non-audio-only events from
one existing Modal-hosted video using the currently deployed predictor/cache.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-path", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--app-name", default=None)
    parser.add_argument(
        "--event-mode",
        choices=("full", "audio-only"),
        default="full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(
        run_preflight(
            media_path=args.media_path,
            app_name=args.app_name,
            audio_only=args.event_mode == "audio-only",
        )
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_preflight_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not report["ok"]:
        raise SystemExit(1)


async def run_preflight(
    *,
    media_path: str,
    app_name: str | None,
    audio_only: bool,
) -> dict[str, Any]:
    modal = import_module("modal")
    app_module = import_module("audience_vectors.modal_app.app")

    resolved_app = app_name or app_module.get_app_name()
    event_mode = "audio_only" if audio_only else "full"
    try:
        cls = modal.Cls.from_name(resolved_app, "TribeV2Predictor")
        predictor = cls()
        result = await predictor.preflight_video.remote.aio(media_path, audio_only)
    except Exception as exc:  # noqa: BLE001
        return build_failure_report(
            app_name=resolved_app,
            media_path=media_path,
            event_mode=event_mode,
            exc=exc,
        )
    return build_success_report(
        app_name=resolved_app,
        media_path=media_path,
        event_mode=event_mode,
        preflight=preflight_to_dict(result),
    )


def preflight_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        payload = result.model_dump()
    elif isinstance(result, dict):
        payload = result
    else:
        payload = json.loads(json.dumps(result, default=str))
    return dict(payload)


def build_success_report(
    *,
    app_name: str,
    media_path: str,
    event_mode: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "attention_capture_tribe_full_preflight_audit",
        "generated_at": datetime.now(UTC).isoformat(),
        "ok": True,
        "app_name": app_name,
        "media_path": media_path,
        "event_mode": event_mode,
        "preflight": preflight,
        "claim_boundary": (
            "This verifies that the deployed Modal TRIBE path can construct "
            "events for one video. It does not validate attentional capture, "
            "does not score Phase 1, and does not replace external labels."
        ),
    }


def build_failure_report(
    *,
    app_name: str,
    media_path: str,
    event_mode: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "attention_capture_tribe_full_preflight_audit",
        "generated_at": datetime.now(UTC).isoformat(),
        "ok": False,
        "app_name": app_name,
        "media_path": media_path,
        "event_mode": event_mode,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "claim_boundary": (
            "This failure withholds the full multimodal TRIBE path. It does not "
            "evaluate the Phase 1 capture-score hypothesis."
        ),
    }


def render_preflight_markdown(report: dict[str, Any]) -> str:
    preflight = report.get("preflight") or {}
    step_seconds = preflight.get("step_seconds") or {}
    lines = [
        "# Attention-Capture TRIBE Full-Preflight Audit",
        "",
        "## Verdict",
        "",
        f"- OK: {report['ok']}",
        f"- App: {report['app_name']}",
        f"- Media path: {report['media_path']}",
        f"- Event mode: {report['event_mode']}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
    ]
    if not report["ok"]:
        lines.extend(
            [
                "## Error",
                "",
                f"- Type: {report.get('error_type')}",
                f"- Message: {report.get('error')}",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(
        [
            "## Preflight",
            "",
            f"- Resolved path exists: {preflight.get('exists')}",
            f"- Duration seconds: {format_float(preflight.get('duration_seconds'))}",
            f"- Event rows: {preflight.get('events_rows')}",
            f"- Event columns: {', '.join(preflight.get('event_columns') or [])}",
            "",
            "## Step Seconds",
            "",
            "| step | seconds |",
            "|---|---:|",
        ]
    )
    if step_seconds:
        for name, seconds in step_seconds.items():
            lines.append(f"| {name} | {format_float(seconds)} |")
    else:
        lines.append("| none | n/a |")
    return "\n".join(lines) + "\n"


def format_float(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
