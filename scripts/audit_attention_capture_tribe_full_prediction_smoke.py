"""Audit whether Modal TRIBE can complete one full-mode video prediction.

This is stricter than `audit_attention_capture_tribe_full_preflight.py`: event
construction can pass while the downstream text model/tokenizer path still
fails. The report stores shapes and errors only, never full activation frames.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from audience_vectors.modal_app.app import get_app_name


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
        run_prediction_smoke(
            media_path=args.media_path,
            app_name=args.app_name or get_app_name(),
            event_mode=args.event_mode,
        )
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_prediction_smoke_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not report["ok"]:
        raise SystemExit(1)


async def run_prediction_smoke(
    *,
    media_path: str,
    app_name: str,
    event_mode: str,
) -> dict[str, Any]:
    import modal  # type: ignore[import-not-found]

    audio_only = event_mode == "audio-only"
    try:
        cls = modal.Cls.from_name(app_name, "TribeV2Predictor")
        predictor = cls()
        result = await predictor.predict_video.remote.aio(media_path, audio_only)
        prediction = prediction_to_summary(result)
        return build_report(
            ok=True,
            app_name=app_name,
            media_path=media_path,
            event_mode=event_mode,
            prediction=prediction,
            error_type=None,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return build_report(
            ok=False,
            app_name=app_name,
            media_path=media_path,
            event_mode=event_mode,
            prediction=None,
            error_type=type(exc).__name__,
            error=str(exc),
        )


def prediction_to_summary(result: Any) -> dict[str, Any]:
    frames = np.asarray(result.frames if hasattr(result, "frames") else result["frames"])
    if frames.ndim == 1:
        frames = frames.reshape(1, -1)
    return {
        "duration_seconds": float(
            result.duration_seconds
            if hasattr(result, "duration_seconds")
            else result["duration_seconds"]
        ),
        "frames_rows": int(frames.shape[0]),
        "frames_cols": int(frames.shape[1]) if frames.ndim >= 2 else 0,
        "frames_dtype": str(frames.dtype),
        "all_finite": bool(np.isfinite(frames).all()),
    }


def build_report(
    *,
    ok: bool,
    app_name: str,
    media_path: str,
    event_mode: str,
    prediction: dict[str, Any] | None,
    error_type: str | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "attention_capture_tribe_full_prediction_smoke_audit",
        "generated_at": datetime.now(UTC).isoformat(),
        "ok": ok,
        "app_name": app_name,
        "media_path": media_path,
        "event_mode": event_mode,
        "prediction": prediction,
        "error_type": error_type,
        "error": error,
        "claim_boundary": (
            "This verifies that the deployed Modal TRIBE path can complete one "
            "video prediction. It does not validate Phase 1, does not estimate "
            "correlation, and does not replace external labels."
        ),
    }


def render_prediction_smoke_markdown(report: dict[str, Any]) -> str:
    prediction = report.get("prediction") or {}
    lines = [
        "# Attention-Capture TRIBE Full-Prediction Smoke Audit",
        "",
        "## Verdict",
        "",
        f"- OK: {report['ok']}",
        f"- App: {report['app_name']}",
        f"- Media path: {report['media_path']}",
        f"- Event mode: {report['event_mode']}",
        f"- Error type: {report['error_type'] or 'none'}",
        f"- Error: {report['error'] or 'none'}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Prediction",
        "",
        f"- Duration seconds: {format_float(prediction.get('duration_seconds'))}",
        f"- Frame rows: {prediction.get('frames_rows', 0)}",
        f"- Frame columns: {prediction.get('frames_cols', 0)}",
        f"- Frame dtype: {prediction.get('frames_dtype', 'n/a')}",
        f"- All finite: {prediction.get('all_finite', False)}",
    ]
    return "\n".join(lines) + "\n"


def format_float(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
