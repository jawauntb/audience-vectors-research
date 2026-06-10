from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_attention_capture_tribe_full_prediction_smoke.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_attention_capture_tribe_full_prediction_smoke",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Prediction:
    frames = [[0.1, 0.2], [0.3, 0.4]]
    duration_seconds = 3.5


def test_prediction_summary_records_shape_without_frame_payload() -> None:
    module = load_module()

    summary = module.prediction_to_summary(Prediction())

    assert summary == {
        "duration_seconds": 3.5,
        "frames_rows": 2,
        "frames_cols": 2,
        "frames_dtype": "float64",
        "all_finite": True,
    }


def test_failure_report_renders_gated_model_error() -> None:
    module = load_module()

    report = module.build_report(
        ok=False,
        app_name="audience-vectors-dev",
        media_path="/bmd-videos/attention_capture/DHF1K/video/003.AVI",
        event_mode="full",
        prediction=None,
        error_type="OSError",
        error="Cannot access gated repo for url https://huggingface.co/meta-llama",
    )
    markdown = module.render_prediction_smoke_markdown(report)

    assert report["ok"] is False
    assert "OSError" in markdown
    assert "Cannot access gated repo" in markdown
