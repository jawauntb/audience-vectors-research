from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_attention_capture_tribe_full_preflight.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_attention_capture_tribe_full_preflight",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_success_report_renders_preflight_details() -> None:
    module = load_module()
    report = module.build_success_report(
        app_name="audience-vectors-dev",
        media_path="/bmd-videos/example.mp4",
        event_mode="full",
        preflight={
            "exists": True,
            "duration_seconds": 3.2,
            "events_rows": 5,
            "event_columns": ["type", "filepath", "start"],
            "step_seconds": {"get_events_dataframe": 1.25},
        },
    )

    markdown = module.render_preflight_markdown(report)

    assert report["ok"] is True
    assert "Event mode: full" in markdown
    assert "Event rows: 5" in markdown
    assert "get_events_dataframe" in markdown


def test_failure_report_renders_error_without_claiming_readiness() -> None:
    module = load_module()
    report = module.build_failure_report(
        app_name="audience-vectors-dev",
        media_path="/bmd-videos/example.mp4",
        event_mode="full",
        exc=RuntimeError("missing model"),
    )

    markdown = module.render_preflight_markdown(report)

    assert report["ok"] is False
    assert "missing model" in markdown
    assert "withholds the full multimodal TRIBE path" in report["claim_boundary"]
