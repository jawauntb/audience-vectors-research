from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "extract_attention_capture_tribe_features_modal_volume.py"
    )
    spec = importlib.util.spec_from_file_location(
        "extract_attention_capture_tribe_features_modal_volume",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_jobs_from_csv_respects_limit(tmp_path: Path) -> None:
    module = load_module()
    source_csv = tmp_path / "labels.csv"
    source_csv.write_text(
        "sample_id,video_path\n"
        "dhf1k_003,/bmd-videos/003.AVI\n"
        "dhf1k_004,/bmd-videos/004.AVI\n",
        encoding="utf-8",
    )

    jobs = module.load_jobs_from_csv(source_csv=source_csv, limit=1)

    assert len(jobs) == 1
    assert jobs[0].sample_id == "dhf1k_003"
    assert jobs[0].media_path == "/bmd-videos/003.AVI"


def test_modal_result_to_dict_accepts_pydantic_like_object() -> None:
    module = load_module()
    result = SimpleNamespace(
        model_dump=lambda: {
            "sample_id": "dhf1k_003",
            "status": "written",
            "frames_rows": 15,
            "frames_cols": 20484,
        }
    )

    payload = module.modal_result_to_dict(result)

    assert payload["sample_id"] == "dhf1k_003"
    assert payload["frames_cols"] == 20484


def test_modal_exception_to_error_result_records_retryable_failure() -> None:
    module = load_module()
    job = module.ModalFeatureJob("dhf1k_146", "/bmd-videos/146.AVI")

    payload = module.modal_exception_to_error_result(
        job=job,
        output_prefix="attention_capture/DHF1K/full",
        event_mode="full",
        exc=RuntimeError("remote expired"),
    )

    assert payload["sample_id"] == "dhf1k_146"
    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
    assert payload["output_path"].endswith("/attention_capture/DHF1K/full/dhf1k_146.npz")


def test_build_report_ready_for_written_and_cached_results(tmp_path: Path) -> None:
    module = load_module()
    jobs = [
        module.ModalFeatureJob("dhf1k_003", "/bmd-videos/003.AVI"),
        module.ModalFeatureJob("dhf1k_004", "/bmd-videos/004.AVI"),
    ]

    report = module.build_report(
        source_csv=tmp_path / "labels.csv",
        sample_id_column="sample_id",
        media_path_column="video_path",
        app_name="audience-vectors-dev",
        output_prefix="attention_capture/DHF1K/full",
        event_mode="full",
        overwrite=False,
        concurrency=8,
        jobs=jobs,
        results=[
            {
                "sample_id": "dhf1k_003",
                "status": "written",
                "frames_rows": 15,
                "frames_cols": 20484,
            },
            {
                "sample_id": "dhf1k_004",
                "status": "cached",
                "frames_rows": 31,
                "frames_cols": 20484,
            },
        ],
    )

    assert report["ready"] is True
    assert report["n_written"] == 1
    assert report["n_cached"] == 1
    assert report["blocking_reasons"] == []


def test_build_report_blocks_on_errors_and_shape_mismatch(tmp_path: Path) -> None:
    module = load_module()
    jobs = [
        module.ModalFeatureJob("dhf1k_003", "/bmd-videos/003.AVI"),
        module.ModalFeatureJob("dhf1k_004", "/bmd-videos/004.AVI"),
    ]

    report = module.build_report(
        source_csv=tmp_path / "labels.csv",
        sample_id_column="sample_id",
        media_path_column="video_path",
        app_name="audience-vectors-dev",
        output_prefix="attention_capture/DHF1K/full",
        event_mode="full",
        overwrite=False,
        concurrency=8,
        jobs=jobs,
        results=[
            {
                "sample_id": "dhf1k_003",
                "status": "error",
                "error_type": "RuntimeError",
                "error": "boom",
            },
            {
                "sample_id": "dhf1k_004",
                "status": "written",
                "frames_rows": 31,
                "frames_cols": 10,
            },
        ],
    )

    assert report["ready"] is False
    assert report["n_errors"] == 1
    assert report["n_shape_mismatches"] == 1
    assert "1 feature extraction jobs failed" in report["blocking_reasons"]
    assert (
        "1 feature files have unexpected vertex count"
        in report["blocking_reasons"]
    )
