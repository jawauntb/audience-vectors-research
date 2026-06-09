from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_launch_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_content_pocket_recognition_launch_assets.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_content_pocket_recognition_launch_assets",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def job(job_id: str, path: str) -> dict:
    return {
        "job_id": job_id,
        "output_video": {"path": path},
    }


def test_required_video_paths_includes_generated_and_old_targets():
    module = load_launch_module()
    design = {
        "session_forms": [
            {
                "session1": {
                    "analysis_encoding_targets": [
                        {"old_video_path": "data/old/a.mp4"},
                    ]
                },
                "session2": {
                    "analysis_recognition_trials": [
                        {"old_video_path": "data/old/a.mp4"},
                        {"old_video_path": "data/old/b.mp4"},
                    ]
                },
            }
        ]
    }
    manifest = {
        "generation_jobs": [
            job("lure_a", "data/generated/lure_a.mp4"),
            job("filler_old_v00", "data/generated/filler_old_v00.mp4"),
        ]
    }

    paths = module.required_video_paths(
        design=design,
        production_manifest=manifest,
    )

    assert paths == [
        "data/generated/filler_old_v00.mp4",
        "data/generated/lure_a.mp4",
        "data/old/a.mp4",
        "data/old/b.mp4",
    ]


def test_filler_trials_wire_old_and_lure_urls_with_balanced_sides():
    module = load_launch_module()
    jobs = {
        "filler_old_v00": job("filler_old_v00", "data/generated/old0.mp4"),
        "filler_lure_v00": job("filler_lure_v00", "data/generated/lure0.mp4"),
        "filler_old_v01": job("filler_old_v01", "data/generated/old1.mp4"),
        "filler_lure_v01": job("filler_lure_v01", "data/generated/lure1.mp4"),
    }
    assets = {
        "data/generated/old0.mp4": {"hosted_url": "https://example.test/old0.mp4", "asset_path": "videos/old0.mp4"},
        "data/generated/lure0.mp4": {"hosted_url": "https://example.test/lure0.mp4", "asset_path": "videos/lure0.mp4"},
        "data/generated/old1.mp4": {"hosted_url": "https://example.test/old1.mp4", "asset_path": "videos/old1.mp4"},
        "data/generated/lure1.mp4": {"hosted_url": "https://example.test/lure1.mp4", "asset_path": "videos/lure1.mp4"},
    }

    trials = module.filler_trials(jobs, assets, count=2)

    assert trials[0]["old_side"] == "left"
    assert trials[0]["old_video_url"].endswith("old0.mp4")
    assert trials[1]["old_side"] == "right"
    assert trials[1]["lure_video_url"].endswith("lure1.mp4")
