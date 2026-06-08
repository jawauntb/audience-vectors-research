from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_video_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_content_pocket_recognition_videos.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_content_pocket_recognition_videos",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def generation_job(tmp_path: Path, *, role: str, output_exists: bool = False) -> dict:
    seed = tmp_path / "seed.png"
    seed.write_bytes(b"seed")
    output = tmp_path / f"{role}.mp4"
    if output_exists:
        output.write_bytes(b"video")
    return {
        "job_id": f"{role}_v00",
        "role": role,
        "matched_id": None,
        "source_pocket": "fresh24_test",
        "old_target_id": None,
        "prompt": "Test prompt.",
        "alpha": 1.0,
        "guidance": 7.5,
        "noise_seed": 123,
        "seed_image": {"path": str(seed)},
        "output_video": {"path": str(output)},
    }


def test_selected_jobs_filters_roles_and_missing(tmp_path):
    module = load_video_module()
    manifest = {
        "generation_jobs": [
            generation_job(tmp_path, role="analysis_lure_video", output_exists=True),
            generation_job(tmp_path, role="filler_old_video", output_exists=False),
        ]
    }

    jobs = module.selected_jobs(
        manifest,
        roles={"filler_old_video"},
        limit=None,
        only_missing=True,
    )

    assert [job["role"] for job in jobs] == ["filler_old_video"]


def test_generation_counts_tracks_present_outputs(tmp_path):
    module = load_video_module()
    present = generation_job(tmp_path, role="analysis_lure_video", output_exists=True)
    missing = generation_job(tmp_path, role="filler_old_video", output_exists=False)
    rows = [
        {**module.job_row(present), "status": "already_present"},
        {**module.job_row(missing), "status": "failed"},
    ]

    counts = module.generation_counts(rows)

    assert counts["requested"] == 2
    assert counts["already_present"] == 1
    assert counts["failed"] == 1
    assert counts["present_after_run"] == 1


def test_contact_sheet_skips_empty_groups(tmp_path):
    module = load_video_module()
    result = module.write_contact_sheet(
        [],
        {},
        out_path=tmp_path / "empty.jpg",
        title="Empty",
    )

    assert result["exists"] is False
    assert result["items"] == 0
