from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_extraction_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "extract_pocket_replay_vjepa.py"
    )
    spec = importlib.util.spec_from_file_location(
        "extract_pocket_replay_vjepa",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_video_jobs_uses_exact_video_stems(tmp_path):
    module = load_extraction_module()
    report_dir = tmp_path / "data" / "reports"
    video_dir = tmp_path / "data" / "generated" / "run"
    report_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    video_path = video_dir / "bo_replay_00_sobol_prompt_search_518_slot00_rep01.mp4"
    video_path.write_bytes(b"fake mp4")
    report_path = report_dir / "report.json"
    report = {
        "rows": [
            {
                "local_video_path": "data/generated/run/"
                "bo_replay_00_sobol_prompt_search_518_slot00_rep01.mp4",
                "replay_tribe_score": 1.25,
                "trial": {
                    "task_id": "sobol_prompt_search_518_slot00",
                    "seed_idx": 0,
                },
            },
            {
                "local_video_path": "data/generated/run/ignored.mp4",
                "replay_tribe_score": None,
                "trial": {"task_id": "ignored", "seed_idx": 1},
            },
        ]
    }

    [job] = module.build_video_jobs(
        report=report,
        report_path=report_path,
        output_dir=tmp_path / "features",
    )

    assert job.sample_id == "bo_replay_00_sobol_prompt_search_518_slot00_rep01"
    assert job.replicate == 1
    assert job.absolute_video_path == video_path
    assert job.output_path.name == f"{job.sample_id}.npz"


def test_result_to_arrays_accepts_object_and_mapping_results():
    module = load_extraction_module()
    obj = SimpleNamespace(
        embedding=[1.0, 2.0],
        duration_seconds=3.5,
        n_frames=12,
    )
    mapping = {
        "embedding": [3.0, 4.0, 5.0],
        "duration_seconds": 1.25,
        "n_frames": 8,
    }

    obj_embedding, obj_duration, obj_frames = module.result_to_arrays(obj)
    map_embedding, map_duration, map_frames = module.result_to_arrays(mapping)

    assert obj_embedding.dtype == np.float32
    assert obj_embedding.tolist() == [1.0, 2.0]
    assert obj_duration == 3.5
    assert obj_frames == 12
    assert map_embedding.tolist() == [3.0, 4.0, 5.0]
    assert map_duration == 1.25
    assert map_frames == 8


def test_summarize_reports_complete_exact_feature_coverage(tmp_path):
    module = load_extraction_module()
    jobs = [
        module.VideoJob(
            sample_id="a",
            task_id="task_a",
            seed_idx=0,
            replicate=0,
            local_video_path="data/generated/a.mp4",
            absolute_video_path=tmp_path / "data" / "generated" / "a.mp4",
            output_path=tmp_path / "features" / "a.npz",
        ),
        module.VideoJob(
            sample_id="b",
            task_id="task_b",
            seed_idx=1,
            replicate=1,
            local_video_path="data/generated/b.mp4",
            absolute_video_path=tmp_path / "data" / "generated" / "b.mp4",
            output_path=tmp_path / "features" / "b.npz",
        ),
    ]
    rows = [
        {"sample_id": "a", "status": "cached", "feature_path": "data/features/a.npz"},
        {"sample_id": "b", "status": "written", "feature_path": "data/features/b.npz"},
    ]

    summary = module.summarize(
        report_path=tmp_path / "data" / "reports" / "report.json",
        output_dir=tmp_path / "features",
        app_name="audience-vectors-dev",
        max_concurrency=2,
        jobs=jobs,
        rows=rows,
    )

    assert summary["n_jobs"] == 2
    assert summary["n_features_available"] == 2
    assert summary["coverage_complete"] is True
    assert summary["status_counts"] == {"cached": 1, "written": 1}
