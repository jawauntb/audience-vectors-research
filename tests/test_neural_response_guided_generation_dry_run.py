from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_dry_run_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_neural_response_guided_generation_dry_run.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_neural_response_guided_generation_dry_run",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_side(
    *,
    run_dir: str,
    replay_idx: int,
    task_id: str,
    rep: int,
    pocket: str,
    policy: str,
) -> dict:
    label = f"bo_replay_{replay_idx:02d}_{task_id}_rep{rep:02d}"
    return {
        "label": pocket,
        "path": f"data/generated/{run_dir}/{label}.mp4",
        "policy": policy,
        "video_label": label,
    }


def make_task(
    *,
    task_id: str,
    positive_task: str,
    control_task: str,
    positive_pocket: str,
    control_pocket: str,
    rep: int,
) -> dict:
    return {
        "task_id": task_id,
        "left": make_side(
            run_dir="bo_descriptor_conditioned_replication_test",
            replay_idx=0,
            task_id=positive_task,
            rep=rep,
            pocket=positive_pocket,
            policy="content_pocket_candidate",
        ),
        "right": make_side(
            run_dir="bo_descriptor_conditioned_replication_test",
            replay_idx=1,
            task_id=control_task,
            rep=rep,
            pocket=control_pocket,
            policy="hard_negative_control",
        ),
        "metadata": {
            "analysis_tier": "primary",
            "target_side": "left",
            "positive_replay_tribe_score": 3.0,
            "control_replay_tribe_score": -2.0,
        },
    }


def write_side_files(root: Path, feature_dir: Path, tasks: list[dict]) -> None:
    vectors = {
        "fresh24_orange_flowers": [1.0, 0.0],
        "fresh24_hanging_clothes": [1.0, 0.1],
        "fresh24_aerial_beach": [-1.0, 0.0],
        "fresh24_city_street": [-1.0, -0.1],
    }
    for task in tasks:
        for side_name in ("left", "right"):
            side = task[side_name]
            video_path = root / side["path"]
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(side["video_label"].encode())
            np.savez(
                feature_dir / f"{side['video_label']}.npz",
                embedding=np.asarray(vectors[side["label"]], dtype=np.float32),
            )


def test_parse_source_pool_and_video_task_id():
    module = load_dry_run_module()

    path = (
        "data/generated/bo_descriptor_conditioned_replication_test/"
        "bo_replay_00_sobol_prompt_search_519_slot10_rep00.mp4"
    )

    assert module.infer_source_pool(path) == "descriptor_conditioned_replication"
    assert (
        module.parse_generation_task_id(
            "bo_replay_00_sobol_prompt_search_519_slot10_rep00"
        )
        == "sobol_prompt_search_519_slot10"
    )


def test_dry_run_surfaces_proxy_disagreement(tmp_path):
    module = load_dry_run_module()
    feature_dir = tmp_path / "data" / "features" / "vjepa"
    feature_dir.mkdir(parents=True)
    tasks = [
        make_task(
            task_id="task-orange-vs-beach",
            positive_task="sobol_prompt_search_519_slot10",
            control_task="sobol_prompt_search_519_slot03",
            positive_pocket="fresh24_orange_flowers",
            control_pocket="fresh24_aerial_beach",
            rep=0,
        ),
        make_task(
            task_id="task-clothes-vs-street",
            positive_task="sobol_prompt_search_520_slot12",
            control_task="sobol_prompt_search_520_slot08",
            positive_pocket="fresh24_hanging_clothes",
            control_pocket="fresh24_city_street",
            rep=1,
        ),
    ]
    write_side_files(tmp_path, feature_dir, tasks)

    task_doc = {"n_tasks": len(tasks), "tasks": tasks}
    tasks_path = write_json(tmp_path / "tasks.json", task_doc)
    summary_path = write_json(
        tmp_path / "summary.json",
        {
            "candidate_records": [
                {
                    "task_id": "sobol_prompt_search_519_slot10",
                    "seed_video_clip_cosine": 0.2,
                },
                {
                    "task_id": "sobol_prompt_search_519_slot03",
                    "seed_video_clip_cosine": 0.9,
                },
                {
                    "task_id": "sobol_prompt_search_520_slot12",
                    "seed_video_clip_cosine": 0.3,
                },
                {
                    "task_id": "sobol_prompt_search_520_slot08",
                    "seed_video_clip_cosine": 0.8,
                },
            ]
        },
    )
    report_path = write_json(
        tmp_path / "report.json",
        {
            "rows": [
                {
                    "label": side["video_label"],
                    "visual_first_status": "retained",
                    "visual_artifact_gate": {"passes_visual_gate": True},
                }
                for task in tasks
                for side in (task["left"], task["right"])
            ]
        },
    )

    payload, markdown = module.build_dry_run(
        tasks_path=tasks_path,
        data_roots=[tmp_path],
        embedding_summary_paths={
            "descriptor_conditioned_replication": summary_path,
        },
        replay_report_paths={
            "descriptor_conditioned_replication": report_path,
        },
        vjepa_feature_dirs={
            "descriptor_conditioned_replication": feature_dir,
        },
    )

    by_metric = payload["aggregate"]["by_metric"]
    assert payload["data_resolution"]["mp4_found_unique_paths"] == 4
    assert payload["data_resolution"]["vjepa_feature_found_side_observations"] == 4
    assert by_metric["tribe_bmd_projection"]["selects_content_pocket_target"] == 2
    assert by_metric["vjepa_centroid_margin"]["selects_content_pocket_target"] == 2
    assert by_metric["clip_seed_video_preservation"]["selects_content_pocket_target"] == 0
    assert payload["aggregate"]["disagreement_examples"]
    assert "proxy-only feasibility dry run" in markdown
