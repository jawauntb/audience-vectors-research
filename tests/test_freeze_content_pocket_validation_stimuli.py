from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_freeze_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "freeze_content_pocket_validation_stimuli.py"
    )
    spec = importlib.util.spec_from_file_location(
        "freeze_content_pocket_validation_stimuli",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_row(
    *,
    root: Path,
    run_dir: str,
    task_id: str,
    pocket: str,
    replicate: int,
    score: float,
    retained: bool = True,
) -> dict:
    label = f"{task_id}_rep{replicate:02d}"
    local_path = f"data/generated/{run_dir}/{label}.mp4"
    path = root / local_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{pocket}:{task_id}:{replicate}".encode())
    return {
        "trial": {"task_id": task_id},
        "replicate_index": replicate,
        "noise_seed": 1000 + replicate,
        "seed": {"bmd_name": pocket},
        "label": label,
        "local_video_path": local_path,
        "generation_error": None,
        "visual_artifact_gate": {"passes_visual_gate": retained},
        "visual_first_retained": retained,
        "replay_tribe_score": score,
    }


def add_candidate(
    rows: list[dict],
    *,
    root: Path,
    run_dir: str,
    task_id: str,
    pocket: str,
    scores: tuple[float, float, float],
    retained: bool = True,
) -> None:
    for replicate, score in enumerate(scores):
        rows.append(
            make_row(
                root=root,
                run_dir=run_dir,
                task_id=task_id,
                pocket=pocket,
                replicate=replicate,
                score=score,
                retained=retained,
            )
        )


def add_controls(
    rows: list[dict],
    *,
    root: Path,
    run_dir: str,
    recipe: int,
) -> None:
    control_specs = [
        ("fresh24_aerial_beach", "03", 0, -8.0),
        ("fresh24_city_street", "08", 1, -9.0),
        ("fresh24_storm_beach", "14", 2, -10.0),
    ]
    for pocket, slot, replicate, score in control_specs:
        rows.append(
            make_row(
                root=root,
                run_dir=run_dir,
                task_id=f"sobol_prompt_search_{recipe}_slot{slot}",
                pocket=pocket,
                replicate=replicate,
                score=score,
            )
        )


def write_report(root: Path, name: str, rows: list[dict]) -> Path:
    report = root / "data" / "reports" / f"{name}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return report


def test_freeze_selects_top_complete_candidates_and_matched_controls(tmp_path):
    module = load_freeze_module()
    primary_rows: list[dict] = []
    boundary_rows: list[dict] = []
    for recipe in (518, 519):
        add_controls(primary_rows, root=tmp_path, run_dir="primary", recipe=recipe)
        add_controls(boundary_rows, root=tmp_path, run_dir="boundary", recipe=recipe)

    add_candidate(
        primary_rows,
        root=tmp_path,
        run_dir="primary",
        task_id="sobol_prompt_search_518_slot10",
        pocket="fresh24_orange_flowers",
        scores=(1.0, 1.0, 1.0),
    )
    add_candidate(
        primary_rows,
        root=tmp_path,
        run_dir="primary",
        task_id="sobol_prompt_search_519_slot10",
        pocket="fresh24_orange_flowers",
        scores=(4.0, 4.0, 4.0),
    )
    add_candidate(
        primary_rows,
        root=tmp_path,
        run_dir="primary",
        task_id="sobol_prompt_search_518_slot12",
        pocket="fresh24_hanging_clothes",
        scores=(3.0, 3.0, 3.0),
    )
    add_candidate(
        boundary_rows,
        root=tmp_path,
        run_dir="boundary",
        task_id="sobol_prompt_search_518_slot18",
        pocket="fresh24_blue_jellyfish",
        scores=(2.0, 2.0, 2.0),
    )
    add_candidate(
        boundary_rows,
        root=tmp_path,
        run_dir="boundary",
        task_id="sobol_prompt_search_519_slot00",
        pocket="fresh24_old_car",
        scores=(1.0, 1.0, 1.0),
    )

    manifest, task_doc, _summary = module.build_freeze(
        primary_report=write_report(tmp_path, "primary", primary_rows),
        boundary_report=write_report(tmp_path, "boundary", boundary_rows),
        primary_candidates_per_pocket=1,
        boundary_candidates_per_pocket=1,
        expected_replicates=3,
    )

    assert task_doc["n_tasks"] == 12
    assert manifest["missing_files"] == []
    selected_tasks = {
        candidate["pocket"]: candidate["task_id"]
        for candidate in manifest["selected_candidates"]
    }
    assert selected_tasks["fresh24_orange_flowers"] == "sobol_prompt_search_519_slot10"
    orange_tasks = [
        task
        for task in task_doc["tasks"]
        if task["metadata"]["positive_pocket"] == "fresh24_orange_flowers"
    ]
    assert [
        task["metadata"]["control_pocket"]
        for task in sorted(orange_tasks, key=lambda item: item["metadata"]["replicate_index"])
    ] == [
        "fresh24_aerial_beach",
        "fresh24_city_street",
        "fresh24_storm_beach",
    ]
    assert {task["metadata"]["sobol_recipe_index"] for task in orange_tasks} == {519}


def test_freeze_skips_visually_rejected_candidate(tmp_path):
    module = load_freeze_module()
    primary_rows: list[dict] = []
    boundary_rows: list[dict] = []
    for recipe in (518, 519):
        add_controls(primary_rows, root=tmp_path, run_dir="primary", recipe=recipe)
        add_controls(boundary_rows, root=tmp_path, run_dir="boundary", recipe=recipe)

    add_candidate(
        primary_rows,
        root=tmp_path,
        run_dir="primary",
        task_id="sobol_prompt_search_518_slot10",
        pocket="fresh24_orange_flowers",
        scores=(8.0, 8.0, 8.0),
        retained=False,
    )
    add_candidate(
        primary_rows,
        root=tmp_path,
        run_dir="primary",
        task_id="sobol_prompt_search_519_slot10",
        pocket="fresh24_orange_flowers",
        scores=(4.0, 4.0, 4.0),
    )
    add_candidate(
        primary_rows,
        root=tmp_path,
        run_dir="primary",
        task_id="sobol_prompt_search_518_slot12",
        pocket="fresh24_hanging_clothes",
        scores=(3.0, 3.0, 3.0),
    )
    add_candidate(
        boundary_rows,
        root=tmp_path,
        run_dir="boundary",
        task_id="sobol_prompt_search_518_slot18",
        pocket="fresh24_blue_jellyfish",
        scores=(2.0, 2.0, 2.0),
    )
    add_candidate(
        boundary_rows,
        root=tmp_path,
        run_dir="boundary",
        task_id="sobol_prompt_search_519_slot00",
        pocket="fresh24_old_car",
        scores=(1.0, 1.0, 1.0),
    )

    manifest, _task_doc, _summary = module.build_freeze(
        primary_report=write_report(tmp_path, "primary", primary_rows),
        boundary_report=write_report(tmp_path, "boundary", boundary_rows),
        primary_candidates_per_pocket=1,
        boundary_candidates_per_pocket=1,
        expected_replicates=3,
    )

    selected_tasks = {
        candidate["pocket"]: candidate["task_id"]
        for candidate in manifest["selected_candidates"]
    }
    assert selected_tasks["fresh24_orange_flowers"] == "sobol_prompt_search_519_slot10"
