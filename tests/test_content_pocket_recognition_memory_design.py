from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_design_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_content_pocket_recognition_memory_design.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_content_pocket_recognition_memory_design",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_report(tmp_path: Path, label: str, *, alpha: float, guidance: float) -> Path:
    report = tmp_path / f"{label}.json"
    report.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "label": label,
                        "noise_seed": 123,
                        "trial": {
                            "alpha": alpha,
                            "guidance": guidance,
                            "prompt": "source prompt",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return report


def stimulus(
    *,
    tmp_path: Path,
    pocket: str,
    role: str,
    label: str,
    score: float,
    recipe: int,
    replicate: int,
) -> dict:
    source_report = write_report(tmp_path, label, alpha=1.5, guidance=6.5)
    return {
        "analysis_tier": "primary",
        "role": role,
        "pocket": pocket,
        "label": label,
        "local_video_path": f"data/generated/{label}.mp4",
        "source_absolute_path": str(tmp_path / f"{label}.mp4"),
        "sha256": f"sha-{label}",
        "recipe_index": recipe,
        "replicate_index": replicate,
        "replay_tribe_score": score,
        "source_report": str(source_report),
    }


def synthetic_manifest(tmp_path: Path) -> dict:
    rows = []
    specs = [
        ("fresh24_orange_flowers", "candidate", 4.0),
        ("fresh24_hanging_clothes", "candidate", 3.0),
        ("fresh24_aerial_beach", "control", -8.0),
        ("fresh24_city_street", "control", -9.0),
        ("fresh24_storm_beach", "control", -10.0),
    ]
    for pocket, role, base_score in specs:
        for index in range(4):
            rows.append(
                stimulus(
                    tmp_path=tmp_path,
                    pocket=pocket,
                    role=role,
                    label=f"{pocket}_{index}",
                    score=base_score + index,
                    recipe=519 + index,
                    replicate=index % 3,
                )
            )
    return {
        "task_pool": {"task_payload_sha256": "freeze-sha"},
        "stimuli": rows,
    }


def test_select_old_targets_keeps_three_variants_per_arm(tmp_path):
    module = load_design_module()
    targets = module.select_old_targets(synthetic_manifest(tmp_path), variants_per_arm=3)

    assert set(targets) == {
        "orange_flowers",
        "hanging_clothes",
        "aerial_beach",
        "city_street",
        "storm_beach",
    }
    assert all(len(arm_targets) == 3 for arm_targets in targets.values())
    assert targets["orange_flowers"][0]["old_video"]["replay_tribe_score"] == 7.0
    assert targets["storm_beach"][0]["old_video"]["replay_tribe_score"] == -10.0
    assert targets["orange_flowers"][0]["old_video"]["source_trial"]["alpha"] == 1.5


def test_lure_requests_require_new_seed_images(tmp_path):
    module = load_design_module()
    targets = module.select_old_targets(synthetic_manifest(tmp_path), variants_per_arm=2)
    requests = module.build_lure_requests(
        targets,
        lure_seed_dir=Path("lure_seeds"),
        noise_seed_base=700000,
    )

    assert len(requests) == 10
    assert all(request["seed_image_status"] == "required_not_committed" for request in requests)
    assert all("lure_seeds" in request["seed_image_required_path"] for request in requests)
    assert all(request["generation_request"]["must_not_optimize_lure_for_memorability"] for request in requests)


def test_session_forms_are_sparse_and_balanced(tmp_path):
    module = load_design_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(synthetic_manifest(tmp_path)), encoding="utf-8")

    design, markdown = module.build_design(
        stimulus_manifest_path=manifest_path,
        variants_per_arm=3,
        n_forms=6,
        session1_filler_count=25,
        session2_filler_recognition_trials=20,
        lure_seed_dir=Path("lure_seeds"),
        noise_seed_base=700000,
    )

    assert design["status"] == "design_not_launchable_until_lures_generated_screened_and_hosted"
    assert len(design["session_forms"]) == 6
    for form in design["session_forms"]:
        arms = [target["arm_id"] for target in form["session1"]["analysis_encoding_targets"]]
        assert len(arms) == len(set(arms)) == 5
        assert len(form["session2"]["analysis_recognition_trials"]) == 5
    assert design["sample_size_plan"]["target_session2_usable"] == 300
    assert "old-vs-lure" in markdown
