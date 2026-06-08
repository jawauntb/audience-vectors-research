from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_prompt_transfer_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_bo_prompt_transfer_manifest.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_bo_prompt_transfer_manifest",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_seed_root(tmp_path: Path) -> Path:
    seed_root = tmp_path / "original"
    seed_dir = seed_root / "seeds"
    seed_dir.mkdir(parents=True)
    rows = [
        {
            "idx": 10,
            "bmd_name": "seed_a",
            "prompt": "prompt a",
            "seed_image": "seeds/a.png",
        },
        {
            "idx": 11,
            "bmd_name": "seed_b",
            "prompt": "prompt b",
            "seed_image": "seeds/b.png",
        },
    ]
    for image_name in ["a.png", "b.png"]:
        (seed_dir / image_name).write_bytes(b"placeholder")
    (seed_dir / "prompts.json").write_text(json.dumps(rows))
    return seed_root


def write_source_trials(tmp_path: Path) -> Path:
    trial_table = tmp_path / "source_trials.json"
    trial_table.write_text(
        json.dumps(
            {
                "all_meta": [
                    {
                        "task_id": "bo_low",
                        "alpha": -1.0,
                        "guidance": 3.0,
                        "seed_idx": 0,
                        "noise_seed": 100,
                        "prompt": "source",
                        "tribe_score": 1.0,
                    },
                    {
                        "task_id": "bo_high",
                        "alpha": 2.0,
                        "guidance": 7.0,
                        "seed_idx": 1,
                        "noise_seed": 200,
                        "prompt": "source",
                        "tribe_score": 5.0,
                    },
                    {
                        "task_id": "sobol_000",
                        "alpha": 0.0,
                        "guidance": 5.0,
                        "seed_idx": 0,
                        "noise_seed": 0,
                        "prompt": "source",
                        "tribe_score": 4.0,
                    },
                ]
            }
        )
    )
    return trial_table


def test_prompt_transfer_manifest_retargets_top_bo_and_controls(tmp_path):
    module = load_prompt_transfer_module()
    manifest = module.build_prompt_transfer_manifest(
        source_trial_table=write_source_trials(tmp_path),
        seed_root=write_seed_root(tmp_path),
        anchor_task_ids=[],
        top_bo_anchors=1,
        target_seed_slots=None,
        replay_seed_pool_size=4,
        sobol_controls_per_seed=2,
        sobol_start_index=8,
        sobol_scramble_seed=42,
    )

    assert manifest["anchor_task_ids"] == ["bo_high"]
    assert [slot["slot"] for slot in manifest["target_seed_slots"]] == [0, 1]
    assert manifest["n_bo_transfer_trials"] == 2
    assert manifest["n_sobol_transfer_trials"] == 4

    rows = manifest["all_meta"]
    bo_rows = [row for row in rows if row["policy"] == "bo_transfer"]
    sobol_rows = [row for row in rows if row["policy"] == "sobol_transfer"]

    assert {row["prompt"] for row in bo_rows} == {"prompt a", "prompt b"}
    assert {row["alpha"] for row in bo_rows} == {2.0}
    assert {row["guidance"] for row in bo_rows} == {7.0}
    assert {row["transfer_source_task_id"] for row in bo_rows} == {"bo_high"}

    assert len(sobol_rows) == 4
    assert {row["prompt"] for row in sobol_rows} == {"prompt a", "prompt b"}
    assert {row["task_id"].split("_slot")[0] for row in sobol_rows} == {
        "sobol_transfer_008",
        "sobol_transfer_009",
    }
