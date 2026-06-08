from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_prompt_search_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_bo_prompt_search_manifest.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_bo_prompt_search_manifest",
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


def test_prompt_search_manifest_balances_sobol_samples_per_available_seed(tmp_path):
    module = load_prompt_search_module()
    manifest = module.build_prompt_search_manifest(
        seed_root=write_seed_root(tmp_path),
        target_seed_slots=None,
        replay_seed_pool_size=4,
        sobol_samples_per_seed=3,
        sobol_start_index=16,
        sobol_scramble_seed=42,
        alpha_range=(-4.0, 4.0),
        guidance_range=(2.0, 8.0),
    )

    assert manifest["kind"] == "bo_prompt_search_trial_table"
    assert [slot["slot"] for slot in manifest["target_seed_slots"]] == [0, 1]
    assert manifest["sobol_samples_per_seed"] == 3
    assert manifest["n_sobol_prompt_search_trials"] == 6

    rows = manifest["all_meta"]
    assert {row["prompt"] for row in rows} == {"prompt a", "prompt b"}
    assert {row["policy"] for row in rows} == {"sobol_prompt_search"}
    assert {row["task_id"].split("_slot")[0] for row in rows} == {
        "sobol_prompt_search_016",
        "sobol_prompt_search_017",
        "sobol_prompt_search_018",
    }
    assert all(-4.0 <= row["alpha"] <= 4.0 for row in rows)
    assert all(2.0 <= row["guidance"] <= 8.0 for row in rows)
