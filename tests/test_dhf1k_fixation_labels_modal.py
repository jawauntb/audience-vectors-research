from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_dhf1k_fixation_labels_modal.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_dhf1k_fixation_labels_modal",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_portable_video_path_preserves_relative_mount_path() -> None:
    module = load_module()

    path = Path("data/attention_capture/DHF1K/video/003.AVI")

    assert module.portable_video_path(path) == "data/attention_capture/DHF1K/video/003.AVI"


def test_select_extreme_tails_for_fixation_density() -> None:
    module = load_module()
    rows = [
        {"sample_id": "dhf1k_001", "mean_fixation_density": 0.1},
        {"sample_id": "dhf1k_002", "mean_fixation_density": 0.2},
        {"sample_id": "dhf1k_003", "mean_fixation_density": 0.8},
        {"sample_id": "dhf1k_004", "mean_fixation_density": 0.9},
    ]

    selected = module.select_extreme_tails(
        rows,
        rank_column="mean_fixation_density",
        count_per_tail=1,
    )

    assert [(row["sample_id"], row["selected_tail"]) for row in selected] == [
        ("dhf1k_001", "low"),
        ("dhf1k_004", "high"),
    ]
