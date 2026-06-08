from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_audit_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_content_pocket_features.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_content_pocket_features",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_visual_descriptors_detect_color_regions():
    module = load_audit_module()
    orange = np.zeros((12, 12, 3), dtype=np.float32)
    orange[..., 0] = 1.0
    orange[..., 1] = 0.45
    descriptors = module.visual_descriptors(orange)

    assert descriptors["orange_fraction"] > 0.95
    assert descriptors["cyan_blue_fraction"] == 0.0
    assert descriptors["warm_minus_cool"] > 0.9


def test_aggregate_candidates_averages_replicates():
    module = load_audit_module()
    rows = [
        {
            "task_id": "candidate_a",
            "pocket": "fresh24_orange_flowers",
            "label": "positive",
            "seed_idx": 10,
            "alpha": 1.0,
            "guidance": 2.0,
            "replay_tribe_score": 3.0,
            "seed_features": {"colorfulness": 0.8},
            "video_features": {"colorfulness": 0.7},
        },
        {
            "task_id": "candidate_a",
            "pocket": "fresh24_orange_flowers",
            "label": "positive",
            "seed_idx": 10,
            "alpha": 1.0,
            "guidance": 2.0,
            "replay_tribe_score": 5.0,
            "seed_features": {"colorfulness": 0.8},
            "video_features": {"colorfulness": 0.9},
        },
    ]

    [candidate] = module.aggregate_candidates(rows)

    assert candidate["n_replicates"] == 2
    assert candidate["mean_replay_tribe_score"] == 4.0
    assert candidate["seed_features"]["colorfulness"] == 0.8
    assert candidate["video_features"]["colorfulness"] == 0.8


def test_feature_gate_accepts_strong_descriptor_separator():
    module = load_audit_module()
    candidates = []
    for value in [0.75, 0.8, 0.85, 0.9]:
        candidates.append(
            {
                "label": "positive",
                "mean_replay_tribe_score": value * 10.0,
                "seed_features": {"colorfulness": value},
                "video_features": {},
            }
        )
    for value in [0.05, 0.1, 0.15]:
        candidates.append(
            {
                "label": "negative_control",
                "mean_replay_tribe_score": -value * 10.0,
                "seed_features": {"colorfulness": value},
                "video_features": {},
            }
        )

    metrics = module.analyze_feature_family(candidates, family="seed")
    gate = module.gate_summary(metrics, [], min_auc=0.85, min_abs_d=1.0)

    assert metrics[0]["feature"] == "colorfulness"
    assert metrics[0]["separation_auc"] == 1.0
    assert gate["accepted"] is True
