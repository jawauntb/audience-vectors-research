from __future__ import annotations

import json

import numpy as np
import pytest

from audience_vectors.bo_replay import (
    load_collaborator_trials,
    replay_summary,
    safe_label,
    score_projection,
    select_trials,
)


def test_load_collaborator_trials_from_all_meta(tmp_path):
    path = tmp_path / "all_results.json"
    path.write_text(
        json.dumps(
            {
                "all_meta": [
                    {
                        "task_id": "sobol_000",
                        "alpha": 1.5,
                        "guidance": 3.25,
                        "seed_idx": 4,
                        "noise_seed": 99,
                        "filename": "trial.mp4",
                        "prompt": "prompt",
                        "tribe_score": 0.7,
                        "clip_score": 0.2,
                        "quality_score": 0.8,
                    }
                ]
            }
        )
    )

    trials = load_collaborator_trials(path)

    assert len(trials) == 1
    assert trials[0].task_id == "sobol_000"
    assert trials[0].alpha == 1.5
    assert trials[0].guidance == 3.25
    assert trials[0].quality_score == 0.8


def test_select_trials_by_top_scores(tmp_path):
    path_trials = [
        {
            "task_id": "a",
            "alpha": 0,
            "guidance": 1,
            "seed_idx": 0,
            "tribe_score": -1,
            "clip_score": 0.9,
            "quality_score": 0.1,
        },
        {
            "task_id": "b",
            "alpha": 0,
            "guidance": 1,
            "seed_idx": 0,
            "tribe_score": 3,
            "clip_score": 0.1,
            "quality_score": 0.4,
        },
        {
            "task_id": "c",
            "alpha": 0,
            "guidance": 1,
            "seed_idx": 0,
            "tribe_score": 2,
            "clip_score": 0.5,
            "quality_score": 0.8,
        },
    ]
    path = tmp_path / "select_trials.json"
    path.write_text(json.dumps(path_trials))
    trials = load_collaborator_trials(path)

    assert [trial.task_id for trial in select_trials(trials, selection="top-tribe")] == [
        "b",
        "c",
    ]
    assert select_trials(trials, task_ids={"a"})[0].task_id == "a"


def test_score_projection_means_frames():
    frames = np.asarray([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    direction = np.asarray([0.6, 0.8], dtype=np.float32)

    assert score_projection(frames, direction) == pytest.approx(2.8)


def test_safe_label_and_replay_summary():
    assert safe_label(" bo 07 / cand 01 ") == "bo_07_cand_01"

    summary = replay_summary(
        [
            {"original_tribe_score": 1.0, "replay_tribe_score": 1.25},
            {"original_tribe_score": 2.0, "replay_tribe_score": 1.75},
            {"original_tribe_score": 3.0},
        ]
    )

    assert summary["n_requested"] == 3
    assert summary["n_scored"] == 2
    assert summary["mean_score_delta_vs_original"] == pytest.approx(0.0)
    assert summary["max_abs_score_delta_vs_original"] == pytest.approx(0.25)
