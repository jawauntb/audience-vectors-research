from __future__ import annotations

import json

import numpy as np
import pytest

from audience_vectors.bo_replay import (
    load_collaborator_trials,
    policy_group_summary,
    replay_summary,
    replicate_summary,
    safe_label,
    score_projection,
    select_trials,
    stratum_policy_summary,
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


def test_select_trials_bo_vs_sobol_applies_budget_per_group(tmp_path):
    path = tmp_path / "select_baseline_trials.json"
    path.write_text(
        json.dumps(
            [
                {
                    "task_id": "sobol_000",
                    "alpha": 0,
                    "guidance": 1,
                    "seed_idx": 0,
                    "tribe_score": 2.0,
                },
                {
                    "task_id": "sobol_001",
                    "alpha": 0,
                    "guidance": 1,
                    "seed_idx": 0,
                    "tribe_score": 5.0,
                },
                {
                    "task_id": "bo01_cand00",
                    "alpha": 0,
                    "guidance": 1,
                    "seed_idx": 0,
                    "tribe_score": 4.0,
                },
                {
                    "task_id": "bo01_cand01",
                    "alpha": 0,
                    "guidance": 1,
                    "seed_idx": 0,
                    "tribe_score": 3.0,
                },
            ]
        )
    )
    trials = load_collaborator_trials(path)

    selected = select_trials(
        trials,
        selection="top-bo-vs-top-sobol",
        max_evals=1,
    )

    assert [trial.task_id for trial in selected] == ["bo01_cand00", "sobol_001"]


def test_select_trials_seed_stratified_bo_vs_sobol_uses_matched_strata(tmp_path):
    path = tmp_path / "select_seed_stratified_trials.json"
    path.write_text(
        json.dumps(
            [
                {
                    "task_id": "sobol_000",
                    "alpha": 0,
                    "guidance": 1,
                    "seed_idx": 0,
                    "prompt": "shared prompt a",
                    "tribe_score": 2.0,
                },
                {
                    "task_id": "bo01_cand00",
                    "alpha": 0,
                    "guidance": 1,
                    "seed_idx": 0,
                    "prompt": "shared prompt a",
                    "tribe_score": 4.0,
                },
                {
                    "task_id": "bo01_cand01",
                    "alpha": 0,
                    "guidance": 1,
                    "seed_idx": 0,
                    "prompt": "shared prompt a",
                    "tribe_score": 6.0,
                },
                {
                    "task_id": "bo02_cand00",
                    "alpha": 0,
                    "guidance": 1,
                    "seed_idx": 1,
                    "prompt": "bo only prompt",
                    "tribe_score": 10.0,
                },
                {
                    "task_id": "sobol_001",
                    "alpha": 0,
                    "guidance": 1,
                    "seed_idx": 2,
                    "prompt": "sobol only prompt",
                    "tribe_score": 9.0,
                },
            ]
        )
    )
    trials = load_collaborator_trials(path)

    selected = select_trials(
        trials,
        selection="seed-stratified-bo-vs-sobol",
        max_evals=1,
    )

    assert [trial.task_id for trial in selected] == ["bo01_cand01", "sobol_000"]


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


def test_replicate_summary_groups_scores_and_ranks_by_mean():
    summary = replicate_summary(
        [
            {
                "trial": {"task_id": "candidate_a", "tribe_score": 1.5},
                "replicate_index": 0,
                "noise_seed": 10,
                "replay_tribe_score": 1.0,
            },
            {
                "trial": {"task_id": "candidate_a", "tribe_score": 1.5},
                "replicate_index": 1,
                "noise_seed": 10010,
                "replay_tribe_score": 3.0,
            },
            {
                "trial": {"task_id": "candidate_b", "tribe_score": 0.25},
                "replicate_index": 0,
                "noise_seed": 22,
                "replay_tribe_score": 0.5,
            },
            {
                "trial": {"task_id": "candidate_b", "tribe_score": 0.25},
                "replicate_index": 1,
                "noise_seed": 10022,
            },
        ]
    )

    assert [item["task_id"] for item in summary] == ["candidate_a", "candidate_b"]
    assert summary[0]["rank_by_mean_replay_tribe_score"] == 1
    assert summary[0]["n_requested"] == 2
    assert summary[0]["n_scored"] == 2
    assert summary[0]["mean_replay_tribe_score"] == pytest.approx(2.0)
    assert summary[0]["std_replay_tribe_score"] == pytest.approx(np.sqrt(2.0))
    assert summary[0]["sem_replay_tribe_score"] == pytest.approx(1.0)
    assert summary[0]["mean_delta_vs_original"] == pytest.approx(0.5)
    assert summary[0]["noise_seeds"] == [10, 10010]
    assert summary[1]["n_requested"] == 2
    assert summary[1]["n_scored"] == 1
    assert summary[1]["std_replay_tribe_score"] == pytest.approx(0.0)


def test_policy_group_summary_compares_candidate_means():
    summary = policy_group_summary(
        [
            {
                "trial": {"task_id": "bo01_cand00", "tribe_score": 4.0},
                "original_tribe_score": 4.0,
                "replay_tribe_score": 2.0,
            },
            {
                "trial": {"task_id": "bo01_cand00", "tribe_score": 4.0},
                "original_tribe_score": 4.0,
                "replay_tribe_score": 4.0,
            },
            {
                "trial": {"task_id": "sobol_000", "tribe_score": 1.0},
                "original_tribe_score": 1.0,
                "replay_tribe_score": 1.0,
            },
            {
                "trial": {"task_id": "sobol_000", "tribe_score": 1.0},
                "original_tribe_score": 1.0,
                "replay_tribe_score": 2.0,
            },
        ]
    )

    assert [item["policy_group"] for item in summary] == ["bo", "sobol"]
    assert summary[0]["n_candidates"] == 1
    assert summary[0]["n_requested"] == 2
    assert summary[0]["n_scored"] == 2
    assert summary[0]["mean_candidate_replay_tribe_score"] == pytest.approx(3.0)
    assert summary[0]["pooled_mean_replay_tribe_score"] == pytest.approx(3.0)
    assert summary[0]["mean_original_tribe_score"] == pytest.approx(4.0)
    assert summary[0]["best_candidate_task_id"] == "bo01_cand00"
    assert summary[1]["mean_candidate_replay_tribe_score"] == pytest.approx(1.5)


def test_stratum_policy_summary_marks_matched_and_unmatched_groups():
    summary = stratum_policy_summary(
        [
            {
                "trial": {
                    "task_id": "bo01_cand00",
                    "prompt": "shared prompt",
                    "tribe_score": 4.0,
                },
                "original_tribe_score": 4.0,
                "replay_tribe_score": 2.0,
            },
            {
                "trial": {
                    "task_id": "bo01_cand00",
                    "prompt": "shared prompt",
                    "tribe_score": 4.0,
                },
                "original_tribe_score": 4.0,
                "replay_tribe_score": 4.0,
            },
            {
                "trial": {
                    "task_id": "sobol_000",
                    "prompt": "shared prompt",
                    "tribe_score": 1.0,
                },
                "original_tribe_score": 1.0,
                "replay_tribe_score": 1.0,
            },
            {
                "trial": {
                    "task_id": "sobol_001",
                    "prompt": "sobol only prompt",
                    "tribe_score": 5.0,
                },
                "original_tribe_score": 5.0,
                "replay_tribe_score": 5.0,
            },
        ]
    )

    shared = [item for item in summary if item["stratum_key"] == "shared prompt"]
    sobol_only = [
        item for item in summary if item["stratum_key"] == "sobol only prompt"
    ]

    assert [item["policy_group"] for item in shared] == ["bo", "sobol"]
    assert all(item["matched_bo_sobol"] for item in shared)
    assert shared[0]["mean_candidate_replay_tribe_score"] == pytest.approx(3.0)
    assert shared[0]["task_ids"] == ["bo01_cand00"]
    assert len(sobol_only) == 1
    assert sobol_only[0]["policy_group"] == "sobol"
    assert not sobol_only[0]["matched_bo_sobol"]
