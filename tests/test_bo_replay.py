from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

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


def load_modal_replay_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "modal_bo_memorability_replay.py"
    )
    spec = importlib.util.spec_from_file_location("modal_bo_memorability_replay", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_validate_run_inputs_requires_artifacts_when_requested(tmp_path):
    trial_table = tmp_path / "trials.json"
    seed_root = tmp_path / "seed_root"
    trial_table.write_text("[]")
    seed_root.mkdir()
    args = Namespace(
        trial_table=trial_table,
        seed_root=seed_root,
        steering_artifact=None,
        cortical_vmem=tmp_path / "missing_v_mem.npz",
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        load_modal_replay_module().validate_run_inputs(args, require_artifacts=True)

    message = str(excinfo.value)
    assert "BO_MEM_STEERING_ARTIFACT" in message
    assert "BO_MEM_CORTICAL_VMEM" in message


def test_parse_args_exposes_svd_generation_controls(monkeypatch):
    module = load_modal_replay_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modal_bo_memorability_replay.py",
            "--svd-num-frames",
            "14",
            "--num-inference-steps",
            "50",
            "--svd-motion-bucket-id",
            "40",
            "--svd-noise-aug-strength",
            "0",
            "--svd-fps",
            "11",
            "--visual-first-retention",
            "complete-candidates",
        ],
    )

    args = module.parse_args()

    assert args.svd_num_frames == 14
    assert args.num_inference_steps == 50
    assert args.svd_motion_bucket_id == 40
    assert args.svd_noise_aug_strength == pytest.approx(0.0)
    assert args.svd_fps == 11
    assert args.visual_first_retention == "complete-candidates"


def test_apply_visual_first_retention_requires_complete_candidate_replicates():
    module = load_modal_replay_module()
    rows = [
        {
            "label": "candidate_a_rep00",
            "trial": {"task_id": "candidate_a"},
            "visual_artifact_gate": {"passes_visual_gate": True},
        },
        {
            "label": "candidate_a_rep01",
            "trial": {"task_id": "candidate_a"},
            "visual_artifact_gate": {"passes_visual_gate": True},
        },
        {
            "label": "candidate_b_rep00",
            "trial": {"task_id": "candidate_b"},
            "visual_artifact_gate": {"passes_visual_gate": True},
        },
        {
            "label": "candidate_b_rep01",
            "trial": {"task_id": "candidate_b"},
            "visual_artifact_gate": {
                "passes_visual_gate": False,
                "artifact_flags": ["tail_sharpness_collapse"],
            },
        },
    ]

    summary = module.apply_visual_first_retention(
        rows,
        mode="complete-candidates",
    )

    assert summary["n_retained_rows"] == 2
    assert summary["n_withheld_rows"] == 2
    assert summary["retained_task_ids"] == ["candidate_a"]
    assert summary["withheld_task_ids"] == ["candidate_b"]
    assert rows[0]["visual_first_status"] == "retained"
    assert rows[2]["visual_first_status"] == "withheld_candidate_has_visual_failure"
    assert rows[3]["visual_first_status"] == "withheld_visual_failure"
    assert summary["withheld_failures"][-1]["artifact_flags"] == [
        "tail_sharpness_collapse"
    ]


def test_apply_visual_first_retention_can_keep_only_passing_videos():
    module = load_modal_replay_module()
    rows = [
        {
            "label": "candidate_a_rep00",
            "trial": {"task_id": "candidate_a"},
            "visual_artifact_gate": {"passes_visual_gate": True},
        },
        {
            "label": "candidate_a_rep01",
            "trial": {"task_id": "candidate_a"},
            "visual_artifact_gate": {"passes_visual_gate": False},
        },
    ]

    summary = module.apply_visual_first_retention(rows, mode="passing-videos")

    assert summary["n_retained_rows"] == 1
    assert summary["n_withheld_rows"] == 1
    assert rows[0]["visual_first_retained"]
    assert not rows[1]["visual_first_retained"]


def test_attach_visual_artifact_gate_summarizes_generated_rows(monkeypatch, tmp_path):
    module = load_modal_replay_module()
    passing_video = tmp_path / "pass.mp4"
    failing_video = tmp_path / "fail.mp4"
    rows: list[dict[str, Any]] = [
        {"label": "passing", "local_video_path": str(passing_video)},
        {"label": "failing", "local_video_path": str(failing_video)},
        {"label": "not-generated"},
    ]

    def fake_summarize_video(path, *, samples, thresholds):
        del thresholds
        passes = Path(path).name == "pass.mp4"
        return {
            "video_path": str(path),
            "sample_count": samples,
            "artifact_flags": [] if passes else ["tail_contrast_collapse"],
            "passes_visual_gate": passes,
        }

    monkeypatch.setattr(module, "summarize_video", fake_summarize_video)

    summary = module.attach_visual_artifact_gate(
        rows,
        samples=5,
        thresholds=module.ArtifactThresholds(),
    )

    assert summary["n_videos"] == 2
    assert summary["n_failed"] == 1
    assert not summary["passes_visual_gate"]
    assert summary["failures"] == [
        {
            "label": "failing",
            "video_path": str(failing_video),
            "artifact_flags": ["tail_contrast_collapse"],
            "error": None,
        }
    ]
    assert rows[0]["visual_artifact_gate"]["passes_visual_gate"]
    assert not rows[1]["visual_artifact_gate"]["passes_visual_gate"]
    assert "visual_artifact_gate" not in rows[2]


def test_attach_visual_artifact_gate_marks_decode_errors(monkeypatch, tmp_path):
    module = load_modal_replay_module()
    video_path = tmp_path / "broken.mp4"
    rows: list[dict[str, Any]] = [
        {"label": "broken", "local_video_path": str(video_path)}
    ]

    def fake_summarize_video(path, *, samples, thresholds):
        del path, samples, thresholds
        raise ValueError("cannot decode")

    monkeypatch.setattr(module, "summarize_video", fake_summarize_video)

    summary = module.attach_visual_artifact_gate(
        rows,
        samples=3,
        thresholds=module.ArtifactThresholds(),
    )

    assert summary["n_videos"] == 1
    assert summary["n_failed"] == 1
    assert not summary["passes_visual_gate"]
    assert rows[0]["visual_artifact_gate"]["artifact_flags"] == ["visual_gate_error"]
    assert "cannot decode" in rows[0]["visual_artifact_gate"]["error"]


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
