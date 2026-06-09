from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_analysis_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analyze_content_pocket_recognition_responses.py"
    )
    spec = importlib.util.spec_from_file_location(
        "analyze_content_pocket_recognition_responses",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARMS = [
    ("orange_flowers", "primary_positive"),
    ("hanging_clothes", "primary_positive"),
    ("aerial_beach", "hard_negative_control"),
    ("city_street", "hard_negative_control"),
    ("storm_beach", "hard_negative_control"),
]


def session1_payload(pid: str, *, form_id: str = "form_00") -> dict:
    return {
        "session_number": 1,
        "participant_id": pid,
        "prolific_pid": pid,
        "form_id": form_id,
        "completed_at": "2026-06-08T12:00:00Z",
        "responses": [
            {
                "trial_id": f"{arm}_encoding",
                "target_id": arm,
                "arm_id": arm,
                "analysis_group": group,
                "media_error": False,
                "exposure_completed": True,
            }
            for arm, group in ARMS
        ],
    }


def session2_payload(
    pid: str,
    *,
    form_id: str = "form_00",
    primary_correct: bool = True,
    hard_correct: bool = False,
    media_error_arm: str | None = None,
) -> dict:
    rows = []
    for arm, group in ARMS:
        is_correct = primary_correct if group == "primary_positive" else hard_correct
        rows.append(
            {
                "trial_id": f"{arm}_recognition",
                "target_id": arm,
                "arm_id": arm,
                "analysis_group": group,
                "old_side": "left",
                "choice_side": "left" if is_correct else "right",
                "correct_choice": "left",
                "is_correct": is_correct,
                "left_media_error": media_error_arm == arm,
                "right_media_error": False,
                "any_media_error": media_error_arm == arm,
            }
        )
    return {
        "session_number": 2,
        "participant_id": pid,
        "prolific_pid": pid,
        "form_id": form_id,
        "completed_at": "2026-06-09T12:00:00Z",
        "responses": rows,
    }


def test_analysis_passes_when_primary_beats_hard_controls():
    module = load_analysis_module()
    payloads = []
    for index in range(10):
        pid = f"P{index:03d}"
        payloads.extend([session1_payload(pid), session2_payload(pid)])

    report = module.analyze_payloads(
        payloads,
        module.AnalysisConfig(min_usable_participants=5),
    )

    assert report["gate"]["status"] == "passed"
    assert report["gate"]["usable_participants"] == 10
    assert report["summaries"]["by_analysis_group"]["primary_positive"]["accuracy"] == 1.0
    assert report["summaries"]["by_analysis_group"]["hard_negative_control"]["accuracy"] == 0.0
    assert report["gate"]["criteria"]["pooled_primary_vs_hard_negative_p_lt_0_05"]


def test_media_error_is_reason_coded_and_blocks_complete_case_gate():
    module = load_analysis_module()
    payloads = [
        session1_payload("P001"),
        session2_payload("P001", media_error_arm="orange_flowers"),
    ]

    report = module.analyze_payloads(
        payloads,
        module.AnalysisConfig(min_usable_participants=1),
    )

    assert report["gate"]["status"] == "underpowered_do_not_interpret"
    assert report["gate"]["usable_participants"] == 0
    assert report["exclusion_counts"]["session2_media_error"] == 1
    assert report["exclusion_counts"]["participant_incomplete_analysis_set"] == 5


def test_loader_accepts_wrapped_csv_payload_and_manual_prefix_exclusion(tmp_path):
    module = load_analysis_module()
    csv_path = tmp_path / "responses.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["response_json"])
        writer.writeheader()
        writer.writerow({"response_json": json.dumps(session1_payload("TEST_001"))})
        writer.writerow({"response_json": json.dumps(session2_payload("TEST_001"))})

    payloads = module.load_payloads([csv_path])
    report = module.analyze_payloads(
        payloads,
        module.AnalysisConfig(
            min_usable_participants=1,
            exclude_participant_prefixes=("TEST_",),
        ),
    )

    assert len(payloads) == 2
    assert report["gate"]["usable_participants"] == 0
    assert report["exclusion_counts"]["manual_participant_prefix_exclusion"] == 5


def test_loader_rejects_missing_response_path(tmp_path):
    module = load_analysis_module()

    with pytest.raises(FileNotFoundError):
        module.load_payloads([tmp_path / "missing_export"])
