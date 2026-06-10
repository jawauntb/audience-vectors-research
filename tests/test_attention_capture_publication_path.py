from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_attention_capture_publication_path.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_attention_capture_publication_path",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_readiness(path: Path, *, snapugc_ready: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "readiness": {
                    "phase1_can_run_now": True,
                    "snapugc_labels_ready": snapugc_ready,
                    "dhf1k_labels_ready": True,
                    "dhf1k_tribe_features_ready": True,
                    "real_manifest_ready": True,
                    "recommended_next_action": "run workflow",
                },
                "tribe_feature_dirs": [
                    {
                        "path": "/tmp/external/tribe_dhf1k_attention_audio_only",
                        "ready_as_feature_cache": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def write_workflow(
    path: Path,
    *,
    dataset: str,
    rho: float,
    passed: bool,
    audio_only: bool = False,
) -> None:
    path.write_text(
        json.dumps(
            {
                "manifest_path": (
                    "phase1_audio_only_manifest.json"
                    if audio_only
                    else "phase1_full_manifest.json"
                ),
                "score_decision": {"scoring_executed": True, "reason": "claim_ready"},
                "preflight": {"claim_ready": True},
                "primary_report": {
                    "n_samples": 350,
                    "n_invalid_capture_denominators": 0,
                    "gate_rho": 0.4,
                    "gate": {
                        "claim_validated": passed,
                        "passed": passed,
                        "rule": "capture_score Spearman rho >= 0.40",
                    },
                    "groups": [
                        {
                            "group": dataset,
                            "metrics": {
                                "capture_score": {
                                    "n": 350,
                                    "rho": rho,
                                    "permutation_p_greater": 0.01,
                                    "gate_passed": passed,
                                }
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def write_feature_cache_audit(
    path: Path,
    *,
    ready: bool = True,
    reproduction_ready: bool = False,
) -> None:
    path.write_text(
        json.dumps(
            {
                "feature_dir": "data/features/tribe_dhf1k_attention_audio_only",
                "ready_for_reuse": ready,
                "ready_for_reproduction": reproduction_ready,
                "archive_uri": None,
                "rerun_commands": (
                    [
                        "uv run python scripts/extract_attention_capture_tribe_features.py"
                    ]
                    if reproduction_ready
                    else []
                ),
                "n_npz_files": 516 if ready else 0,
                "n_expected_sample_ids": 516,
                "n_missing_expected_sample_ids": 0 if ready else 1,
                "n_bad_npz": 0,
                "n_shape_mismatches": 0,
                "aggregate_sha256": "a" * 64,
                "blocking_reasons": [] if ready else ["missing sample"],
            }
        ),
        encoding="utf-8",
    )


def test_publication_audit_blocks_current_failed_audio_only_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    for name in module.DEFAULT_TOKEN_ENVS:
        monkeypatch.delenv(name, raising=False)
    readiness = tmp_path / "readiness.json"
    workflow = tmp_path / "dhf1k_audio_only_workflow.json"
    write_readiness(readiness, snapugc_ready=False)
    write_workflow(workflow, dataset="DHF1K", rho=-0.03, passed=False, audio_only=True)

    report = module.build_publication_path_report(
        readiness_json=readiness,
        workflow_jsons=[workflow],
        min_paper_datasets=2,
    )

    assert report["publication_ready"] is False
    assert report["phase2_ready"] is False
    assert (
        "current H2 capture_score failed the Phase 1 rho gate"
        in report["blocking_reasons"]
    )
    assert (
        "no SnapUGC/VQualA retention label CSV is mounted" in report["blocking_reasons"]
    )
    assert any("audio-only" in reason for reason in report["blocking_reasons"])
    assert any("feature cache" in warning for warning in report["warnings"])


def test_publication_audit_records_feature_cache_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    for name in module.DEFAULT_TOKEN_ENVS:
        monkeypatch.delenv(name, raising=False)
    readiness = tmp_path / "readiness.json"
    workflow = tmp_path / "dhf1k_audio_only_workflow.json"
    cache = tmp_path / "feature_cache.json"
    write_readiness(readiness, snapugc_ready=False)
    write_workflow(workflow, dataset="DHF1K", rho=-0.03, passed=False, audio_only=True)
    write_feature_cache_audit(cache)

    report = module.build_publication_path_report(
        readiness_json=readiness,
        workflow_jsons=[workflow],
        feature_cache_audits=[cache],
        min_paper_datasets=2,
    )
    markdown = module.render_publication_markdown(report)

    assert report["feature_cache_audit_summaries"][0]["ready_for_reuse"] is True
    assert report["feature_cache_audit_summaries"][0]["ready_for_reproduction"] is False
    assert any("checksum provenance" in warning for warning in report["warnings"])
    assert not any(
        "archived or regenerated" in warning for warning in report["warnings"]
    )
    assert "Feature Cache Evidence" in markdown
    assert "data/features/tribe_dhf1k_attention_audio_only" in markdown
    assert "aaaaaaaaaaaa" in markdown


def test_publication_audit_accepts_feature_cache_rerun_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    for name in module.DEFAULT_TOKEN_ENVS:
        monkeypatch.delenv(name, raising=False)
    readiness = tmp_path / "readiness.json"
    workflow = tmp_path / "dhf1k_audio_only_workflow.json"
    cache = tmp_path / "feature_cache.json"
    write_readiness(readiness, snapugc_ready=False)
    write_workflow(workflow, dataset="DHF1K", rho=-0.03, passed=False, audio_only=True)
    write_feature_cache_audit(cache, reproduction_ready=True)

    report = module.build_publication_path_report(
        readiness_json=readiness,
        workflow_jsons=[workflow],
        feature_cache_audits=[cache],
        min_paper_datasets=2,
    )
    markdown = module.render_publication_markdown(report)

    cache_summary = report["feature_cache_audit_summaries"][0]
    assert cache_summary["ready_for_reuse"] is True
    assert cache_summary["ready_for_reproduction"] is True
    assert cache_summary["n_rerun_commands"] == 1
    assert not any("feature cache" in warning for warning in report["warnings"])
    assert "True | True | 516 | 516 | 1 | aaaaaaaaaaaa" in markdown


def test_publication_audit_accepts_multidataset_retention_and_token_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    monkeypatch.setenv("HF_TOKEN", "present-but-not-reported")
    readiness = tmp_path / "readiness.json"
    dhf1k = tmp_path / "dhf1k_workflow.json"
    snapugc = tmp_path / "snapugc_workflow.json"
    write_readiness(readiness, snapugc_ready=True)
    write_workflow(dhf1k, dataset="DHF1K", rho=0.45, passed=True)
    write_workflow(snapugc, dataset="SnapUGC", rho=0.48, passed=True)

    report = module.build_publication_path_report(
        readiness_json=readiness,
        workflow_jsons=[dhf1k, snapugc],
        min_paper_datasets=2,
    )

    assert report["publication_ready"] is True
    assert report["paper_claim_allowed"] is True
    assert report["phase2_ready"] is True
    assert report["blocking_reasons"] == []
    assert report["credential_audit"]["entries"][0]["present"] is True
    assert "present-but-not-reported" not in json.dumps(report)


def test_publication_markdown_renders_workflow_table(tmp_path: Path) -> None:
    module = load_module()
    workflow = tmp_path / "workflow.json"
    write_workflow(workflow, dataset="DHF1K", rho=0.12, passed=False)

    report = module.build_publication_path_report(
        readiness_json=None,
        workflow_jsons=[workflow],
        min_paper_datasets=1,
    )
    markdown = module.render_publication_markdown(report)

    assert "Attention-Capture Publication Path Audit" in markdown
    assert "Workflow Evidence" in markdown
    assert "0.1200" in markdown
