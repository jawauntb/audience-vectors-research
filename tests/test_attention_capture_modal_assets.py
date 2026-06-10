from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_attention_capture_modal_assets.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_attention_capture_modal_assets",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_modal_unblock_summary_blocks_without_retention_labels_or_token() -> None:
    module = load_module()
    volume_report = {
        "audits": [
            {
                "label_candidates": [],
                "dataset_candidates": [
                    {"path": "activation_geometry", "claim_blocked": False}
                ],
                "feature_candidates": [],
            }
        ]
    }
    secret_report = {"any_present": False}

    summary = module.summarize_publication_unblocks(
        volume_report=volume_report,
        secret_report=secret_report,
    )

    assert summary["retention_labels_maybe_available"] is False
    assert summary["external_dataset_dirs_maybe_available"] is True
    assert summary["full_multimodal_token_env_present"] is False
    assert summary["blocking_reasons"] == [
        "no Modal-hosted SnapUGC/VQualA retention label candidate found",
        "no Modal secret exposes a HuggingFace token env name",
    ]


def test_modal_unblock_summary_ignores_claim_blocked_labels() -> None:
    module = load_module()
    volume_report = {
        "audits": [
            {
                "label_candidates": [
                    {
                        "path": "fixtures/synthetic_ecr_labels.csv",
                        "claim_blocked": True,
                    }
                ],
                "dataset_candidates": [],
                "feature_candidates": [],
            }
        ]
    }
    secret_report = {"any_present": True}

    summary = module.summarize_publication_unblocks(
        volume_report=volume_report,
        secret_report=secret_report,
    )

    assert summary["retention_labels_maybe_available"] is False
    assert summary["full_multimodal_token_env_present"] is True
    assert summary["blocking_reasons"] == [
        "no Modal-hosted SnapUGC/VQualA retention label candidate found"
    ]


def test_modal_unblock_summary_accepts_retention_label_and_token() -> None:
    module = load_module()
    volume_report = {
        "audits": [
            {
                "label_candidates": [
                    {
                        "path": "snapugc/ecr_labels.csv",
                        "claim_blocked": False,
                    }
                ],
                "dataset_candidates": [],
                "feature_candidates": [
                    {
                        "path": "snapugc/tribe_features",
                        "claim_blocked": False,
                    }
                ],
            }
        ]
    }
    secret_report = {"any_present": True}

    summary = module.summarize_publication_unblocks(
        volume_report=volume_report,
        secret_report=secret_report,
    )

    assert summary["retention_labels_maybe_available"] is True
    assert summary["feature_caches_maybe_available"] is True
    assert summary["blocking_reasons"] == []
