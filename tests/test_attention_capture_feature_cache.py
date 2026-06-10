from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_attention_capture_feature_cache.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_attention_capture_feature_cache",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_feature(
    path: Path,
    sample_id: str,
    *,
    vertices: int = 4,
    media_path: str | None = None,
) -> None:
    np.savez_compressed(
        path,
        frames=np.zeros((2, vertices), dtype=np.float32),
        duration_seconds=np.array(1.25, dtype=np.float32),
        sample_id=np.array(sample_id),
        media_path=np.array(media_path or f"video/{sample_id}.AVI"),
        transport=np.array("bytes"),
        event_mode=np.array("audio-only"),
    )


def write_manifest(path: Path, sample_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {"sample_id": sample_id, "tribe_feature_path": f"{sample_id}.npz"}
                    for sample_id in sample_ids
                ]
            }
        ),
        encoding="utf-8",
    )


def test_feature_cache_audit_accepts_complete_cache(tmp_path: Path) -> None:
    module = load_module()
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    write_feature(feature_dir / "dhf1k_001.npz", "dhf1k_001")
    write_feature(feature_dir / "dhf1k_002.npz", "dhf1k_002")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, ["dhf1k_001", "dhf1k_002"])

    report = module.audit_feature_cache(
        feature_dir=feature_dir,
        display_feature_dir="data/features/tribe_dhf1k_attention_audio_only",
        manifest_paths=[manifest],
        expected_vertices=4,
    )

    assert report["schema_version"] == 2
    assert report["ready_for_reuse"] is True
    assert report["ready_for_reproduction"] is False
    assert report["feature_dir"] == "data/features/tribe_dhf1k_attention_audio_only"
    assert report["n_npz_files"] == 2
    assert report["n_expected_sample_ids"] == 2
    assert report["n_missing_expected_sample_ids"] == 0
    assert report["event_mode_counts"] == {"audio-only": 2}
    assert report["transport_counts"] == {"bytes": 2}
    assert report["frame_shape_counts"] == {"2x4": 2}
    assert len(report["aggregate_sha256"]) == 64
    assert report["files"][0]["path"] == "dhf1k_001.npz"
    assert report["files"][0]["media_path"] == "video/dhf1k_001.AVI"
    markdown = module.render_feature_cache_markdown(report)
    assert "Ready for reuse: True" in markdown
    assert "Ready for reproduction: False" in markdown
    assert "audio-only=2" in markdown


def test_feature_cache_audit_records_deterministic_rerun_path(tmp_path: Path) -> None:
    module = load_module()
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    write_feature(feature_dir / "dhf1k_001.npz", "dhf1k_001")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, ["dhf1k_001"])

    report = module.audit_feature_cache(
        feature_dir=feature_dir,
        manifest_paths=[manifest],
        expected_vertices=4,
        rerun_commands=[
            "uv run python scripts/extract_attention_capture_tribe_features.py"
        ],
    )
    markdown = module.render_feature_cache_markdown(report)

    assert report["ready_for_reuse"] is True
    assert report["ready_for_reproduction"] is True
    assert report["archive_uri"] is None
    assert report["rerun_commands"] == [
        "uv run python scripts/extract_attention_capture_tribe_features.py"
    ]
    assert "Ready for reproduction: True" in markdown
    assert "extract_attention_capture_tribe_features.py" in markdown


def test_feature_cache_audit_normalizes_absolute_media_metadata(tmp_path: Path) -> None:
    module = load_module()
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    write_feature(
        feature_dir / "dhf1k_001.npz",
        "dhf1k_001",
        media_path=(
            "/Users/example/.codex/worktrees/old/isc_mod/"
            "data/attention_capture/DHF1K/video/001.AVI"
        ),
    )

    report = module.audit_feature_cache(
        feature_dir=feature_dir,
        expected_vertices=4,
    )

    assert report["files"][0]["media_path"] == (
        "data/attention_capture/DHF1K/video/001.AVI"
    )


def test_feature_cache_audit_blocks_missing_manifest_sample(tmp_path: Path) -> None:
    module = load_module()
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    write_feature(feature_dir / "dhf1k_001.npz", "dhf1k_001")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, ["dhf1k_001", "dhf1k_002"])

    report = module.audit_feature_cache(
        feature_dir=feature_dir,
        manifest_paths=[manifest],
        expected_vertices=4,
    )

    assert report["ready_for_reuse"] is False
    assert report["n_missing_expected_sample_ids"] == 1
    assert report["missing_expected_sample_ids"] == ["dhf1k_002"]
    assert "1 manifest sample ids are missing" in report["blocking_reasons"]


def test_feature_cache_audit_blocks_shape_mismatch(tmp_path: Path) -> None:
    module = load_module()
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    write_feature(feature_dir / "dhf1k_001.npz", "dhf1k_001", vertices=3)

    report = module.audit_feature_cache(
        feature_dir=feature_dir,
        expected_vertices=4,
    )

    assert report["ready_for_reuse"] is False
    assert report["n_shape_mismatches"] == 1
    assert "1 NPZ files have unexpected frames shape" in report["blocking_reasons"]
