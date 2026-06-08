from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_dhf1k_attention_labels.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_dhf1k_attention_labels",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_png(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8), mode="L").save(path)


def make_fake_dhf1k_video(root: Path, video_id: str, map_value: int) -> None:
    video_dir = root / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / f"{video_id}.AVI").write_bytes(b"fake video")
    maps_dir = root / "annotation" / video_id / "maps"
    fixation_dir = root / "annotation" / video_id / "fixation"
    write_png(maps_dir / "0001.png", np.full((4, 4), map_value))
    fixation = np.zeros((4, 4), dtype=np.uint8)
    fixation[1, 1] = 255
    write_png(fixation_dir / "0001.png", fixation)


def test_build_rows_reads_dhf1k_maps_and_video_paths(tmp_path: Path) -> None:
    module = load_module()
    make_fake_dhf1k_video(tmp_path, "001", 32)
    make_fake_dhf1k_video(tmp_path, "002", 128)

    rows = module.build_rows(dhf1k_root=tmp_path, split="train", limit=2)

    assert [row.sample_id for row in rows] == ["dhf1k_001", "dhf1k_002"]
    assert rows[0].split == "train"
    assert rows[0].n_map_frames == 1
    assert rows[0].n_fixation_frames == 1
    assert rows[1].mean_map_intensity > rows[0].mean_map_intensity
    assert rows[0].video_path.endswith("001.AVI")


def test_select_extreme_tails_labels_high_and_low_rows(tmp_path: Path) -> None:
    module = load_module()
    for index, value in enumerate((16, 64, 128, 240), start=1):
        make_fake_dhf1k_video(tmp_path, f"{index:03d}", value)
    rows = module.build_rows(dhf1k_root=tmp_path, split="train", limit=4)

    selected = module.select_extreme_tails(
        rows,
        rank_column="mean_map_intensity",
        count_per_tail=1,
    )

    assert [(row.sample_id, row.selected_tail) for row in selected] == [
        ("dhf1k_001", "low"),
        ("dhf1k_004", "high"),
    ]
    audit = module.summarize_rows(
        selected,
        dhf1k_root=tmp_path,
        split="train",
        rank_column="mean_map_intensity",
        extreme_count_per_tail=1,
        min_rows=2,
        min_distinct_rank_values=2,
    )
    assert audit["n_rows"] == 2
    assert audit["experiment"] == "dhf1k_attention_label_audit"
    assert audit["labels_csv"] is None
    assert audit["metrics"]["mean_map_intensity"]["n"] == 2
    assert audit["ready_for_manifest_alignment"] is True
    assert audit["recommended_ground_truth_column"] == "mean_map_intensity"


def test_dhf1k_audit_blocks_degenerate_rank_column(tmp_path: Path) -> None:
    module = load_module()
    for index in range(1, 4):
        make_fake_dhf1k_video(tmp_path, f"{index:03d}", 64)
    rows = module.build_rows(dhf1k_root=tmp_path, split="train", limit=3)

    audit = module.summarize_rows(
        rows,
        dhf1k_root=tmp_path,
        split="train",
        rank_column="mean_map_intensity",
        extreme_count_per_tail=None,
        min_rows=3,
        min_distinct_rank_values=2,
    )

    assert audit["ready_for_manifest_alignment"] is False
    assert audit["rank_column_ready"] is False
    assert audit["recommended_ground_truth_column"] is None
    assert any("zero variance" in reason for reason in audit["blocking_reasons"])


def test_select_extreme_tails_requires_disjoint_tails(tmp_path: Path) -> None:
    module = load_module()
    for index, value in enumerate((16, 64, 128), start=1):
        make_fake_dhf1k_video(tmp_path, f"{index:03d}", value)
    rows = module.build_rows(dhf1k_root=tmp_path, split="train", limit=3)

    with pytest.raises(ValueError, match="disjoint extreme tails"):
        module.select_extreme_tails(
            rows,
            rank_column="mean_map_intensity",
            count_per_tail=2,
        )

    with pytest.raises(ValueError, match="positive"):
        module.select_extreme_tails(
            rows,
            rank_column="mean_map_intensity",
            count_per_tail=0,
        )
