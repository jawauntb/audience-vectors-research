from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_production_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_content_pocket_recognition_stimulus_production.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_content_pocket_recognition_stimulus_production",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_design(tmp_path: Path) -> dict:
    return {
        "source_task_payload_sha256": "freeze-sha",
        "lure_generation_requests": [
            {
                "lure_id": "orange_flowers_lure_v00",
                "target_id": "orange_flowers_old_v00",
                "pocket": "fresh24_orange_flowers",
                "prompt": "Different orange flowers.",
                "distinctiveness_requirements": ["same broad category"],
                "generation_request": {
                    "alpha": 2.5,
                    "guidance": 7.5,
                    "suggested_noise_seed": 765190,
                },
            },
            {
                "lure_id": "hanging_clothes_lure_v00",
                "target_id": "hanging_clothes_old_v00",
                "pocket": "fresh24_hanging_clothes",
                "prompt": "Different hanging clothes.",
                "distinctiveness_requirements": ["same broad category"],
                "generation_request": {
                    "alpha": 3.0,
                    "guidance": 8.0,
                    "suggested_noise_seed": 765200,
                },
            },
        ],
        "tmp_path": str(tmp_path),
    }


def write_design(tmp_path: Path) -> Path:
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(synthetic_design(tmp_path)), encoding="utf-8")
    return design_path


def test_build_manifest_counts_missing_seed_images(tmp_path):
    module = load_production_module()
    manifest, markdown = module.build_manifest(
        design_path=write_design(tmp_path),
        seed_root=tmp_path / "seeds",
        video_out_dir=tmp_path / "videos",
        seed_screening_result=tmp_path / "screening.json",
        video_screening_result=tmp_path / "video_screening.json",
        filler_old_count=4,
        filler_recognition_count=3,
    )

    counts = manifest["artifact_counts"]
    assert counts["seed_image_requests"] == 9
    assert counts["generation_jobs"] == 9
    assert counts["seed_images_present"] == 0
    assert counts["seed_images_missing"] == 9
    assert manifest["status"] == "missing_seed_images_not_ready_for_generation"
    assert "near-duplicate" in markdown


def test_existing_seed_image_gets_hashed(tmp_path):
    module = load_production_module()
    seed = tmp_path / "seeds" / "analysis_lures" / "orange_flowers_lure_v00.png"
    seed.parent.mkdir(parents=True)
    seed.write_bytes(b"not really an image but enough for inventory")

    manifest, _markdown = module.build_manifest(
        design_path=write_design(tmp_path),
        seed_root=tmp_path / "seeds",
        video_out_dir=tmp_path / "videos",
        seed_screening_result=tmp_path / "screening.json",
        video_screening_result=tmp_path / "video_screening.json",
        filler_old_count=0,
        filler_recognition_count=0,
    )

    requests = {
        request["request_id"]: request
        for request in manifest["seed_image_requests"]
    }
    assert requests["orange_flowers_lure_v00"]["status"] == "present"
    assert requests["orange_flowers_lure_v00"]["seed_image"]["sha256"]
    assert requests["hanging_clothes_lure_v00"]["status"] == "missing_seed_image"


def test_screened_seed_images_are_ready_for_svd_generation(tmp_path):
    module = load_production_module()
    seed_root = tmp_path / "seeds"
    for name in ("orange_flowers_lure_v00", "hanging_clothes_lure_v00"):
        seed = seed_root / "analysis_lures" / f"{name}.png"
        seed.parent.mkdir(parents=True, exist_ok=True)
        seed.write_bytes(b"seed bytes")
    screening_result = tmp_path / "screening.json"
    screening_result.write_text(
        json.dumps(
            {
                "status": "seed_image_screen_passed_for_svd_generation",
                "accepted_for_svd_generation": True,
            }
        ),
        encoding="utf-8",
    )

    manifest, markdown = module.build_manifest(
        design_path=write_design(tmp_path),
        seed_root=seed_root,
        video_out_dir=tmp_path / "videos",
        seed_screening_result=screening_result,
        video_screening_result=tmp_path / "video_screening.json",
        filler_old_count=0,
        filler_recognition_count=0,
    )

    assert manifest["status"] == "seed_images_screened_ready_for_svd_generation"
    assert manifest["seed_image_screening"]["accepted_for_svd_generation"] is True
    assert "Manual image distinctiveness" not in "\n".join(
        manifest["launch_blockers"]
    )
    assert "Generate SVD MP4s" in markdown


def test_screened_videos_are_ready_for_hosting_review(tmp_path):
    module = load_production_module()
    seed_root = tmp_path / "seeds"
    video_out_dir = tmp_path / "videos"
    for name in ("orange_flowers_lure_v00", "hanging_clothes_lure_v00"):
        seed = seed_root / "analysis_lures" / f"{name}.png"
        seed.parent.mkdir(parents=True, exist_ok=True)
        seed.write_bytes(b"seed bytes")
        video = video_out_dir / "analysis_lures" / f"{name}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video bytes")
    seed_screening = tmp_path / "seed_screening.json"
    seed_screening.write_text(
        json.dumps(
            {
                "status": "seed_image_screen_passed_for_svd_generation",
                "accepted_for_svd_generation": True,
            }
        ),
        encoding="utf-8",
    )
    video_screening = tmp_path / "video_screening.json"
    video_screening.write_text(
        json.dumps(
            {
                "status": "agent_video_screen_passed_for_hosting",
                "accepted_for_hosting": True,
            }
        ),
        encoding="utf-8",
    )

    manifest, markdown = module.build_manifest(
        design_path=write_design(tmp_path),
        seed_root=seed_root,
        video_out_dir=video_out_dir,
        seed_screening_result=seed_screening,
        video_screening_result=video_screening,
        filler_old_count=0,
        filler_recognition_count=0,
    )

    assert manifest["status"] == "recognition_videos_screened_ready_for_hosting"
    assert manifest["video_screening"]["accepted_for_hosting"] is True
    assert "Generated MP4 visual screening" not in "\n".join(
        manifest["launch_blockers"]
    )
    assert "host accepted videos" in markdown


def test_filler_recognition_count_cannot_exceed_old_count(tmp_path):
    module = load_production_module()

    try:
        module.build_manifest(
            design_path=write_design(tmp_path),
            seed_root=tmp_path / "seeds",
            video_out_dir=tmp_path / "videos",
            seed_screening_result=tmp_path / "screening.json",
            video_screening_result=tmp_path / "video_screening.json",
            filler_old_count=2,
            filler_recognition_count=3,
        )
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
    else:
        raise AssertionError("expected ValueError")
