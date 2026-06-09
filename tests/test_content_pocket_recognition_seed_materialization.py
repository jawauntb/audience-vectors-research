from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_materializer_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "materialize_content_pocket_recognition_seed_images.py"
    )
    spec = importlib.util.spec_from_file_location(
        "materialize_content_pocket_recognition_seed_images",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_seed_prompt_keeps_recognition_constraints():
    module = load_materializer_module()
    prompt = module.render_seed_prompt(
        {
            "request_id": "orange_flowers_lure_v00",
            "role": "analysis_lure_seed",
            "source_pocket": "fresh24_orange_flowers",
            "prompt": "Different orange flowers.",
            "requirements": [
                "same broad category as orange flowers",
                "not a close-up near-duplicate of the frozen target clip",
            ],
        }
    )

    assert "image-to-video seed" in prompt
    assert "not a close-up near-duplicate" in prompt
    assert "no text" in prompt
    assert "do not optimize for memorability" in prompt


def test_selected_requests_filters_roles_and_missing(tmp_path):
    module = load_materializer_module()
    present = tmp_path / "present.png"
    present.write_bytes(b"fake")
    manifest = {
        "seed_image_requests": [
            {
                "request_id": "a",
                "role": "analysis_lure_seed",
                "seed_image": {"path": str(present)},
            },
            {
                "request_id": "b",
                "role": "filler_old_seed",
                "seed_image": {"path": str(tmp_path / "missing.png")},
            },
        ]
    }

    requests = module.selected_requests(
        manifest,
        roles={"filler_old_seed"},
        request_ids=None,
        limit=None,
        only_missing=True,
    )

    assert [request["request_id"] for request in requests] == ["b"]


def test_selected_requests_filters_request_ids(tmp_path):
    module = load_materializer_module()
    manifest = {
        "seed_image_requests": [
            {
                "request_id": "filler_old_v06",
                "role": "filler_old_seed",
                "seed_image": {"path": str(tmp_path / "old.png")},
            },
            {
                "request_id": "filler_lure_v06",
                "role": "filler_lure_seed",
                "seed_image": {"path": str(tmp_path / "lure.png")},
            },
        ]
    }

    requests = module.selected_requests(
        manifest,
        roles=None,
        request_ids={"filler_lure_v06"},
        limit=None,
        only_missing=False,
    )

    assert [request["request_id"] for request in requests] == ["filler_lure_v06"]


def test_contact_sheet_skips_empty_groups(tmp_path):
    module = load_materializer_module()
    result = module.build_contact_sheet(
        rows=[],
        out_path=tmp_path / "sheet.jpg",
        title="Empty",
    )

    assert result["exists"] is False
    assert result["items"] == 0
