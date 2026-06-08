from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_audit_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_bo_content_axes.py"
    spec = importlib.util.spec_from_file_location("audit_bo_content_axes", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_seed_root(tmp_path: Path) -> Path:
    seed_root = tmp_path / "original"
    seed_dir = seed_root / "seeds"
    seed_dir.mkdir(parents=True)
    (seed_dir / "present.png").write_bytes(b"placeholder")
    rows = [
        {
            "idx": 1,
            "bmd_name": "present",
            "prompt": "present prompt",
            "seed_image": "seeds/present.png",
        },
        {
            "idx": 2,
            "bmd_name": "missing",
            "prompt": "missing prompt",
            "seed_image": "seeds/missing.png",
        },
    ]
    (seed_dir / "prompts.json").write_text(json.dumps(rows))
    return seed_root


def test_seed_availability_counts_present_and_missing_images(tmp_path):
    module = load_audit_module()
    summary = module.seed_availability(write_seed_root(tmp_path))

    assert summary["n_catalog_rows"] == 2
    assert summary["n_available_seed_images"] == 1
    assert summary["n_missing_seed_images"] == 1
    assert summary["available"][0]["bmd_name"] == "present"
    assert summary["missing"][0]["bmd_name"] == "missing"


def test_prompt_conditioning_audit_detects_metadata_only_svd_replay(tmp_path):
    module = load_audit_module()
    generator = tmp_path / "svd_generator.py"
    replay = tmp_path / "replay.py"
    generator.write_text(
        """
class SVDGenerator:
    def generate(self, image_bytes, guidance_scale=3.0):
        pass
"""
    )
    replay.write_text(
        """
def generate_videos_on_modal(generator, image):
    generator.generate.spawn(image, guidance_scale=3.0)
"""
    )

    audit = module.prompt_conditioning_audit(
        replay_script=replay,
        svd_generator=generator,
    )

    assert audit["generator_accepts_prompt"] is False
    assert audit["replay_passes_prompt"] is False
    assert audit["current_prompt_axis"] == "metadata_only"
