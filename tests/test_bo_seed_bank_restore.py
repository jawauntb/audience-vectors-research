from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

from PIL import Image


def load_restore_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "restore_bo_seed_bank.py"
    )
    spec = importlib.util.spec_from_file_location("restore_bo_seed_bank", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def image_bytes(*, size: tuple[int, int] = (96, 48)) -> bytes:
    image = Image.new("RGB", size, color=(12, 34, 56))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def write_seed_root(tmp_path: Path) -> Path:
    seed_root = tmp_path / "original"
    seed_dir = seed_root / "seeds"
    seed_dir.mkdir(parents=True)
    (seed_dir / "existing_seed.png").write_bytes(image_bytes(size=(640, 352)))
    rows = [
        {
            "idx": 0,
            "bmd_name": "existing",
            "prompt": "existing prompt",
            "seed_image": "seeds/existing_seed.png",
            "source_image": "https://example.invalid/existing.jpg",
        },
        {
            "idx": 1,
            "bmd_name": "missing",
            "prompt": "missing prompt",
            "seed_image": "seeds/missing_seed.png",
            "source_image": "https://example.invalid/missing.jpg",
        },
        {
            "idx": 2,
            "bmd_name": "unrestorable",
            "prompt": "unrestorable prompt",
            "seed_image": "seeds/unrestorable_seed.png",
        },
    ]
    (seed_dir / "prompts.json").write_text(json.dumps(rows))
    return seed_root


def test_restore_seed_bank_dry_run_reports_restorable_missing_images(tmp_path):
    module = load_restore_module()
    seed_root = write_seed_root(tmp_path)

    report = module.restore_seed_bank(
        seed_root=seed_root,
        dry_run=True,
        overwrite=False,
        image_size=(640, 352),
    )

    assert report["counts"] == {
        "dry_run_restorable": 1,
        "existing": 1,
        "missing_source_image": 1,
    }
    assert not (seed_root / "seeds" / "missing_seed.png").exists()


def test_restore_seed_bank_writes_normalized_png(tmp_path):
    module = load_restore_module()
    seed_root = write_seed_root(tmp_path)

    report = module.restore_seed_bank(
        seed_root=seed_root,
        dry_run=False,
        overwrite=False,
        image_size=(640, 352),
        downloader=lambda url: image_bytes(size=(960, 528)),
    )

    restored = seed_root / "seeds" / "missing_seed.png"
    assert report["counts"]["restored"] == 1
    assert restored.exists()
    with Image.open(restored) as image:
        assert image.format == "PNG"
        assert image.size == (640, 352)

    restored_row = [row for row in report["rows"] if row["status"] == "restored"][0]
    assert restored_row["source_width"] == 960
    assert restored_row["source_height"] == 528
    assert restored_row["width"] == 640
    assert restored_row["height"] == 352
