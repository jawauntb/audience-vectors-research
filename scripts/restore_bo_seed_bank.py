"""Restore missing BO replay seed images from the catalog source URLs.

The collaborator seed catalog includes 24 prompt rows, but the repository may
only have a subset of the raw image files locally. This script restores missing
images from each row's ``source_image`` URL, normalizes them to the replay seed
format, and writes a lightweight JSON report. Restored images are raw data and
should stay local unless the artifact policy explicitly changes.
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
INTAKE_ROOT = (
    REPO_ROOT
    / "research_program"
    / "neurips_memorability_selector"
    / "collaborator_inputs"
    / "camilo_bo_memorability"
)
DEFAULT_SEED_ROOT = INTAKE_ROOT / "original"
DEFAULT_REPORT_PATH = Path("data/reports/bo_seed_bank_restore_20260608.json")
DEFAULT_IMAGE_SIZE = (640, 352)
DEFAULT_TIMEOUT_SECONDS = 30.0


def load_seed_catalog(seed_root: Path) -> list[dict[str, Any]]:
    """Load the seed prompt catalog."""
    prompts_path = seed_root / "seeds" / "prompts.json"
    rows = json.loads(prompts_path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{prompts_path} does not contain a list")
    return rows


def fetch_url(url: str, *, timeout_seconds: float) -> bytes:
    """Fetch one source image URL."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "audience-vectors-seed-bank-restore/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def normalize_image_bytes(
    image_bytes: bytes,
    *,
    image_size: tuple[int, int],
) -> tuple[bytes, dict[str, Any]]:
    """Decode an image, normalize to RGB PNG, and return metadata."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        source_size = image.size
        normalized = image.convert("RGB")
        if normalized.size != image_size:
            normalized = normalized.resize(image_size, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        normalized.save(output, format="PNG")

    payload = output.getvalue()
    return payload, {
        "source_width": source_size[0],
        "source_height": source_size[1],
        "width": image_size[0],
        "height": image_size[1],
        "format": "PNG",
        "n_bytes": len(payload),
    }


def restore_seed_bank(
    *,
    seed_root: Path,
    dry_run: bool,
    overwrite: bool,
    image_size: tuple[int, int],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    downloader: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    """Restore missing seed files and return a report payload."""
    rows = load_seed_catalog(seed_root)
    restored_rows: list[dict[str, Any]] = []

    for row in rows:
        seed_image = row.get("seed_image")
        source_image = row.get("source_image")
        item: dict[str, Any] = {
            "idx": int(row["idx"]),
            "bmd_name": str(row["bmd_name"]),
            "seed_image": str(seed_image) if seed_image else None,
            "source_image": str(source_image) if source_image else None,
        }

        if not seed_image:
            item["status"] = "no_seed_image"
            restored_rows.append(item)
            continue

        image_path = seed_root / str(seed_image)
        item["image_path"] = str(image_path)
        if image_path.exists() and not overwrite:
            item["status"] = "existing"
            restored_rows.append(item)
            continue

        if not source_image:
            item["status"] = "missing_source_image"
            restored_rows.append(item)
            continue

        if dry_run:
            item["status"] = "dry_run_restorable"
            restored_rows.append(item)
            continue

        fetcher = downloader or (
            lambda url: fetch_url(url, timeout_seconds=timeout_seconds)
        )
        restored_bytes, metadata = normalize_image_bytes(
            fetcher(str(source_image)),
            image_size=image_size,
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = image_path.with_suffix(f"{image_path.suffix}.tmp")
        tmp_path.write_bytes(restored_bytes)
        tmp_path.replace(image_path)
        item.update(metadata)
        item["status"] = "restored"
        restored_rows.append(item)

    counts = Counter(str(item["status"]) for item in restored_rows)
    return {
        "schema_version": 1,
        "kind": "bo_seed_bank_restore",
        "seed_root": str(seed_root),
        "dry_run": dry_run,
        "overwrite": overwrite,
        "image_size": list(image_size),
        "n_catalog_rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "rows": restored_rows,
    }


def parse_image_size(raw: str) -> tuple[int, int]:
    """Parse WIDTH,HEIGHT."""
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("image size must be WIDTH,HEIGHT")
    try:
        width, height = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("image size values must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("image size values must be positive")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--image-size",
        type=parse_image_size,
        default=DEFAULT_IMAGE_SIZE,
        help="Normalized output image size as WIDTH,HEIGHT.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = restore_seed_bank(
        seed_root=args.seed_root,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        image_size=args.image_size,
        timeout_seconds=args.timeout_seconds,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "report_path": str(args.report_path),
                "n_catalog_rows": report["n_catalog_rows"],
                "counts": report["counts"],
                "image_size": report["image_size"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
