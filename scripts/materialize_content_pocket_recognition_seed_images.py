"""Materialize recognition-memory seed images with the OpenAI Image API.

This script reads the content-pocket recognition stimulus production manifest,
generates any missing seed images, writes local PNG files under ``data/``, and
builds committed contact sheets for screening. The full-resolution seed images
remain local data-lake artifacts; the committed result JSON records hashes and
generation metadata.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from PIL import Image, ImageDraw, ImageFont

ARTIFACT_DIR = Path(
    "research_program/neurips_memorability_selector/collaborator_inputs/"
    "camilo_bo_memorability"
)
DEFAULT_PRODUCTION_MANIFEST = (
    ARTIFACT_DIR / "content_pocket_recognition_stimulus_production_manifest_20260608.json"
)
DEFAULT_ENV_FILE = Path("/Users/jawaun/superoptimizers/.env")
DEFAULT_OUT_JSON = ARTIFACT_DIR / "content_pocket_recognition_seed_materialization_20260608.json"
DEFAULT_OUT_MD = ARTIFACT_DIR / "content_pocket_recognition_seed_materialization_20260608.md"
DEFAULT_CONTACT_SHEET_DIR = (
    ARTIFACT_DIR / "content_pocket_recognition_seed_screening_sheets_20260608"
)
OPENAI_IMAGE_ENDPOINT = "https://api.openai.com/v1/images/generations"


@dataclass(frozen=True)
class GenerationConfig:
    """OpenAI image generation settings."""

    model: str
    size: str
    quality: str
    timeout_seconds: float
    max_retries: int
    retry_sleep_seconds: float


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
        "bytes": path.stat().st_size if exists else None,
    }


def api_key_from_env_file(env_file: Path) -> str | None:
    if not env_file.exists():
        return None
    values = dotenv_values(env_file)
    value = values.get("OPENAI_API_KEY")
    return str(value) if value else None


def api_key_from_doppler(env_file: Path) -> str | None:
    doppler = os.environ.get("DOPPLER_TOKEN")
    if not doppler and env_file.exists():
        env_values = dotenv_values(env_file)
        doppler = str(env_values.get("DOPPLER_TOKEN") or "")
        if doppler:
            os.environ["DOPPLER_TOKEN"] = doppler
    if not doppler:
        return None
    try:
        result = subprocess.run(
            ["doppler", "secrets", "get", "OPENAI_API_KEY", "--plain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def resolve_api_key(env_file: Path, *, use_doppler: bool) -> str:
    env_value = os.environ.get("OPENAI_API_KEY")
    if env_value and not use_doppler:
        return env_value
    if use_doppler:
        doppler_value = api_key_from_doppler(env_file)
        if doppler_value:
            return doppler_value
    file_value = api_key_from_env_file(env_file)
    if file_value:
        return file_value
    raise RuntimeError(
        "OPENAI_API_KEY not found in environment, env file, or Doppler fallback"
    )


def sanitize_error(message: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_*.-]+", "sk-REDACTED", message)


def render_seed_prompt(request: dict[str, Any]) -> str:
    requirements = "\n".join(
        f"- {item}" for item in request.get("requirements", [])
    )
    source_pocket = request.get("source_pocket") or "unrelated filler"
    return (
        "Create one photorealistic still image for an image-to-video seed.\n"
        "Use a natural 16:9 landscape composition, realistic lighting, and a "
        "clear central subject. The image must be suitable for a short natural "
        "video generated from a still seed.\n\n"
        f"Seed role: {request['role']}\n"
        f"Source category: {source_pocket}\n"
        f"Base scene prompt: {request['prompt']}\n\n"
        "Hard requirements:\n"
        f"{requirements}\n"
        "- no text, lettering, captions, logos, watermarks, signs, UI, borders, "
        "or poster-like layout\n"
        "- no surreal collage, illustration, heavy stylization, or obvious "
        "synthetic artifacts\n"
        "- do not optimize for memorability or visual salience; make a neutral, "
        "naturalistic seed image\n"
        "- preserve the broad category while changing exact composition enough "
        "to support old-vs-lure recognition memory"
    )


def decode_image_response(payload: dict[str, Any]) -> bytes:
    try:
        image_base64 = payload["data"][0]["b64_json"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"image response missing data[0].b64_json: {payload}") from exc
    return base64.b64decode(image_base64)


def openai_generate_image(
    *,
    api_key: str,
    prompt: str,
    config: GenerationConfig,
) -> tuple[bytes, dict[str, Any]]:
    body = {
        "model": config.model,
        "prompt": prompt,
        "n": 1,
        "size": config.size,
        "quality": config.quality,
        "output_format": "png",
        "background": "opaque",
    }
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: str | None = None
    for attempt in range(config.max_retries + 1):
        request = urllib.request.Request(
            OPENAI_IMAGE_ENDPOINT,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return decode_image_response(payload), payload
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = sanitize_error(f"HTTP {exc.code}: {error_body[:1000]}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except (TimeoutError, OSError) as exc:
            last_error = sanitize_error(repr(exc))
        if attempt < config.max_retries:
            time.sleep(config.retry_sleep_seconds * (attempt + 1))
    raise RuntimeError(last_error or "unknown OpenAI image generation error")


def materialize_one(
    *,
    request: dict[str, Any],
    api_key: str,
    config: GenerationConfig,
    overwrite: bool,
) -> dict[str, Any]:
    out_path = Path(str(request["seed_image"]["path"]))
    started = time.monotonic()
    row: dict[str, Any] = {
        "request_id": request["request_id"],
        "role": request["role"],
        "source_pocket": request.get("source_pocket"),
        "matched_id": request.get("matched_id"),
        "seed_image": file_inventory(out_path),
        "generation_seconds": None,
        "status": None,
        "error": None,
    }
    if out_path.exists() and not overwrite:
        row["status"] = "already_present"
        return row

    prompt = render_seed_prompt(request)
    try:
        image_bytes, payload = openai_generate_image(
            api_key=api_key,
            prompt=prompt,
            config=config,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_bytes(image_bytes)
        with Image.open(tmp_path) as image:
            image.verify()
        tmp_path.replace(out_path)
        row["status"] = "generated"
        row["seed_image"] = file_inventory(out_path)
        row["openai_response"] = {
            "created": payload.get("created"),
            "usage": payload.get("usage"),
            "model": config.model,
            "size": config.size,
            "quality": config.quality,
        }
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = sanitize_error(repr(exc))
    row["generation_seconds"] = time.monotonic() - started
    return row


def selected_requests(
    manifest: dict[str, Any],
    *,
    roles: set[str] | None,
    request_ids: set[str] | None,
    limit: int | None,
    only_missing: bool,
) -> list[dict[str, Any]]:
    requests = list(manifest["seed_image_requests"])
    if roles:
        requests = [request for request in requests if request["role"] in roles]
    if request_ids:
        requests = [
            request
            for request in requests
            if str(request["request_id"]) in request_ids
        ]
    if only_missing:
        requests = [
            request
            for request in requests
            if not Path(str(request["seed_image"]["path"])).exists()
        ]
    if limit is not None:
        requests = requests[:limit]
    return requests


def build_contact_sheet(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    title: str,
    columns: int = 5,
    thumb_size: tuple[int, int] = (256, 144),
    label_height: int = 44,
) -> dict[str, Any]:
    present = [
        row
        for row in rows
        if Path(str(row["seed_image"]["path"])).exists()
    ]
    if not present:
        return {"path": str(out_path), "exists": False, "items": 0}

    rows_count = (len(present) + columns - 1) // columns
    width = columns * thumb_size[0]
    header_height = 42
    cell_height = thumb_size[1] + label_height
    height = header_height + rows_count * cell_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)

    for index, row in enumerate(present):
        image_path = Path(str(row["seed_image"]["path"]))
        col = index % columns
        grid_row = index // columns
        x = col * thumb_size[0]
        y = header_height + grid_row * cell_height
        with Image.open(image_path).convert("RGB") as image:
            image.thumbnail(thumb_size)
            thumb = Image.new("RGB", thumb_size, "white")
            ox = (thumb_size[0] - image.width) // 2
            oy = (thumb_size[1] - image.height) // 2
            thumb.paste(image, (ox, oy))
        sheet.paste(thumb, (x, y))
        label = f"{row['request_id']}\n{row.get('source_pocket') or ''}"
        draw.text((x + 4, y + thumb_size[1] + 4), label, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    return {
        "path": str(out_path),
        "exists": True,
        "items": len(present),
        "sha256": sha256_file(out_path),
        "bytes": out_path.stat().st_size,
    }


def build_contact_sheets(
    *,
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> list[dict[str, Any]]:
    sheets = []
    groups = [
        ("analysis_lures", "analysis_lure_seed", "Analysis lure seed images"),
        ("filler_old", "filler_old_seed", "Filler old-target seed images"),
        ("filler_lures", "filler_lure_seed", "Filler lure seed images"),
    ]
    for filename, role, title in groups:
        role_rows = [row for row in rows if row["role"] == role]
        sheets.append(
            {
                "role": role,
                **build_contact_sheet(
                    rows=role_rows,
                    out_path=out_dir / f"{filename}.jpg",
                    title=title,
                ),
            }
        )
    return sheets


def image_thumb(path: Path | None, size: tuple[int, int]) -> Image.Image:
    if path is None or not path.exists():
        image = Image.new("RGB", size, (235, 235, 235))
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), "missing", fill=(0, 0, 0), font=ImageFont.load_default())
        return image
    with Image.open(path).convert("RGB") as image:
        image.thumbnail(size)
        thumb = Image.new("RGB", size, "white")
        ox = (size[0] - image.width) // 2
        oy = (size[1] - image.height) // 2
        thumb.paste(image, (ox, oy))
        return thumb


def extract_video_frame(video_path: Path, out_path: Path) -> Path | None:
    if not video_path.exists():
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss",
        "1",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(out_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 or not out_path.exists():
        return None
    return out_path


def resolve_old_video_path(old_video: dict[str, Any]) -> Path | None:
    for key in ("source_absolute_path", "local_video_path"):
        raw = old_video.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if path.exists():
            return path
    return None


def build_pair_contact_sheet(
    *,
    pairs: list[dict[str, Any]],
    out_path: Path,
    title: str,
    left_title: str,
    right_title: str,
    thumb_size: tuple[int, int] = (256, 144),
    label_height: int = 48,
) -> dict[str, Any]:
    if not pairs:
        return {"path": str(out_path), "exists": False, "items": 0}

    header_height = 64
    row_height = thumb_size[1] + label_height
    width = thumb_size[0] * 2
    height = header_height + row_height * len(pairs)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    draw.text((8, 34), left_title, fill=(0, 0, 0), font=font)
    draw.text((thumb_size[0] + 8, 34), right_title, fill=(0, 0, 0), font=font)

    for index, pair in enumerate(pairs):
        y = header_height + index * row_height
        left_path = Path(str(pair["left_path"])) if pair.get("left_path") else None
        right_path = Path(str(pair["right_path"])) if pair.get("right_path") else None
        sheet.paste(image_thumb(left_path, thumb_size), (0, y))
        sheet.paste(image_thumb(right_path, thumb_size), (thumb_size[0], y))
        label = f"{pair['pair_id']}\n{pair.get('source_pocket') or ''}"
        draw.text((4, y + thumb_size[1] + 4), label, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    return {
        "path": str(out_path),
        "exists": True,
        "items": len(pairs),
        "sha256": sha256_file(out_path),
        "bytes": out_path.stat().st_size,
    }


def build_analysis_pair_contact_sheet(
    *,
    design: dict[str, Any],
    rows_by_id: dict[str, dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        for request in design["lure_generation_requests"]:
            lure_id = str(request["lure_id"])
            row = rows_by_id.get(lure_id)
            old_path = resolve_old_video_path(request["matched_old_video"])
            old_frame = None
            if old_path is not None:
                old_frame = extract_video_frame(old_path, temp_dir / f"{lure_id}_old.jpg")
            pairs.append(
                {
                    "pair_id": f"{request['target_id']} vs {lure_id}",
                    "source_pocket": request.get("pocket"),
                    "left_path": str(old_frame) if old_frame else None,
                    "right_path": row["seed_image"]["path"] if row else None,
                }
            )
        return build_pair_contact_sheet(
            pairs=pairs,
            out_path=out_dir / "analysis_old_vs_lure_pairs.jpg",
            title="Analysis old clip frame vs generated lure seed",
            left_title="Old clip frame",
            right_title="Generated lure seed",
        )


def build_filler_pair_contact_sheet(
    *,
    rows_by_id: dict[str, dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    lure_rows = [
        row for row in rows_by_id.values() if row["role"] == "filler_lure_seed"
    ]
    for row in sorted(lure_rows, key=lambda item: str(item["request_id"])):
        old_id = str(row["matched_id"])
        old_row = rows_by_id.get(old_id)
        pairs.append(
            {
                "pair_id": f"{old_id} vs {row['request_id']}",
                "source_pocket": row.get("source_pocket"),
                "left_path": old_row["seed_image"]["path"] if old_row else None,
                "right_path": row["seed_image"]["path"],
            }
        )
    return build_pair_contact_sheet(
        pairs=pairs,
        out_path=out_dir / "filler_old_vs_lure_pairs.jpg",
        title="Filler old seed vs generated filler lure seed",
        left_title="Filler old seed",
        right_title="Filler lure seed",
    )


def render_markdown(result: dict[str, Any]) -> str:
    counts = result["counts"]
    lines = [
        "# Content-Pocket Recognition Seed Materialization Result",
        "",
        f"Date: {result['created_at_utc']}",
        "",
        "## Discovery-Regime Audit",
        "",
        "Question: can the recognition-memory production manifest's seed-image",
        "requests be materialized without accepting near-duplicate lures?",
        "",
        "Current regime:",
        "",
        "- Artifact types: seed PNGs, seed hashes, generation metadata, contact",
        "  sheets, screening status.",
        "- Operations: OpenAI Image API generation from production-manifest prompts",
        "  and requirements; contact-sheet construction.",
        "- Gates/verifiers: all 60 seed images present, manual contact-sheet review",
        "  before SVD generation, no human-memory claim until recognition data.",
        "- Known limitation: this result materializes seed images only. It does not",
        "  generate SVD videos or validate memorability.",
        "",
        "Action class: production search inside the accepted recognition-memory",
        "validation regime.",
        "",
        "## Counts",
        "",
        f"- Requested: {counts['requested']}",
        f"- Generated: {counts['generated']}",
        f"- Already present: {counts['already_present']}",
        f"- Failed: {counts['failed']}",
        f"- Seed images present after run: {counts['present_after_run']}",
        "",
        "## Contact Sheets",
        "",
    ]
    for sheet in result["contact_sheets"]:
        status = "present" if sheet.get("exists") else "missing"
        lines.append(f"- `{sheet['path']}` ({status}, items={sheet.get('items', 0)})")
    lines.extend(
        [
            "",
            "## Next Action",
            "",
            "Review the contact sheets for same-category match, non-duplication, no",
            "text/watermarks, and no obvious artifacts. Only after image screening",
            "passes should the SVD generation jobs in the production manifest run.",
            "",
        ]
    )
    return "\n".join(lines)


def run_materialization(
    *,
    manifest_path: Path,
    out_json: Path,
    out_md: Path,
    contact_sheet_dir: Path,
    env_file: Path,
    use_doppler: bool,
    roles: set[str] | None,
    request_ids: set[str] | None,
    limit: int | None,
    only_missing: bool,
    overwrite: bool,
    dry_run: bool,
    workers: int,
    config: GenerationConfig,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    requests = selected_requests(
        manifest,
        roles=roles,
        request_ids=request_ids,
        limit=limit,
        only_missing=only_missing,
    )
    rows: list[dict[str, Any]]
    if dry_run:
        rows = [
            {
                "request_id": request["request_id"],
                "role": request["role"],
                "source_pocket": request.get("source_pocket"),
                "matched_id": request.get("matched_id"),
                "seed_image": file_inventory(Path(str(request["seed_image"]["path"]))),
                "status": "dry_run",
                "error": None,
            }
            for request in requests
        ]
    else:
        api_key = resolve_api_key(env_file, use_doppler=use_doppler)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    materialize_one,
                    request=request,
                    api_key=api_key,
                    config=config,
                    overwrite=overwrite,
                )
                for request in requests
            ]
            rows = []
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                print(
                    f"[seed-materialize] {row['status']}: "
                    f"{row['request_id']} -> {row['seed_image']['path']}",
                    flush=True,
                )
                rows.append(row)
            rows.sort(key=lambda row: str(row["request_id"]))

    contact_sheets = build_contact_sheets(rows=rows, out_dir=contact_sheet_dir)
    rows_by_id = {str(row["request_id"]): row for row in rows}
    design_path = Path(str(manifest["source_recognition_design"]))
    if design_path.exists():
        design = load_json(design_path)
        contact_sheets.append(
            {
                "role": "analysis_old_vs_lure_pair",
                **build_analysis_pair_contact_sheet(
                    design=design,
                    rows_by_id=rows_by_id,
                    out_dir=contact_sheet_dir,
                ),
            }
        )
    contact_sheets.append(
        {
            "role": "filler_old_vs_lure_pair",
            **build_filler_pair_contact_sheet(
                rows_by_id=rows_by_id,
                out_dir=contact_sheet_dir,
            ),
        }
    )
    counts = {
        "requested": len(rows),
        "generated": sum(1 for row in rows if row["status"] == "generated"),
        "already_present": sum(1 for row in rows if row["status"] == "already_present"),
        "failed": sum(1 for row in rows if row["status"] == "failed"),
        "present_after_run": sum(
            1 for row in rows if Path(str(row["seed_image"]["path"])).exists()
        ),
    }
    result = {
        "schema_version": "content_pocket_recognition_seed_materialization.v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_production_manifest": str(manifest_path),
        "model": config.model,
        "size": config.size,
        "quality": config.quality,
        "dry_run": dry_run,
        "counts": counts,
        "rows": rows,
        "contact_sheets": contact_sheets,
        "claim_boundary": [
            "Seed materialization is not human-memory evidence.",
            "Generated seed images require manual screening before SVD generation.",
            "Near-duplicate lures must be rejected, not averaged away.",
        ],
    }
    write_json(out_json, result)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(result), encoding="utf-8")
    return result


def parse_roles(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PRODUCTION_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--contact-sheet-dir", type=Path, default=DEFAULT_CONTACT_SHEET_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--use-doppler", action="store_true")
    parser.add_argument("--model", default="gpt-image-2")
    parser.add_argument("--size", default="1536x864")
    parser.add_argument("--quality", default="low")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=8.0)
    parser.add_argument("--roles", help="comma-separated roles to generate")
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    config = GenerationConfig(
        model=args.model,
        size=args.size,
        quality=args.quality,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    result = run_materialization(
        manifest_path=args.manifest,
        out_json=args.out_json,
        out_md=args.out_md,
        contact_sheet_dir=args.contact_sheet_dir,
        env_file=args.env_file,
        use_doppler=args.use_doppler,
        roles=parse_roles(args.roles),
        request_ids=set(args.request_id) if args.request_id else None,
        limit=args.limit,
        only_missing=not args.include_existing,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        workers=args.workers,
        config=config,
    )
    print(f"[done] wrote {args.out_json}")
    print(f"[done] wrote {args.out_md}")
    print(f"[done] counts: {result['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
