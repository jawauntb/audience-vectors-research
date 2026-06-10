"""Ingest official DHF1K videos into the Modal volume used by TRIBE.

The heavy work runs in Modal: download the official `video.rar`, extract it on
the container SSD, copy the requested videos into `bmd-videos-v1`, then commit
the volume. The local process only sends the expected video IDs and writes the
audit report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "attention-capture-dhf1k-video-ingest"
DEFAULT_MODAL_VOLUME_NAME = "bmd-videos-v1"
DEFAULT_MODAL_ROOT = "/bmd-videos"
DEFAULT_TARGET_PREFIX = "attention_capture/DHF1K"
OFFICIAL_DHF1K_FOLDER_ID = "1sW0tf9RQMO4RR7SyKhU8Kmbm4jwkFGpQ"
OFFICIAL_DHF1K_VIDEO_FILE_ID = "1UEFQmRdDbtVT-ePjMZVrv9oVV0ra631s"
OFFICIAL_DHF1K_VIDEO_ARCHIVE = "video.rar"
VIDEO_SUFFIXES = (".AVI", ".avi", ".mp4", ".MP4", ".mov", ".MOV")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unar")
    .pip_install("gdown")
)
app = modal.App(APP_NAME)
bmd_videos_volume = modal.Volume.from_name(
    DEFAULT_MODAL_VOLUME_NAME,
    create_if_missing=True,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--video-id-column", default="video_id")
    parser.add_argument("--target-prefix", default=DEFAULT_TARGET_PREFIX)
    parser.add_argument(
        "--google-drive-video-file-id",
        default=OFFICIAL_DHF1K_VIDEO_FILE_ID,
        help="Official DHF1K video.rar Google Drive file id.",
    )
    parser.add_argument(
        "--copy-all-videos",
        action="store_true",
        help="Copy every extracted DHF1K video instead of only IDs in --labels-csv.",
    )
    parser.add_argument(
        "--expected-min-videos",
        type=int,
        default=350,
        help="Minimum copied/existing target videos required for ready=true.",
    )
    return parser.parse_args()


@app.function(
    image=image,
    cpu=4.0,
    memory=8 * 1024,
    ephemeral_disk=512 * 1024,
    timeout=8 * 60 * 60,
    volumes={DEFAULT_MODAL_ROOT: bmd_videos_volume},
)
def ingest_dhf1k_videos(
    *,
    expected_video_ids: list[str],
    target_prefix: str = DEFAULT_TARGET_PREFIX,
    google_drive_video_file_id: str = OFFICIAL_DHF1K_VIDEO_FILE_ID,
    copy_all_videos: bool = False,
    expected_min_videos: int = 350,
) -> dict[str, Any]:
    try:
        return ingest_dhf1k_videos_impl(
            expected_video_ids=expected_video_ids,
            target_prefix=target_prefix,
            google_drive_video_file_id=google_drive_video_file_id,
            copy_all_videos=copy_all_videos,
            expected_min_videos=expected_min_videos,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "ready": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "target_prefix": target_prefix,
            "google_drive_video_file_id": google_drive_video_file_id,
            "copy_all_videos": copy_all_videos,
            "expected_video_ids": expected_video_ids,
            "claim_boundary": ingest_claim_boundary(),
        }


@app.local_entrypoint()
def main(
    labels_csv: str,
    output_json: str,
    output_md: str,
    sample_id_column: str = "sample_id",
    video_id_column: str = "video_id",
    target_prefix: str = DEFAULT_TARGET_PREFIX,
    google_drive_video_file_id: str = OFFICIAL_DHF1K_VIDEO_FILE_ID,
    copy_all_videos: bool = False,
    expected_min_videos: int = 350,
) -> None:
    labels_csv_path = Path(labels_csv)
    output_json_path = Path(output_json)
    output_md_path = Path(output_md)
    expected_video_ids = expected_video_ids_from_labels(
        labels_csv_path,
        sample_id_column=sample_id_column,
        video_id_column=video_id_column,
    )
    ingest = ingest_dhf1k_videos.remote(
        expected_video_ids=expected_video_ids,
        target_prefix=target_prefix,
        google_drive_video_file_id=google_drive_video_file_id,
        copy_all_videos=copy_all_videos,
        expected_min_videos=expected_min_videos,
    )
    report = {
        "schema_version": 1,
        "experiment": "dhf1k_modal_video_ingest",
        "generated_at": datetime.now(UTC).isoformat(),
        "labels_csv": str(labels_csv_path),
        "sample_id_column": sample_id_column,
        "video_id_column": video_id_column,
        "n_expected_video_ids": len(expected_video_ids),
        "modal_volume_name": DEFAULT_MODAL_VOLUME_NAME,
        "modal_root": DEFAULT_MODAL_ROOT,
        "target_prefix": target_prefix,
        "official_source": {
            "folder_id": OFFICIAL_DHF1K_FOLDER_ID,
            "video_file_id": google_drive_video_file_id,
            "video_archive": OFFICIAL_DHF1K_VIDEO_ARCHIVE,
            "folder_url": (
                "https://drive.google.com/drive/folders/"
                f"{OFFICIAL_DHF1K_FOLDER_ID}"
            ),
        },
        "ingest": ingest,
        "ready": bool(ingest.get("ready")),
        "claim_boundary": ingest_claim_boundary(),
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md_path.write_text(render_ingest_markdown(report), encoding="utf-8")
    print(f"wrote {output_json_path}")
    print(f"wrote {output_md_path}")
    if not report["ready"]:
        raise SystemExit(1)


def ingest_dhf1k_videos_impl(
    *,
    expected_video_ids: list[str],
    target_prefix: str,
    google_drive_video_file_id: str,
    copy_all_videos: bool,
    expected_min_videos: int,
) -> dict[str, Any]:
    import gdown  # type: ignore[import-not-found]

    started = time.monotonic()
    staging_root = Path("/tmp/dhf1k_video_ingest")
    archive_path = staging_root / OFFICIAL_DHF1K_VIDEO_ARCHIVE
    extract_dir = staging_root / "extracted"
    reset_dir(staging_root)
    extract_dir.mkdir(parents=True, exist_ok=True)

    download_started = time.monotonic()
    downloaded = gdown.download(
        id=google_drive_video_file_id,
        output=str(archive_path),
        quiet=False,
        use_cookies=False,
    )
    download_seconds = time.monotonic() - download_started
    if downloaded is None or not archive_path.exists():
        raise RuntimeError("gdown did not produce the DHF1K video archive")

    extract_started = time.monotonic()
    subprocess.run(
        ["unar", "-o", str(extract_dir), str(archive_path)],
        check=True,
        timeout=6 * 60 * 60,
    )
    extract_seconds = time.monotonic() - extract_started

    source_video_dir = find_extracted_video_dir(extract_dir)
    target_video_dir = Path(DEFAULT_MODAL_ROOT) / target_prefix.strip("/") / "video"
    target_video_dir.mkdir(parents=True, exist_ok=True)
    copy_report = copy_videos_to_volume(
        source_video_dir=source_video_dir,
        target_video_dir=target_video_dir,
        expected_video_ids=expected_video_ids,
        copy_all_videos=copy_all_videos,
    )
    bmd_videos_volume.commit()
    n_ready_videos = copy_report["n_copied"] + copy_report["n_existing"]
    ready = (
        bool(copy_report["ok"])
        and n_ready_videos >= expected_min_videos
        and not copy_report["missing_expected_video_ids"]
    )
    return {
        "ok": bool(copy_report["ok"]),
        "ready": ready,
        "target_video_dir": str(target_video_dir),
        "source_video_dir": str(source_video_dir),
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "download_seconds": download_seconds,
        "extract_seconds": extract_seconds,
        "total_seconds": time.monotonic() - started,
        "copy_all_videos": copy_all_videos,
        "expected_min_videos": expected_min_videos,
        "google_drive_video_file_id": google_drive_video_file_id,
        "copy_report": copy_report,
        "claim_boundary": ingest_claim_boundary(),
    }


def expected_video_ids_from_labels(
    labels_csv: Path,
    *,
    sample_id_column: str,
    video_id_column: str,
) -> list[str]:
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{labels_csv} is missing a CSV header")
        ids = [
            normalize_video_id(row.get(video_id_column) or row[sample_id_column])
            for row in reader
        ]
    return dedupe(ids)


def normalize_video_id(value: str) -> str:
    tail = value.rsplit("_", 1)[-1]
    if not tail.isdigit():
        raise ValueError(f"cannot normalize DHF1K video id from {value!r}")
    return f"{int(tail):03d}"


def find_extracted_video_dir(extract_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for root, _, files in os.walk(extract_dir):
        root_path = Path(root)
        if root_path.name.lower() != "video":
            continue
        count = sum(1 for name in files if Path(name).suffix in VIDEO_SUFFIXES)
        if count:
            candidates.append((count, root_path))
    if not candidates:
        raise RuntimeError(f"no extracted DHF1K video directory found under {extract_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def copy_videos_to_volume(
    *,
    source_video_dir: Path,
    target_video_dir: Path,
    expected_video_ids: list[str],
    copy_all_videos: bool,
) -> dict[str, Any]:
    copied: list[dict[str, Any]] = []
    existing: list[dict[str, Any]] = []
    missing_expected_video_ids: list[str] = []
    if copy_all_videos:
        source_paths = sorted(
            path
            for path in source_video_dir.iterdir()
            if path.is_file() and path.suffix in VIDEO_SUFFIXES
        )
    else:
        source_paths = []
        for video_id in expected_video_ids:
            path = find_video_path(source_video_dir, video_id)
            if path is None:
                missing_expected_video_ids.append(video_id)
                continue
            source_paths.append(path)

    for source_path in source_paths:
        target_path = target_video_dir / source_path.name
        row = {
            "source_path": str(source_path),
            "target_path": str(target_path),
            "size_bytes": source_path.stat().st_size,
        }
        if target_path.exists() and target_path.stat().st_size == source_path.stat().st_size:
            existing.append(row)
            continue
        shutil.copy2(source_path, target_path)
        copied.append(row)

    return {
        "ok": not missing_expected_video_ids,
        "source_video_count": count_video_files(source_video_dir),
        "target_video_count": count_video_files(target_video_dir),
        "n_copied": len(copied),
        "n_existing": len(existing),
        "n_missing_expected": len(missing_expected_video_ids),
        "missing_expected_video_ids": missing_expected_video_ids,
        "copied_preview": copied[:25],
        "existing_preview": existing[:25],
    }


def find_video_path(video_dir: Path, video_id: str) -> Path | None:
    for candidate_id in video_id_variants(video_id):
        for suffix in VIDEO_SUFFIXES:
            candidate = video_dir / f"{candidate_id}{suffix}"
            if candidate.exists():
                return candidate
    return None


def video_id_variants(video_id: str) -> list[str]:
    value = int(video_id)
    return [f"{value:03d}", f"{value:04d}"]


def count_video_files(video_dir: Path) -> int:
    if not video_dir.is_dir():
        return 0
    return sum(
        1
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix in VIDEO_SUFFIXES
    )


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ingest_claim_boundary() -> str:
    return (
        "This ingests public DHF1K media into a Modal Volume for later TRIBE "
        "feature extraction. It does not compute labels, score TRIBE, or "
        "validate attentional capture."
    )


def render_ingest_markdown(report: dict[str, Any]) -> str:
    ingest = report["ingest"]
    copy_report = ingest.get("copy_report") or {}
    lines = [
        "# DHF1K Modal Video Ingest",
        "",
        f"- Ready: **{report['ready']}**",
        f"- Labels CSV: `{report['labels_csv']}`",
        f"- Expected video IDs: **{report['n_expected_video_ids']}**",
        f"- Modal volume: `{report['modal_volume_name']}`",
        f"- Target prefix: `{report['modal_root'].rstrip('/')}/{report['target_prefix'].strip('/')}`",
        f"- Official Drive folder: `{report['official_source']['folder_id']}`",
        f"- Official video archive file: `{report['official_source']['video_file_id']}`",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Result",
        "",
        f"- OK: {ingest.get('ok')}",
        f"- Error: {ingest.get('error') or 'none'}",
        f"- Archive size: {int(ingest.get('archive_size_bytes') or 0):,} bytes",
        f"- Download seconds: {format_float(ingest.get('download_seconds'))}",
        f"- Extract seconds: {format_float(ingest.get('extract_seconds'))}",
        f"- Total seconds: {format_float(ingest.get('total_seconds'))}",
        f"- Copied: {copy_report.get('n_copied', 0)}",
        f"- Existing: {copy_report.get('n_existing', 0)}",
        f"- Missing expected: {copy_report.get('n_missing_expected', 0)}",
        f"- Target video count: {copy_report.get('target_video_count', 0)}",
    ]
    missing = copy_report.get("missing_expected_video_ids") or []
    if missing:
        lines.extend(["", "## Missing Expected IDs", ""])
        lines.extend(f"- `{video_id}`" for video_id in missing[:50])
    return "\n".join(lines) + "\n"


def format_float(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.2f}"


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


if __name__ == "__main__":
    main()
