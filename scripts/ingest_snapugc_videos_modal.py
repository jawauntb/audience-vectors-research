"""Ingest official SnapUGC/VQualA train videos into a Modal Volume.

The heavy work runs in Modal: download the Google Drive folder containing the
split `train_videos_split` archive, extract it on the container SSD, copy the
videos into `bmd-videos-v1`, then commit the volume. The local process only
launches the job and writes a small provenance report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "attention-capture-snapugc-video-ingest"
DEFAULT_MODAL_VOLUME_NAME = "bmd-videos-v1"
DEFAULT_MODAL_ROOT = "/bmd-videos"
DEFAULT_TARGET_PREFIX = "attention_capture/SnapUGC"
OFFICIAL_SNAPUGC_TRAIN_VIDEO_FOLDER_ID = "134gJflcaQ7Dhg5EUKfLdeXW61fj1fiNo"
OFFICIAL_SNAPUGC_TRAIN_LABEL_FILE_ID = "1Mv5Esq5gGuxRTayabRUb5NmHwEN3JdbD"
DEFAULT_SPLIT_ARCHIVE_STEM = "train_videos_split"
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")
OFFICIAL_SNAPUGC_TRAIN_VIDEO_PART_IDS = {
    "train_videos_split.z01": "1t7bcW1Ep0FElD716VUo-EPS1pkE_zcFG",
    "train_videos_split.z02": "1Bg4_a3WMJsvn-NygAUHA2VsnZ5sjE08u",
    "train_videos_split.z03": "1-UZHu_cUSSjxm3S8xncXn8EYlf-RoRXV",
    "train_videos_split.z04": "1ICZDXvQ0TNhteE72zwAjO-q76_fW3tH_",
    "train_videos_split.z05": "1p0K-kMjpWkzxsVy0gNQHxtTNFG0aE0Ls",
    "train_videos_split.z06": "1KeTMybOYJ8CrADrriNmKHxRK9YCJNaiX",
    "train_videos_split.z07": "1FLqeIvvNBMWFyxFP5s8OuYApwCew-0xF",
    "train_videos_split.z08": "1-UmjuxgCFXop2xHEXMxT4pzO_eV5xzvm",
    "train_videos_split.z09": "1wrAH9OrV2QpJCuyzT6zC9cqXTx32QNd0",
    "train_videos_split.z10": "1TsFrmTLvN90rrI5qanOIaOM5_Cc8RsJO",
    "train_videos_split.z11": "1oSGu3h4PncJ8XFCKlTkkL-SOeeMPdI9K",
    "train_videos_split.z12": "1KZLXxNCJ29CsZ5_cV7vYlAU8iyDCZXTN",
    "train_videos_split.z13": "1Tr88-m4ypjFAusjs-vaFkcd6HWoaCyPy",
    "train_videos_split.z14": "1Oxy8chmPThdYBz0aA4lkogZEY8fLPnUN",
    "train_videos_split.zip": "1oTsaYeeZRfYsAepATu5JQUenjxvqTga5",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("p7zip-full")
    .pip_install("gdown")
)
app = modal.App(APP_NAME)
bmd_videos_volume = modal.Volume.from_name(
    DEFAULT_MODAL_VOLUME_NAME,
    create_if_missing=True,
)


class ArchiveDownloadError(RuntimeError):
    def __init__(self, message: str, *, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class SplitZipParts:
    main_zip: Path
    numbered_parts: list[Path]
    missing_part_suffixes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--google-drive-folder-id",
        default=OFFICIAL_SNAPUGC_TRAIN_VIDEO_FOLDER_ID,
        help="Google Drive folder containing train_videos_split.z01...zip.",
    )
    parser.add_argument(
        "--google-drive-label-file-id",
        default=OFFICIAL_SNAPUGC_TRAIN_LABEL_FILE_ID,
        help=(
            "Optional train_data.csv Google Drive file id. Use an empty string "
            "to skip label CSV download into the Modal volume."
        ),
    )
    parser.add_argument("--split-archive-stem", default=DEFAULT_SPLIT_ARCHIVE_STEM)
    parser.add_argument("--target-prefix", default=DEFAULT_TARGET_PREFIX)
    parser.add_argument(
        "--expected-min-videos",
        type=int,
        default=1,
        help="Minimum copied/existing target videos required for ready=true.",
    )
    parser.add_argument("--preview-limit", type=int, default=25)
    return parser.parse_args()


@app.function(
    image=image,
    cpu=4.0,
    memory=16 * 1024,
    ephemeral_disk=768 * 1024,
    timeout=12 * 60 * 60,
    volumes={DEFAULT_MODAL_ROOT: bmd_videos_volume},
)
def ingest_snapugc_train_videos(
    *,
    google_drive_folder_id: str = OFFICIAL_SNAPUGC_TRAIN_VIDEO_FOLDER_ID,
    google_drive_label_file_id: str | None = OFFICIAL_SNAPUGC_TRAIN_LABEL_FILE_ID,
    split_archive_stem: str = DEFAULT_SPLIT_ARCHIVE_STEM,
    target_prefix: str = DEFAULT_TARGET_PREFIX,
    expected_min_videos: int = 1,
    preview_limit: int = 25,
) -> dict[str, Any]:
    try:
        return ingest_snapugc_train_videos_impl(
            google_drive_folder_id=google_drive_folder_id,
            google_drive_label_file_id=google_drive_label_file_id,
            split_archive_stem=split_archive_stem,
            target_prefix=target_prefix,
            expected_min_videos=expected_min_videos,
            preview_limit=preview_limit,
    )
    except Exception as exc:  # noqa: BLE001
        commit_volume_best_effort()
        return {
            "ok": False,
            "ready": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "archive_report": getattr(exc, "report", None),
            "google_drive_folder_id": google_drive_folder_id,
            "google_drive_label_file_id": google_drive_label_file_id,
            "split_archive_stem": split_archive_stem,
            "target_prefix": target_prefix,
            "claim_boundary": ingest_claim_boundary(),
        }


@app.local_entrypoint()
def main(
    output_json: str,
    output_md: str,
    google_drive_folder_id: str = OFFICIAL_SNAPUGC_TRAIN_VIDEO_FOLDER_ID,
    google_drive_label_file_id: str = OFFICIAL_SNAPUGC_TRAIN_LABEL_FILE_ID,
    split_archive_stem: str = DEFAULT_SPLIT_ARCHIVE_STEM,
    target_prefix: str = DEFAULT_TARGET_PREFIX,
    expected_min_videos: int = 1,
    preview_limit: int = 25,
) -> None:
    output_json_path = Path(output_json)
    output_md_path = Path(output_md)
    label_file_id = google_drive_label_file_id or None
    ingest = ingest_snapugc_train_videos.remote(
        google_drive_folder_id=google_drive_folder_id,
        google_drive_label_file_id=label_file_id,
        split_archive_stem=split_archive_stem,
        target_prefix=target_prefix,
        expected_min_videos=expected_min_videos,
        preview_limit=preview_limit,
    )
    report = {
        "schema_version": 1,
        "experiment": "snapugc_modal_video_ingest",
        "generated_at": datetime.now(UTC).isoformat(),
        "modal_volume_name": DEFAULT_MODAL_VOLUME_NAME,
        "modal_root": DEFAULT_MODAL_ROOT,
        "target_prefix": target_prefix,
        "official_source": {
            "train_video_folder_id": google_drive_folder_id,
            "train_video_folder_url": (
                "https://drive.google.com/drive/folders/"
                f"{google_drive_folder_id}"
            ),
            "train_label_file_id": label_file_id,
            "train_label_file_url": (
                "https://drive.google.com/file/d/"
                f"{label_file_id}/view"
                if label_file_id
                else None
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


def ingest_snapugc_train_videos_impl(
    *,
    google_drive_folder_id: str,
    google_drive_label_file_id: str | None,
    split_archive_stem: str,
    target_prefix: str,
    expected_min_videos: int,
    preview_limit: int,
) -> dict[str, Any]:
    import gdown  # type: ignore[import-not-found]

    started = time.monotonic()
    staging_root = Path("/tmp/snapugc_video_ingest")
    extract_dir = staging_root / "extracted"
    reset_dir(staging_root)
    extract_dir.mkdir(parents=True, exist_ok=True)

    download_started = time.monotonic()
    archive_report = download_split_archive_parts(
        gdown_module=gdown,
        google_drive_folder_id=google_drive_folder_id,
        split_archive_stem=split_archive_stem,
        target_prefix=target_prefix,
        preview_limit=preview_limit,
    )
    download_seconds = time.monotonic() - download_started
    downloaded_paths = [Path(path) for path in archive_report["paths"]]
    if not downloaded_paths:
        raise RuntimeError("gdown did not download any SnapUGC archive parts")

    split_parts = find_split_zip_parts(
        Path(archive_report["archive_dir"]),
        stem=split_archive_stem,
    )
    if split_parts.missing_part_suffixes:
        missing = ", ".join(split_parts.missing_part_suffixes)
        raise RuntimeError(f"split archive is missing parts: {missing}")

    label_report = maybe_download_label_csv(
        google_drive_label_file_id=google_drive_label_file_id,
        target_prefix=target_prefix,
    )

    extract_started = time.monotonic()
    subprocess.run(
        ["7z", "x", str(split_parts.main_zip), f"-o{extract_dir}", "-y"],
        check=True,
        timeout=8 * 60 * 60,
    )
    extract_seconds = time.monotonic() - extract_started

    source_video_paths = collect_video_paths(extract_dir)
    target_video_dir = Path(DEFAULT_MODAL_ROOT) / target_prefix.strip("/") / "video"
    target_video_dir.mkdir(parents=True, exist_ok=True)
    copy_report = copy_videos_to_volume(
        source_video_paths=source_video_paths,
        target_video_dir=target_video_dir,
        preview_limit=preview_limit,
    )
    bmd_videos_volume.commit()
    n_ready_videos = copy_report["n_copied"] + copy_report["n_existing"]
    ready = (
        bool(copy_report["ok"])
        and n_ready_videos >= expected_min_videos
        and len(source_video_paths) >= expected_min_videos
    )
    return {
        "ok": bool(copy_report["ok"]),
        "ready": ready,
        "target_video_dir": str(target_video_dir),
        "target_label_csv": label_report.get("target_label_csv"),
        "archive_report": archive_report,
        "extract_dir": str(extract_dir),
        "downloaded_path_count": len(downloaded_paths),
        "downloaded_paths_preview": [str(path) for path in downloaded_paths[:preview_limit]],
        "split_archive": {
            "main_zip": str(split_parts.main_zip),
            "numbered_parts": [str(path) for path in split_parts.numbered_parts],
            "n_numbered_parts": len(split_parts.numbered_parts),
            "missing_part_suffixes": split_parts.missing_part_suffixes,
            "total_size_bytes": split_zip_size_bytes(split_parts),
        },
        "n_source_videos": len(source_video_paths),
        "expected_min_videos": expected_min_videos,
        "download_seconds": download_seconds,
        "extract_seconds": extract_seconds,
        "total_seconds": time.monotonic() - started,
        "label_report": label_report,
        "copy_report": copy_report,
        "claim_boundary": ingest_claim_boundary(),
    }


def download_split_archive_parts(
    *,
    gdown_module: Any,
    google_drive_folder_id: str,
    split_archive_stem: str,
    target_prefix: str,
    preview_limit: int,
) -> dict[str, Any]:
    archive_dir = archive_cache_dir(
        target_prefix=target_prefix,
        split_archive_stem=split_archive_stem,
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    if is_official_snapugc_archive(
        google_drive_folder_id=google_drive_folder_id,
        split_archive_stem=split_archive_stem,
    ):
        return download_official_split_archive_parts(
            gdown_module=gdown_module,
            archive_dir=archive_dir,
            preview_limit=preview_limit,
        )

    downloaded_paths = gdown_module.download_folder(
        id=google_drive_folder_id,
        output=str(archive_dir),
        quiet=False,
        use_cookies=False,
    )
    commit_volume_best_effort()
    if not downloaded_paths:
        raise RuntimeError("gdown did not download any SnapUGC archive parts")
    paths = [str(Path(path)) for path in downloaded_paths]
    return {
        "ok": True,
        "mode": "drive_folder",
        "archive_dir": str(archive_dir),
        "paths": paths,
        "paths_preview": paths[:preview_limit],
        "n_paths": len(paths),
        "n_downloaded": len(paths),
        "n_existing": 0,
        "n_failed": 0,
    }


def download_official_split_archive_parts(
    *,
    gdown_module: Any,
    archive_dir: Path,
    preview_limit: int,
    max_attempts: int = 2,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    paths: list[str] = []
    for filename, file_id in OFFICIAL_SNAPUGC_TRAIN_VIDEO_PART_IDS.items():
        target_path = archive_dir / filename
        row: dict[str, Any] = {
            "filename": filename,
            "file_id": file_id,
            "path": str(target_path),
        }
        if archive_part_ready(target_path):
            row.update(
                {
                    "status": "existing",
                    "size_bytes": target_path.stat().st_size,
                    "attempts": 0,
                }
            )
            files.append(row)
            paths.append(str(target_path))
            continue

        temp_path = target_path.with_name(f"{target_path.name}.download")
        errors: list[str] = []
        for attempt in range(1, max_attempts + 1):
            if temp_path.exists():
                temp_path.unlink()
            try:
                downloaded = gdown_module.download(
                    id=file_id,
                    output=str(temp_path),
                    quiet=False,
                    use_cookies=False,
                )
                if downloaded is None or not archive_part_ready(temp_path):
                    raise RuntimeError(f"gdown did not produce {filename}")
                temp_path.replace(target_path)
                row.update(
                    {
                        "status": "downloaded",
                        "size_bytes": target_path.stat().st_size,
                        "attempts": attempt,
                    }
                )
                files.append(row)
                paths.append(str(target_path))
                bmd_videos_volume.commit()
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if temp_path.exists():
                    temp_path.unlink()
                if attempt < max_attempts:
                    time.sleep(15 * attempt)
        else:
            row.update(
                {
                    "status": "failed",
                    "attempts": max_attempts,
                    "errors": errors,
                }
            )
            files.append(row)
            report = build_archive_download_report(
                archive_dir=archive_dir,
                files=files,
                paths=paths,
                mode="official_file_ids",
                preview_limit=preview_limit,
            )
            commit_volume_best_effort()
            raise ArchiveDownloadError(
                f"failed to download {filename} after {max_attempts} attempts",
                report=report,
            )

    report = build_archive_download_report(
        archive_dir=archive_dir,
        files=files,
        paths=paths,
        mode="official_file_ids",
        preview_limit=preview_limit,
    )
    bmd_videos_volume.commit()
    return report


def archive_cache_dir(*, target_prefix: str, split_archive_stem: str) -> Path:
    return (
        Path(DEFAULT_MODAL_ROOT)
        / target_prefix.strip("/")
        / "archive"
        / safe_name(split_archive_stem)
    )


def is_official_snapugc_archive(
    *,
    google_drive_folder_id: str,
    split_archive_stem: str,
) -> bool:
    return (
        google_drive_folder_id == OFFICIAL_SNAPUGC_TRAIN_VIDEO_FOLDER_ID
        and split_archive_stem == DEFAULT_SPLIT_ARCHIVE_STEM
    )


def archive_part_ready(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def build_archive_download_report(
    *,
    archive_dir: Path,
    files: list[dict[str, Any]],
    paths: list[str],
    mode: str,
    preview_limit: int,
) -> dict[str, Any]:
    return {
        "ok": all(file["status"] != "failed" for file in files),
        "mode": mode,
        "archive_dir": str(archive_dir),
        "paths": paths,
        "paths_preview": paths[:preview_limit],
        "n_paths": len(paths),
        "n_files": len(files),
        "n_downloaded": sum(1 for file in files if file["status"] == "downloaded"),
        "n_existing": sum(1 for file in files if file["status"] == "existing"),
        "n_failed": sum(1 for file in files if file["status"] == "failed"),
        "files": files,
    }


def commit_volume_best_effort() -> None:
    try:
        bmd_videos_volume.commit()
    except Exception:  # noqa: BLE001
        pass


def maybe_download_label_csv(
    *,
    google_drive_label_file_id: str | None,
    target_prefix: str,
) -> dict[str, Any]:
    if not google_drive_label_file_id:
        return {"downloaded": False, "target_label_csv": None}

    import gdown  # type: ignore[import-not-found]

    label_dir = Path(DEFAULT_MODAL_ROOT) / target_prefix.strip("/") / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    target_label_csv = label_dir / "train_data.csv"
    if target_label_csv.exists() and target_label_csv.stat().st_size > 0:
        return {
            "downloaded": False,
            "existing": True,
            "target_label_csv": str(target_label_csv),
            "size_bytes": target_label_csv.stat().st_size,
            "google_drive_label_file_id": google_drive_label_file_id,
        }
    downloaded = gdown.download(
        id=google_drive_label_file_id,
        output=str(target_label_csv),
        quiet=False,
        use_cookies=False,
    )
    if downloaded is None or not target_label_csv.exists():
        raise RuntimeError("gdown did not produce SnapUGC train_data.csv")
    return {
        "downloaded": True,
        "target_label_csv": str(target_label_csv),
        "size_bytes": target_label_csv.stat().st_size,
        "google_drive_label_file_id": google_drive_label_file_id,
    }


def find_split_zip_parts(root: Path, *, stem: str) -> SplitZipParts:
    candidates: list[SplitZipParts] = []
    for main_zip in root.rglob(f"{stem}.zip"):
        numbered_parts = sorted(
            (
                path
                for path in main_zip.parent.glob(f"{stem}.z[0-9][0-9]")
                if split_part_number(path) is not None
            ),
            key=lambda path: split_part_number(path) or 0,
        )
        candidates.append(
            SplitZipParts(
                main_zip=main_zip,
                numbered_parts=numbered_parts,
                missing_part_suffixes=missing_part_suffixes(numbered_parts),
            )
        )
    if not candidates:
        raise RuntimeError(f"no {stem}.zip found under {root}")
    return max(candidates, key=lambda parts: len(parts.numbered_parts))


def split_part_number(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if len(suffix) != 4 or not suffix.startswith(".z"):
        return None
    number = suffix[2:]
    if not number.isdigit():
        return None
    return int(number)


def missing_part_suffixes(numbered_parts: list[Path]) -> list[str]:
    if not numbered_parts:
        return ["z01"]
    present = {split_part_number(path) for path in numbered_parts}
    max_part = max(part for part in present if part is not None)
    return [
        f"z{part:02d}"
        for part in range(1, max_part + 1)
        if part not in present
    ]


def split_zip_size_bytes(parts: SplitZipParts) -> int:
    return parts.main_zip.stat().st_size + sum(
        path.stat().st_size for path in parts.numbered_parts
    )


def collect_video_paths(root: Path) -> list[Path]:
    suffixes = {suffix.lower() for suffix in VIDEO_SUFFIXES}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def copy_videos_to_volume(
    *,
    source_video_paths: list[Path],
    target_video_dir: Path,
    preview_limit: int,
) -> dict[str, Any]:
    copied: list[dict[str, Any]] = []
    existing: list[dict[str, Any]] = []
    overwritten: list[dict[str, Any]] = []
    target_names = unique_target_names(source_video_paths)
    for source_path in source_video_paths:
        target_path = target_video_dir / target_names[source_path]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "source_path": str(source_path),
            "target_path": str(target_path),
            "size_bytes": source_path.stat().st_size,
        }
        if target_path.exists():
            if target_path.stat().st_size == source_path.stat().st_size:
                existing.append(row)
                continue
            overwritten.append(
                {
                    **row,
                    "previous_size_bytes": target_path.stat().st_size,
                }
            )
        shutil.copy2(source_path, target_path)
        copied.append(row)

    return {
        "ok": bool(source_video_paths),
        "n_source_videos": len(source_video_paths),
        "target_video_count": count_video_files(target_video_dir),
        "n_copied": len(copied),
        "n_existing": len(existing),
        "n_overwritten": len(overwritten),
        "copied_preview": copied[:preview_limit],
        "existing_preview": existing[:preview_limit],
        "overwritten_preview": overwritten[:preview_limit],
    }


def unique_target_names(source_video_paths: list[Path]) -> dict[Path, str]:
    counts: dict[str, int] = {}
    for path in source_video_paths:
        counts[path.name] = counts.get(path.name, 0) + 1
    names: dict[Path, str] = {}
    for path in source_video_paths:
        if counts[path.name] == 1:
            names[path] = path.name
            continue
        parent = safe_name(path.parent.name)
        names[path] = f"{parent}__{path.name}"
    return names


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def count_video_files(video_dir: Path) -> int:
    if not video_dir.is_dir():
        return 0
    suffixes = {suffix.lower() for suffix in VIDEO_SUFFIXES}
    return sum(
        1
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ingest_claim_boundary() -> str:
    return (
        "This ingests SnapUGC/VQualA media into a Modal Volume for later TRIBE "
        "feature extraction. It does not validate labels, score TRIBE, or "
        "support Phase 1/2/3 claims by itself."
    )


def render_ingest_markdown(report: dict[str, Any]) -> str:
    ingest = report["ingest"]
    copy_report = ingest.get("copy_report") or {}
    split_archive = ingest.get("split_archive") or {}
    label_report = ingest.get("label_report") or {}
    archive_report = ingest.get("archive_report") or {}
    downloaded_path_count = ingest.get("downloaded_path_count")
    if not isinstance(downloaded_path_count, int):
        downloaded_path_count = int(archive_report.get("n_paths") or 0)
    lines = [
        "# SnapUGC Modal Video Ingest",
        "",
        f"- Ready: **{report['ready']}**",
        f"- Modal volume: `{report['modal_volume_name']}`",
        f"- Target prefix: `{report['modal_root'].rstrip('/')}/{report['target_prefix'].strip('/')}`",
        f"- Train video folder: `{report['official_source']['train_video_folder_id']}`",
        f"- Train label file: `{report['official_source']['train_label_file_id'] or 'skipped'}`",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Result",
        "",
        f"- OK: {ingest.get('ok')}",
        f"- Error: {ingest.get('error') or 'none'}",
        f"- Archive cache: `{archive_report.get('archive_dir') or 'none'}`",
        f"- Archive download mode: {archive_report.get('mode') or 'none'}",
        f"- Archive downloaded: {archive_report.get('n_downloaded', 0)}",
        f"- Archive existing: {archive_report.get('n_existing', 0)}",
        f"- Archive failed: {archive_report.get('n_failed', 0)}",
        f"- Downloaded/cached paths: {downloaded_path_count}",
        f"- Split numbered parts: {split_archive.get('n_numbered_parts', 0)}",
        f"- Split archive size: {int(split_archive.get('total_size_bytes') or 0):,} bytes",
        f"- Source videos found: {ingest.get('n_source_videos', 0)}",
        f"- Copied: {copy_report.get('n_copied', 0)}",
        f"- Existing: {copy_report.get('n_existing', 0)}",
        f"- Overwritten: {copy_report.get('n_overwritten', 0)}",
        f"- Target video count: {copy_report.get('target_video_count', 0)}",
        f"- Label CSV downloaded: {label_report.get('downloaded', False)}",
        f"- Target label CSV: `{label_report.get('target_label_csv') or 'none'}`",
        f"- Download seconds: {format_float(ingest.get('download_seconds'))}",
        f"- Extract seconds: {format_float(ingest.get('extract_seconds'))}",
        f"- Total seconds: {format_float(ingest.get('total_seconds'))}",
    ]
    missing = split_archive.get("missing_part_suffixes") or []
    if missing:
        lines.extend(["", "## Missing Split Parts", ""])
        lines.extend(f"- `{part}`" for part in missing)
    return "\n".join(lines) + "\n"


def format_float(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.2f}"


if __name__ == "__main__":
    main()
