"""Audit whether DHF1K Phase 1 videos are mounted for Modal TRIBE runs.

The local process reads the committed DHF1K label CSV and writes reports. The
existence check runs inside Modal next to the `bmd-videos-v1` volume that the
deployed TRIBE predictor already mounts at `/bmd-videos`.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import modal

APP_NAME = "attention-capture-dhf1k-modal-media"
DEFAULT_MODAL_VOLUME_NAME = "bmd-videos-v1"
DEFAULT_MODAL_ROOT = "/bmd-videos"
DEFAULT_MODAL_PREFIX = "attention_capture/DHF1K"
VIDEO_SUFFIXES = (".AVI", ".avi", ".mp4", ".MP4", ".mov", ".MOV")

image = modal.Image.debian_slim(python_version="3.12")
app = modal.App(APP_NAME)
dhf1k_media_volume = modal.Volume.from_name(
    DEFAULT_MODAL_VOLUME_NAME,
    create_if_missing=False,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--output-modal-csv",
        type=Path,
        default=None,
        help=(
            "Optional copy of --labels-csv with video_path rewritten to the "
            "Modal path expected by TRIBE. Extra provenance columns are added."
        ),
    )
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--video-id-column", default="video_id")
    parser.add_argument("--media-path-column", default="video_path")
    parser.add_argument("--modal-root", default=DEFAULT_MODAL_ROOT)
    parser.add_argument("--modal-prefix", default=DEFAULT_MODAL_PREFIX)
    parser.add_argument("--preview-limit", type=int, default=25)
    return parser.parse_args()


@app.function(
    image=image,
    cpu=1.0,
    memory=1024,
    timeout=10 * 60,
    volumes={DEFAULT_MODAL_ROOT: dhf1k_media_volume},
)
def audit_modal_media_rows(
    rows: list[dict[str, Any]],
    *,
    modal_root: str = DEFAULT_MODAL_ROOT,
    modal_prefix: str = DEFAULT_MODAL_PREFIX,
    preview_limit: int = 25,
) -> dict[str, Any]:
    root = Path(modal_root)
    prefix_path = root / modal_prefix.strip("/")
    video_dir = prefix_path / "video"
    checked_rows = [check_modal_media_row(row) for row in rows]
    found = [row for row in checked_rows if row["found"]]
    missing = [row for row in checked_rows if not row["found"]]
    zero_byte = [
        row
        for row in checked_rows
        if row["found"] and int(row.get("size_bytes") or 0) <= 0
    ]
    return {
        "modal_root": modal_root,
        "modal_prefix": modal_prefix,
        "volume_mount_exists": root.exists(),
        "prefix_exists": prefix_path.exists(),
        "video_dir_exists": video_dir.exists(),
        "n_expected": len(checked_rows),
        "n_found": len(found),
        "n_missing": len(missing),
        "n_zero_byte_found": len(zero_byte),
        "found_preview": found[:preview_limit],
        "missing_preview": missing[:preview_limit],
        "zero_byte_preview": zero_byte[:preview_limit],
        "checked_rows": checked_rows,
    }


@app.local_entrypoint()
def main(
    labels_csv: str,
    output_json: str,
    output_md: str,
    output_modal_csv: str | None = None,
    sample_id_column: str = "sample_id",
    video_id_column: str = "video_id",
    media_path_column: str = "video_path",
    modal_root: str = DEFAULT_MODAL_ROOT,
    modal_prefix: str = DEFAULT_MODAL_PREFIX,
    preview_limit: int = 25,
) -> None:
    labels_csv_path = Path(labels_csv)
    output_json_path = Path(output_json)
    output_md_path = Path(output_md)
    output_modal_csv_path = Path(output_modal_csv) if output_modal_csv else None
    expected_rows = build_expected_rows(
        labels_csv_path,
        sample_id_column=sample_id_column,
        video_id_column=video_id_column,
        media_path_column=media_path_column,
        modal_root=modal_root,
        modal_prefix=modal_prefix,
    )
    remote_audit = audit_modal_media_rows.remote(
        expected_rows,
        modal_root=modal_root,
        modal_prefix=modal_prefix,
        preview_limit=preview_limit,
    )
    if output_modal_csv_path is not None:
        write_modal_path_csv(
            labels_csv=labels_csv_path,
            output_csv=output_modal_csv_path,
            checked_rows=remote_audit["checked_rows"],
            sample_id_column=sample_id_column,
            media_path_column=media_path_column,
        )
    report = build_report(
        labels_csv=labels_csv_path,
        output_modal_csv=output_modal_csv_path,
        sample_id_column=sample_id_column,
        video_id_column=video_id_column,
        media_path_column=media_path_column,
        modal_root=modal_root,
        modal_prefix=modal_prefix,
        remote_audit=remote_audit,
    )
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md_path.write_text(render_dhf1k_modal_media_markdown(report), encoding="utf-8")
    print(f"wrote {output_json_path}")
    print(f"wrote {output_md_path}")
    if output_modal_csv_path is not None:
        print(f"wrote {output_modal_csv_path}")


def build_expected_rows(
    labels_csv: Path,
    *,
    sample_id_column: str,
    video_id_column: str,
    media_path_column: str,
    modal_root: str,
    modal_prefix: str,
) -> list[dict[str, Any]]:
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{labels_csv} is missing a CSV header")
        rows = [dict(row) for row in reader]

    expected: list[dict[str, Any]] = []
    for row in rows:
        sample_id = required_cell(row, sample_id_column, labels_csv)
        video_id = row.get(video_id_column) or video_id_from_sample_id(sample_id)
        source_video_path = required_cell(row, media_path_column, labels_csv)
        expected_video_path = modal_video_path_from_source(
            source_video_path,
            modal_root=modal_root,
            modal_prefix=modal_prefix,
            video_id=video_id,
        )
        expected.append(
            {
                "sample_id": sample_id,
                "video_id": video_id,
                "source_video_path": source_video_path,
                "expected_modal_video_path": expected_video_path,
                "candidate_modal_paths": candidate_modal_video_paths(
                    expected_video_path,
                    video_id=video_id,
                ),
            }
        )
    return expected


def check_modal_media_row(row: dict[str, Any]) -> dict[str, Any]:
    for candidate in row.get("candidate_modal_paths") or []:
        path = Path(str(candidate))
        if path.is_file():
            stat = path.stat()
            return {
                **row,
                "found": True,
                "found_path": str(path),
                "size_bytes": int(stat.st_size),
            }
    return {
        **row,
        "found": False,
        "found_path": None,
        "size_bytes": None,
    }


def build_report(
    *,
    labels_csv: Path,
    output_modal_csv: Path | None,
    sample_id_column: str,
    video_id_column: str,
    media_path_column: str,
    modal_root: str,
    modal_prefix: str,
    remote_audit: dict[str, Any],
) -> dict[str, Any]:
    ready = (
        bool(remote_audit.get("volume_mount_exists"))
        and int(remote_audit.get("n_expected") or 0) > 0
        and int(remote_audit.get("n_missing") or 0) == 0
        and int(remote_audit.get("n_zero_byte_found") or 0) == 0
    )
    return {
        "schema_version": 1,
        "experiment": "dhf1k_modal_media_audit",
        "generated_at": datetime.now(UTC).isoformat(),
        "labels_csv": str(labels_csv),
        "output_modal_csv": str(output_modal_csv) if output_modal_csv else None,
        "sample_id_column": sample_id_column,
        "video_id_column": video_id_column,
        "media_path_column": media_path_column,
        "modal_volume_name": DEFAULT_MODAL_VOLUME_NAME,
        "modal_root": modal_root,
        "modal_prefix": modal_prefix,
        "ready_for_full_feature_extraction": ready,
        "blocking_reasons": blocking_reasons(remote_audit),
        "remote_audit": remote_audit,
        "recommended_full_extraction_command": recommended_full_extraction_command(
            output_modal_csv=output_modal_csv,
            media_path_column=media_path_column,
        ),
        "claim_boundary": (
            "This Modal CPU audit checks whether the DHF1K media files needed "
            "for full-mode TRIBE extraction are mounted. It does not score "
            "TRIBE features or validate attentional capture."
        ),
    }


def blocking_reasons(remote_audit: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not remote_audit.get("volume_mount_exists"):
        reasons.append(f"Modal mount {DEFAULT_MODAL_ROOT} is not visible")
    if int(remote_audit.get("n_expected") or 0) == 0:
        reasons.append("labels CSV produced zero expected DHF1K media rows")
    if not remote_audit.get("video_dir_exists"):
        reasons.append(
            "DHF1K video directory is not mounted at "
            f"{remote_audit.get('modal_root')}/{str(remote_audit.get('modal_prefix')).strip('/')}/video"
        )
    missing = int(remote_audit.get("n_missing") or 0)
    if missing:
        reasons.append(
            f"{missing} expected DHF1K videos are missing from Modal volume "
            f"{DEFAULT_MODAL_VOLUME_NAME}"
        )
    zero_byte = int(remote_audit.get("n_zero_byte_found") or 0)
    if zero_byte:
        reasons.append(f"{zero_byte} mounted DHF1K videos are zero-byte files")
    return reasons


def modal_video_path_from_source(
    source_video_path: str,
    *,
    modal_root: str,
    modal_prefix: str,
    video_id: str,
) -> str:
    normalized = source_video_path.replace("\\", "/")
    if normalized.startswith(f"{modal_root.rstrip('/')}/"):
        return normalized
    relative = dhf1k_relative_video_path(normalized, video_id=video_id)
    return join_modal_path(modal_root, modal_prefix, relative)


def dhf1k_relative_video_path(source_video_path: str, *, video_id: str) -> PurePosixPath:
    path = PurePosixPath(source_video_path)
    parts = path.parts
    lower_parts = [part.lower() for part in parts]
    if "dhf1k" in lower_parts:
        idx = lower_parts.index("dhf1k")
        suffix = parts[idx + 1 :]
        if suffix:
            return PurePosixPath(*suffix)
    return PurePosixPath("video") / f"{int(video_id):03d}.AVI"


def join_modal_path(
    modal_root: str,
    modal_prefix: str,
    relative_path: PurePosixPath,
) -> str:
    parts = [
        modal_root.strip("/"),
        modal_prefix.strip("/"),
        str(relative_path).strip("/"),
    ]
    return "/" + "/".join(part for part in parts if part)


def candidate_modal_video_paths(expected_path: str, *, video_id: str) -> list[str]:
    path = PurePosixPath(expected_path)
    parent = path.parent
    suffixes = [path.suffix, *VIDEO_SUFFIXES]
    candidates = [expected_path]
    for candidate_id in video_id_variants(video_id):
        for suffix in suffixes:
            if suffix:
                candidates.append(str(parent / f"{candidate_id}{suffix}"))
    return dedupe(candidates)


def video_id_variants(video_id: str) -> list[str]:
    value = int(video_id)
    return [f"{value:03d}", f"{value:04d}"]


def video_id_from_sample_id(sample_id: str) -> str:
    tail = sample_id.rsplit("_", 1)[-1]
    if not tail.isdigit():
        raise ValueError(f"cannot derive DHF1K video_id from sample_id={sample_id!r}")
    return f"{int(tail):03d}"


def write_modal_path_csv(
    *,
    labels_csv: Path,
    output_csv: Path,
    checked_rows: list[dict[str, Any]],
    sample_id_column: str,
    media_path_column: str,
) -> None:
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{labels_csv} is missing a CSV header")
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames)

    by_sample_id = {str(row["sample_id"]): row for row in checked_rows}
    extra_columns = (
        "source_video_path",
        "modal_media_found",
        "modal_media_size_bytes",
    )
    for column in extra_columns:
        if column not in fieldnames:
            fieldnames.append(column)
    for row in rows:
        sample_id = required_cell(row, sample_id_column, labels_csv)
        checked = by_sample_id[sample_id]
        source_video_path = row.get(media_path_column, "")
        row[media_path_column] = (
            str(checked.get("found_path"))
            if checked.get("found_path")
            else str(checked["expected_modal_video_path"])
        )
        row["source_video_path"] = source_video_path
        row["modal_media_found"] = str(bool(checked.get("found")))
        row["modal_media_size_bytes"] = (
            "" if checked.get("size_bytes") is None else str(checked["size_bytes"])
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def recommended_full_extraction_command(
    *,
    output_modal_csv: Path | None,
    media_path_column: str,
) -> str | None:
    if output_modal_csv is None:
        return None
    return (
        "uv run --extra modal python "
        "scripts/extract_attention_capture_tribe_features.py "
        f"--source-csv {output_modal_csv} "
        "--output-dir data/features/tribe_dhf1k_attention_full "
        "--sample-id-column sample_id "
        f"--media-path-column {media_path_column} "
        "--transport path "
        "--event-mode full "
        "--concurrency 8"
    )


def render_dhf1k_modal_media_markdown(report: dict[str, Any]) -> str:
    audit = report["remote_audit"]
    lines = [
        "# DHF1K Modal Media Audit",
        "",
        f"- Labels CSV: `{report['labels_csv']}`",
        f"- Modal path CSV: `{report['output_modal_csv'] or 'not written'}`",
        f"- Modal volume: `{report['modal_volume_name']}`",
        f"- Modal prefix: `{report['modal_root'].rstrip('/')}/{report['modal_prefix'].strip('/')}`",
        f"- Expected videos: **{audit['n_expected']}**",
        f"- Found videos: **{audit['n_found']}**",
        f"- Missing videos: **{audit['n_missing']}**",
        f"- Zero-byte videos: **{audit['n_zero_byte_found']}**",
        f"- Ready for full feature extraction: **{report['ready_for_full_feature_extraction']}**",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Blocking Reasons",
        "",
    ]
    reasons = report["blocking_reasons"]
    lines.extend(f"- {reason}" for reason in reasons) if reasons else lines.append(
        "- none",
    )
    lines.extend(["", "## Missing Preview", ""])
    missing = audit.get("missing_preview") or []
    if not missing:
        lines.append("- none")
    else:
        lines.extend(
            f"- `{row['sample_id']}` -> `{row['expected_modal_video_path']}`"
            for row in missing
        )
    lines.extend(["", "## Found Preview", ""])
    found = audit.get("found_preview") or []
    if not found:
        lines.append("- none")
    else:
        lines.extend(
            f"- `{row['sample_id']}` -> `{row['found_path']}` "
            f"({int(row['size_bytes']):,} bytes)"
            for row in found
        )
    command = report.get("recommended_full_extraction_command")
    if command:
        lines.extend(
            [
                "",
                "## Full-Mode Extraction Command",
                "",
                "Run this only after the audit reports ready:",
                "",
                "```bash",
                command,
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def required_cell(row: dict[str, str], column: str, path: Path) -> str:
    value = row.get(column)
    if value is None or value == "":
        raise ValueError(f"{path} has a row missing required column {column!r}")
    return value


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
