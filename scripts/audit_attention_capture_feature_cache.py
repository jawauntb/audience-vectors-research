"""Build a portable checksum/provenance audit for TRIBE feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument(
        "--display-feature-dir",
        default=None,
        help=(
            "Portable path to record in the report. Defaults to --feature-dir. "
            "Use this when auditing an external cache mounted elsewhere."
        ),
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=[],
        help="Phase 1 manifest whose sample ids should be present in the cache.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--expected-vertices", type=int, default=20484)
    parser.add_argument(
        "--archive-uri",
        default=None,
        help=(
            "Optional durable artifact location for the audited cache, such as "
            "an object-storage URI. Credential-bearing URLs should not be used."
        ),
    )
    parser.add_argument(
        "--rerun-command",
        action="append",
        default=[],
        help=(
            "Deterministic command that can regenerate or re-audit the cache. "
            "May be passed multiple times."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_feature_cache(
        feature_dir=args.feature_dir,
        display_feature_dir=args.display_feature_dir,
        manifest_paths=args.manifest,
        expected_vertices=args.expected_vertices,
        archive_uri=args.archive_uri,
        rerun_commands=args.rerun_command,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_feature_cache_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    if not report["ready_for_reuse"]:
        raise SystemExit(1)


def audit_feature_cache(
    *,
    feature_dir: Path,
    display_feature_dir: str | None = None,
    manifest_paths: list[Path] | None = None,
    expected_vertices: int = 20484,
    archive_uri: str | None = None,
    rerun_commands: list[str] | None = None,
) -> dict[str, Any]:
    manifest_paths = manifest_paths or []
    rerun_commands = rerun_commands or []
    feature_paths = sorted(feature_dir.glob("*.npz"))
    file_audits = [
        audit_feature_file(path, feature_dir=feature_dir) for path in feature_paths
    ]
    expected_sample_ids = expected_ids_from_manifests(manifest_paths)
    actual_sample_ids = {
        str(item["sample_id"]) for item in file_audits if item["sample_id"] is not None
    }
    missing_expected = sorted(expected_sample_ids - actual_sample_ids)
    extra_sample_ids = (
        sorted(actual_sample_ids - expected_sample_ids) if expected_sample_ids else []
    )
    bad_npz = [item for item in file_audits if item["read_error"]]
    shape_mismatches = [
        item
        for item in file_audits
        if item["frames_shape"] is None
        or len(item["frames_shape"]) != 2
        or item["frames_shape"][1] != expected_vertices
    ]
    aggregate_sha256 = aggregate_file_digest(file_audits)
    blocking_reasons = feature_cache_blockers(
        feature_dir=feature_dir,
        file_audits=file_audits,
        missing_expected=missing_expected,
        bad_npz=bad_npz,
        shape_mismatches=shape_mismatches,
    )
    return {
        "schema_version": 2,
        "experiment": "attention_capture_feature_cache_audit",
        "feature_dir": display_feature_dir or str(feature_dir),
        "source_feature_dir_exists": feature_dir.is_dir(),
        "manifest_paths": [str(path) for path in manifest_paths],
        "expected_vertices": expected_vertices,
        "n_npz_files": len(file_audits),
        "n_expected_sample_ids": len(expected_sample_ids),
        "n_missing_expected_sample_ids": len(missing_expected),
        "n_extra_sample_ids": len(extra_sample_ids),
        "n_bad_npz": len(bad_npz),
        "n_shape_mismatches": len(shape_mismatches),
        "total_bytes": sum(int(item["size_bytes"]) for item in file_audits),
        "aggregate_sha256": aggregate_sha256,
        "archive_uri": archive_uri,
        "rerun_commands": rerun_commands,
        "ready_for_reproduction": bool(
            not blocking_reasons and (archive_uri or rerun_commands)
        ),
        "event_mode_counts": dict(
            Counter(str(item["event_mode"]) for item in file_audits)
        ),
        "transport_counts": dict(
            Counter(str(item["transport"]) for item in file_audits)
        ),
        "frame_shape_counts": dict(
            Counter(shape_key(item["frames_shape"]) for item in file_audits)
        ),
        "ready_for_reuse": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "missing_expected_sample_ids": missing_expected[:50],
        "extra_sample_ids": extra_sample_ids[:50],
        "bad_npz_files": [item["path"] for item in bad_npz[:50]],
        "shape_mismatch_files": [item["path"] for item in shape_mismatches[:50]],
        "files": file_audits,
        "claim_boundary": (
            "This audit verifies cached TRIBE feature artifact integrity and "
            "manifest coverage. It does not validate attentional capture."
        ),
    }


def audit_feature_file(path: Path, *, feature_dir: Path) -> dict[str, Any]:
    digest = sha256_file(path)
    size = path.stat().st_size
    metadata: dict[str, Any] = {
        "sample_id": None,
        "media_path": None,
        "transport": None,
        "event_mode": None,
        "duration_seconds": None,
        "frames_shape": None,
        "frames_dtype": None,
    }
    read_error = None
    try:
        with np.load(path, allow_pickle=False) as payload:
            frames = payload["frames"]
            metadata["frames_shape"] = list(frames.shape)
            metadata["frames_dtype"] = str(frames.dtype)
            for key in ("sample_id", "media_path", "transport", "event_mode"):
                if key in payload:
                    metadata[key] = scalar_value(payload[key])
            metadata["media_path"] = portable_metadata_path(metadata["media_path"])
            if "duration_seconds" in payload:
                metadata["duration_seconds"] = float(
                    np.asarray(payload["duration_seconds"])
                )
    except Exception as exc:  # noqa: BLE001
        read_error = str(exc)
    return {
        "path": portable_child_path(path, feature_dir=feature_dir),
        "sha256": digest,
        "size_bytes": size,
        "read_error": read_error,
        **metadata,
    }


def expected_ids_from_manifests(manifest_paths: list[Path]) -> set[str]:
    expected: set[str] = set()
    for path in manifest_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for sample in payload.get("samples") or []:
            if not isinstance(sample, dict):
                continue
            sample_id = sample.get("sample_id")
            if isinstance(sample_id, str) and sample_id:
                expected.add(sample_id)
    return expected


def feature_cache_blockers(
    *,
    feature_dir: Path,
    file_audits: list[dict[str, Any]],
    missing_expected: list[str],
    bad_npz: list[dict[str, Any]],
    shape_mismatches: list[dict[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if not feature_dir.is_dir():
        blockers.append("feature directory does not exist")
    if not file_audits:
        blockers.append("feature directory has no NPZ files")
    if missing_expected:
        blockers.append(f"{len(missing_expected)} manifest sample ids are missing")
    if bad_npz:
        blockers.append(f"{len(bad_npz)} NPZ files could not be read")
    if shape_mismatches:
        blockers.append(
            f"{len(shape_mismatches)} NPZ files have unexpected frames shape"
        )
    return blockers


def aggregate_file_digest(file_audits: list[dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for item in file_audits:
        line = f"{item['path']}\t{item['size_bytes']}\t{item['sha256']}\n"
        hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scalar_value(value: np.ndarray) -> str | float | int | None:
    out = np.asarray(value).item()
    if isinstance(out, bytes):
        return out.decode("utf-8")
    if isinstance(out, str | int | float):
        return out
    return str(out)


def portable_child_path(path: Path, *, feature_dir: Path) -> str:
    try:
        return str(path.relative_to(feature_dir))
    except ValueError:
        return path.name


def portable_metadata_path(value: object) -> object:
    if not isinstance(value, str):
        return value
    for marker, prefix in (
        ("/data/attention_capture/", "data/attention_capture/"),
        ("/data/features/", "data/features/"),
    ):
        if marker in value:
            return f"{prefix}{value.split(marker, 1)[1]}"
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def shape_key(shape: Any) -> str:
    if not isinstance(shape, list):
        return "missing"
    return "x".join(str(part) for part in shape)


def render_feature_cache_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Attention-Capture Feature Cache Audit",
        "",
        "## Verdict",
        "",
        f"- Feature dir: {report['feature_dir']}",
        f"- Ready for reuse: {report['ready_for_reuse']}",
        f"- NPZ files: {report['n_npz_files']}",
        f"- Expected sample ids: {report['n_expected_sample_ids']}",
        f"- Missing expected sample ids: {report['n_missing_expected_sample_ids']}",
        f"- Extra sample ids: {report['n_extra_sample_ids']}",
        f"- Bad NPZ files: {report['n_bad_npz']}",
        f"- Shape mismatches: {report['n_shape_mismatches']}",
        f"- Total bytes: {report['total_bytes']}",
        f"- Aggregate SHA-256: {report['aggregate_sha256']}",
        f"- Archive URI: {report['archive_uri'] or 'n/a'}",
        f"- Rerun commands: {len(report['rerun_commands'])}",
        f"- Ready for reproduction: {report['ready_for_reproduction']}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Blocking Reasons",
        "",
    ]
    blockers = report["blocking_reasons"]
    lines.extend(f"- {reason}" for reason in blockers) if blockers else lines.append(
        "- none",
    )
    lines.extend(["", "## Counts", ""])
    lines.append(f"- Event modes: {format_counts(report['event_mode_counts'])}")
    lines.append(f"- Transports: {format_counts(report['transport_counts'])}")
    lines.append(f"- Frame shapes: {format_counts(report['frame_shape_counts'])}")
    lines.extend(["", "## Reproduction Path", ""])
    if report["archive_uri"]:
        lines.append(f"- Archive URI: {report['archive_uri']}")
    else:
        lines.append("- Archive URI: n/a")
    rerun_commands = report["rerun_commands"]
    if rerun_commands:
        lines.extend(f"- `{command}`" for command in rerun_commands)
    else:
        lines.append("- Rerun commands: n/a")
    lines.extend(["", "## File Preview", ""])
    lines.extend(
        [
            "| path | sample_id | shape | event_mode | size | sha256 prefix |",
            "|---|---|---|---|---:|---|",
        ],
    )
    for item in report["files"][:20]:
        lines.append(
            "| "
            f"{item['path']} | {item['sample_id'] or 'n/a'} | "
            f"{shape_key(item['frames_shape'])} | {item['event_mode'] or 'n/a'} | "
            f"{item['size_bytes']} | {str(item['sha256'])[:12]} |"
        )
    if not report["files"]:
        lines.append("| none | n/a | n/a | n/a | 0 | n/a |")
    return "\n".join(lines) + "\n"


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


if __name__ == "__main__":
    main()
