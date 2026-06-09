"""Audit local data readiness for Phase 1 attention-capture validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

VIDEO_SUFFIXES = (".AVI", ".avi", ".mp4", ".MP4", ".mov", ".MOV")
SNAPUGC_HINTS = (
    "snapugc",
    "vquala",
    "ecr",
    "effective_completion",
    "completion_rate",
    "retention",
)
ID_COLUMN_HINTS = ("sample_id", "video_id", "video", "id")
GROUND_TRUTH_HINTS = ("ecr", "completion", "retention", "engagement")
CLAIM_BLOCKED_HINTS = ("synthetic", "fixture", "smoke", "control", "not_attention")
MAX_DISCOVERY_FILES = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        default=[],
        help="Root to scan. May be passed multiple times.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--feature-sample-limit", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_roots = args.search_root or default_search_roots()
    report = build_readiness_report(
        search_roots=search_roots,
        repo_root=Path.cwd(),
        feature_sample_limit=args.feature_sample_limit,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_readiness_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


def default_search_roots() -> list[Path]:
    return [
        Path("."),
        Path("data"),
        Path("/Users/jawaun/isc_mod/data"),
        Path("/Users/jawaun/data"),
        Path("/Users/jawaun/datasets"),
    ]


def build_readiness_report(
    *,
    search_roots: list[Path],
    repo_root: Path,
    feature_sample_limit: int = 8,
) -> dict[str, Any]:
    roots = [audit_search_root(path) for path in search_roots]
    existing_roots = [Path(root["path"]) for root in roots if root["exists"]]
    dhf1k_candidates = find_dhf1k_candidates(existing_roots)
    dhf1k_label_audits = find_dhf1k_label_audits(existing_roots)
    snapugc_candidates = find_snapugc_label_csvs(existing_roots)
    feature_dirs = find_tribe_feature_dirs(
        existing_roots,
        sample_limit=feature_sample_limit,
    )
    roi_masks = audit_roi_masks(repo_root)
    manifests = find_phase1_manifests(existing_roots)
    readiness = derive_readiness(
        dhf1k_candidates=dhf1k_candidates,
        dhf1k_label_audits=dhf1k_label_audits,
        snapugc_candidates=snapugc_candidates,
        feature_dirs=feature_dirs,
        roi_masks=roi_masks,
        manifests=manifests,
    )

    return {
        "schema_version": 1,
        "experiment": "phase1_attention_capture_data_readiness",
        "search_roots": roots,
        "dhf1k_candidates": dhf1k_candidates,
        "dhf1k_label_audits": dhf1k_label_audits,
        "snapugc_label_candidates": snapugc_candidates,
        "tribe_feature_dirs": feature_dirs,
        "roi_masks": roi_masks,
        "phase1_manifests": manifests,
        "readiness": readiness,
        "claim_boundary": (
            "This report audits local data availability only. It does not score "
            "TRIBE features or validate attentional capture."
        ),
    }


def audit_search_root(path: Path) -> dict[str, Any]:
    resolved = path.expanduser()
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "is_dir": resolved.is_dir(),
    }


def find_dhf1k_candidates(search_roots: list[Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in search_roots:
        for candidate in candidate_dataset_roots(root):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            audit = audit_dhf1k_root(candidate)
            if audit["has_video_dir"] or audit["has_annotation_dir"]:
                candidates.append(audit)
    return sorted(candidates, key=lambda item: item["path"])


def candidate_dataset_roots(root: Path) -> list[Path]:
    roots = [root]
    if not root.is_dir():
        return roots
    for child in safe_iterdir(root):
        if child.is_dir() and ("dhf1k" in child.name.lower() or child.name == "DHF1K"):
            roots.append(child)
    return roots


def audit_dhf1k_root(path: Path) -> dict[str, Any]:
    video_dir = path / "video"
    annotation_dir = path / "annotation"
    video_count = count_direct_files(video_dir, VIDEO_SUFFIXES)
    map_video_count = count_annotation_video_dirs(annotation_dir, "maps")
    fixation_video_count = count_annotation_video_dirs(annotation_dir, "fixation")
    return {
        "path": str(path),
        "has_video_dir": video_dir.is_dir(),
        "has_annotation_dir": annotation_dir.is_dir(),
        "n_videos": video_count,
        "n_annotation_map_video_dirs": map_video_count,
        "n_fixation_video_dirs": fixation_video_count,
        "ready_for_label_build": bool(video_count and map_video_count),
    }


def find_dhf1k_label_audits(search_roots: list[Path]) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for path in iter_matching_files(search_roots, "*.json"):
        audit = audit_dhf1k_label_audit(path)
        if audit is not None:
            audits.append(audit)
    return sorted(audits, key=lambda item: item["path"])


def audit_dhf1k_label_audit(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("experiment") != "dhf1k_attention_label_audit":
        return None

    labels_csv = payload.get("labels_csv")
    labels_csv_exists = path_exists_from_audit(labels_csv, audit_path=path)
    ready_flag = bool(payload.get("ready_for_manifest_alignment"))
    return {
        "path": str(path),
        "labels_csv": labels_csv,
        "labels_csv_exists": labels_csv_exists,
        "rank_column": payload.get("rank_column"),
        "recommended_ground_truth_column": payload.get(
            "recommended_ground_truth_column",
        ),
        "n_rows": payload.get("n_rows"),
        "ready_for_manifest_alignment": ready_flag,
        "ready_for_handoff": ready_flag and labels_csv_exists,
        "blocking_reasons": payload.get("blocking_reasons") or [],
    }


def find_snapugc_label_csvs(search_roots: list[Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in iter_matching_files(search_roots, "*.csv"):
        audit = audit_label_csv(path)
        if audit["candidate"]:
            candidates.append(audit)
    return sorted(candidates, key=lambda item: item["path"])


def audit_label_csv(path: Path) -> dict[str, Any]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            n_rows = sum(1 for _ in reader)
    except (OSError, UnicodeDecodeError, csv.Error):
        header = []
        n_rows = 0

    normalized = [column.strip().lower() for column in header]
    haystack = " ".join([path.name.lower(), *normalized])
    claim_blocked = any(hint in str(path).lower() for hint in CLAIM_BLOCKED_HINTS)
    has_dataset_hint = any(hint in haystack for hint in SNAPUGC_HINTS)
    has_id = any(column in ID_COLUMN_HINTS for column in normalized)
    has_ground_truth = any(
        any(hint in column for hint in GROUND_TRUTH_HINTS)
        for column in normalized
    )
    return {
        "path": str(path),
        "columns": header,
        "n_rows": n_rows,
        "candidate": bool(
            has_dataset_hint and has_id and has_ground_truth and not claim_blocked
        ),
        "claim_blocked": claim_blocked,
        "has_id_column": has_id,
        "has_ground_truth_column": has_ground_truth,
    }


def find_tribe_feature_dirs(
    search_roots: list[Path],
    *,
    sample_limit: int,
) -> list[dict[str, Any]]:
    by_parent: dict[Path, list[Path]] = {}
    for path in iter_matching_files(search_roots, "*.npz"):
        by_parent.setdefault(path.parent, []).append(path)

    audits = [
        audit_feature_dir(parent, paths, sample_limit=sample_limit)
        for parent, paths in by_parent.items()
    ]
    candidates = [
        audit
        for audit in audits
        if audit["n_frames_npz_sampled"] > 0 or "tribe" in audit["path"].lower()
    ]
    return sorted(
        candidates,
        key=lambda item: (item["n_frames_npz_sampled"], item["n_npz_files"]),
        reverse=True,
    )


def audit_feature_dir(
    parent: Path,
    paths: list[Path],
    *,
    sample_limit: int,
) -> dict[str, Any]:
    sampled = sorted(paths)[:sample_limit]
    frame_shapes: dict[str, int] = {}
    frames_npz = 0
    claim_blocked = any(hint in str(parent).lower() for hint in CLAIM_BLOCKED_HINTS)
    for path in sampled:
        shape = npz_frames_shape(path)
        if shape is None:
            continue
        frames_npz += 1
        frame_shapes[shape] = frame_shapes.get(shape, 0) + 1
    return {
        "path": str(parent),
        "n_npz_files": len(paths),
        "n_sampled": len(sampled),
        "n_frames_npz_sampled": frames_npz,
        "frame_shape_counts": frame_shapes,
        "claim_blocked": claim_blocked,
        "ready_as_feature_cache": bool(frames_npz) and not claim_blocked,
    }


def audit_roi_masks(repo_root: Path) -> dict[str, Any]:
    base = repo_root / "research_program" / "dopamine_detox_attention_capture"
    results = base / "results"
    disjoint = results / "destrieux_roi_masks_disjoint_20260608.npz"
    overlapping = results / "destrieux_roi_masks_20260608.npz"
    return {
        "disjoint": audit_roi_mask_file(disjoint, repo_root=repo_root),
        "overlapping": audit_roi_mask_file(overlapping, repo_root=repo_root),
        "ready_for_primary_scoring": disjoint.exists(),
    }


def audit_roi_mask_file(path: Path, *, repo_root: Path) -> dict[str, Any]:
    return {
        "path": display_path(path, repo_root=repo_root),
        "exists": path.exists(),
    }


def find_phase1_manifests(search_roots: list[Path]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in iter_matching_files(search_roots, "phase1*manifest*.json"):
        manifests.append(audit_phase1_manifest(path))
    return sorted(manifests, key=lambda item: item["path"])


def audit_phase1_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    samples = payload.get("samples")
    status = str(payload.get("status") or "unspecified")
    sample_rows = [sample for sample in samples or [] if isinstance(sample, dict)]
    datasets = sorted(
        {
            str(sample.get("dataset") or "unknown")
            for sample in sample_rows
        },
    )
    claim_blocked = is_claim_blocked_manifest(status=status, datasets=datasets)
    provenance = audit_manifest_provenance(
        payload,
        n_samples=len(samples) if isinstance(samples, list) else 0,
        datasets=datasets,
        required=not claim_blocked,
    )
    return {
        "path": str(path),
        "status": status,
        "n_samples": len(samples) if isinstance(samples, list) else 0,
        "datasets": datasets,
        "claim_blocked": claim_blocked,
        "provenance_required": provenance["required"],
        "provenance_ready": provenance["ready"],
        "provenance_blocking_reasons": provenance["blocking_reasons"],
        "alignment_audit": provenance["alignment_audit"],
        "ready_for_workflow": bool(samples)
        and not claim_blocked
        and provenance["ready"],
    }


def is_claim_blocked_manifest(*, status: str, datasets: list[str]) -> bool:
    haystack = " ".join([status, *datasets]).lower()
    return any(term in haystack for term in CLAIM_BLOCKED_HINTS)


def audit_manifest_provenance(
    manifest: dict[str, Any],
    *,
    n_samples: int,
    datasets: list[str],
    required: bool,
) -> dict[str, Any]:
    alignment = alignment_audit_metadata(manifest)
    reasons: list[str] = []
    if required:
        if not isinstance(alignment, dict):
            reasons.append(
                "claim-updatable manifest requires metadata.alignment_audit"
            )
        else:
            reasons.extend(
                manifest_alignment_blocking_reasons(
                    alignment,
                    n_samples=n_samples,
                    datasets=datasets,
                ),
            )

    return {
        "required": required,
        "ready": not reasons,
        "blocking_reasons": reasons,
        "alignment_audit": alignment_audit_summary(alignment),
    }


def alignment_audit_metadata(manifest: dict[str, Any]) -> dict[str, Any] | None:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return None
    alignment = metadata.get("alignment_audit")
    return alignment if isinstance(alignment, dict) else None


def alignment_audit_summary(alignment: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(alignment, dict):
        return {
            "path": None,
            "sha256": None,
            "ready_for_manifest_build": None,
            "n_aligned_features": None,
            "n_missing_features": None,
            "label_audit_ready": None,
        }
    label_audit = alignment.get("label_audit")
    label_audit_ready = (
        label_audit.get("ready_for_manifest_alignment")
        if isinstance(label_audit, dict)
        else None
    )
    return {
        "path": alignment.get("path"),
        "sha256": alignment.get("sha256"),
        "ready_for_manifest_build": alignment.get("ready_for_manifest_build"),
        "n_aligned_features": alignment.get("n_aligned_features"),
        "n_missing_features": alignment.get("n_missing_features"),
        "label_audit_ready": label_audit_ready,
    }


def manifest_alignment_blocking_reasons(
    alignment: dict[str, Any],
    *,
    n_samples: int,
    datasets: list[str],
) -> list[str]:
    reasons: list[str] = []
    if alignment.get("ready_for_manifest_build") is not True:
        reasons.append("metadata.alignment_audit is not ready for manifest build")
    if not isinstance(alignment.get("path"), str) or not alignment.get("path"):
        reasons.append("metadata.alignment_audit.path is missing")
    if not looks_like_sha256(alignment.get("sha256")):
        reasons.append("metadata.alignment_audit.sha256 is missing or invalid")

    n_aligned = optional_int(alignment.get("n_aligned_features"))
    if n_aligned is None:
        reasons.append("metadata.alignment_audit.n_aligned_features is missing")
    elif n_aligned < n_samples:
        reasons.append(
            "metadata.alignment_audit.n_aligned_features is below manifest "
            "sample count"
        )

    n_missing = optional_int(alignment.get("n_missing_features"))
    if n_missing is None:
        reasons.append("metadata.alignment_audit.n_missing_features is missing")
    elif n_missing:
        reasons.append(
            f"metadata.alignment_audit reports {n_missing} missing features"
        )

    reasons.extend(dhf1k_manifest_label_audit_reasons(alignment, datasets=datasets))
    return reasons


def dhf1k_manifest_label_audit_reasons(
    alignment: dict[str, Any],
    *,
    datasets: list[str],
) -> list[str]:
    if not any(dataset.upper() == "DHF1K" for dataset in datasets):
        return []
    label_audit = alignment.get("label_audit")
    if not isinstance(label_audit, dict):
        return ["DHF1K manifests require alignment_audit.label_audit"]
    if label_audit.get("ready_for_manifest_alignment") is not True:
        return ["alignment_audit.label_audit is not ready"]
    return []


def derive_readiness(
    *,
    dhf1k_candidates: list[dict[str, Any]],
    dhf1k_label_audits: list[dict[str, Any]],
    snapugc_candidates: list[dict[str, Any]],
    feature_dirs: list[dict[str, Any]],
    roi_masks: dict[str, Any],
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    dhf1k_root_ready = any(item["ready_for_label_build"] for item in dhf1k_candidates)
    dhf1k_label_ready = any(item["ready_for_handoff"] for item in dhf1k_label_audits)
    dhf1k_label_audit_present = bool(dhf1k_label_audits)
    snapugc_ready = bool(snapugc_candidates)
    features_ready = any(item["ready_as_feature_cache"] for item in feature_dirs)
    masks_ready = bool(roi_masks["ready_for_primary_scoring"])
    manifest_ready = any(item["ready_for_workflow"] for item in manifests)
    manifest_blockers = manifest_readiness_blockers(manifests)
    blockers = readiness_blockers(
        dhf1k_root_ready=dhf1k_root_ready,
        dhf1k_label_ready=dhf1k_label_ready,
        dhf1k_label_audit_present=dhf1k_label_audit_present,
        snapugc_ready=snapugc_ready,
        features_ready=features_ready,
        masks_ready=masks_ready,
        manifest_ready=manifest_ready,
        manifest_blockers=manifest_blockers,
    )
    return {
        "dhf1k_root_ready_for_label_build": dhf1k_root_ready,
        "dhf1k_label_audit_ready": dhf1k_label_ready,
        "dhf1k_labels_ready": dhf1k_label_ready,
        "snapugc_labels_ready": snapugc_ready,
        "tribe_features_ready": features_ready,
        "roi_masks_ready": masks_ready,
        "real_manifest_ready": manifest_ready,
        "phase1_can_run_now": bool(manifest_ready and masks_ready),
        "blocking_reasons": blockers,
        "recommended_next_action": recommended_next_action(
            dhf1k_root_ready=dhf1k_root_ready,
            dhf1k_label_ready=dhf1k_label_ready,
            snapugc_ready=snapugc_ready,
            features_ready=features_ready,
            masks_ready=masks_ready,
            manifest_ready=manifest_ready,
            manifest_blockers=manifest_blockers,
        ),
    }


def manifest_readiness_blockers(manifests: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for manifest in manifests:
        if manifest["claim_blocked"] or manifest["ready_for_workflow"]:
            continue
        reasons = manifest.get("provenance_blocking_reasons") or []
        if not reasons:
            reasons = ["manifest is not workflow-ready"]
        blockers.append(
            f"{manifest['path']}: " + "; ".join(str(reason) for reason in reasons)
        )
    return blockers


def readiness_blockers(
    *,
    dhf1k_root_ready: bool,
    dhf1k_label_ready: bool,
    dhf1k_label_audit_present: bool,
    snapugc_ready: bool,
    features_ready: bool,
    masks_ready: bool,
    manifest_ready: bool,
    manifest_blockers: list[str],
) -> list[str]:
    blockers: list[str] = []
    blockers.extend(
        f"manifest not workflow-ready: {reason}"
        for reason in manifest_blockers
    )
    has_claim_updatable_manifest = bool(manifest_ready or manifest_blockers)
    if not (dhf1k_label_ready or snapugc_ready or has_claim_updatable_manifest):
        if dhf1k_root_ready:
            blockers.append("DHF1K root found but no ready DHF1K label audit found")
        else:
            blockers.append("no external attention-label source found")
    if dhf1k_label_audit_present and not dhf1k_label_ready and not manifest_ready:
        blockers.append("DHF1K label audit is not ready for manifest alignment")
    if not features_ready and not manifest_ready:
        blockers.append("no cached TRIBE feature directory found")
    if not masks_ready:
        blockers.append("disjoint ROI mask NPZ is missing")
    return blockers


def recommended_next_action(
    *,
    dhf1k_root_ready: bool,
    dhf1k_label_ready: bool,
    snapugc_ready: bool,
    features_ready: bool,
    masks_ready: bool,
    manifest_ready: bool,
    manifest_blockers: list[str],
) -> str:
    if manifest_ready and masks_ready:
        return "run scripts/run_attention_capture_phase1_workflow.py"
    if manifest_blockers:
        return "fix Phase 1 manifest provenance, then rerun the guarded workflow"
    if dhf1k_label_ready and features_ready:
        return "build the DHF1K Phase 1 manifest, then run the guarded workflow"
    if snapugc_ready and features_ready:
        return "build the SnapUGC Phase 1 manifest, then run the guarded workflow"
    if dhf1k_label_ready and not features_ready:
        return "extract DHF1K TRIBE features from the audited DHF1K labels"
    if snapugc_ready and not features_ready:
        return "extract TRIBE features for the SnapUGC/VQualA label CSV"
    if dhf1k_root_ready:
        return "build DHF1K labels and confirm ready_for_manifest_alignment=true"
    return "acquire or mount external DHF1K/SnapUGC labels and videos"


def render_readiness_markdown(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    lines = [
        "# Phase 1 Data Readiness Audit",
        "",
        "## Verdict",
        "",
        f"- Phase 1 can run now: {readiness['phase1_can_run_now']}",
        (
            "- DHF1K root ready for label build: "
            f"{readiness['dhf1k_root_ready_for_label_build']}"
        ),
        f"- DHF1K label audit ready: {readiness['dhf1k_label_audit_ready']}",
        f"- DHF1K labels ready: {readiness['dhf1k_labels_ready']}",
        f"- SnapUGC labels ready: {readiness['snapugc_labels_ready']}",
        f"- TRIBE features ready: {readiness['tribe_features_ready']}",
        f"- ROI masks ready: {readiness['roi_masks_ready']}",
        f"- Real manifest ready: {readiness['real_manifest_ready']}",
        f"- Recommended next action: {readiness['recommended_next_action']}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Blocking Reasons",
        "",
    ]
    blockers = readiness["blocking_reasons"]
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append(
        "- none",
    )
    lines.extend(render_dhf1k_section(report["dhf1k_candidates"]))
    lines.extend(render_dhf1k_label_audit_section(report["dhf1k_label_audits"]))
    lines.extend(render_snapugc_section(report["snapugc_label_candidates"]))
    lines.extend(render_feature_section(report["tribe_feature_dirs"]))
    lines.extend(render_manifest_section(report["phase1_manifests"]))
    return "\n".join(lines) + "\n"


def render_dhf1k_section(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## DHF1K Candidates",
        "",
        "| path | videos | map dirs | fixation dirs | ready |",
        "|---|---:|---:|---:|---|",
    ]
    if not items:
        lines.append("| none | 0 | 0 | 0 | False |")
    for item in items:
        lines.append(
            "| "
            f"{item['path']} | {item['n_videos']} | "
            f"{item['n_annotation_map_video_dirs']} | "
            f"{item['n_fixation_video_dirs']} | "
            f"{item['ready_for_label_build']} |"
        )
    return lines


def render_dhf1k_label_audit_section(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## DHF1K Label Audits",
        "",
        "| path | labels CSV exists | rank column | rows | ready |",
        "|---|---|---|---:|---|",
    ]
    if not items:
        lines.append("| none | False | n/a | 0 | False |")
    for item in items:
        lines.append(
            "| "
            f"{item['path']} | {item['labels_csv_exists']} | "
            f"{item['rank_column'] or 'n/a'} | "
            f"{item['n_rows'] or 0} | {item['ready_for_handoff']} |"
        )
    return lines


def render_snapugc_section(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## SnapUGC/VQualA Label Candidates",
        "",
        "| path | rows | columns |",
        "|---|---:|---|",
    ]
    if not items:
        lines.append("| none | 0 | n/a |")
    for item in items:
        lines.append(
            "| "
            f"{item['path']} | {item['n_rows']} | "
            f"{', '.join(item['columns'])} |"
        )
    return lines


def render_feature_section(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## TRIBE Feature Cache Candidates",
        "",
        "| path | npz files | sampled frames arrays | claim blocked | ready |",
        "|---|---:|---:|---|---|",
    ]
    if not items:
        lines.append("| none | 0 | 0 | False | False |")
    for item in items[:20]:
        lines.append(
            "| "
            f"{item['path']} | {item['n_npz_files']} | "
            f"{item['n_frames_npz_sampled']} | "
            f"{item['claim_blocked']} | "
            f"{item['ready_as_feature_cache']} |"
        )
    return lines


def render_manifest_section(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Phase 1 Manifests",
        "",
        (
            "| path | status | samples | claim blocked | provenance required | "
            "provenance ready | workflow-ready |"
        ),
        "|---|---|---:|---|---|---|---|",
    ]
    if not items:
        lines.append("| none | n/a | 0 | True | False | False | False |")
    for item in items:
        lines.append(
            "| "
            f"{item['path']} | {item['status']} | {item['n_samples']} | "
            f"{item['claim_blocked']} | {item['provenance_required']} | "
            f"{item['provenance_ready']} | {item['ready_for_workflow']} |"
        )
    return lines


def count_direct_files(path: Path, suffixes: tuple[str, ...]) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for child in safe_iterdir(path) if child.suffix in suffixes)


def count_annotation_video_dirs(annotation_dir: Path, child_name: str) -> int:
    if not annotation_dir.is_dir():
        return 0
    count = 0
    for child in safe_iterdir(annotation_dir):
        if child.is_dir() and any((child / child_name).glob("*.png")):
            count += 1
    return count


def iter_matching_files(search_roots: list[Path], pattern: str) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists():
            continue
        try:
            iterator = root.rglob(pattern) if root.is_dir() else iter([root])
            for path in iterator:
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                paths.append(path)
                if len(paths) >= MAX_DISCOVERY_FILES:
                    return paths
        except OSError:
            continue
    return paths


def npz_frames_shape(path: Path) -> str | None:
    try:
        payload = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if "frames" not in payload:
        return None
    frames = np.asarray(payload["frames"])
    return "x".join(str(dim) for dim in frames.shape)


def looks_like_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def optional_int(value: object) -> int | None:
    if not isinstance(value, int | float | str):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def display_path(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def path_exists_from_audit(raw_path: object, *, audit_path: Path) -> bool:
    if not raw_path:
        return False
    candidate = Path(str(raw_path)).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.extend([Path.cwd() / candidate, audit_path.parent / candidate])
    return any(path.exists() for path in candidates)


def safe_iterdir(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except OSError:
        return []


if __name__ == "__main__":
    main()
