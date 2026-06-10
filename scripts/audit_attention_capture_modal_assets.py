"""Audit Modal-hosted assets for attention-capture publication unblocks.

The local process only launches the Modal job and writes the report. Dataset and
cache discovery runs inside Modal CPU containers mounted to known volumes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import modal

APP_NAME = "attention-capture-modal-assets"
MOUNT_ROOT = Path("/modal-volumes")
DEFAULT_VOLUME_NAMES = (
    "rde-activation-results",
    "audience-analyzer-runs-v1",
    "wan22-lora-data-v1",
    "wan22-lora-outputs-v1",
    "wan22-lora-cache-v1",
    "svd-weights-v1",
    "svd-outputs-v1",
    "wan22-weights-v1",
    "wan22-outputs-v1",
    "cogvideox-outputs-v1",
    "cogvideox-weights-v1",
    "tribe-v2-weights-v1",
    "vjepa-weights-v1",
    "bmd-videos-v1",
    "fr-dev-data",
    "fr-prd-data",
    "fr-stg-data",
    "flytrap-review-prod-data",
    "flytrap-review-data",
    "tac-docker-data",
)
DEFAULT_SECRET_NAMES = (
    "underlying-analyzer-env",
    "fr-dev-internal-api",
    "fr-dev-github-app",
    "fr-dev-llm-api-keys",
    "fr-prd-internal-api",
    "fr-prd-github-app",
    "fr-prd-llm-api-keys",
    "fr-stg-internal-api",
    "fr-stg-github-app",
    "fr-stg-llm-api-keys",
    "flytrap-review-prod-internal-api",
    "flytrap-review-prod-github-app",
    "flytrap-review-prod-llm-api-keys",
    "flytrap-review-staging-internal-api",
    "flytrap-review-staging-github-app",
    "flytrap-review-staging-llm-api-keys",
    "internal-api",
    "github-app",
    "llm-api-keys",
)
TOKEN_ENVS = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)
LABEL_HINTS = (
    "snapugc",
    "vquala",
    "ecr",
    "effective_completion",
    "completion_rate",
    "retention",
    "engagement",
)
DATASET_HINTS = (
    "dhf1k",
    "snapugc",
    "vquala",
    "memento",
)
FEATURE_HINTS = (
    "tribe",
    "feature",
    "activation",
)
CLAIM_BLOCKED_HINTS = (
    "synthetic",
    "fixture",
    "smoke",
    "control",
    "not_attention",
)


def safe_mount_name(volume_name: str) -> str:
    return volume_name.replace("-", "_")


VOLUME_MOUNTS: dict[str | PurePosixPath, Any] = {
    str(MOUNT_ROOT / safe_mount_name(name)): modal.Volume.from_name(
        name,
        create_if_missing=False,
    )
    for name in DEFAULT_VOLUME_NAMES
}

image = modal.Image.debian_slim(python_version="3.12")
app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=1.0,
    memory=1024,
    timeout=15 * 60,
    volumes=VOLUME_MOUNTS,
)
def audit_modal_volumes(
    max_entries_per_volume: int = 20_000,
    max_depth: int = 5,
    preview_limit: int = 80,
) -> dict[str, Any]:
    return build_modal_volume_report(
        volume_names=DEFAULT_VOLUME_NAMES,
        max_entries_per_volume=max_entries_per_volume,
        max_depth=max_depth,
        preview_limit=preview_limit,
    )


@app.function(
    image=image,
    cpu=1.0,
    memory=256,
    timeout=2 * 60,
    secrets=[modal.Secret.from_name(name) for name in DEFAULT_SECRET_NAMES],
)
def audit_modal_secret_presence() -> dict[str, Any]:
    entries = [
        {"env": name, "present": bool(os.environ.get(name))} for name in TOKEN_ENVS
    ]
    matching_env_names = sorted(
        name
        for name in os.environ
        if "HF" in name.upper() or "HUGGINGFACE" in name.upper()
    )
    return {
        "secret_names_checked": list(DEFAULT_SECRET_NAMES),
        "token_envs_checked": list(TOKEN_ENVS),
        "entries": entries,
        "any_present": any(entry["present"] for entry in entries),
        "matching_env_names": matching_env_names,
        "claim_boundary": (
            "This reports environment variable presence only. It never reports "
            "secret values and does not prove access to any gated HuggingFace model."
        ),
    }


@app.local_entrypoint()
def main(
    output_json: str,
    output_md: str,
    max_entries_per_volume: int = 20_000,
    max_depth: int = 5,
    preview_limit: int = 80,
) -> None:
    volume_report = audit_modal_volumes.remote(
        max_entries_per_volume=max_entries_per_volume,
        max_depth=max_depth,
        preview_limit=preview_limit,
    )
    secret_report = audit_modal_secret_presence.remote()
    report = {
        "schema_version": 1,
        "experiment": "attention_capture_modal_asset_audit",
        "generated_at": datetime.now(UTC).isoformat(),
        "volume_report": volume_report,
        "secret_report": secret_report,
        "configuration": {
            "volume_names": list(DEFAULT_VOLUME_NAMES),
            "secret_names": list(DEFAULT_SECRET_NAMES),
        },
        "publication_unblocks": summarize_publication_unblocks(
            volume_report=volume_report,
            secret_report=secret_report,
        ),
        "claim_boundary": (
            "This Modal CPU audit checks remote asset availability only. It does "
            "not score TRIBE features or validate attentional capture."
        ),
    }
    output_json_path = Path(output_json)
    output_md_path = Path(output_md)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md_path.write_text(render_modal_asset_markdown(report), encoding="utf-8")
    print(f"wrote {output_json_path}")
    print(f"wrote {output_md_path}")


def build_modal_volume_report(
    *,
    volume_names: Iterable[str],
    max_entries_per_volume: int,
    max_depth: int,
    preview_limit: int,
) -> dict[str, Any]:
    audits = [
        audit_one_volume(
            volume_name=name,
            mount_path=MOUNT_ROOT / safe_mount_name(name),
            max_entries=max_entries_per_volume,
            max_depth=max_depth,
            preview_limit=preview_limit,
        )
        for name in volume_names
    ]
    return {
        "volume_names_checked": list(volume_names),
        "max_entries_per_volume": max_entries_per_volume,
        "max_depth": max_depth,
        "audits": audits,
        "n_label_candidates": sum(len(audit["label_candidates"]) for audit in audits),
        "n_dataset_candidates": sum(
            len(audit["dataset_candidates"]) for audit in audits
        ),
        "n_feature_candidates": sum(
            len(audit["feature_candidates"]) for audit in audits
        ),
    }


def audit_one_volume(
    *,
    volume_name: str,
    mount_path: Path,
    max_entries: int,
    max_depth: int,
    preview_limit: int,
) -> dict[str, Any]:
    n_entries = 0
    n_files = 0
    n_dirs = 0
    truncated = False
    label_candidates: list[dict[str, Any]] = []
    dataset_candidates: list[dict[str, Any]] = []
    feature_candidates: list[dict[str, Any]] = []

    if not mount_path.exists():
        return empty_volume_audit(volume_name=volume_name, mount_path=mount_path)

    for root, dirs, files in os.walk(mount_path):
        root_path = Path(root)
        depth = relative_depth(root_path, mount_path)
        if depth >= max_depth:
            dirs[:] = []

        n_entries, n_dirs, truncated = scan_directory_candidates(
            root_path=root_path,
            dirs=dirs,
            mount_path=mount_path,
            dataset_candidates=dataset_candidates,
            feature_candidates=feature_candidates,
            preview_limit=preview_limit,
            max_entries=max_entries,
            n_entries=n_entries,
            n_dirs=n_dirs,
        )
        if truncated:
            break

        n_entries, n_files, truncated = scan_file_candidates(
            root_path=root_path,
            files=files,
            mount_path=mount_path,
            label_candidates=label_candidates,
            feature_candidates=feature_candidates,
            preview_limit=preview_limit,
            max_entries=max_entries,
            n_entries=n_entries,
            n_files=n_files,
        )
        if truncated:
            break

    return {
        "volume": volume_name,
        "mount_path": str(mount_path),
        "exists": True,
        "n_entries_seen": n_entries,
        "n_files_seen": n_files,
        "n_dirs_seen": n_dirs,
        "truncated": truncated,
        "label_candidates": label_candidates,
        "dataset_candidates": dataset_candidates,
        "feature_candidates": feature_candidates,
    }


def empty_volume_audit(*, volume_name: str, mount_path: Path) -> dict[str, Any]:
    return {
        "volume": volume_name,
        "mount_path": str(mount_path),
        "exists": False,
        "n_entries_seen": 0,
        "n_files_seen": 0,
        "n_dirs_seen": 0,
        "truncated": False,
        "label_candidates": [],
        "dataset_candidates": [],
        "feature_candidates": [],
    }


def scan_directory_candidates(
    *,
    root_path: Path,
    dirs: list[str],
    mount_path: Path,
    dataset_candidates: list[dict[str, Any]],
    feature_candidates: list[dict[str, Any]],
    preview_limit: int,
    max_entries: int,
    n_entries: int,
    n_dirs: int,
) -> tuple[int, int, bool]:
    for dirname in sorted(dirs):
        n_entries += 1
        n_dirs += 1
        add_directory_candidate(
            root_path / dirname,
            mount_path=mount_path,
            dataset_candidates=dataset_candidates,
            feature_candidates=feature_candidates,
            preview_limit=preview_limit,
        )
        if n_entries >= max_entries:
            return n_entries, n_dirs, True
    return n_entries, n_dirs, False


def add_directory_candidate(
    candidate: Path,
    *,
    mount_path: Path,
    dataset_candidates: list[dict[str, Any]],
    feature_candidates: list[dict[str, Any]],
    preview_limit: int,
) -> None:
    candidate_text = str(candidate)
    if has_any_hint(candidate_text, DATASET_HINTS):
        append_preview(
            dataset_candidates,
            candidate_entry(candidate, mount_path=mount_path, kind="dir"),
            preview_limit=preview_limit,
        )
    if has_any_hint(candidate_text, FEATURE_HINTS):
        append_preview(
            feature_candidates,
            candidate_entry(candidate, mount_path=mount_path, kind="dir"),
            preview_limit=preview_limit,
        )


def scan_file_candidates(
    *,
    root_path: Path,
    files: list[str],
    mount_path: Path,
    label_candidates: list[dict[str, Any]],
    feature_candidates: list[dict[str, Any]],
    preview_limit: int,
    max_entries: int,
    n_entries: int,
    n_files: int,
) -> tuple[int, int, bool]:
    for filename in sorted(files):
        n_entries += 1
        n_files += 1
        add_file_candidate(
            root_path / filename,
            mount_path=mount_path,
            label_candidates=label_candidates,
            feature_candidates=feature_candidates,
            preview_limit=preview_limit,
        )
        if n_entries >= max_entries:
            return n_entries, n_files, True
    return n_entries, n_files, False


def add_file_candidate(
    candidate: Path,
    *,
    mount_path: Path,
    label_candidates: list[dict[str, Any]],
    feature_candidates: list[dict[str, Any]],
    preview_limit: int,
) -> None:
    candidate_text = str(candidate)
    if is_label_candidate(candidate_text):
        append_preview(
            label_candidates,
            label_candidate_entry(candidate, mount_path=mount_path),
            preview_limit=preview_limit,
        )
    if has_any_hint(candidate_text, FEATURE_HINTS):
        append_preview(
            feature_candidates,
            candidate_entry(candidate, mount_path=mount_path, kind="file"),
            preview_limit=preview_limit,
        )


def is_label_candidate(path_text: str) -> bool:
    suffix = Path(path_text).suffix.lower()
    return suffix in {".csv", ".json", ".jsonl", ".parquet"} and has_any_hint(
        path_text,
        LABEL_HINTS,
    )


def label_candidate_entry(path: Path, *, mount_path: Path) -> dict[str, Any]:
    entry = candidate_entry(path, mount_path=mount_path, kind="file")
    if path.suffix.lower() == ".csv":
        entry["csv_header_preview"] = csv_header_preview(path)
    return entry


def candidate_entry(path: Path, *, mount_path: Path, kind: str) -> dict[str, Any]:
    rel = str(path.relative_to(mount_path))
    return {
        "path": rel,
        "kind": kind,
        "claim_blocked": has_any_hint(rel, CLAIM_BLOCKED_HINTS),
    }


def csv_header_preview(path: Path) -> list[str]:
    try:
        first_line = path.open(encoding="utf-8").readline().strip()
    except (OSError, UnicodeDecodeError):
        return []
    return [cell.strip() for cell in first_line.split(",")[:40] if cell.strip()]


def append_preview(
    items: list[dict[str, Any]], item: dict[str, Any], *, preview_limit: int
) -> None:
    if len(items) < preview_limit:
        items.append(item)


def relative_depth(path: Path, root: Path) -> int:
    if path == root:
        return 0
    return len(path.relative_to(root).parts)


def has_any_hint(value: str, hints: Iterable[str]) -> bool:
    lowered = value.lower()
    return any(hint in lowered for hint in hints)


def summarize_publication_unblocks(
    *,
    volume_report: dict[str, Any],
    secret_report: dict[str, Any],
) -> dict[str, Any]:
    label_candidates = [
        candidate
        for audit in volume_report["audits"]
        for candidate in audit["label_candidates"]
        if not candidate["claim_blocked"]
    ]
    dataset_candidates = [
        candidate
        for audit in volume_report["audits"]
        for candidate in audit["dataset_candidates"]
        if not candidate["claim_blocked"]
    ]
    feature_candidates = [
        candidate
        for audit in volume_report["audits"]
        for candidate in audit["feature_candidates"]
        if not candidate["claim_blocked"]
    ]
    return {
        "retention_labels_maybe_available": bool(label_candidates),
        "external_dataset_dirs_maybe_available": bool(dataset_candidates),
        "feature_caches_maybe_available": bool(feature_candidates),
        "full_multimodal_token_env_present": bool(secret_report["any_present"]),
        "blocking_reasons": modal_unblock_reasons(
            label_candidates=label_candidates,
            token_present=bool(secret_report["any_present"]),
        ),
    }


def modal_unblock_reasons(
    *,
    label_candidates: list[dict[str, Any]],
    token_present: bool,
) -> list[str]:
    reasons: list[str] = []
    if not label_candidates:
        reasons.append("no Modal-hosted SnapUGC/VQualA retention label candidate found")
    if not token_present:
        reasons.append("no Modal secret exposes a HuggingFace token env name")
    return reasons


def render_modal_asset_markdown(report: dict[str, Any]) -> str:
    unblocks = report["publication_unblocks"]
    secret = report["secret_report"]
    volume = report["volume_report"]
    configuration = report.get("configuration") or {}
    lines = [
        "# Attention-Capture Modal Asset Audit",
        "",
        "## Verdict",
        "",
        f"- Retention labels maybe available: {unblocks['retention_labels_maybe_available']}",
        f"- External dataset dirs maybe available: {unblocks['external_dataset_dirs_maybe_available']}",
        f"- Feature caches maybe available: {unblocks['feature_caches_maybe_available']}",
        f"- Full multimodal token env present: {unblocks['full_multimodal_token_env_present']}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Configuration",
        "",
        f"- Volumes checked: {len(configuration.get('volume_names') or [])}",
        f"- Secrets checked: {len(configuration.get('secret_names') or [])}",
        "",
        "## Blocking Reasons",
        "",
    ]
    reasons = unblocks["blocking_reasons"]
    lines.extend(f"- {reason}" for reason in reasons) if reasons else lines.append(
        "- none"
    )
    lines.extend(
        [
            "",
            "## Secret Presence",
            "",
            f"- Secrets checked: {', '.join(secret['secret_names_checked'])}",
            f"- Token envs checked: {', '.join(secret['token_envs_checked'])}",
            f"- Matching env names: {', '.join(secret['matching_env_names']) or 'none'}",
            "",
            "## Volume Summary",
            "",
            "| volume | entries | files | dirs | truncated | labels | datasets | features |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ],
    )
    for audit in volume["audits"]:
        lines.append(
            "| "
            f"{audit['volume']} | {audit['n_entries_seen']} | "
            f"{audit['n_files_seen']} | {audit['n_dirs_seen']} | "
            f"{audit['truncated']} | {len(audit['label_candidates'])} | "
            f"{len(audit['dataset_candidates'])} | "
            f"{len(audit['feature_candidates'])} |"
        )
    lines.extend(
        render_candidate_section("Label Candidates", volume, "label_candidates")
    )
    lines.extend(
        render_candidate_section("Dataset Candidates", volume, "dataset_candidates")
    )
    lines.extend(
        render_candidate_section("Feature Candidates", volume, "feature_candidates")
    )
    return "\n".join(lines) + "\n"


def render_candidate_section(
    title: str,
    volume_report: dict[str, Any],
    key: str,
) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        "| volume | path | kind | claim blocked |",
        "|---|---|---|---|",
    ]
    found = False
    for audit in volume_report["audits"]:
        for candidate in audit[key]:
            found = True
            lines.append(
                "| "
                f"{audit['volume']} | {candidate['path']} | "
                f"{candidate['kind']} | {candidate['claim_blocked']} |"
            )
    if not found:
        lines.append("| none | n/a | n/a | False |")
    return lines


if __name__ == "__main__":
    # Modal invokes the local_entrypoint above.
    pass
