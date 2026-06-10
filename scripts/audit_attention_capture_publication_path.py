"""Audit whether attention-capture evidence is ready for a paper claim."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_TOKEN_ENVS = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-json", type=Path, default=None)
    parser.add_argument(
        "--workflow-json",
        action="append",
        type=Path,
        default=[],
        help="Phase 1 workflow JSON to include. May be passed multiple times.",
    )
    parser.add_argument(
        "--feature-cache-audit",
        action="append",
        type=Path,
        default=[],
        help="Feature-cache checksum audit JSON. May be passed multiple times.",
    )
    parser.add_argument(
        "--modal-asset-audit",
        action="append",
        type=Path,
        default=[],
        help="Modal-hosted asset audit JSON. May be passed multiple times.",
    )
    parser.add_argument(
        "--tribe-full-preflight-audit",
        action="append",
        type=Path,
        default=[],
        help="TRIBE full-mode Modal preflight audit JSON. May be passed multiple times.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--min-paper-datasets", type=int, default=2)
    parser.add_argument(
        "--token-env",
        action="append",
        default=[],
        help="Environment variable that can unlock gated HuggingFace text models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_publication_path_report(
        readiness_json=args.readiness_json,
        workflow_jsons=args.workflow_json,
        feature_cache_audits=args.feature_cache_audit,
        modal_asset_audits=args.modal_asset_audit,
        tribe_full_preflight_audits=args.tribe_full_preflight_audit,
        min_paper_datasets=args.min_paper_datasets,
        token_envs=tuple(args.token_env or DEFAULT_TOKEN_ENVS),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.write_text(render_publication_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


def build_publication_path_report(
    *,
    readiness_json: Path | None,
    workflow_jsons: list[Path],
    feature_cache_audits: list[Path] | None = None,
    modal_asset_audits: list[Path] | None = None,
    tribe_full_preflight_audits: list[Path] | None = None,
    min_paper_datasets: int = 2,
    token_envs: tuple[str, ...] = DEFAULT_TOKEN_ENVS,
) -> dict[str, Any]:
    readiness = load_readiness(readiness_json)
    workflows = [summarize_workflow(path) for path in workflow_jsons]
    cache_audits = [
        summarize_feature_cache_audit(path) for path in feature_cache_audits or []
    ]
    modal_audits = [
        summarize_modal_asset_audit(path) for path in modal_asset_audits or []
    ]
    tribe_full_preflights = [
        summarize_tribe_full_preflight_audit(path)
        for path in tribe_full_preflight_audits or []
    ]
    credential_audit = audit_text_model_credentials(token_envs)
    completed_real_workflows = [
        workflow for workflow in workflows if workflow["scoring_executed"]
    ]
    passed_workflows = [
        workflow for workflow in completed_real_workflows if workflow["gate_passed"]
    ]
    scored_datasets = sorted(
        {
            dataset
            for workflow in completed_real_workflows
            for dataset in workflow["datasets"]
            if dataset != "pooled"
        },
    )
    retention_labels_ready = bool(
        readiness.get("readiness", {}).get("snapugc_labels_ready"),
    )
    modal_retention_labels_available = any(
        audit["retention_labels_maybe_available"] for audit in modal_audits
    )
    modal_token_present = any(
        audit["full_multimodal_token_env_present"] for audit in modal_audits
    )
    tribe_full_preflight_ready = any(
        audit["ok"] and audit["event_mode"] == "full" for audit in tribe_full_preflights
    )
    full_multimodal_ready = (
        credential_audit["any_present"]
        or modal_token_present
        or tribe_full_preflight_ready
    )
    has_audio_only_workflow = any(workflow["audio_only"] for workflow in workflows)
    phase1_gate_passed = bool(passed_workflows)
    enough_datasets_for_paper = len(scored_datasets) >= min_paper_datasets

    blockers = publication_blockers(
        workflows=workflows,
        phase1_gate_passed=phase1_gate_passed,
        retention_labels_ready=retention_labels_ready,
        modal_asset_audits_supplied=bool(modal_audits),
        modal_retention_labels_available=modal_retention_labels_available,
        tribe_full_preflight_audits_supplied=bool(tribe_full_preflights),
        full_multimodal_ready=full_multimodal_ready,
        has_audio_only_workflow=has_audio_only_workflow,
        enough_datasets_for_paper=enough_datasets_for_paper,
        min_paper_datasets=min_paper_datasets,
    )
    warnings = publication_warnings(
        readiness=readiness,
        workflows=workflows,
        cache_audits=cache_audits,
        modal_audits=modal_audits,
        tribe_full_preflights=tribe_full_preflights,
    )
    return {
        "schema_version": 1,
        "experiment": "attention_capture_publication_path_audit",
        "generated_at": datetime.now(UTC).isoformat(),
        "readiness_json": str(readiness_json) if readiness_json is not None else None,
        "workflow_jsons": [str(path) for path in workflow_jsons],
        "feature_cache_audit_jsons": [str(path) for path in feature_cache_audits or []],
        "modal_asset_audit_jsons": [str(path) for path in modal_asset_audits or []],
        "tribe_full_preflight_audit_jsons": [
            str(path) for path in tribe_full_preflight_audits or []
        ],
        "min_paper_datasets": min_paper_datasets,
        "readiness_summary": summarize_readiness(readiness),
        "credential_audit": credential_audit,
        "full_multimodal_ready": full_multimodal_ready,
        "feature_cache_audit_summaries": cache_audits,
        "modal_asset_audit_summaries": modal_audits,
        "tribe_full_preflight_summaries": tribe_full_preflights,
        "workflow_summaries": workflows,
        "phase1_gate_passed": phase1_gate_passed,
        "phase2_ready": phase1_gate_passed,
        "publication_ready": not blockers,
        "paper_claim_allowed": not blockers,
        "blocking_reasons": blockers,
        "warnings": warnings,
        "next_actions": next_actions(blockers, warnings),
        "claim_boundary": (
            "This audit decides whether current evidence can support the "
            "attention-capture paper claim. It is stricter than data readiness: "
            "a runnable manifest is not enough when the scoring gate failed or "
            "required external or full-mode evidence is absent."
        ),
    }


def load_readiness(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_workflow(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    primary = payload.get("primary_report") or {}
    gate = primary.get("gate") or {}
    groups = list(primary.get("groups") or [])
    pooled = primary.get("pooled")
    if isinstance(pooled, dict):
        groups.append(pooled)
    datasets = sorted(
        {
            str(group.get("group") or "unknown")
            for group in groups
            if isinstance(group, dict)
        },
    )
    capture_metrics: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        metric = capture_score_metric(group)
        if metric is not None:
            capture_metrics.append(metric)
    best_capture = best_metric(capture_metrics)
    manifest_path = str(payload.get("manifest_path") or "")
    return {
        "path": str(path),
        "manifest_path": manifest_path,
        "primary_label": payload.get("primary_label"),
        "scoring_executed": bool(
            (payload.get("score_decision") or {}).get("scoring_executed"),
        ),
        "score_decision_reason": (payload.get("score_decision") or {}).get("reason"),
        "claim_ready": bool((payload.get("preflight") or {}).get("claim_ready")),
        "gate_passed": bool(gate.get("claim_validated") or gate.get("passed")),
        "gate_rule": gate.get("rule"),
        "gate_rho": primary.get("gate_rho") or payload.get("gate_rho"),
        "datasets": datasets,
        "n_samples": primary.get("n_samples"),
        "n_invalid_capture_denominators": primary.get(
            "n_invalid_capture_denominators",
        ),
        "best_capture_score": best_capture,
        "audio_only": "audio_only" in str(path) or "audio_only" in manifest_path,
    }


def capture_score_metric(group: dict[str, Any]) -> dict[str, Any] | None:
    metrics = group.get("metrics")
    if not isinstance(metrics, dict):
        return None
    capture = metrics.get("capture_score")
    if not isinstance(capture, dict):
        return None
    rho = capture.get("rho")
    if not isinstance(rho, int | float):
        return None
    return {
        "group": group.get("group"),
        "n": capture.get("n"),
        "rho": float(rho),
        "permutation_p_greater": capture.get("permutation_p_greater"),
        "gate_passed": bool(capture.get("gate_passed")),
    }


def best_metric(metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not metrics:
        return None
    return max(metrics, key=lambda metric: float(metric["rho"]))


def audit_text_model_credentials(token_envs: tuple[str, ...]) -> dict[str, Any]:
    entries = [
        {"env": name, "present": bool(os.environ.get(name))} for name in token_envs
    ]
    return {
        "token_envs_checked": list(token_envs),
        "entries": entries,
        "any_present": any(entry["present"] for entry in entries),
        "claim_boundary": (
            "Credential values are never reported. Presence only indicates that "
            "the full text-model path may be runnable; it does not prove access "
            "to any specific gated model."
        ),
    }


def summarize_feature_cache_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "feature_dir": payload.get("feature_dir"),
        "ready_for_reuse": bool(payload.get("ready_for_reuse")),
        "n_npz_files": payload.get("n_npz_files"),
        "n_expected_sample_ids": payload.get("n_expected_sample_ids"),
        "n_missing_expected_sample_ids": payload.get("n_missing_expected_sample_ids"),
        "n_bad_npz": payload.get("n_bad_npz"),
        "n_shape_mismatches": payload.get("n_shape_mismatches"),
        "aggregate_sha256": payload.get("aggregate_sha256"),
        "archive_uri": payload.get("archive_uri"),
        "n_rerun_commands": len(payload.get("rerun_commands") or []),
        "ready_for_reproduction": bool(payload.get("ready_for_reproduction")),
        "blocking_reasons": payload.get("blocking_reasons") or [],
    }


def summarize_modal_asset_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    unblocks = payload.get("publication_unblocks") or {}
    volume_report = payload.get("volume_report") or {}
    secret_report = payload.get("secret_report") or {}
    audits = volume_report.get("audits") or []
    return {
        "path": str(path),
        "retention_labels_maybe_available": bool(
            unblocks.get("retention_labels_maybe_available"),
        ),
        "external_dataset_dirs_maybe_available": bool(
            unblocks.get("external_dataset_dirs_maybe_available"),
        ),
        "feature_caches_maybe_available": bool(
            unblocks.get("feature_caches_maybe_available"),
        ),
        "full_multimodal_token_env_present": bool(
            unblocks.get("full_multimodal_token_env_present"),
        ),
        "blocking_reasons": list(unblocks.get("blocking_reasons") or []),
        "n_volumes_checked": len(volume_report.get("volume_names_checked") or []),
        "n_label_candidates": volume_report.get("n_label_candidates", 0),
        "n_dataset_candidates": volume_report.get("n_dataset_candidates", 0),
        "n_feature_candidates": volume_report.get("n_feature_candidates", 0),
        "n_truncated_volumes": sum(
            1
            for audit in audits
            if isinstance(audit, dict) and bool(audit.get("truncated"))
        ),
        "secret_names_checked": list(secret_report.get("secret_names_checked") or []),
        "matching_env_names": list(secret_report.get("matching_env_names") or []),
        "claim_boundary": payload.get("claim_boundary"),
    }


def summarize_tribe_full_preflight_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    preflight = payload.get("preflight") or {}
    return {
        "path": str(path),
        "ok": bool(payload.get("ok")),
        "app_name": payload.get("app_name"),
        "media_path": payload.get("media_path"),
        "event_mode": payload.get("event_mode"),
        "events_rows": preflight.get("events_rows"),
        "duration_seconds": preflight.get("duration_seconds"),
        "event_columns": list(preflight.get("event_columns") or []),
        "error_type": payload.get("error_type"),
        "error": payload.get("error"),
        "claim_boundary": payload.get("claim_boundary"),
    }


def summarize_readiness(readiness: dict[str, Any]) -> dict[str, Any]:
    values = readiness.get("readiness") if isinstance(readiness, dict) else None
    if not isinstance(values, dict):
        return {}
    keys = (
        "phase1_can_run_now",
        "snapugc_labels_ready",
        "dhf1k_labels_ready",
        "dhf1k_tribe_features_ready",
        "real_manifest_ready",
        "recommended_next_action",
    )
    return {key: values.get(key) for key in keys}


def publication_blockers(
    *,
    workflows: list[dict[str, Any]],
    phase1_gate_passed: bool,
    retention_labels_ready: bool,
    modal_asset_audits_supplied: bool,
    modal_retention_labels_available: bool,
    tribe_full_preflight_audits_supplied: bool,
    full_multimodal_ready: bool,
    has_audio_only_workflow: bool,
    enough_datasets_for_paper: bool,
    min_paper_datasets: int,
) -> list[str]:
    blockers: list[str] = []
    if not workflows:
        blockers.append("no Phase 1 workflow reports supplied")
    elif not phase1_gate_passed:
        blockers.append("current H2 capture_score failed the Phase 1 rho gate")
    if not retention_labels_ready:
        if modal_asset_audits_supplied and modal_retention_labels_available:
            blockers.append(
                "Modal SnapUGC/VQualA retention label candidates still need an "
                "audited retention manifest"
            )
        elif modal_asset_audits_supplied:
            blockers.append(
                "no SnapUGC/VQualA retention label CSV is mounted or available "
                "in audited Modal volumes"
            )
        else:
            blockers.append("no SnapUGC/VQualA retention label CSV is mounted")
    if has_audio_only_workflow and not full_multimodal_ready:
        if tribe_full_preflight_audits_supplied:
            blockers.append(
                "completed TRIBE workflows are audio-only and no successful "
                "full multimodal TRIBE preflight is available"
            )
        elif modal_asset_audits_supplied:
            blockers.append(
                "completed TRIBE workflows are audio-only and no local or Modal "
                "HuggingFace text model token or full-mode preflight is present"
            )
        else:
            blockers.append(
                "completed TRIBE workflows are audio-only and no HuggingFace "
                "text model token or full-mode preflight is present"
            )
    if not enough_datasets_for_paper:
        blockers.append(
            f"fewer than {min_paper_datasets} external datasets have completed "
            "claim-ready workflow reports"
        )
    return blockers


def publication_warnings(
    *,
    readiness: dict[str, Any],
    workflows: list[dict[str, Any]],
    cache_audits: list[dict[str, Any]],
    modal_audits: list[dict[str, Any]],
    tribe_full_preflights: list[dict[str, Any]],
) -> list[str]:
    warnings = feature_cache_warnings(readiness=readiness, cache_audits=cache_audits)
    if any(workflow["best_capture_score"] is None for workflow in workflows):
        warnings.append("at least one workflow lacks a capture_score metric")
    if any(audit["n_truncated_volumes"] for audit in modal_audits):
        warnings.append(
            "at least one Modal asset audit hit its per-volume scan limit; rerun "
            "with a larger --max-entries-per-volume before treating absence as final"
        )
    if any(not audit["ok"] for audit in tribe_full_preflights):
        warnings.append("at least one TRIBE full-mode preflight audit failed")
    return warnings


def feature_cache_warnings(
    *,
    readiness: dict[str, Any],
    cache_audits: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    feature_dirs = (
        readiness.get("tribe_feature_dirs") if isinstance(readiness, dict) else None
    )
    ready_cache_audit = any(audit["ready_for_reuse"] for audit in cache_audits)
    reproduction_ready_cache_audit = any(
        audit["ready_for_reproduction"] for audit in cache_audits
    )
    cache_reproduction_warning_emitted = False
    if isinstance(feature_dirs, list):
        absolute_feature_dirs = [
            str(item.get("path"))
            for item in feature_dirs
            if isinstance(item, dict) and str(item.get("path", "")).startswith("/")
        ]
        if absolute_feature_dirs and reproduction_ready_cache_audit:
            pass
        elif absolute_feature_dirs and ready_cache_audit:
            warnings.append(
                "TRIBE feature cache has checksum provenance, but the cache is "
                "still external to git and needs an archive location or "
                "deterministic rerun path"
            )
            cache_reproduction_warning_emitted = True
        elif absolute_feature_dirs:
            warnings.append(
                "TRIBE feature cache is external to the repo and should be "
                "archived or regenerated for reproducibility"
            )
            cache_reproduction_warning_emitted = True
    for audit in cache_audits:
        if not audit["ready_for_reuse"]:
            warnings.append(f"feature cache audit is not reusable: {audit['path']}")
        elif (
            not audit["ready_for_reproduction"]
            and not cache_reproduction_warning_emitted
        ):
            warnings.append(
                "feature cache audit lacks archive URI or deterministic rerun "
                f"commands: {audit['path']}"
            )
    return warnings


def next_actions(blockers: list[str], warnings: list[str]) -> list[str]:
    actions: list[str] = []
    if any("capture_score failed" in blocker for blocker in blockers):
        actions.append(
            "Do not enter Phase 2/3 neutralization from the current H2 score; "
            "either acquire retention labels for an independent test or "
            "preregister a revised score before evaluating held-out data."
        )
    if any("retention label" in blocker for blocker in blockers):
        actions.append(
            "Mount granted SnapUGC/VQualA labels and build a retention manifest "
            "with alignment-audit provenance."
        )
    if any(
        "HuggingFace" in blocker or "full multimodal TRIBE preflight" in blocker
        for blocker in blockers
    ):
        actions.append(
            "Provide a HuggingFace token with access to the gated TRIBE text "
            "model path or pass a full-mode TRIBE preflight from cached Modal "
            "weights, then rerun full multimodal feature extraction."
        )
    if any("external datasets" in blocker for blocker in blockers):
        actions.append(
            "Require at least one held-out external validation dataset before "
            "claiming publication readiness."
        )
    if any("feature cache" in warning for warning in warnings):
        actions.append(
            "Add an object-storage/archive location or deterministic rerun "
            "instructions for the external TRIBE feature cache."
        )
    return dedupe(actions)


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def render_publication_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Attention-Capture Publication Path Audit",
        "",
        "## Verdict",
        "",
        f"- Publication ready: {report['publication_ready']}",
        f"- Paper claim allowed: {report['paper_claim_allowed']}",
        f"- Phase 2 ready: {report['phase2_ready']}",
        f"- Phase 1 gate passed: {report['phase1_gate_passed']}",
        f"- Full multimodal path ready: {report['full_multimodal_ready']}",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Blocking Reasons",
        "",
    ]
    blockers = report["blocking_reasons"]
    lines.extend(f"- {blocker}" for blocker in blockers) if blockers else lines.append(
        "- none",
    )
    lines.extend(["", "## Warnings", ""])
    warnings = report["warnings"]
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append(
        "- none",
    )
    lines.extend(["", "## Next Actions", ""])
    actions = report["next_actions"]
    lines.extend(f"- {action}" for action in actions) if actions else lines.append(
        "- none",
    )
    lines.extend(render_workflow_table(report["workflow_summaries"]))
    lines.extend(render_feature_cache_table(report["feature_cache_audit_summaries"]))
    lines.extend(render_modal_asset_table(report["modal_asset_audit_summaries"]))
    lines.extend(
        render_tribe_full_preflight_table(report["tribe_full_preflight_summaries"])
    )
    return "\n".join(lines) + "\n"


def render_workflow_table(workflows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Workflow Evidence",
        "",
        "| workflow | datasets | gate | best rho | p | n | invalid denominators |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    if not workflows:
        lines.append("| none | n/a | False | n/a | n/a | 0 | 0 |")
        return lines
    for workflow in workflows:
        metric = workflow["best_capture_score"] or {}
        lines.append(
            "| "
            f"{workflow['path']} | "
            f"{', '.join(workflow['datasets']) or 'n/a'} | "
            f"{workflow['gate_passed']} | "
            f"{format_float(metric.get('rho'))} | "
            f"{format_float(metric.get('permutation_p_greater'))} | "
            f"{metric.get('n') or 0} | "
            f"{workflow['n_invalid_capture_denominators'] or 0} |"
        )
    return lines


def render_feature_cache_table(cache_audits: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Feature Cache Evidence",
        "",
        (
            "| audit | feature dir | ready | reproduction | npz files | "
            "expected ids | rerun cmds | aggregate sha256 |"
        ),
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    if not cache_audits:
        lines.append("| none | n/a | False | False | 0 | 0 | 0 | n/a |")
        return lines
    for audit in cache_audits:
        lines.append(
            "| "
            f"{audit['path']} | {audit['feature_dir'] or 'n/a'} | "
            f"{audit['ready_for_reuse']} | {audit['ready_for_reproduction']} | "
            f"{audit['n_npz_files'] or 0} | "
            f"{audit['n_expected_sample_ids'] or 0} | "
            f"{audit['n_rerun_commands'] or 0} | "
            f"{str(audit['aggregate_sha256'] or 'n/a')[:12]} |"
        )
    return lines


def render_modal_asset_table(modal_audits: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Modal Asset Evidence",
        "",
        (
            "| audit | volumes | labels | datasets | features | modal token | "
            "truncated | blockers |"
        ),
        "|---|---:|---|---|---|---|---:|---|",
    ]
    if not modal_audits:
        lines.append("| none | 0 | False | False | False | False | 0 | n/a |")
        return lines
    for audit in modal_audits:
        blockers = "; ".join(audit["blocking_reasons"]) or "none"
        lines.append(
            "| "
            f"{table_cell(audit['path'])} | "
            f"{audit['n_volumes_checked']} | "
            f"{audit['retention_labels_maybe_available']} | "
            f"{audit['external_dataset_dirs_maybe_available']} | "
            f"{audit['feature_caches_maybe_available']} | "
            f"{audit['full_multimodal_token_env_present']} | "
            f"{audit['n_truncated_volumes']} | "
            f"{table_cell(blockers)} |"
        )
    return lines


def render_tribe_full_preflight_table(preflights: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## TRIBE Full-Preflight Evidence",
        "",
        "| audit | ok | event mode | events | duration | media | error |",
        "|---|---|---|---:|---:|---|---|",
    ]
    if not preflights:
        lines.append("| none | False | n/a | 0 | n/a | n/a | n/a |")
        return lines
    for audit in preflights:
        lines.append(
            "| "
            f"{table_cell(audit['path'])} | "
            f"{audit['ok']} | "
            f"{audit['event_mode'] or 'n/a'} | "
            f"{audit['events_rows'] or 0} | "
            f"{format_float(audit['duration_seconds'])} | "
            f"{table_cell(audit['media_path'] or 'n/a')} | "
            f"{table_cell(audit['error'] or 'none')} |"
        )
    return lines


def format_float(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.4f}"


def table_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    main()
