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
    min_paper_datasets: int = 2,
    token_envs: tuple[str, ...] = DEFAULT_TOKEN_ENVS,
) -> dict[str, Any]:
    readiness = load_readiness(readiness_json)
    workflows = [summarize_workflow(path) for path in workflow_jsons]
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
    full_multimodal_ready = credential_audit["any_present"]
    has_audio_only_workflow = any(workflow["audio_only"] for workflow in workflows)
    phase1_gate_passed = bool(passed_workflows)
    enough_datasets_for_paper = len(scored_datasets) >= min_paper_datasets

    blockers = publication_blockers(
        workflows=workflows,
        phase1_gate_passed=phase1_gate_passed,
        retention_labels_ready=retention_labels_ready,
        full_multimodal_ready=full_multimodal_ready,
        has_audio_only_workflow=has_audio_only_workflow,
        enough_datasets_for_paper=enough_datasets_for_paper,
        min_paper_datasets=min_paper_datasets,
    )
    warnings = publication_warnings(
        readiness=readiness,
        workflows=workflows,
    )
    return {
        "schema_version": 1,
        "experiment": "attention_capture_publication_path_audit",
        "generated_at": datetime.now(UTC).isoformat(),
        "readiness_json": str(readiness_json) if readiness_json is not None else None,
        "workflow_jsons": [str(path) for path in workflow_jsons],
        "min_paper_datasets": min_paper_datasets,
        "readiness_summary": summarize_readiness(readiness),
        "credential_audit": credential_audit,
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
            "retention/full-multimodal evidence is absent."
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
        {"env": name, "present": bool(os.environ.get(name))}
        for name in token_envs
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
        blockers.append("no SnapUGC/VQualA retention label CSV is mounted")
    if has_audio_only_workflow and not full_multimodal_ready:
        blockers.append(
            "completed TRIBE workflows are audio-only and no HuggingFace text "
            "model token is present"
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
) -> list[str]:
    warnings: list[str] = []
    feature_dirs = readiness.get("tribe_feature_dirs") if isinstance(readiness, dict) else None
    if isinstance(feature_dirs, list):
        absolute_feature_dirs = [
            str(item.get("path"))
            for item in feature_dirs
            if isinstance(item, dict) and str(item.get("path", "")).startswith("/")
        ]
        if absolute_feature_dirs:
            warnings.append(
                "TRIBE feature cache is external to the repo and should be "
                "archived or regenerated for reproducibility"
            )
    if any(workflow["best_capture_score"] is None for workflow in workflows):
        warnings.append("at least one workflow lacks a capture_score metric")
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
    if any("HuggingFace" in blocker for blocker in blockers):
        actions.append(
            "Provide a HuggingFace token with access to the gated TRIBE text "
            "model path, then rerun full multimodal feature extraction."
        )
    if any("external datasets" in blocker for blocker in blockers):
        actions.append(
            "Require at least one held-out external validation dataset before "
            "claiming publication readiness."
        )
    if any("feature cache" in warning for warning in warnings):
        actions.append(
            "Create a non-git artifact plan for the external TRIBE feature "
            "cache: checksum manifest, object storage, or deterministic rerun "
            "instructions."
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
        f"- Full multimodal credential present: {report['credential_audit']['any_present']}",
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


def format_float(value: object) -> str:
    if not isinstance(value, int | float):
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
