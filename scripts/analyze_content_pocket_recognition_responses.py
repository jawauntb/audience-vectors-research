"""Analyze two-session content-pocket recognition-memory responses.

This script consumes JSON payloads emitted by the Prolific launch pages and
turns them into an auditable gate report. It deliberately keeps excluded rows in
the output with explicit reasons, because the response export is part of the
evidence trail rather than a disposable preprocessing step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(
    "research_program/neurips_memorability_selector/collaborator_inputs/"
    "camilo_bo_memorability"
)
DEFAULT_OUT_JSON = (
    ARTIFACT_DIR / "content_pocket_recognition_response_analysis_result_20260608.json"
)
DEFAULT_OUT_MD = (
    ARTIFACT_DIR / "content_pocket_recognition_response_analysis_result_20260608.md"
)
DEFAULT_MIN_USABLE_PARTICIPANTS = 200

PRIMARY_GROUP = "primary_positive"
HARD_NEGATIVE_GROUP = "hard_negative_control"
ANALYSIS_GROUPS = {PRIMARY_GROUP, HARD_NEGATIVE_GROUP}
PRIMARY_ARMS = {"orange_flowers", "hanging_clothes"}
HARD_NEGATIVE_ARMS = {"aerial_beach", "city_street", "storm_beach"}
REQUIRED_ANALYSIS_ARMS = PRIMARY_ARMS | HARD_NEGATIVE_ARMS


@dataclass(frozen=True)
class AnalysisConfig:
    """Pre-registered response-analysis settings."""

    min_usable_participants: int = DEFAULT_MIN_USABLE_PARTICIPANTS
    exclude_participant_ids: frozenset[str] = frozenset()
    exclude_participant_prefixes: tuple[str, ...] = ()
    exclude_study_ids: frozenset[str] = frozenset()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def response_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.suffix.lower() in {".json", ".jsonl", ".csv"}
            )
        elif path.exists() and path.suffix.lower() in {".json", ".jsonl", ".csv"}:
            files.append(path)
        elif not path.exists():
            raise FileNotFoundError(f"Response path does not exist: {path}")
    return files


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def unwrap_payload(value: Any) -> dict[str, Any] | None:
    value = maybe_json(value)
    if isinstance(value, dict) and "responses" in value and "session_number" in value:
        return value
    if isinstance(value, dict):
        for key in ("payload", "body", "data", "json", "response_json"):
            if key in value:
                payload = unwrap_payload(value[key])
                if payload is not None:
                    return payload
    return None


def payloads_from_json(path: Path) -> list[dict[str, Any]]:
    raw = load_json(path)
    values = raw if isinstance(raw, list) else [raw]
    payloads = []
    for value in values:
        payload = unwrap_payload(value)
        if payload is not None:
            payloads.append({**payload, "_source_file": str(path)})
    return payloads


def payloads_from_jsonl(path: Path) -> list[dict[str, Any]]:
    payloads = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = unwrap_payload(line)
        if payload is not None:
            payloads.append(
                {
                    **payload,
                    "_source_file": str(path),
                    "_source_line": line_number,
                }
            )
    return payloads


def payloads_from_csv(path: Path) -> list[dict[str, Any]]:
    payloads = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 2):
            payload = unwrap_payload(row)
            if payload is not None:
                payloads.append(
                    {
                        **payload,
                        "_source_file": str(path),
                        "_source_row": row_number,
                    }
                )
    return payloads


def load_payloads(paths: list[Path]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in response_files(paths):
        suffix = path.suffix.lower()
        if suffix == ".json":
            payloads.extend(payloads_from_json(path))
        elif suffix == ".jsonl":
            payloads.extend(payloads_from_jsonl(path))
        elif suffix == ".csv":
            payloads.extend(payloads_from_csv(path))
    return payloads


def participant_id(payload: dict[str, Any]) -> str:
    return str(
        payload.get("prolific_pid")
        or payload.get("participant_id")
        or payload.get("participantId")
        or "unknown"
    ).strip()


def completed_at(payload: dict[str, Any]) -> str:
    value = payload.get("completed_at") or payload.get("completedAt") or ""
    return str(value)


def boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return None


def select_latest_payloads(
    payloads: list[dict[str, Any]],
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        raw_session_number = payload.get("session_number")
        if raw_session_number is None:
            continue
        try:
            session_number = int(raw_session_number)
        except (TypeError, ValueError):
            continue
        grouped[(participant_id(payload), session_number)].append(payload)

    selected: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates = []
    for key, items in grouped.items():
        items.sort(key=completed_at)
        selected[key] = items[-1]
        if len(items) > 1:
            duplicates.append(
                {
                    "participant_id": key[0],
                    "session_number": key[1],
                    "payload_count": len(items),
                    "selected_completed_at": completed_at(items[-1]),
                }
            )
    return selected, {"duplicate_session_payloads": duplicates}


def manual_exclusion_reasons(
    participant: str,
    payloads: dict[int, dict[str, Any]],
    config: AnalysisConfig,
) -> list[str]:
    reasons = []
    if participant in config.exclude_participant_ids:
        reasons.append("manual_participant_exclusion")
    if any(participant.startswith(prefix) for prefix in config.exclude_participant_prefixes):
        reasons.append("manual_participant_prefix_exclusion")
    study_ids = {
        str(payload.get("prolific_study_id") or "")
        for payload in payloads.values()
        if payload.get("prolific_study_id")
    }
    if study_ids & set(config.exclude_study_ids):
        reasons.append("manual_study_exclusion")
    return reasons


def response_index(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if payload is None:
        return {}
    rows = payload.get("responses") or []
    return {str(row.get("target_id")): row for row in rows if row.get("target_id")}


def session2_media_error(row: dict[str, Any]) -> bool:
    return any(
        boolish(row.get(field)) is True
        for field in ("any_media_error", "left_media_error", "right_media_error")
    )


def evaluate_trial(
    *,
    participant: str,
    session1_payload: dict[str, Any] | None,
    session2_payload: dict[str, Any],
    session1_rows_by_target: dict[str, dict[str, Any]],
    row: dict[str, Any],
    manual_reasons: list[str],
) -> dict[str, Any]:
    target_id = str(row.get("target_id") or "")
    analysis_group = str(row.get("analysis_group") or "")
    arm_id = str(row.get("arm_id") or "")
    reasons = list(manual_reasons)

    if session1_payload is None:
        reasons.append("missing_session1_payload")
    elif str(session1_payload.get("form_id")) != str(session2_payload.get("form_id")):
        reasons.append("form_mismatch")

    exposure = session1_rows_by_target.get(target_id)
    if analysis_group in ANALYSIS_GROUPS:
        if exposure is None:
            reasons.append("missing_session1_exposure")
        else:
            if boolish(exposure.get("media_error")) is True:
                reasons.append("session1_media_error")
            if boolish(exposure.get("exposure_completed")) is False:
                reasons.append("session1_incomplete_exposure")

    if session2_media_error(row):
        reasons.append("session2_media_error")

    is_correct = boolish(row.get("is_correct"))
    if is_correct is None:
        reasons.append("missing_correctness")

    return {
        "participant_id": participant,
        "form_id": str(session2_payload.get("form_id") or ""),
        "session2_completed_at": completed_at(session2_payload),
        "trial_id": row.get("trial_id"),
        "target_id": target_id,
        "arm_id": arm_id,
        "analysis_group": analysis_group,
        "choice_side": row.get("choice_side"),
        "correct_choice": row.get("correct_choice"),
        "is_correct": is_correct,
        "response_time_ms": row.get("response_time_ms"),
        "old_side": row.get("old_side"),
        "session1_media_error": boolish(exposure.get("media_error")) if exposure else None,
        "session1_exposure_completed": (
            boolish(exposure.get("exposure_completed")) if exposure else None
        ),
        "session2_media_error": session2_media_error(row),
        "exclusion_reasons": sorted(set(reasons)),
        "usable_trial_before_complete_case_filter": not reasons,
        "usable_for_primary_gate": False,
    }


def build_trial_rows(
    payloads: list[dict[str, Any]],
    config: AnalysisConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, selection_report = select_latest_payloads(payloads)
    participants = sorted({participant for participant, _session in selected})
    session_participants = {
        session: sorted(participant for participant, payload_session in selected if payload_session == session)
        for session in (1, 2)
    }

    trial_rows = []
    participant_reasons: dict[str, list[str]] = {}
    for participant in participants:
        session_payloads = {
            session: selected[(participant, session)]
            for session in (1, 2)
            if (participant, session) in selected
        }
        reasons = manual_exclusion_reasons(participant, session_payloads, config)
        if 2 not in session_payloads:
            participant_reasons[participant] = [*reasons, "missing_session2_payload"]
            continue
        if 1 not in session_payloads:
            participant_reasons[participant] = [*reasons, "missing_session1_payload"]

        session1_payload = session_payloads.get(1)
        session2_payload = session_payloads[2]
        session1_rows_by_target = response_index(session1_payload)
        for row in session2_payload.get("responses") or []:
            trial_rows.append(
                evaluate_trial(
                    participant=participant,
                    session1_payload=session1_payload,
                    session2_payload=session2_payload,
                    session1_rows_by_target=session1_rows_by_target,
                    row=row,
                    manual_reasons=reasons,
                )
            )

    complete_participants = complete_case_participants(trial_rows)
    for row in trial_rows:
        if row["analysis_group"] not in ANALYSIS_GROUPS:
            continue
        if row["participant_id"] not in complete_participants:
            row["exclusion_reasons"] = sorted(
                {*row["exclusion_reasons"], "participant_incomplete_analysis_set"}
            )
        row["usable_for_primary_gate"] = not row["exclusion_reasons"]

    participant_exclusion_counts = Counter(
        reason
        for reasons in participant_reasons.values()
        for reason in reasons
    )
    return trial_rows, {
        **selection_report,
        "participants_seen": participants,
        "session_participants": session_participants,
        "participant_exclusions": participant_reasons,
        "participant_exclusion_counts": dict(sorted(participant_exclusion_counts.items())),
        "complete_case_participants": sorted(complete_participants),
    }


def complete_case_participants(trial_rows: list[dict[str, Any]]) -> set[str]:
    arms_by_participant: dict[str, set[str]] = defaultdict(set)
    for row in trial_rows:
        if row["analysis_group"] not in ANALYSIS_GROUPS:
            continue
        if row["usable_trial_before_complete_case_filter"]:
            arms_by_participant[row["participant_id"]].add(row["arm_id"])
    return {
        participant
        for participant, arms in arms_by_participant.items()
        if REQUIRED_ANALYSIS_ARMS.issubset(arms)
    }


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    phat = correct / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(1 for row in rows if row["is_correct"] is True)
    low, high = wilson_interval(correct, n)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "wilson_95_ci": [low, high],
    }


def two_proportion_z_test(
    *,
    correct_a: int,
    n_a: int,
    correct_b: int,
    n_b: int,
) -> dict[str, float | None]:
    if n_a == 0 or n_b == 0:
        return {"z": None, "p_two_sided": None}
    p_a = correct_a / n_a
    p_b = correct_b / n_b
    pooled = (correct_a + correct_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return {"z": None, "p_two_sided": None}
    z = (p_a - p_b) / se
    return {"z": z, "p_two_sided": math.erfc(abs(z) / math.sqrt(2))}


def exact_sign_test_p_value(positive: int, negative: int) -> float | None:
    n = positive + negative
    if n == 0:
        return None
    extreme = max(positive, negative)
    tail = sum(math.comb(n, k) for k in range(extreme, n + 1)) / (2**n)
    return min(1.0, 2 * tail)


def participant_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row["usable_for_primary_gate"]]
    by_participant: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in usable:
        by_participant[row["participant_id"]][row["analysis_group"]].append(row["is_correct"])

    paired = []
    for participant, groups in sorted(by_participant.items()):
        if PRIMARY_GROUP not in groups or HARD_NEGATIVE_GROUP not in groups:
            continue
        primary_mean = sum(groups[PRIMARY_GROUP]) / len(groups[PRIMARY_GROUP])
        hard_mean = sum(groups[HARD_NEGATIVE_GROUP]) / len(groups[HARD_NEGATIVE_GROUP])
        paired.append(
            {
                "participant_id": participant,
                "primary_accuracy": primary_mean,
                "hard_negative_accuracy": hard_mean,
                "difference": primary_mean - hard_mean,
            }
        )

    diffs = [row["difference"] for row in paired]
    positive = sum(1 for diff in diffs if diff > 0)
    negative = sum(1 for diff in diffs if diff < 0)
    return {
        "paired_participants": len(paired),
        "mean_primary_minus_hard_negative": sum(diffs) / len(diffs) if diffs else None,
        "positive_differences": positive,
        "negative_differences": negative,
        "two_sided_sign_test_p": exact_sign_test_p_value(positive, negative),
        "rows": paired,
    }


def group_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row["usable_for_primary_gate"]]
    by_group = {
        group: summarize_rows([row for row in usable if row["analysis_group"] == group])
        for group in (PRIMARY_GROUP, HARD_NEGATIVE_GROUP)
    }
    by_arm = {
        arm: summarize_rows([row for row in usable if row["arm_id"] == arm])
        for arm in sorted(REQUIRED_ANALYSIS_ARMS)
    }
    primary = by_group[PRIMARY_GROUP]
    hard = by_group[HARD_NEGATIVE_GROUP]
    primary_vs_hard = two_proportion_z_test(
        correct_a=int(primary["correct"]),
        n_a=int(primary["n"]),
        correct_b=int(hard["correct"]),
        n_b=int(hard["n"]),
    )
    return {
        "by_analysis_group": by_group,
        "by_arm": by_arm,
        "primary_vs_hard_negative_two_proportion_z": primary_vs_hard,
        "participant_effects": participant_effects(rows),
    }


def greater_than(value: float | None, threshold: float | None) -> bool:
    return value is not None and threshold is not None and value > threshold


def gate_report(
    *,
    rows: list[dict[str, Any]],
    summaries: dict[str, Any],
    min_usable_participants: int,
) -> dict[str, Any]:
    usable_participants = summaries["participant_effects"]["paired_participants"]
    primary_accuracy = summaries["by_analysis_group"][PRIMARY_GROUP]["accuracy"]
    hard_accuracy = summaries["by_analysis_group"][HARD_NEGATIVE_GROUP]["accuracy"]
    orange_accuracy = summaries["by_arm"]["orange_flowers"]["accuracy"]
    hanging_accuracy = summaries["by_arm"]["hanging_clothes"]["accuracy"]
    z_p = summaries["primary_vs_hard_negative_two_proportion_z"]["p_two_sided"]

    criteria = {
        "minimum_usable_participants": usable_participants >= min_usable_participants,
        "pooled_primary_accuracy_exceeds_hard_negative": greater_than(
            primary_accuracy,
            hard_accuracy,
        ),
        "orange_flowers_accuracy_exceeds_hard_negative": greater_than(
            orange_accuracy,
            hard_accuracy,
        ),
        "hanging_clothes_accuracy_exceeds_hard_negative": greater_than(
            hanging_accuracy,
            hard_accuracy,
        ),
        "pooled_primary_accuracy_above_chance": greater_than(primary_accuracy, 0.5),
        "orange_flowers_accuracy_above_chance": greater_than(orange_accuracy, 0.5),
        "hanging_clothes_accuracy_above_chance": greater_than(hanging_accuracy, 0.5),
        "pooled_primary_vs_hard_negative_p_lt_0_05": (
            z_p is not None and z_p < 0.05
        ),
    }
    if not rows:
        status = "no_response_data"
    elif not criteria["minimum_usable_participants"]:
        status = "underpowered_do_not_interpret"
    elif all(criteria.values()):
        status = "passed"
    else:
        status = "failed"
    return {
        "status": status,
        "min_usable_participants": min_usable_participants,
        "usable_participants": usable_participants,
        "criteria": criteria,
        "claim_boundary": (
            "Only a passed gate on delayed Session 2 data supports a human "
            "recognition-memory claim for the primary pockets."
        ),
    }


def analyze_payloads(payloads: list[dict[str, Any]], config: AnalysisConfig) -> dict[str, Any]:
    trial_rows, selection_report = build_trial_rows(payloads, config)
    summaries = group_summaries(trial_rows)
    exclusion_counts = Counter(
        reason
        for row in trial_rows
        for reason in row["exclusion_reasons"]
    )
    usable_rows = [row for row in trial_rows if row["usable_for_primary_gate"]]
    return {
        "schema_version": "content_pocket_recognition_response_analysis.v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status": "analysis_complete" if payloads else "no_response_data",
        "config": {
            "min_usable_participants": config.min_usable_participants,
            "exclude_participant_ids": sorted(config.exclude_participant_ids),
            "exclude_participant_prefixes": list(config.exclude_participant_prefixes),
            "exclude_study_ids": sorted(config.exclude_study_ids),
        },
        "counts": {
            "payloads_loaded": len(payloads),
            "participants_seen": len(selection_report["participants_seen"]),
            "session1_participants": len(selection_report["session_participants"][1]),
            "session2_participants": len(selection_report["session_participants"][2]),
            "trial_rows": len(trial_rows),
            "usable_primary_gate_rows": len(usable_rows),
            "usable_primary_gate_participants": summaries["participant_effects"][
                "paired_participants"
            ],
        },
        "gate": gate_report(
            rows=trial_rows,
            summaries=summaries,
            min_usable_participants=config.min_usable_participants,
        ),
        "summaries": summaries,
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "selection_report": selection_report,
        "trial_rows": trial_rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    gate = report["gate"]
    summaries = report["summaries"]
    counts = report["counts"]
    lines = [
        "# Content-Pocket Recognition Response Analysis",
        "",
        f"Date: {report['created_at_utc']}",
        "",
        "## Discovery-Regime Audit",
        "",
        "Question: do primary SVD content pockets produce actual delayed",
        "old-vs-lure recognition-memory gains over hard-negative controls?",
        "",
        "Action class: human-validation analysis inside the recognition-memory",
        "regime. This can validate human memory only after enough delayed",
        "Session 2 data clear the gate.",
        "",
        "## Status",
        "",
        f"- Status: `{gate['status']}`",
        f"- Payloads loaded: {counts['payloads_loaded']}",
        f"- Participants seen: {counts['participants_seen']}",
        f"- Session 1 participants: {counts['session1_participants']}",
        f"- Session 2 participants: {counts['session2_participants']}",
        f"- Usable gate participants: {gate['usable_participants']}",
        f"- Minimum usable participants: {gate['min_usable_participants']}",
        "",
        "## Gate Criteria",
        "",
        "| criterion | passed |",
        "|---|---:|",
    ]
    for name, passed in gate["criteria"].items():
        lines.append(f"| `{name}` | {str(passed).lower()} |")

    lines.extend(
        [
            "",
            "## Accuracy Summary",
            "",
            "| group | n | correct | accuracy | Wilson 95% CI |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for group, item in summaries["by_analysis_group"].items():
        low, high = item["wilson_95_ci"]
        ci = "n/a" if low is None else f"[{low:.3f}, {high:.3f}]"
        accuracy = "n/a" if item["accuracy"] is None else f"{item['accuracy']:.3f}"
        lines.append(f"| `{group}` | {item['n']} | {item['correct']} | {accuracy} | {ci} |")

    z_test = summaries["primary_vs_hard_negative_two_proportion_z"]
    lines.extend(
        [
            "",
            "## Primary Contrast",
            "",
            f"- Two-proportion z: {z_test['z'] if z_test['z'] is not None else 'n/a'}",
            "- Two-sided p: "
            f"{z_test['p_two_sided'] if z_test['p_two_sided'] is not None else 'n/a'}",
            "- Paired participant mean primary-minus-hard-negative: "
            f"{summaries['participant_effects']['mean_primary_minus_hard_negative']}",
            "- Paired sign-test p: "
            f"{summaries['participant_effects']['two_sided_sign_test_p']}",
            "",
            "## Arm Summary",
            "",
            "| arm | n | correct | accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm, item in summaries["by_arm"].items():
        accuracy = "n/a" if item["accuracy"] is None else f"{item['accuracy']:.3f}"
        lines.append(f"| `{arm}` | {item['n']} | {item['correct']} | {accuracy} |")

    lines.extend(["", "## Exclusions", ""])
    if report["exclusion_counts"]:
        lines.extend(["| reason | rows |", "|---|---:|"])
        for reason, count in report["exclusion_counts"].items():
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("No trial-level exclusions recorded.")

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- {gate['claim_boundary']}",
            "- Underpowered, dry-run, media-error, missing-session, and incomplete",
            "  complete-case outputs are not human memorability evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        type=Path,
        action="append",
        default=[],
        help="Response file or directory containing .json, .jsonl, or .csv payloads.",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--min-usable-participants",
        type=int,
        default=DEFAULT_MIN_USABLE_PARTICIPANTS,
    )
    parser.add_argument("--exclude-participant-id", action="append", default=[])
    parser.add_argument("--exclude-participant-prefix", action="append", default=[])
    parser.add_argument("--exclude-study-id", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_usable_participants <= 0:
        raise ValueError("--min-usable-participants must be positive")
    config = AnalysisConfig(
        min_usable_participants=args.min_usable_participants,
        exclude_participant_ids=frozenset(args.exclude_participant_id),
        exclude_participant_prefixes=tuple(args.exclude_participant_prefix),
        exclude_study_ids=frozenset(args.exclude_study_id),
    )
    payloads = load_payloads(args.responses)
    report = analyze_payloads(payloads, config)
    write_json(args.out_json, report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"[done] wrote {args.out_json}")
    print(f"[done] wrote {args.out_md}")
    print(f"[done] gate status: {report['gate']['status']}")
    print(f"[done] usable gate participants: {report['gate']['usable_participants']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
