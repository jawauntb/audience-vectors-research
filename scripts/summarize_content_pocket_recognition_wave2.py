"""Summarize Wave 2 content-pocket recognition-memory exports.

The June 2026 Wave 1 webhook inbox hit an early retention cap, so this report
uses Prolific-confirmed Session 1 completion plus deterministic form assignment
and complete Session 2 recognition payloads. Raw exports can contain participant
metadata; committed outputs from this script contain only aggregate counts and
deidentified hashes where needed for audit provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

PRIMARY_GROUP = "primary_positive"
HARD_NEGATIVE_GROUP = "hard_negative_control"
FILLER_GROUP = "unrelated_filler"
PRIMARY_ARMS = ("orange_flowers", "hanging_clothes")
HARD_NEGATIVE_ARMS = ("aerial_beach", "city_street", "storm_beach")
ARM_ORDER = (*PRIMARY_ARMS, *HARD_NEGATIVE_ARMS)

POSITIVE_COLOR = "#2f855a"
NEGATIVE_COLOR = "#2b6cb0"
FILLER_COLOR = "#718096"
INK = "#1a202c"
MUTED = "#4a5568"
GRID = "#e2e8f0"


@dataclass(frozen=True)
class SummaryRow:
    """Aggregate binary recognition result."""

    name: str
    correct: int
    n: int
    accuracy: float
    wilson95_low: float
    wilson95_high: float
    exact_binom_p_vs_0_5: float
    media_errors: int
    median_rt_ms: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webhook-csv", required=True, type=Path)
    parser.add_argument("--prolific-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    parser.add_argument(
        "--hash-salt",
        default="content-pocket-20260611",
        help="Salt for deidentified participant hashes.",
    )
    parser.add_argument("--bootstrap-iters", type=int, default=50_000)
    parser.add_argument("--permutation-iters", type=int, default=100_000)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def maybe_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped or stripped[0] != "{":
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def hash_participant(participant_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{participant_id}".encode()).hexdigest()[:12]


def load_payloads(webhook_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for row in webhook_rows:
        if row.get("method") != "POST":
            continue
        payload = maybe_json(row.get("content") or "")
        if payload is None:
            continue
        payload["_webhook_uuid"] = row.get("uuid")
        payload["_webhook_created_at"] = row.get("created_at")
        payloads.append(payload)
    return payloads


def is_real_complete_session2(payload: dict[str, Any]) -> bool:
    participant_id = str(payload.get("prolific_pid") or "")
    responses = payload.get("responses") or []
    return (
        payload.get("session_number") == 2
        and bool(participant_id)
        and not participant_id.startswith("CODEX")
        and payload.get("n_trials") == 25
        and isinstance(responses, list)
        and len(responses) == 25
    )


def is_real_session1(payload: dict[str, Any]) -> bool:
    participant_id = str(payload.get("prolific_pid") or "")
    responses = payload.get("responses") or []
    return (
        payload.get("session_number") == 1
        and bool(participant_id)
        and not participant_id.startswith("CODEX")
        and payload.get("n_trials") == 30
        and isinstance(responses, list)
        and len(responses) == 30
    )


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (center - half, center + half)


def exact_binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value computed in log space."""

    if n == 0:
        return math.nan
    logp = math.log(p)
    logq = math.log1p(-p)
    log_probs = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * logp
        + (n - i) * logq
        for i in range(n + 1)
    ]
    observed = log_probs[k]
    included = [value for value in log_probs if value <= observed + 1e-12]
    maximum = max(included)
    return min(
        1.0,
        math.exp(maximum) * sum(math.exp(value - maximum) for value in included),
    )


def summarize_rows(
    name: str,
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> SummaryRow:
    selected = [row for row in rows if predicate(row)]
    n = len(selected)
    correct = sum(boolish(row["is_correct"]) for row in selected)
    low, high = wilson_interval(correct, n)
    rts = [
        int(row["response_time_ms"])
        for row in selected
        if isinstance(row.get("response_time_ms"), int)
    ]
    return SummaryRow(
        name=name,
        correct=correct,
        n=n,
        accuracy=correct / n if n else math.nan,
        wilson95_low=low,
        wilson95_high=high,
        exact_binom_p_vs_0_5=exact_binomial_two_sided(correct, n),
        media_errors=sum(boolish(row["any_media_error"]) for row in selected),
        median_rt_ms=float(statistics.median(rts)) if rts else None,
    )


def paired_primary_contrast(
    rows: list[dict[str, Any]],
    *,
    bootstrap_iters: int,
    permutation_iters: int,
    rng_seed: int,
) -> dict[str, Any]:
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_participant[row["participant_hash"]].append(row)

    diffs = []
    primary_accs = []
    hard_negative_accs = []
    for participant_rows in by_participant.values():
        primary = [
            row for row in participant_rows if row["analysis_group"] == PRIMARY_GROUP
        ]
        hard_negative = [
            row
            for row in participant_rows
            if row["analysis_group"] == HARD_NEGATIVE_GROUP
        ]
        if len(primary) != 2 or len(hard_negative) != 3:
            continue
        primary_acc = sum(boolish(row["is_correct"]) for row in primary) / 2
        hard_negative_acc = (
            sum(boolish(row["is_correct"]) for row in hard_negative) / 3
        )
        primary_accs.append(primary_acc)
        hard_negative_accs.append(hard_negative_acc)
        diffs.append(primary_acc - hard_negative_acc)

    rng = random.Random(rng_seed)
    boot_means = [
        statistics.mean(diffs[rng.randrange(len(diffs))] for _ in diffs)
        for _ in range(bootstrap_iters)
    ]
    boot_means.sort()

    observed = abs(statistics.mean(diffs))
    extreme = 0
    for _ in range(permutation_iters):
        mean = sum(diff if rng.random() < 0.5 else -diff for diff in diffs) / len(
            diffs
        )
        if abs(mean) >= observed - 1e-15:
            extreme += 1

    return {
        "participants": len(diffs),
        "primary_mean_accuracy": statistics.mean(primary_accs),
        "hard_negative_mean_accuracy": statistics.mean(hard_negative_accs),
        "mean_difference": statistics.mean(diffs),
        "bootstrap95_low": boot_means[int(0.025 * len(boot_means))],
        "bootstrap95_high": boot_means[int(0.975 * len(boot_means))],
        "sign_flip_permutation_p": (extreme + 1) / (permutation_iters + 1),
    }


def arm_contrast(
    rows: list[dict[str, Any]],
    arm_id: str,
    *,
    bootstrap_iters: int,
    permutation_iters: int,
    rng_seed: int,
) -> dict[str, Any]:
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_participant[row["participant_hash"]].append(row)

    diffs = []
    arm_accs = []
    hard_negative_accs = []
    for participant_rows in by_participant.values():
        arm_rows = [row for row in participant_rows if row["arm_id"] == arm_id]
        hard_negative = [
            row
            for row in participant_rows
            if row["analysis_group"] == HARD_NEGATIVE_GROUP
        ]
        if len(arm_rows) != 1 or len(hard_negative) != 3:
            continue
        arm_acc = float(boolish(arm_rows[0]["is_correct"]))
        hard_negative_acc = (
            sum(boolish(row["is_correct"]) for row in hard_negative) / 3
        )
        arm_accs.append(arm_acc)
        hard_negative_accs.append(hard_negative_acc)
        diffs.append(arm_acc - hard_negative_acc)

    rng = random.Random(rng_seed)
    boot_means = [
        statistics.mean(diffs[rng.randrange(len(diffs))] for _ in diffs)
        for _ in range(bootstrap_iters)
    ]
    boot_means.sort()
    observed = abs(statistics.mean(diffs))
    extreme = 0
    for _ in range(permutation_iters):
        mean = sum(diff if rng.random() < 0.5 else -diff for diff in diffs) / len(
            diffs
        )
        if abs(mean) >= observed - 1e-15:
            extreme += 1

    return {
        "participants": len(diffs),
        "arm_mean_accuracy": statistics.mean(arm_accs),
        "hard_negative_mean_accuracy": statistics.mean(hard_negative_accs),
        "mean_difference": statistics.mean(diffs),
        "bootstrap95_low": boot_means[int(0.025 * len(boot_means))],
        "bootstrap95_high": boot_means[int(0.975 * len(boot_means))],
        "sign_flip_permutation_p": (extreme + 1) / (permutation_iters + 1),
    }


def build_response_rows(
    session2_payloads: list[dict[str, Any]],
    prolific_by_participant: dict[str, dict[str, str]],
    salt: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    response_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    for payload in session2_payloads:
        participant_id = str(payload["prolific_pid"])
        participant_hash = hash_participant(participant_id, salt)
        prolific_row = prolific_by_participant.get(participant_id, {})
        participant_response_rows = []
        for index, response in enumerate(payload["responses"]):
            row = {
                "participant_hash": participant_hash,
                "form_id": payload.get("form_id"),
                "trial_index": index,
                "trial_id": response.get("trial_id"),
                "target_id": response.get("target_id"),
                "arm_id": response.get("arm_id"),
                "analysis_group": response.get("analysis_group"),
                "old_side": response.get("old_side"),
                "choice_side": response.get("choice_side"),
                "is_correct": boolish(response.get("is_correct")),
                "any_media_error": boolish(response.get("any_media_error")),
                "response_time_ms": int(response.get("response_time_ms") or 0),
            }
            response_rows.append(row)
            participant_response_rows.append(row)

        participant_rows.append(
            {
                "participant_hash": participant_hash,
                "status": prolific_row.get("Status"),
                "completion_code": prolific_row.get("Completion code"),
                "form_id": payload.get("form_id"),
                "n_correct": sum(
                    boolish(row["is_correct"]) for row in participant_response_rows
                ),
                "n_trials": len(participant_response_rows),
                "filler_correct": sum(
                    boolish(row["is_correct"])
                    for row in participant_response_rows
                    if row["analysis_group"] == FILLER_GROUP
                ),
                "filler_trials": sum(
                    1
                    for row in participant_response_rows
                    if row["analysis_group"] == FILLER_GROUP
                ),
                "pocket_correct": sum(
                    boolish(row["is_correct"])
                    for row in participant_response_rows
                    if row["analysis_group"] != FILLER_GROUP
                ),
                "pocket_trials": sum(
                    1
                    for row in participant_response_rows
                    if row["analysis_group"] != FILLER_GROUP
                ),
                "any_media_errors": sum(
                    boolish(row["any_media_error"])
                    for row in participant_response_rows
                ),
                "median_rt_ms": statistics.median(
                    row["response_time_ms"] for row in participant_response_rows
                ),
                "total_rt_ms": sum(
                    row["response_time_ms"] for row in participant_response_rows
                ),
            }
        )
    return response_rows, participant_rows


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def render_accuracy_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    by_name = {row["name"]: row for row in rows}
    selected = [
        by_name[name]
        for name in [
            "arm:orange_flowers",
            "arm:hanging_clothes",
            "arm:aerial_beach",
            "arm:city_street",
            "arm:storm_beach",
            "unrelated_filler",
        ]
    ]
    label_map = {
        "arm:orange_flowers": "Orange flowers",
        "arm:hanging_clothes": "Hanging clothes",
        "arm:aerial_beach": "Aerial beach",
        "arm:city_street": "City street",
        "arm:storm_beach": "Storm beach",
        "unrelated_filler": "Fillers",
    }
    color_map = {
        "arm:orange_flowers": POSITIVE_COLOR,
        "arm:hanging_clothes": POSITIVE_COLOR,
        "arm:aerial_beach": NEGATIVE_COLOR,
        "arm:city_street": NEGATIVE_COLOR,
        "arm:storm_beach": NEGATIVE_COLOR,
        "unrelated_filler": FILLER_COLOR,
    }

    width = 1180
    height = 620
    left = 190
    top = 100
    chart_w = 760
    chart_h = 360
    row_gap = chart_h / (len(selected) - 1)

    def x(value: float) -> float:
        return left + value * chart_w

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{left}" y="46" font-family="Inter, Arial, sans-serif" font-size="27" font-weight="800" fill="{INK}">Delayed recognition accuracy by content arm</text>',
        f'<text x="{left}" y="76" font-family="Inter, Arial, sans-serif" font-size="15" fill="{MUTED}">Wave 2 Prolific old-vs-lure task, media-error trials excluded; points show Wilson 95% CIs.</text>',
    ]

    for tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        tick_x = x(tick)
        lines.append(
            f'<line x1="{tick_x:.1f}" y1="{top - 10}" x2="{tick_x:.1f}" y2="{top + chart_h + 20}" stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{tick_x:.1f}" y="{top + chart_h + 48}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="{MUTED}">{int(tick * 100)}%</text>'
        )

    lines.append(
        f'<line x1="{x(0.5):.1f}" y1="{top - 16}" x2="{x(0.5):.1f}" y2="{top + chart_h + 24}" stroke="#a0aec0" stroke-dasharray="6 6" stroke-width="2"/>'
    )
    lines.append(
        f'<text x="{x(0.5) + 8:.1f}" y="{top - 6}" font-family="Inter, Arial, sans-serif" font-size="12" fill="{MUTED}">chance</text>'
    )

    for index, row in enumerate(selected):
        y = top + index * row_gap
        color = color_map[row["name"]]
        name = label_map[row["name"]]
        lines.extend(
            [
                f'<text x="{left - 16}" y="{y + 5:.1f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="700" fill="{INK}">{svg_escape(name)}</text>',
                f'<line x1="{x(row["wilson95_low"]):.1f}" y1="{y:.1f}" x2="{x(row["wilson95_high"]):.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="6" stroke-linecap="round" opacity="0.35"/>',
                f'<circle cx="{x(row["accuracy"]):.1f}" cy="{y:.1f}" r="11" fill="{color}"/>',
                f'<text x="{x(row["accuracy"]) + 20:.1f}" y="{y + 5:.1f}" font-family="Inter, Arial, sans-serif" font-size="15" font-weight="700" fill="{INK}">{pct(row["accuracy"])} ({row["correct"]}/{row["n"]})</text>',
            ]
        )

    legend_y = 530
    legend = [
        (POSITIVE_COLOR, "Primary positive pockets"),
        (NEGATIVE_COLOR, "Hard negative controls"),
        (FILLER_COLOR, "Unrelated fillers"),
    ]
    legend_x = left
    for color, label in legend:
        lines.append(
            f'<rect x="{legend_x}" y="{legend_y}" width="18" height="18" rx="4" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{legend_x + 26}" y="{legend_y + 14}" font-family="Inter, Arial, sans-serif" font-size="14" fill="{MUTED}">{label}</text>'
        )
        legend_x += 280

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_contrast_svg(path: Path, summary: dict[str, Any]) -> None:
    contrast = summary["paired_primary_vs_hard_negative_no_media_errors"]
    primary = next(
        row for row in summary["no_media_error_summaries"] if row["name"] == "primary_positive"
    )
    hard_negative = next(
        row
        for row in summary["no_media_error_summaries"]
        if row["name"] == "hard_negative_control"
    )

    width = 1120
    height = 560
    left = 120
    base_y = 360
    bar_w = 160
    gap = 92
    scale_h = 260

    def bar_h(value: float) -> float:
        return value * scale_h

    primary_h = bar_h(primary["accuracy"])
    hard_h = bar_h(hard_negative["accuracy"])
    diff = contrast["mean_difference"]
    diff_low = contrast["bootstrap95_low"]
    diff_high = contrast["bootstrap95_high"]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{left}" y="48" font-family="Inter, Arial, sans-serif" font-size="28" font-weight="800" fill="{INK}">Primary pockets exceed hard controls</text>',
        f'<text x="{left}" y="78" font-family="Inter, Arial, sans-serif" font-size="15" fill="{MUTED}">Paired participant-level contrast after excluding media-error trials.</text>',
    ]
    for tick in [0.5, 0.7, 0.9]:
        y = base_y - bar_h(tick)
        lines.append(
            f'<line x1="{left - 10}" y1="{y:.1f}" x2="{left + 2 * bar_w + gap + 20}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 22}" y="{y + 5:.1f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="13" fill="{MUTED}">{int(tick * 100)}%</text>'
        )

    primary_x = left + 40
    hard_x = primary_x + bar_w + gap
    lines.extend(
        [
            f'<rect x="{primary_x}" y="{base_y - primary_h:.1f}" width="{bar_w}" height="{primary_h:.1f}" rx="10" fill="{POSITIVE_COLOR}"/>',
            f'<rect x="{hard_x}" y="{base_y - hard_h:.1f}" width="{bar_w}" height="{hard_h:.1f}" rx="10" fill="{NEGATIVE_COLOR}"/>',
            f'<text x="{primary_x + bar_w / 2:.1f}" y="{base_y + 34}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="15" font-weight="700" fill="{INK}">Primary positives</text>',
            f'<text x="{hard_x + bar_w / 2:.1f}" y="{base_y + 34}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="15" font-weight="700" fill="{INK}">Hard controls</text>',
            f'<text x="{primary_x + bar_w / 2:.1f}" y="{base_y - primary_h - 14:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="18" font-weight="800" fill="{INK}">{pct(primary["accuracy"])}</text>',
            f'<text x="{hard_x + bar_w / 2:.1f}" y="{base_y - hard_h - 14:.1f}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="18" font-weight="800" fill="{INK}">{pct(hard_negative["accuracy"])}</text>',
        ]
    )

    callout_x = 650
    callout_y = 146
    lines.extend(
        [
            f'<rect x="{callout_x}" y="{callout_y}" width="360" height="190" rx="14" fill="#edf7f0" stroke="#9ae6b4"/>',
            f'<text x="{callout_x + 24}" y="{callout_y + 44}" font-family="Inter, Arial, sans-serif" font-size="18" font-weight="800" fill="{INK}">Paired lift</text>',
            f'<text x="{callout_x + 24}" y="{callout_y + 86}" font-family="Inter, Arial, sans-serif" font-size="34" font-weight="900" fill="{POSITIVE_COLOR}">+{diff * 100:.1f} pp</text>',
            f'<text x="{callout_x + 24}" y="{callout_y + 120}" font-family="Inter, Arial, sans-serif" font-size="14" fill="{MUTED}">Bootstrap 95% CI: +{diff_low * 100:.1f} to +{diff_high * 100:.1f} pp</text>',
            f'<text x="{callout_x + 24}" y="{callout_y + 148}" font-family="Inter, Arial, sans-serif" font-size="14" fill="{MUTED}">Sign-flip p = {format_p(contrast["sign_flip_permutation_p"])}</text>',
            f'<text x="{callout_x + 24}" y="{callout_y + 176}" font-family="Inter, Arial, sans-serif" font-size="14" fill="{MUTED}">Paired participants: {contrast["participants"]}</text>',
        ]
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_arm_contrast_svg(path: Path, summary: dict[str, Any]) -> None:
    contrasts = summary["arm_contrasts_no_media_errors"]
    rows = [
        ("orange_flowers", "Orange flowers", contrasts["orange_flowers"]),
        ("hanging_clothes", "Hanging clothes", contrasts["hanging_clothes"]),
    ]

    width = 980
    height = 420
    left = 210
    top = 145
    chart_w = 610
    row_gap = 95
    x_min = -0.05
    x_max = 0.25

    def x(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * chart_w

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        (
            f'<text x="{left}" y="48" font-family="Inter, Arial, sans-serif" '
            f'font-size="27" font-weight="800" fill="{INK}">'
            "Pocket-specific lift vs hard controls</text>"
        ),
        (
            f'<text x="{left}" y="78" font-family="Inter, Arial, sans-serif" '
            f'font-size="15" fill="{MUTED}">'
            "Paired participant-level differences with bootstrap 95% CIs.</text>"
        ),
    ]

    for tick in [-0.05, 0.0, 0.1, 0.2, 0.25]:
        tick_x = x(tick)
        lines.append(
            f'<line x1="{tick_x:.1f}" y1="{top - 42}" '
            f'x2="{tick_x:.1f}" y2="{top + row_gap + 42}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{tick_x:.1f}" y="{top + row_gap + 75}" '
            'text-anchor="middle" font-family="Inter, Arial, sans-serif" '
            f'font-size="13" fill="{MUTED}">{tick * 100:+.0f} pp</text>'
        )

    lines.append(
        f'<line x1="{x(0):.1f}" y1="{top - 48}" '
        f'x2="{x(0):.1f}" y2="{top + row_gap + 48}" '
        'stroke="#a0aec0" stroke-dasharray="6 6" stroke-width="2"/>'
    )

    for index, (_, label, row) in enumerate(rows):
        y = top + index * row_gap
        low = row["bootstrap95_low"]
        high = row["bootstrap95_high"]
        diff = row["mean_difference"]
        p_value = row["sign_flip_permutation_p"]
        lines.extend(
            [
                (
                    f'<text x="{left - 18}" y="{y + 5:.1f}" '
                    'text-anchor="end" font-family="Inter, Arial, sans-serif" '
                    f'font-size="16" font-weight="700" fill="{INK}">'
                    f"{svg_escape(label)}</text>"
                ),
                (
                    f'<line x1="{x(low):.1f}" y1="{y:.1f}" '
                    f'x2="{x(high):.1f}" y2="{y:.1f}" '
                    f'stroke="{POSITIVE_COLOR}" stroke-width="7" '
                    'stroke-linecap="round" opacity="0.32"/>'
                ),
                (
                    f'<circle cx="{x(diff):.1f}" cy="{y:.1f}" r="11" '
                    f'fill="{POSITIVE_COLOR}"/>'
                ),
                (
                    f'<text x="{x(high) + 18:.1f}" y="{y + 5:.1f}" '
                    'font-family="Inter, Arial, sans-serif" font-size="15" '
                    f'font-weight="700" fill="{INK}">'
                    f"+{diff * 100:.1f} pp, p = {format_p(p_value)}</text>"
                ),
            ]
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Content-Pocket Recognition-Memory Wave 2 Result",
        "",
        f"Date: {summary['generated_at']}",
        "",
        "## Status",
        "",
        "- Status: `human_recognition_wave2_analyzed`",
        "- Claim level: narrow delayed human old-vs-lure recognition evidence.",
        "- Not claimed: broad human memorability, measured-BMD validation, or prompt-conditioned generation control.",
        "",
        "## Figures",
        "",
        "![Delayed recognition accuracy by content arm](figures/content_pocket_recognition_accuracy_20260612.svg)",
        "",
        "![Primary pockets exceed hard controls](figures/content_pocket_recognition_contrast_20260612.svg)",
        "",
        "![Pocket-specific lift vs hard controls](figures/content_pocket_recognition_arm_contrasts_20260612.svg)",
        "",
        "## Data Integrity",
        "",
        f"- Prolific Wave 2 rows: {summary['prolific_rows']}.",
        f"- Prolific statuses: `{summary['prolific_status_counts']}`.",
        f"- Complete matched Wave 2 webhook payloads: {summary['complete_matched_participants']}.",
        (
            "- Timed-out submissions without complete webhook payloads: "
            f"{summary['timed_out_missing_webhook_count']}."
        ),
        f"- Visible Wave 1 payload subset: {summary['session1_visible_payloads']}; overlap with complete Wave 2: {summary['session1_session2_overlap']}; form mismatches in overlap: {summary['session1_session2_form_mismatches']}.",
        "",
        "The Wave 1 webhook inbox was capped before the paid upgrade, so the full Wave 1 JSON payload set is not available. The recognition result therefore relies on Prolific-confirmed Wave 1 completion plus deterministic form assignment, with the visible Wave 1 subset used as a provenance check.",
        "",
        "## Primary Recognition Results",
        "",
        "| Group | Correct / n | Accuracy | Wilson 95% CI | p vs 0.5 | Media errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["no_media_error_summaries"]:
        lines.append(
            "| {name} | {correct}/{n} | {accuracy:.3f} | [{wilson95_low:.3f}, {wilson95_high:.3f}] | {p:.3g} | {media_errors} |".format(
                name=row["name"],
                correct=row["correct"],
                n=row["n"],
                accuracy=row["accuracy"],
                wilson95_low=row["wilson95_low"],
                wilson95_high=row["wilson95_high"],
                p=row["exact_binom_p_vs_0_5"],
                media_errors=row["media_errors"],
            )
        )
    contrast = summary["paired_primary_vs_hard_negative_no_media_errors"]
    all_contrast = summary["paired_primary_vs_hard_negative_all_trials"]
    lines.extend(
        [
            "",
            "## Paired Positive-Vs-Hard-Negative Contrast",
            "",
            "- Primary analysis excludes trials with media-error flags.",
            f"- Complete paired participants: {contrast['participants']}.",
            f"- Mean primary-positive minus hard-negative accuracy difference: {contrast['mean_difference']:.3f}.",
            f"- Bootstrap 95% CI: [{contrast['bootstrap95_low']:.3f}, {contrast['bootstrap95_high']:.3f}].",
            f"- Sign-flip permutation p-value: {contrast['sign_flip_permutation_p']:.4g}.",
            "",
            "All-trial sensitivity:",
            "",
            f"- Complete paired participants: {all_contrast['participants']}.",
            f"- Mean difference: {all_contrast['mean_difference']:.3f}.",
            f"- Bootstrap 95% CI: [{all_contrast['bootstrap95_low']:.3f}, {all_contrast['bootstrap95_high']:.3f}].",
            f"- Sign-flip permutation p-value: {all_contrast['sign_flip_permutation_p']:.4g}.",
            "",
            "## Individual Pocket Contrasts",
            "",
            "| Pocket | Mean lift vs hard controls | Bootstrap 95% CI | Sign-flip p | Participants |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm, row in summary["arm_contrasts_no_media_errors"].items():
        label = arm.replace("_", " ")
        lines.append(
            "| {label} | {diff:+.1f} pp | [{low:+.1f}, {high:+.1f}] pp | {p:.4g} | {n} |".format(
                label=label,
                diff=row["mean_difference"] * 100,
                low=row["bootstrap95_low"] * 100,
                high=row["bootstrap95_high"] * 100,
                p=row["sign_flip_permutation_p"],
                n=row["participants"],
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This wave supports a narrow human recognition-memory claim for the primary content-pocket pair. Hanging clothes is individually robust; orange flowers is high in absolute recognition but weaker as a standalone contrast against the hard-negative pool. Because the original full analysis plan named a larger minimum usable sample, this result should be written as a strong Wave 2 human-validation draft result rather than as the final large-sample confirmation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    prolific_rows = load_csv(args.prolific_csv)
    webhook_rows = load_csv(args.webhook_csv)
    payloads = load_payloads(webhook_rows)
    session1_payloads = [payload for payload in payloads if is_real_session1(payload)]
    session2_payloads = [
        payload for payload in payloads if is_real_complete_session2(payload)
    ]
    prolific_by_participant = {row["Participant id"]: row for row in prolific_rows}

    response_rows, participant_rows = build_response_rows(
        session2_payloads,
        prolific_by_participant,
        args.hash_salt,
    )
    response_rows_no_media = [row for row in response_rows if not row["any_media_error"]]

    all_summaries = [
        summarize_rows("all_trials", response_rows, lambda row: True),
        summarize_rows(
            "all_nonfiller_pockets",
            response_rows,
            lambda row: row["analysis_group"] != FILLER_GROUP,
        ),
        summarize_rows(
            "primary_positive",
            response_rows,
            lambda row: row["analysis_group"] == PRIMARY_GROUP,
        ),
        summarize_rows(
            "hard_negative_control",
            response_rows,
            lambda row: row["analysis_group"] == HARD_NEGATIVE_GROUP,
        ),
        summarize_rows(
            "unrelated_filler",
            response_rows,
            lambda row: row["analysis_group"] == FILLER_GROUP,
        ),
    ]
    no_media_summaries = [
        summarize_rows(
            "primary_positive",
            response_rows_no_media,
            lambda row: row["analysis_group"] == PRIMARY_GROUP,
        ),
        summarize_rows(
            "hard_negative_control",
            response_rows_no_media,
            lambda row: row["analysis_group"] == HARD_NEGATIVE_GROUP,
        ),
        summarize_rows(
            "unrelated_filler",
            response_rows_no_media,
            lambda row: row["analysis_group"] == FILLER_GROUP,
        ),
    ]
    for arm in ARM_ORDER:
        all_summaries.append(
            summarize_rows(
                f"arm:{arm}",
                response_rows,
                lambda row, arm=arm: row["arm_id"] == arm,
            )
        )
        no_media_summaries.append(
            summarize_rows(
                f"arm:{arm}",
                response_rows_no_media,
                lambda row, arm=arm: row["arm_id"] == arm,
            )
        )
    session1_by_participant = {payload["prolific_pid"]: payload for payload in session1_payloads}
    session2_by_participant = {payload["prolific_pid"]: payload for payload in session2_payloads}
    overlap = sorted(set(session1_by_participant) & set(session2_by_participant))
    form_mismatches = [
        hash_participant(participant_id, args.hash_salt)
        for participant_id in overlap
        if session1_by_participant[participant_id].get("form_id")
        != session2_by_participant[participant_id].get("form_id")
    ]

    complete_participant_ids = set(session2_by_participant)
    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "webhook_rows": len(webhook_rows),
        "webhook_post_payloads": len(payloads),
        "prolific_rows": len(prolific_rows),
        "prolific_status_counts": dict(Counter(row["Status"] for row in prolific_rows)),
        "complete_matched_participants": len(session2_payloads),
        "matched_status_counts": dict(Counter(row["status"] for row in participant_rows)),
        "form_counts": dict(Counter(row["form_id"] for row in participant_rows)),
        "session1_visible_payloads": len(session1_payloads),
        "session1_session2_overlap": len(overlap),
        "session1_session2_form_mismatches": len(form_mismatches),
        "timed_out_missing_webhook_count": sum(
            1 for row in prolific_rows if row["Participant id"] not in complete_participant_ids
        ),
        "all_summaries": [asdict(row) for row in all_summaries],
        "no_media_error_summaries": [asdict(row) for row in no_media_summaries],
        "paired_primary_vs_hard_negative_all_trials": paired_primary_contrast(
            response_rows,
            bootstrap_iters=args.bootstrap_iters,
            permutation_iters=args.permutation_iters,
            rng_seed=20260611,
        ),
        "paired_primary_vs_hard_negative_no_media_errors": paired_primary_contrast(
            response_rows_no_media,
            bootstrap_iters=args.bootstrap_iters,
            permutation_iters=args.permutation_iters,
            rng_seed=20260612,
        ),
        "arm_contrasts_no_media_errors": {
            arm: arm_contrast(
                response_rows_no_media,
                arm,
                bootstrap_iters=args.bootstrap_iters,
                permutation_iters=args.permutation_iters,
                rng_seed=20260613 + index,
            )
            for index, arm in enumerate(PRIMARY_ARMS)
        },
    }

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    summary["figures"] = {
        "accuracy": str(args.figure_dir / "content_pocket_recognition_accuracy_20260612.svg"),
        "contrast": str(args.figure_dir / "content_pocket_recognition_contrast_20260612.svg"),
        "arm_contrasts": str(
            args.figure_dir / "content_pocket_recognition_arm_contrasts_20260612.svg"
        ),
    }
    render_accuracy_svg(
        args.figure_dir / "content_pocket_recognition_accuracy_20260612.svg",
        summary["no_media_error_summaries"],
    )
    render_contrast_svg(
        args.figure_dir / "content_pocket_recognition_contrast_20260612.svg",
        summary,
    )
    render_arm_contrast_svg(
        args.figure_dir / "content_pocket_recognition_arm_contrasts_20260612.svg",
        summary,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.out_md.write_text(markdown_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
