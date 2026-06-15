"""Build a counterbalanced form plan for the confirmatory recognition study."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(
    "research_program/neurips_memorability_selector/experiments/"
    "content_pocket_confirmatory_recognition_20260615"
)
DEFAULT_CONFIG = EXPERIMENT_DIR / "confirmatory_study_config_20260615.json"
DEFAULT_OUT_JSON = EXPERIMENT_DIR / "confirmatory_form_plan_20260615.json"
DEFAULT_OUT_MD = EXPERIMENT_DIR / "confirmatory_form_plan_20260615.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def condition_for(*, family_index: int, form_index: int) -> str:
    return (
        "selector_top"
        if (family_index + form_index) % 2 == 0
        else "quality_matched_control"
    )


def build_form_plan(config: dict[str, Any]) -> dict[str, Any]:
    families = config["content_families"]
    form_count = int(config["session_design"]["form_count"])
    analysis_per_participant = int(
        config["session_design"]["analysis_families_per_participant"]
    )
    fillers_per_participant = int(config["session_design"]["fillers_per_participant"])
    if len(families) != analysis_per_participant:
        raise ValueError(
            "content family count must match analysis_families_per_participant"
        )
    if form_count % 2 != 0:
        raise ValueError("form_count must be even for this counterbalancing rule")

    forms = []
    condition_counts_by_family: dict[str, Counter[str]] = {
        str(family["family_id"]): Counter() for family in families
    }
    condition_counts_by_form: dict[str, Counter[str]] = {}
    for form_index in range(form_count):
        form_id = f"confirmatory_form_{form_index:02d}"
        trials = []
        form_counter: Counter[str] = Counter()
        ordered_family_indices = [
            (form_index + offset) % len(families) for offset in range(len(families))
        ]
        for family_index in ordered_family_indices:
            family = families[family_index]
            family_id = str(family["family_id"])
            condition = condition_for(
                family_index=family_index,
                form_index=form_index,
            )
            old_item_id = f"{family_id}_{condition}_old"
            lure_item_id = f"{family_id}_{condition}_lure"
            trials.append(
                {
                    "family_id": family_id,
                    "condition": condition,
                    "old_item_id": old_item_id,
                    "lure_item_id": lure_item_id,
                    "session1_slot": len(trials) + 1,
                    "session2_trial_id": f"{form_id}_{family_id}_recognition",
                }
            )
            form_counter[condition] += 1
            condition_counts_by_family[family_id][condition] += 1
        condition_counts_by_form[form_id] = form_counter
        forms.append(
            {
                "form_id": form_id,
                "analysis_trials": trials,
                "filler_old_count": fillers_per_participant,
                "filler_recognition_count": fillers_per_participant,
            }
        )

    target_delayed = int(config["participant_plan"]["target_usable_delayed_n"])
    retention = float(config["participant_plan"]["expected_session2_retention"])
    recommended_session1 = math.ceil(target_delayed / retention)

    return {
        "created_at_utc": config["manifest_created_at_utc"],
        "experiment_id": config["experiment_id"],
        "source_config": str(DEFAULT_CONFIG),
        "form_count": form_count,
        "forms": forms,
        "balance_checks": {
            "condition_counts_by_form": {
                form_id: dict(counter)
                for form_id, counter in condition_counts_by_form.items()
            },
            "condition_counts_by_family": {
                family_id: dict(counter)
                for family_id, counter in condition_counts_by_family.items()
            },
            "recommended_session1_recruit_for_target": recommended_session1,
        },
    }


def validate_plan(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for form in plan["forms"]:
        counts = Counter(trial["condition"] for trial in form["analysis_trials"])
        if counts["selector_top"] != counts["quality_matched_control"]:
            errors.append(f"{form['form_id']} is not condition-balanced: {dict(counts)}")
        families = [trial["family_id"] for trial in form["analysis_trials"]]
        if len(families) != len(set(families)):
            errors.append(f"{form['form_id']} repeats a family")

    for family_id, counts in plan["balance_checks"]["condition_counts_by_family"].items():
        if counts.get("selector_top") != counts.get("quality_matched_control"):
            errors.append(f"{family_id} is not balanced across forms: {counts}")
    return errors


def render_markdown(plan: dict[str, Any], errors: list[str]) -> str:
    lines = [
        "# Confirmatory Recognition Form Plan",
        "",
        f"Created: `{plan['created_at_utc']}`",
        f"Experiment: `{plan['experiment_id']}`",
        "",
        "## Balance Summary",
        "",
        f"- Forms: `{plan['form_count']}`",
        "- Per form: 12 analysis trials, 6 selector-top, 6 matched-control.",
        "- Per family across forms: 4 selector-top assignments and 4 matched-control assignments.",
        "- Family order rotates by form to reduce deterministic order effects.",
        (
            "- Recommended Session 1 recruit for target delayed sample: "
            f"`{plan['balance_checks']['recommended_session1_recruit_for_target']}`"
        ),
        "",
        "## Validation",
        "",
    ]
    if errors:
        lines.append("Validation errors:")
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("Validation passed.")

    lines.extend(["", "## Forms", ""])
    for form in plan["forms"]:
        counts = Counter(trial["condition"] for trial in form["analysis_trials"])
        lines.extend(
            [
                f"### {form['form_id']}",
                "",
                f"- Conditions: `{dict(counts)}`",
                f"- Fillers: `{form['filler_old_count']}` old and recognition pairs",
                "",
            ]
        )
        for trial in form["analysis_trials"]:
            lines.append(
                "- "
                f"`{trial['session2_trial_id']}`: "
                f"`{trial['family_id']}` / `{trial['condition']}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    plan = build_form_plan(config)
    errors = validate_plan(plan)
    if errors:
        plan["validation_status"] = "failed"
        plan["validation_errors"] = errors
    else:
        plan["validation_status"] = "passed"
        plan["validation_errors"] = []
    write_json(args.out_json, plan)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(plan, errors), encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
