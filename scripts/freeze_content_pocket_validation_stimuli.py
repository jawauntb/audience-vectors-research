"""Freeze the content-pocket human/BMD validation stimulus set.

The freeze takes the accepted primary descriptor-conditioned replay report and
the blue-jellyfish/old-car boundary replay report, selects complete retained
task-level candidates, and writes:

- a detailed stimulus manifest with exact MP4 hashes,
- a compact pairwise task pool compatible with build_selector_prolific_survey,
- and a Markdown launch note for human/BMD validation review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path(
    "research_program/neurips_memorability_selector/collaborator_inputs/"
    "camilo_bo_memorability"
)
DEFAULT_PRIMARY_REPORT = Path(
    "data/reports/"
    "bo_descriptor_conditioned_replication_sobol518_523_x5_noise250k_reps3_"
    "steps50_motion5_noise0_20260608.json"
)
DEFAULT_BOUNDARY_REPORT = Path(
    "data/reports/"
    "bo_boundary_pocket_audit_sobol518_523_x5_noise350k_reps3_steps50_motion5_"
    "noise0_20260608.json"
)
DEFAULT_TASKS = ARTIFACT_DIR / "content_pocket_validation_pairwise_tasks_20260608.json"
DEFAULT_MANIFEST = (
    ARTIFACT_DIR / "content_pocket_validation_stimuli_manifest_20260608.json"
)
DEFAULT_SUMMARY = (
    ARTIFACT_DIR / "content_pocket_validation_stimuli_manifest_20260608.md"
)

PRIMARY_POCKETS = ("fresh24_orange_flowers", "fresh24_hanging_clothes")
BOUNDARY_POCKETS = ("fresh24_blue_jellyfish", "fresh24_old_car")
HARD_NEGATIVE_CONTROLS = (
    "fresh24_aerial_beach",
    "fresh24_city_street",
    "fresh24_storm_beach",
)
CONTROL_BY_REPLICATE = {
    0: "fresh24_aerial_beach",
    1: "fresh24_city_street",
    2: "fresh24_storm_beach",
}
RECIPE_RE = re.compile(r"sobol_prompt_search_(?P<recipe>\d+)_slot(?P<slot>\d+)")


@dataclass(frozen=True)
class ReplayRow:
    """One scored generated-video replicate from a replay report."""

    report_kind: str
    report_path: Path
    task_id: str
    recipe_index: int
    slot_index: int
    replicate_index: int
    pocket: str
    label: str
    local_video_path: str
    absolute_video_path: Path
    replay_tribe_score: float
    noise_seed: int
    visual_first_retained: bool
    passes_visual_gate: bool
    generation_error: str | None


@dataclass(frozen=True)
class Candidate:
    """A task-level candidate made of the complete stochastic replicate set."""

    tier: str
    pocket: str
    task_id: str
    recipe_index: int
    slot_index: int
    rows: tuple[ReplayRow, ...]

    @property
    def mean_tribe_score(self) -> float:
        return sum(row.replay_tribe_score for row in self.rows) / len(self.rows)

    @property
    def min_tribe_score(self) -> float:
        return min(row.replay_tribe_score for row in self.rows)

    @property
    def max_tribe_score(self) -> float:
        return max(row.replay_tribe_score for row in self.rows)

    @property
    def rows_by_replicate(self) -> dict[int, ReplayRow]:
        return {row.replicate_index: row for row in self.rows}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def report_worktree_root(report_path: Path) -> Path:
    """Infer the repo root for a data/reports replay report."""
    if report_path.parent.name == "reports" and report_path.parent.parent.name == "data":
        return report_path.parent.parent.parent
    return Path.cwd()


def parse_recipe(task_id: str) -> tuple[int, int]:
    match = RECIPE_RE.fullmatch(task_id)
    if match is None:
        raise ValueError(f"Cannot parse Sobol recipe and slot from task_id={task_id!r}")
    return int(match.group("recipe")), int(match.group("slot"))


def replay_rows(report_path: Path, report_kind: str) -> list[ReplayRow]:
    report_path = report_path.resolve()
    report = load_json(report_path)
    root = report_worktree_root(report_path)
    rows: list[ReplayRow] = []
    for raw in report.get("rows", []):
        task_id = str(raw["trial"]["task_id"])
        recipe_index, slot_index = parse_recipe(task_id)
        local_video_path = str(raw["local_video_path"])
        score = raw.get("replay_tribe_score")
        if score is None:
            continue
        visual_gate = raw.get("visual_artifact_gate") or {}
        rows.append(
            ReplayRow(
                report_kind=report_kind,
                report_path=report_path,
                task_id=task_id,
                recipe_index=recipe_index,
                slot_index=slot_index,
                replicate_index=int(raw["replicate_index"]),
                pocket=str(raw["seed"]["bmd_name"]),
                label=str(raw["label"]),
                local_video_path=local_video_path,
                absolute_video_path=root / local_video_path,
                replay_tribe_score=float(score),
                noise_seed=int(raw["noise_seed"]),
                visual_first_retained=bool(raw.get("visual_first_retained")),
                passes_visual_gate=bool(visual_gate.get("passes_visual_gate")),
                generation_error=raw.get("generation_error"),
            )
        )
    return rows


def candidate_is_complete(rows: list[ReplayRow], expected_replicates: int) -> bool:
    if len(rows) != expected_replicates:
        return False
    if sorted(row.replicate_index for row in rows) != list(range(expected_replicates)):
        return False
    return all(
        row.visual_first_retained
        and row.passes_visual_gate
        and row.generation_error is None
        for row in rows
    )


def complete_candidates(
    rows: list[ReplayRow],
    *,
    pockets: tuple[str, ...],
    tier: str,
    expected_replicates: int,
) -> list[Candidate]:
    by_task: dict[tuple[str, str], list[ReplayRow]] = defaultdict(list)
    for row in rows:
        if row.pocket in pockets:
            by_task[(row.pocket, row.task_id)].append(row)

    candidates: list[Candidate] = []
    for (pocket, task_id), group in by_task.items():
        ordered = sorted(group, key=lambda row: row.replicate_index)
        if not candidate_is_complete(ordered, expected_replicates):
            continue
        first = ordered[0]
        candidates.append(
            Candidate(
                tier=tier,
                pocket=pocket,
                task_id=task_id,
                recipe_index=first.recipe_index,
                slot_index=first.slot_index,
                rows=tuple(ordered),
            )
        )
    return candidates


def select_top_candidates(
    rows: list[ReplayRow],
    *,
    pockets: tuple[str, ...],
    tier: str,
    per_pocket: int,
    expected_replicates: int,
) -> list[Candidate]:
    candidates = complete_candidates(
        rows,
        pockets=pockets,
        tier=tier,
        expected_replicates=expected_replicates,
    )
    selected: list[Candidate] = []
    for pocket in pockets:
        pocket_candidates = [candidate for candidate in candidates if candidate.pocket == pocket]
        pocket_candidates.sort(
            key=lambda candidate: (
                candidate.mean_tribe_score,
                candidate.min_tribe_score,
                -candidate.recipe_index,
            ),
            reverse=True,
        )
        selected.extend(pocket_candidates[:per_pocket])
    return selected


def control_index(rows: list[ReplayRow]) -> dict[tuple[str, int, int], ReplayRow]:
    controls: dict[tuple[str, int, int], ReplayRow] = {}
    for row in rows:
        if row.pocket not in HARD_NEGATIVE_CONTROLS:
            continue
        key = (row.pocket, row.recipe_index, row.replicate_index)
        controls[key] = row
    return controls


def compact_pocket_name(pocket: str) -> str:
    return pocket.removeprefix("fresh24_")


def side_payload(row: ReplayRow, policy: str) -> dict[str, Any]:
    return {
        "policy": policy,
        "path": row.local_video_path,
        "label": row.pocket,
        "video_label": row.label,
    }


def task_payload(
    *,
    index: int,
    candidate: Candidate,
    positive_row: ReplayRow,
    control_row: ReplayRow,
    analysis_tier: str,
) -> dict[str, Any]:
    target = side_payload(positive_row, "content_pocket_candidate")
    baseline = side_payload(control_row, "hard_negative_control")
    target_side = "left" if index % 2 == 0 else "right"
    left = target if target_side == "left" else baseline
    right = baseline if target_side == "left" else target
    control_name = compact_pocket_name(control_row.pocket)
    pocket_name = compact_pocket_name(candidate.pocket)
    return {
        "task_id": (
            f"content_pocket_{analysis_tier}_{pocket_name}_{candidate.task_id}_"
            f"rep{positive_row.replicate_index:02d}_vs_{control_name}"
        ),
        "seed": (
            f"{candidate.pocket}:{candidate.task_id}:"
            f"rep{positive_row.replicate_index:02d}:{control_row.pocket}"
        ),
        "comparison": f"{analysis_tier}_content_pocket_vs_hard_negative",
        "left": left,
        "right": right,
        "target_policy": "content_pocket_candidate",
        "baseline_policy": "hard_negative_control",
        "question": "Which clip feels more memorable?",
        "metadata": {
            "analysis_tier": analysis_tier,
            "positive_pocket": candidate.pocket,
            "control_pocket": control_row.pocket,
            "sobol_recipe_index": candidate.recipe_index,
            "positive_slot_index": candidate.slot_index,
            "control_slot_index": control_row.slot_index,
            "replicate_index": positive_row.replicate_index,
            "positive_replay_tribe_score": positive_row.replay_tribe_score,
            "control_replay_tribe_score": control_row.replay_tribe_score,
            "candidate_mean_tribe_score": candidate.mean_tribe_score,
            "target_side": target_side,
        },
    }


def stimulus_record(row: ReplayRow, role: str, analysis_tier: str) -> dict[str, Any]:
    exists = row.absolute_video_path.exists()
    return {
        "role": role,
        "analysis_tier": analysis_tier,
        "pocket": row.pocket,
        "task_id": row.task_id,
        "recipe_index": row.recipe_index,
        "slot_index": row.slot_index,
        "replicate_index": row.replicate_index,
        "label": row.label,
        "local_video_path": row.local_video_path,
        "source_absolute_path": str(row.absolute_video_path),
        "exists": exists,
        "video_bytes": row.absolute_video_path.stat().st_size if exists else None,
        "sha256": sha256_bytes(row.absolute_video_path),
        "replay_tribe_score": row.replay_tribe_score,
        "noise_seed": row.noise_seed,
        "visual_first_retained": row.visual_first_retained,
        "passes_visual_gate": row.passes_visual_gate,
        "source_report": str(row.report_path),
    }


def candidate_record(candidate: Candidate) -> dict[str, Any]:
    return {
        "analysis_tier": candidate.tier,
        "pocket": candidate.pocket,
        "task_id": candidate.task_id,
        "recipe_index": candidate.recipe_index,
        "slot_index": candidate.slot_index,
        "n_replicates": len(candidate.rows),
        "mean_tribe_score": candidate.mean_tribe_score,
        "min_tribe_score": candidate.min_tribe_score,
        "max_tribe_score": candidate.max_tribe_score,
        "replicate_scores": [
            {
                "replicate_index": row.replicate_index,
                "label": row.label,
                "replay_tribe_score": row.replay_tribe_score,
            }
            for row in candidate.rows
        ],
    }


def build_freeze(
    *,
    primary_report: Path,
    boundary_report: Path,
    primary_candidates_per_pocket: int,
    boundary_candidates_per_pocket: int,
    expected_replicates: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    primary_rows = replay_rows(primary_report, "primary_descriptor_conditioned_replication")
    boundary_rows = replay_rows(boundary_report, "boundary_pocket_audit")
    selected_candidates = [
        *select_top_candidates(
            primary_rows,
            pockets=PRIMARY_POCKETS,
            tier="primary",
            per_pocket=primary_candidates_per_pocket,
            expected_replicates=expected_replicates,
        ),
        *select_top_candidates(
            boundary_rows,
            pockets=BOUNDARY_POCKETS,
            tier="exploratory_boundary",
            per_pocket=boundary_candidates_per_pocket,
            expected_replicates=expected_replicates,
        ),
    ]
    controls_by_report = {
        "primary": control_index(primary_rows),
        "exploratory_boundary": control_index(boundary_rows),
    }

    tasks: list[dict[str, Any]] = []
    stimulus_by_path: dict[str, dict[str, Any]] = {}
    missing_control_keys: list[dict[str, Any]] = []
    for candidate in selected_candidates:
        candidate_controls = controls_by_report[candidate.tier]
        for replicate_index in range(expected_replicates):
            positive_row = candidate.rows_by_replicate[replicate_index]
            control_pocket = CONTROL_BY_REPLICATE[replicate_index]
            control_key = (control_pocket, candidate.recipe_index, replicate_index)
            control_row = candidate_controls.get(control_key)
            if control_row is None:
                missing_control_keys.append(
                    {
                        "candidate": candidate.task_id,
                        "pocket": candidate.pocket,
                        "missing_control_key": list(control_key),
                    }
                )
                continue
            tasks.append(
                task_payload(
                    index=len(tasks),
                    candidate=candidate,
                    positive_row=positive_row,
                    control_row=control_row,
                    analysis_tier=candidate.tier,
                )
            )
            for row, role in ((positive_row, "candidate"), (control_row, "control")):
                stimulus_by_path.setdefault(
                    row.local_video_path,
                    stimulus_record(row, role, candidate.tier),
                )

    missing_files = [
        stimulus
        for stimulus in stimulus_by_path.values()
        if stimulus["exists"] is not True
    ]
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    task_doc = {
        "schema_version": "content_pocket_pairwise_tasks.v1",
        "created_at_utc": now,
        "study": "content_pocket_human_bmd_validation_20260608",
        "n_tasks": len(tasks),
        "tasks": tasks,
    }
    manifest = {
        "schema_version": "content_pocket_stimulus_freeze.v1",
        "created_at_utc": now,
        "status": "prelaunch_stimulus_freeze_not_launched",
        "source_reports": {
            "primary_descriptor_conditioned_replication": str(primary_report.resolve()),
            "boundary_pocket_audit": str(boundary_report.resolve()),
        },
        "selection_policy": {
            "primary_candidates_per_pocket": primary_candidates_per_pocket,
            "boundary_candidates_per_pocket": boundary_candidates_per_pocket,
            "expected_replicates_per_candidate": expected_replicates,
            "candidate_selection": (
                "top complete visual-retained task-level candidates by mean TRIBE "
                "score within each pocket"
            ),
            "control_matching": (
                "same Sobol recipe index and stochastic replicate index; "
                "rep00=aerial beach, rep01=city street, rep02=storm beach"
            ),
            "side_assignment": "deterministic alternating target side by frozen task order",
        },
        "analysis_tiers": {
            "primary": {
                "pockets": list(PRIMARY_POCKETS),
                "claim_status": (
                    "TRIBE/V-JEPA compute-proxy candidates; generated-video CLIP "
                    "failed prospectively; human/BMD validation not yet run"
                ),
            },
            "exploratory_boundary": {
                "pockets": list(BOUNDARY_POCKETS),
                "claim_status": (
                    "TRIBE positive and CLIP-side boundary positive; exact V-JEPA "
                    "did not pass the boundary audit"
                ),
            },
            "hard_negative_controls": list(HARD_NEGATIVE_CONTROLS),
        },
        "selected_candidates": [candidate_record(candidate) for candidate in selected_candidates],
        "stimuli": sorted(
            stimulus_by_path.values(),
            key=lambda item: (
                item["analysis_tier"],
                item["role"],
                item["pocket"],
                item["recipe_index"],
                item["replicate_index"],
            ),
        ),
        "tasks": tasks,
        "task_pool": {
            "n_tasks": len(tasks),
            "comparisons": dict(sorted(Counter(task["comparison"] for task in tasks).items())),
            "n_unique_video_paths": len(stimulus_by_path),
            "task_payload_sha256": sha256_json(tasks),
            "video_path_set_sha256": sha256_json(sorted(stimulus_by_path)),
        },
        "missing_files": missing_files,
        "missing_control_keys": missing_control_keys,
        "launch_blockers": [
            "Human/IRB approval and participant compensation are not recorded in this artifact.",
            "Every selected MP4 must be manually screened before launch.",
            "Hosted HTTPS URLs must replace local data-lake paths before Prolific launch.",
            "The packet remains proxy-selected until human or measured-BMD results clear.",
        ],
    }
    return manifest, task_doc, render_markdown_summary(manifest)


def format_score(value: float) -> str:
    return f"{value:.4f}"


def render_markdown_summary(manifest: dict[str, Any]) -> str:
    candidate_lines = [
        "| tier | pocket | task | recipe | reps | mean TRIBE | min | max |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in manifest["selected_candidates"]:
        candidate_lines.append(
            "| {tier} | `{pocket}` | `{task}` | {recipe} | {reps} | {mean} | {min_score} | {max_score} |".format(
                tier=candidate["analysis_tier"],
                pocket=candidate["pocket"],
                task=candidate["task_id"],
                recipe=candidate["recipe_index"],
                reps=candidate["n_replicates"],
                mean=format_score(candidate["mean_tribe_score"]),
                min_score=format_score(candidate["min_tribe_score"]),
                max_score=format_score(candidate["max_tribe_score"]),
            )
        )

    task_lines = [
        "| comparison | tasks |",
        "|---|---:|",
    ]
    for comparison, count in manifest["task_pool"]["comparisons"].items():
        task_lines.append(f"| `{comparison}` | {count} |")

    return "\n".join(
        [
            "# Content-Pocket Validation Stimulus Freeze",
            "",
            f"Date: {manifest['created_at_utc']}",
            "",
            "## Status",
            "",
            "Prelaunch stimulus freeze only. These MP4s are proxy-selected and have",
            "not yet cleared a human memorability or measured-BMD gate.",
            "",
            "Primary analysis remains V-JEPA-caveated: orange flowers and hanging",
            "clothes are TRIBE/V-JEPA compute-proxy candidates, while generated-video",
            "CLIP did not pass prospectively. Blue jellyfish and old car are",
            "exploratory boundary arms because exact V-JEPA did not pass their",
            "boundary audit.",
            "",
            "## Frozen Task Pool",
            "",
            *task_lines,
            "",
            f"Unique MP4 paths: {manifest['task_pool']['n_unique_video_paths']}",
            f"Task payload SHA-256: `{manifest['task_pool']['task_payload_sha256']}`",
            f"Video path set SHA-256: `{manifest['task_pool']['video_path_set_sha256']}`",
            "",
            "## Selected Candidates",
            "",
            *candidate_lines,
            "",
            "## Control Matching",
            "",
            "Each positive replicate is paired with the hard-negative control from the",
            "same Sobol recipe index and stochastic replicate index:",
            "",
            "- `rep00` -> `fresh24_aerial_beach`",
            "- `rep01` -> `fresh24_city_street`",
            "- `rep02` -> `fresh24_storm_beach`",
            "",
            "## Output Artifacts",
            "",
            "- `content_pocket_validation_stimuli_manifest_20260608.json`",
            "- `content_pocket_validation_pairwise_tasks_20260608.json`",
            "- `content_pocket_validation_prolific_survey_20260608.html`",
            "",
            "## Launch Blockers",
            "",
            *[f"- {blocker}" for blocker in manifest["launch_blockers"]],
            "",
            "## Next Action",
            "",
            "Manually screen the frozen MP4s, host the screened videos at stable HTTPS",
            "URLs, then build the blinded forced-choice survey from the frozen task",
            "JSON. Keep any BMD/measured-brain transfer report on the same frozen",
            "stimulus set so the human and BMD gates adjudicate identical clips.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-report", type=Path, default=DEFAULT_PRIMARY_REPORT)
    parser.add_argument("--boundary-report", type=Path, default=DEFAULT_BOUNDARY_REPORT)
    parser.add_argument("--out-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--primary-candidates-per-pocket", type=int, default=2)
    parser.add_argument("--boundary-candidates-per-pocket", type=int, default=2)
    parser.add_argument("--expected-replicates", type=int, default=3)
    args = parser.parse_args()

    manifest, task_doc, summary = build_freeze(
        primary_report=args.primary_report,
        boundary_report=args.boundary_report,
        primary_candidates_per_pocket=args.primary_candidates_per_pocket,
        boundary_candidates_per_pocket=args.boundary_candidates_per_pocket,
        expected_replicates=args.expected_replicates,
    )
    if manifest["missing_control_keys"]:
        raise ValueError(f"Missing matched controls: {manifest['missing_control_keys']}")
    if manifest["missing_files"]:
        raise ValueError(f"Missing selected MP4 files: {manifest['missing_files']}")

    write_json(args.out_manifest, manifest)
    write_json(args.out_tasks, task_doc)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(summary, encoding="utf-8")
    print(f"[done] wrote {args.out_manifest}")
    print(f"[done] wrote {args.out_tasks}")
    print(f"[done] wrote {args.out_summary}")
    print(f"[done] frozen tasks: {task_doc['n_tasks']}")
    print(f"[done] unique MP4 paths: {manifest['task_pool']['n_unique_video_paths']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
