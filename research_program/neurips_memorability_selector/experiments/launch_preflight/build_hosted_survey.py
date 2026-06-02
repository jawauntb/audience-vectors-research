#!/usr/bin/env python3
"""Prepare and validate the V-JEPA Prolific survey for hosting.

This does not launch a study. It creates review/preflight artifacts:

- a hosted-video URL map template,
- a frozen task/randomization metadata record,
- and, once real hosted URLs are provided, a hostable survey HTML file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS_DIR = DEFAULT_EXPERIMENTS_DIR / "prolific_launch_assets_2026-06-01"
DEFAULT_TASKS = DEFAULT_EXPERIMENTS_DIR / "current_selector_pairwise_tasks_with_vjepa.json"
DEFAULT_MANIFEST = DEFAULT_EXPERIMENTS_DIR / "current_selector_manifest_with_vjepa.json"
DEFAULT_SURVEY = DEFAULT_EXPERIMENTS_DIR / "current_selector_prolific_survey_with_vjepa.html"
DEFAULT_URL_MAP = DEFAULT_ASSETS_DIR / "hosted_video_url_map.template.json"
DEFAULT_FREEZE = DEFAULT_ASSETS_DIR / "task_randomization_freeze.json"
DEFAULT_HOSTED_SURVEY = DEFAULT_ASSETS_DIR / "current_selector_prolific_survey_with_vjepa.hosted.html"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_tasks(task_doc: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = task_doc.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("Task file must contain a top-level tasks list")
    return tasks


def task_video_paths(tasks: list[dict[str, Any]]) -> list[str]:
    paths: set[str] = set()
    for task in tasks:
        for side in ("left", "right"):
            value = task.get(side, {}).get("path")
            if not isinstance(value, str) or not value:
                raise ValueError(f"Task {task.get('task_id')} is missing {side}.path")
            paths.add(value)
    return sorted(paths)


def validate_tasks(task_doc: dict[str, Any], manifest_doc: dict[str, Any] | None = None) -> dict[str, Any]:
    tasks = extract_tasks(task_doc)
    declared_count = task_doc.get("n_tasks")
    if declared_count is not None and declared_count != len(tasks):
        raise ValueError(f"n_tasks={declared_count} but found {len(tasks)} tasks")

    task_ids = [task.get("task_id") for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        duplicates = [task_id for task_id, count in Counter(task_ids).items() if count > 1]
        raise ValueError(f"Duplicate task_id values: {duplicates}")

    seeds: set[str] = set()
    for task in tasks:
        seed = task.get("seed")
        if not isinstance(seed, str) or not seed:
            raise ValueError(f"Task {task.get('task_id')} is missing seed")
        seeds.add(seed)
    sorted_seeds = sorted(seeds)
    comparisons = Counter(task.get("comparison") for task in tasks)
    video_paths = task_video_paths(tasks)

    if manifest_doc is not None:
        manifest_missing = manifest_doc.get("missing_files")
        if manifest_missing not in (None, []):
            raise ValueError(f"Manifest reports missing files: {manifest_missing}")

    return {
        "n_tasks": len(tasks),
        "n_seeds": len(sorted_seeds),
        "n_unique_video_paths": len(video_paths),
        "comparisons": dict(sorted(comparisons.items())),
        "seeds": sorted_seeds,
        "video_paths": video_paths,
    }


def parse_survey_metadata(survey_html: str) -> dict[str, Any]:
    trials_match = re.search(r"const\s+TRIALS_PER_PARTICIPANT\s*=\s*(\d+);", survey_html)
    asset_base_match = re.search(r"const\s+ASSET_BASE\s*=\s*([\"'])(.*?)\1;", survey_html)
    if not trials_match:
        raise ValueError("Could not find TRIALS_PER_PARTICIPANT in survey HTML")
    if not asset_base_match:
        raise ValueError("Could not find ASSET_BASE in survey HTML")
    return {
        "trials_per_participant": int(trials_match.group(1)),
        "asset_base": asset_base_match.group(2),
        "randomization": {
            "participant_seed_source": "PROLIFIC_PID query parameter or entered participant id; falls back to anonymous/Date.now in the existing HTML",
            "hash_function": "32-bit FNV-1a style hashString in survey HTML",
            "prng": "mulberry32",
            "balancing": "shuffle comparison families, take floor(trials_per_participant / n_comparisons) per family, then fill remaining trials from unused tasks",
            "final_order": "selected trials are shuffled once more before display",
        },
    }


def build_url_template(video_paths: list[str], output: Path) -> None:
    entries = [
        {
            "local_path": path,
            "hosted_url": "",
            "screened": False,
            "screening_notes": "",
        }
        for path in video_paths
    ]
    write_json(
        output,
        {
            "schema_version": "hosted_video_url_map.v1",
            "prepared_for": "current_selector_prolific_survey_with_vjepa.html",
            "instructions": [
                "Fill hosted_url with stable HTTPS URLs for every local_path.",
                "Set screened to true only after sensitive-content screening is complete.",
                "Run build_hosted_survey.py --url-map <this file> --hosted-survey <output.html>.",
            ],
            "videos": entries,
        },
    )


def load_url_map(path: Path, required_paths: list[str]) -> dict[str, str]:
    doc = load_json(path)
    videos = doc.get("videos")
    if not isinstance(videos, list):
        raise ValueError("URL map must contain a videos list")

    mapping: dict[str, str] = {}
    unscreened: list[str] = []
    for item in videos:
        local_path = item.get("local_path")
        hosted_url = item.get("hosted_url")
        if not isinstance(local_path, str):
            raise ValueError("Every URL map item needs local_path")
        if not isinstance(hosted_url, str):
            raise ValueError(f"{local_path} has non-string hosted_url")
        mapping[local_path] = hosted_url
        if item.get("screened") is not True:
            unscreened.append(local_path)

    missing = sorted(set(required_paths) - set(mapping))
    if missing:
        raise ValueError(f"URL map is missing {len(missing)} required paths")
    blanks = [path for path in required_paths if not mapping[path]]
    if blanks:
        raise ValueError(f"URL map still has {len(blanks)} blank hosted_url values")
    non_https = [path for path in required_paths if not mapping[path].startswith("https://")]
    if non_https:
        raise ValueError(f"URL map has {len(non_https)} non-HTTPS hosted_url values")
    if unscreened:
        raise ValueError(f"URL map has {len(unscreened)} unscreened videos")
    return mapping


def render_hosted_survey(survey_html: str, mapping: dict[str, str], output: Path) -> None:
    rendered = survey_html
    for local_path, hosted_url in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        rendered = rendered.replace(local_path, hosted_url)
    rendered = re.sub(r"const\s+ASSET_BASE\s*=\s*([\"']).*?\1;", 'const ASSET_BASE = "";', rendered)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def build_freeze(
    task_path: Path,
    manifest_path: Path,
    survey_path: Path,
    task_doc: dict[str, Any],
    validation: dict[str, Any],
    survey_metadata: dict[str, Any],
    output: Path,
) -> None:
    tasks = extract_tasks(task_doc)
    freeze = {
        "schema_version": "selector_preflight_freeze.v1",
        "prepared_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "prelaunch_not_launched",
        "source_files": {
            str(task_path.relative_to(DEFAULT_EXPERIMENTS_DIR)): sha256_bytes(task_path),
            str(manifest_path.relative_to(DEFAULT_EXPERIMENTS_DIR)): sha256_bytes(manifest_path),
            str(survey_path.relative_to(DEFAULT_EXPERIMENTS_DIR)): sha256_bytes(survey_path),
        },
        "task_pool": {
            "n_tasks": validation["n_tasks"],
            "n_seeds": validation["n_seeds"],
            "n_unique_video_paths": validation["n_unique_video_paths"],
            "comparisons": validation["comparisons"],
            "task_order_sha256": sha256_json([task["task_id"] for task in tasks]),
            "task_payload_sha256": sha256_json(tasks),
            "video_path_set_sha256": sha256_json(validation["video_paths"]),
        },
        "survey_randomization": survey_metadata,
        "launch_blockers": [
            "Faculty/PI and IRB determination not recorded in this preflight artifact.",
            "Hosted video URLs must be filled into hosted_video_url_map.template.json.",
            "Stimulus screening must be marked complete for every hosted video.",
            "Completion code, compensation, and response export endpoint remain to be finalized.",
        ],
    }
    write_json(output, freeze)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--survey", type=Path, default=DEFAULT_SURVEY)
    parser.add_argument("--url-template", type=Path, default=DEFAULT_URL_MAP)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--url-map", type=Path, help="Filled hosted URL map to validate and render")
    parser.add_argument("--hosted-survey", type=Path, default=DEFAULT_HOSTED_SURVEY)
    args = parser.parse_args()

    task_doc = load_json(args.tasks)
    manifest_doc = load_json(args.manifest)
    survey_html = args.survey.read_text(encoding="utf-8")
    validation = validate_tasks(task_doc, manifest_doc)
    survey_metadata = parse_survey_metadata(survey_html)

    build_url_template(validation["video_paths"], args.url_template)
    build_freeze(args.tasks, args.manifest, args.survey, task_doc, validation, survey_metadata, args.freeze)

    print(f"Validated {validation['n_tasks']} tasks across {validation['n_seeds']} seeds")
    print(f"Wrote URL template: {args.url_template}")
    print(f"Wrote freeze metadata: {args.freeze}")

    if args.url_map:
        mapping = load_url_map(args.url_map, validation["video_paths"])
        render_hosted_survey(survey_html, mapping, args.hosted_survey)
        print(f"Wrote hosted survey: {args.hosted_survey}")
    else:
        print("No hosted survey rendered because --url-map was not provided")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
