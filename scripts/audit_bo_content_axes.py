"""Audit which content axes are actually available to BO/SVD replay.

The BO replay trial tables carry prompt text, seed indices, and seed images.
For the current Modal SVD image-to-video runner, prompt text is provenance and
stratification metadata; the generation call is image-conditioned and does not
pass prompt text into ``SVDGenerator.generate``. This script makes that audit
explicit before we design prompt- or seed-broadened experiments.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
INTAKE_ROOT = (
    REPO_ROOT
    / "research_program"
    / "neurips_memorability_selector"
    / "collaborator_inputs"
    / "camilo_bo_memorability"
)
DEFAULT_SEED_ROOT = INTAKE_ROOT / "original"
DEFAULT_REPLAY_SCRIPT = REPO_ROOT / "scripts" / "modal_bo_memorability_replay.py"
DEFAULT_SVD_GENERATOR = (
    REPO_ROOT / "src" / "audience_vectors" / "modal_app" / "functions" / "svd_generator.py"
)


def seed_catalog_rows(seed_root: Path) -> list[dict[str, Any]]:
    prompts_path = seed_root / "seeds" / "prompts.json"
    rows = json.loads(prompts_path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"{prompts_path} does not contain a list")
    return rows


def seed_availability(seed_root: Path) -> dict[str, Any]:
    rows = seed_catalog_rows(seed_root)
    available: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    no_image: list[dict[str, Any]] = []
    for row in rows:
        seed_image = row.get("seed_image")
        item = {
            "idx": int(row["idx"]),
            "bmd_name": str(row["bmd_name"]),
            "prompt": str(row["prompt"]),
            "seed_image": str(seed_image) if seed_image else None,
        }
        if not seed_image:
            no_image.append(item)
            continue
        image_path = seed_root / str(seed_image)
        item["image_path"] = str(image_path)
        if image_path.exists():
            available.append(item)
        else:
            missing.append(item)

    return {
        "seed_root": str(seed_root),
        "n_catalog_rows": len(rows),
        "n_available_seed_images": len(available),
        "n_missing_seed_images": len(missing),
        "n_rows_without_seed_image": len(no_image),
        "available": available,
        "missing": missing,
        "without_seed_image": no_image,
    }


def function_arg_names(path: Path, *, class_name: str, function_name: str) -> list[str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == function_name:
                return [arg.arg for arg in item.args.args]
    raise ValueError(f"{class_name}.{function_name} not found in {path}")


def spawn_keywords(path: Path, *, enclosing_function: str) -> list[str]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != enclosing_function:
            continue
        names: list[str] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if isinstance(func, ast.Attribute) and func.attr == "spawn":
                names.extend(keyword.arg for keyword in child.keywords if keyword.arg)
        return sorted(set(names))
    raise ValueError(f"{enclosing_function} not found in {path}")


def prompt_conditioning_audit(
    *,
    replay_script: Path,
    svd_generator: Path,
) -> dict[str, Any]:
    generator_args = function_arg_names(
        svd_generator,
        class_name="SVDGenerator",
        function_name="generate",
    )
    replay_spawn_keywords = spawn_keywords(
        replay_script,
        enclosing_function="generate_videos_on_modal",
    )
    generator_accepts_prompt = "prompt" in generator_args
    replay_passes_prompt = "prompt" in replay_spawn_keywords
    return {
        "replay_script": str(replay_script),
        "svd_generator": str(svd_generator),
        "generator_generate_args": generator_args,
        "replay_spawn_keywords": replay_spawn_keywords,
        "generator_accepts_prompt": generator_accepts_prompt,
        "replay_passes_prompt": replay_passes_prompt,
        "current_prompt_axis": (
            "generation_conditioning"
            if generator_accepts_prompt and replay_passes_prompt
            else "metadata_only"
        ),
    }


def build_content_axes_audit(
    *,
    seed_root: Path,
    replay_script: Path,
    svd_generator: Path,
) -> dict[str, Any]:
    availability = seed_availability(seed_root)
    prompt_audit = prompt_conditioning_audit(
        replay_script=replay_script,
        svd_generator=svd_generator,
    )
    return {
        "schema_version": 1,
        "kind": "bo_svd_content_axes_audit",
        "seed_availability": availability,
        "prompt_conditioning": prompt_audit,
        "actionable_axes": {
            "prompt_rewriting": (
                "blocked_until_generator_plumbing_changes"
                if prompt_audit["current_prompt_axis"] == "metadata_only"
                else "available"
            ),
            "seed_image_selection": (
                "available"
                if availability["n_available_seed_images"] > 1
                else "insufficient_seed_images"
            ),
            "seed_bank_expansion": (
                "needed"
                if availability["n_missing_seed_images"] > 0
                else "optional"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    parser.add_argument("--replay-script", type=Path, default=DEFAULT_REPLAY_SCRIPT)
    parser.add_argument("--svd-generator", type=Path, default=DEFAULT_SVD_GENERATOR)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/reports/bo_svd_content_axes_audit_20260608.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_content_axes_audit(
        seed_root=args.seed_root,
        replay_script=args.replay_script,
        svd_generator=args.svd_generator,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(audit, indent=2))
    print(
        json.dumps(
            {
                "report_path": str(args.report_path),
                "n_catalog_rows": audit["seed_availability"]["n_catalog_rows"],
                "n_available_seed_images": audit["seed_availability"][
                    "n_available_seed_images"
                ],
                "n_missing_seed_images": audit["seed_availability"][
                    "n_missing_seed_images"
                ],
                "current_prompt_axis": audit["prompt_conditioning"][
                    "current_prompt_axis"
                ],
                "actionable_axes": audit["actionable_axes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
