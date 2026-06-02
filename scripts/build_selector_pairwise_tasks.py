"""Build blinded pairwise human-eval tasks from a selector manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def add_task(
    tasks: list[dict[str, Any]],
    *,
    rng: random.Random,
    seed: str,
    comparison: str,
    policy_a: str,
    policy_b: str,
    manifest_row: dict[str, Any],
) -> None:
    paths = manifest_row["video_paths"]
    labels = manifest_row["labels"]
    if paths.get(policy_a) is None or paths.get(policy_b) is None:
        return
    if paths[policy_a] == paths[policy_b]:
        return

    left_policy, right_policy = policy_a, policy_b
    if rng.random() < 0.5:
        left_policy, right_policy = right_policy, left_policy

    tasks.append(
        {
            "task_id": f"{seed}__{comparison}",
            "seed": seed,
            "comparison": comparison,
            "left": {
                "policy": left_policy,
                "path": paths[left_policy],
                "label": labels[left_policy],
            },
            "right": {
                "policy": right_policy,
                "path": paths[right_policy],
                "label": labels[right_policy],
            },
            "target_policy": policy_a,
            "baseline_policy": policy_b,
            "question": "Which video would be more memorable to you after seeing many similar clips?",
        }
    )


def build_tasks(
    *,
    manifest: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    comparisons = [
        ("product_vs_base", "product_selected", "base"),
        ("product_vs_single_lora", "product_selected", "single_lora"),
        ("product_vs_raw_best", "product_selected", "raw_best_of_n"),
        ("product_vs_clip_seed_image", "product_selected", "clip_seed_image_best"),
        ("product_vs_clip_prompt", "product_selected", "clip_prompt_best"),
        (
            "product_vs_clip_preservation",
            "product_selected",
            "clip_preservation_best",
        ),
        (
            "product_vs_vjepa_memorability",
            "product_selected",
            "vjepa_memorability_best",
        ),
        ("gated_vs_base", "gated_best_of_n", "base"),
        ("gated_vs_clip_preservation", "gated_best_of_n", "clip_preservation_best"),
        ("gated_vs_vjepa_memorability", "gated_best_of_n", "vjepa_memorability_best"),
    ]

    for row in manifest["rows"]:
        for comparison, policy_a, policy_b in comparisons:
            add_task(
                tasks,
                rng=rng,
                seed=row["seed"],
                comparison=comparison,
                policy_a=policy_a,
                policy_b=policy_b,
                manifest_row=row,
            )

    rng.shuffle(tasks)

    return {
        "schema_version": 1,
        "source_manifest": manifest["source"],
        "n_seeds": manifest["n_seeds"],
        "n_tasks": len(tasks),
        "comparisons": comparisons,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "research_program/neurips_memorability_selector/experiments/"
            "current_selector_manifest.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "research_program/neurips_memorability_selector/experiments/"
            "current_selector_pairwise_tasks.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260528)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    tasks = build_tasks(manifest=manifest, rng=random.Random(args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tasks, indent=2) + "\n")

    print(f"[done] wrote {args.out}")
    print(f"[done] tasks: {tasks['n_tasks']}")


if __name__ == "__main__":
    main()
