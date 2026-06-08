"""Build a prompt-transfer trial table for BO memorability replay.

The saved collaborator BO table mostly supports two prompt strata. This script
tests a different question: do the best BO parameter recipes transfer across
the available image-backed prompt strata, or are they prompt-pocket artifacts?

It writes a replay-compatible trial table whose ``all_meta`` rows can be passed
to ``scripts/modal_bo_memorability_replay.py --trial-table``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audience_vectors.bo_replay import (
    CollaboratorBOTrial,
    load_collaborator_trials,
    safe_label,
    sort_by_score,
    trial_policy_group,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INTAKE_ROOT = (
    REPO_ROOT
    / "research_program"
    / "neurips_memorability_selector"
    / "collaborator_inputs"
    / "camilo_bo_memorability"
)
DEFAULT_TRIAL_TABLE = INTAKE_ROOT / "raw_results" / "gpu_run_3obj_all_results.json"
DEFAULT_SEED_ROOT = INTAKE_ROOT / "original"
DEFAULT_REPLAY_SEED_POOL_SIZE = 16


@dataclass(frozen=True)
class SeedSlot:
    """One optimizer seed slot mapped through the replay seed pool."""

    slot: int
    source_idx: int
    bmd_name: str
    prompt: str
    image_path: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "source_idx": self.source_idx,
            "bmd_name": self.bmd_name,
            "prompt": self.prompt,
            "image_path": str(self.image_path),
        }


def load_available_seed_rows(seed_root: Path) -> list[dict[str, Any]]:
    """Load seed prompt rows whose image files are present locally."""
    prompts_path = seed_root / "seeds" / "prompts.json"
    rows = json.loads(prompts_path.read_text())
    available: list[dict[str, Any]] = []
    for row in rows:
        seed_image = row.get("seed_image")
        if not seed_image:
            continue
        image_path = seed_root / str(seed_image)
        if not image_path.exists():
            continue
        available.append(
            {
                "idx": int(row["idx"]),
                "bmd_name": str(row["bmd_name"]),
                "prompt": str(row["prompt"]),
                "image_path": image_path,
            }
        )
    if not available:
        raise FileNotFoundError(f"no available seed images under {seed_root}")
    return available


def build_replay_seed_pool(
    seed_root: Path,
    *,
    pool_size: int = DEFAULT_REPLAY_SEED_POOL_SIZE,
) -> list[SeedSlot]:
    """Mirror the replay script's cycling seed-pool semantics."""
    if pool_size <= 0:
        raise ValueError("pool size must be positive")

    available = load_available_seed_rows(seed_root)
    seed_pool: list[SeedSlot] = []
    for slot in range(pool_size):
        row = available[slot % len(available)]
        seed_pool.append(
            SeedSlot(
                slot=slot,
                source_idx=int(row["idx"]),
                bmd_name=str(row["bmd_name"]),
                prompt=str(row["prompt"]),
                image_path=Path(row["image_path"]),
            )
        )
    return seed_pool


def select_seed_slots(
    seed_root: Path,
    *,
    target_slots: list[int] | None = None,
    pool_size: int = DEFAULT_REPLAY_SEED_POOL_SIZE,
) -> list[SeedSlot]:
    """Select target seed slots, defaulting to one slot per available image."""
    seed_pool = build_replay_seed_pool(seed_root, pool_size=pool_size)
    if target_slots:
        invalid = [slot for slot in target_slots if slot < 0 or slot >= pool_size]
        if invalid:
            raise ValueError(f"seed slots out of range 0..{pool_size - 1}: {invalid}")
        return [seed_pool[slot] for slot in target_slots]

    selected: list[SeedSlot] = []
    seen_names: set[str] = set()
    for slot in seed_pool:
        if slot.bmd_name in seen_names:
            continue
        selected.append(slot)
        seen_names.add(slot.bmd_name)
    return selected


def select_anchor_trials(
    trials: list[CollaboratorBOTrial],
    *,
    anchor_task_ids: list[str],
    top_bo_anchors: int,
) -> list[CollaboratorBOTrial]:
    """Select BO recipes to retarget across prompt strata."""
    if anchor_task_ids:
        by_id = {trial.task_id: trial for trial in trials}
        missing = sorted(set(anchor_task_ids) - set(by_id))
        if missing:
            raise ValueError(f"unknown anchor task ids: {', '.join(missing)}")
        return [by_id[task_id] for task_id in anchor_task_ids]

    if top_bo_anchors <= 0:
        raise ValueError("--top-bo-anchors must be positive")
    bo_trials = [
        trial for trial in trials if trial_policy_group(trial.task_id) == "bo"
    ]
    anchors = sort_by_score(bo_trials, score_name="tribe_score")[:top_bo_anchors]
    if not anchors:
        raise ValueError("no BO anchors found in source trial table")
    return anchors


def bo_transfer_row(
    anchor: CollaboratorBOTrial,
    *,
    anchor_rank: int,
    seed_slot: SeedSlot,
) -> dict[str, Any]:
    """Retarget one BO alpha/guidance recipe to one prompt seed slot."""
    task_id = safe_label(f"bo_transfer_{anchor.task_id}_slot{seed_slot.slot:02d}")
    return {
        "task_id": task_id,
        "policy": "bo_transfer",
        "transfer_source_task_id": anchor.task_id,
        "alpha": anchor.alpha,
        "guidance": anchor.guidance,
        "seed_idx": seed_slot.slot,
        "noise_seed": int(anchor.noise_seed + 50_000 + anchor_rank * 1000 + seed_slot.slot),
        "filename": None,
        "prompt": seed_slot.prompt,
        "tribe_score": None,
        "clip_score": None,
        "quality_score": None,
        "source_anchor_tribe_score": anchor.tribe_score,
        "source_anchor_clip_score": anchor.clip_score,
        "source_anchor_quality_score": anchor.quality_score,
        "target_seed": seed_slot.to_json(),
    }


def draw_sobol_points(count: int, *, scramble_seed: int) -> np.ndarray:
    """Draw 2D Sobol points for alpha/guidance controls."""
    from torch.quasirandom import SobolEngine  # noqa: PLC0415

    engine = SobolEngine(dimension=2, scramble=True, seed=scramble_seed)
    return np.asarray(engine.draw(count), dtype=np.float64)


def sobol_transfer_row(
    *,
    sobol_index: int,
    point: np.ndarray,
    seed_slot: SeedSlot,
) -> dict[str, Any]:
    """Create one matched Sobol alpha/guidance control for a target prompt."""
    task_id = safe_label(f"sobol_transfer_{sobol_index:03d}_slot{seed_slot.slot:02d}")
    return {
        "task_id": task_id,
        "policy": "sobol_transfer",
        "alpha": float(point[0] * 20.0 - 10.0),
        "guidance": float(point[1] * 9.0 + 1.0),
        "seed_idx": seed_slot.slot,
        "noise_seed": sobol_index,
        "filename": None,
        "prompt": seed_slot.prompt,
        "tribe_score": None,
        "clip_score": None,
        "quality_score": None,
        "target_seed": seed_slot.to_json(),
    }


def build_prompt_transfer_manifest(
    *,
    source_trial_table: Path,
    seed_root: Path,
    anchor_task_ids: list[str],
    top_bo_anchors: int,
    target_seed_slots: list[int] | None,
    replay_seed_pool_size: int,
    sobol_controls_per_seed: int,
    sobol_start_index: int,
    sobol_scramble_seed: int,
) -> dict[str, Any]:
    """Build a replay-compatible prompt-transfer manifest."""
    if sobol_controls_per_seed < 0:
        raise ValueError("--sobol-controls-per-seed must be >= 0")
    if sobol_start_index < 0:
        raise ValueError("--sobol-start-index must be >= 0")

    trials = load_collaborator_trials(source_trial_table)
    anchors = select_anchor_trials(
        trials,
        anchor_task_ids=anchor_task_ids,
        top_bo_anchors=top_bo_anchors,
    )
    seed_slots = select_seed_slots(
        seed_root,
        target_slots=target_seed_slots,
        pool_size=replay_seed_pool_size,
    )

    rows: list[dict[str, Any]] = []
    for anchor_rank, anchor in enumerate(anchors):
        for seed_slot in seed_slots:
            rows.append(
                bo_transfer_row(
                    anchor,
                    anchor_rank=anchor_rank,
                    seed_slot=seed_slot,
                )
            )

    sobol_points = draw_sobol_points(
        sobol_start_index + sobol_controls_per_seed,
        scramble_seed=sobol_scramble_seed,
    )[sobol_start_index:]
    for offset, point in enumerate(sobol_points):
        sobol_index = sobol_start_index + offset
        for seed_slot in seed_slots:
            rows.append(
                sobol_transfer_row(
                    sobol_index=sobol_index,
                    point=point,
                    seed_slot=seed_slot,
                )
            )

    return {
        "schema_version": 1,
        "kind": "bo_prompt_transfer_trial_table",
        "source_trial_table": str(source_trial_table),
        "seed_root": str(seed_root),
        "replay_seed_pool_size": replay_seed_pool_size,
        "top_bo_anchors": top_bo_anchors,
        "anchor_task_ids": [anchor.task_id for anchor in anchors],
        "target_seed_slots": [slot.to_json() for slot in seed_slots],
        "sobol_controls_per_seed": sobol_controls_per_seed,
        "sobol_start_index": sobol_start_index,
        "sobol_scramble_seed": sobol_scramble_seed,
        "n_bo_transfer_trials": len(anchors) * len(seed_slots),
        "n_sobol_transfer_trials": sobol_controls_per_seed * len(seed_slots),
        "all_meta": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trial-table", type=Path, default=DEFAULT_TRIAL_TABLE)
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    parser.add_argument("--anchor-task-id", action="append", default=[])
    parser.add_argument("--top-bo-anchors", type=int, default=3)
    parser.add_argument("--target-seed-slot", type=int, action="append")
    parser.add_argument(
        "--replay-seed-pool-size",
        type=int,
        default=DEFAULT_REPLAY_SEED_POOL_SIZE,
    )
    parser.add_argument("--sobol-controls-per-seed", type=int, default=3)
    parser.add_argument("--sobol-start-index", type=int, default=256)
    parser.add_argument("--sobol-scramble-seed", type=int, default=42)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/reports/bo_prompt_transfer_trial_table.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_prompt_transfer_manifest(
        source_trial_table=args.source_trial_table,
        seed_root=args.seed_root,
        anchor_task_ids=list(args.anchor_task_id),
        top_bo_anchors=args.top_bo_anchors,
        target_seed_slots=args.target_seed_slot,
        replay_seed_pool_size=args.replay_seed_pool_size,
        sobol_controls_per_seed=args.sobol_controls_per_seed,
        sobol_start_index=args.sobol_start_index,
        sobol_scramble_seed=args.sobol_scramble_seed,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "report_path": str(args.report_path),
                "anchors": manifest["anchor_task_ids"],
                "target_seed_slots": [
                    slot["bmd_name"] for slot in manifest["target_seed_slots"]
                ],
                "n_trials": len(manifest["all_meta"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
