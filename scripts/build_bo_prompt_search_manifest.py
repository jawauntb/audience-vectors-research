"""Build a per-prompt Sobol search trial table for BO memorability replay.

The saved BO table and prompt-transfer stress test show prompt-pocket behavior:
old high-scoring alpha/guidance recipes do not transfer across prompt strata.
This builder creates the next diagnostic panel: a balanced Sobol alpha/guidance
search inside every locally image-backed prompt slot.

The output is replay-compatible with
``scripts/modal_bo_memorability_replay.py --trial-table``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from audience_vectors.bo_prompt_manifests import (
    SeedSlot,
    draw_sobol_points,
    select_seed_slots,
)
from audience_vectors.bo_replay import safe_label

REPO_ROOT = Path(__file__).resolve().parents[1]
INTAKE_ROOT = (
    REPO_ROOT
    / "research_program"
    / "neurips_memorability_selector"
    / "collaborator_inputs"
    / "camilo_bo_memorability"
)
DEFAULT_SEED_ROOT = INTAKE_ROOT / "original"
DEFAULT_REPLAY_SEED_POOL_SIZE = 16
DEFAULT_ALPHA_RANGE = (-10.0, 10.0)
DEFAULT_GUIDANCE_RANGE = (1.0, 10.0)


def scale_unit_interval(value: float, *, low: float, high: float) -> float:
    """Scale a unit-interval Sobol coordinate to an inclusive search range."""
    return float(value * (high - low) + low)


def sobol_prompt_search_row(
    *,
    sobol_index: int,
    point: np.ndarray,
    seed_slot: SeedSlot,
    alpha_range: tuple[float, float],
    guidance_range: tuple[float, float],
    noise_seed_offset: int,
) -> dict[str, Any]:
    """Create one prompt-local Sobol alpha/guidance search row."""
    task_id = safe_label(
        f"sobol_prompt_search_{sobol_index:03d}_slot{seed_slot.slot:02d}"
    )
    return {
        "task_id": task_id,
        "policy": "sobol_prompt_search",
        "alpha": scale_unit_interval(
            float(point[0]),
            low=alpha_range[0],
            high=alpha_range[1],
        ),
        "guidance": scale_unit_interval(
            float(point[1]),
            low=guidance_range[0],
            high=guidance_range[1],
        ),
        "seed_idx": seed_slot.slot,
        "noise_seed": sobol_index + noise_seed_offset,
        "filename": None,
        "prompt": seed_slot.prompt,
        "tribe_score": None,
        "clip_score": None,
        "quality_score": None,
        "target_seed": seed_slot.to_json(),
    }


def validate_range(name: str, values: tuple[float, float]) -> None:
    low, high = values
    if not low < high:
        raise ValueError(f"{name} lower bound must be < upper bound")


def build_prompt_search_manifest(
    *,
    seed_root: Path,
    target_seed_slots: list[int] | None,
    replay_seed_pool_size: int,
    sobol_samples_per_seed: int,
    sobol_start_index: int,
    sobol_scramble_seed: int,
    noise_seed_offset: int,
    alpha_range: tuple[float, float],
    guidance_range: tuple[float, float],
) -> dict[str, Any]:
    """Build a replay-compatible prompt-local Sobol search manifest."""
    if sobol_samples_per_seed <= 0:
        raise ValueError("--sobol-samples-per-seed must be positive")
    if sobol_start_index < 0:
        raise ValueError("--sobol-start-index must be >= 0")
    if noise_seed_offset < 0:
        raise ValueError("--noise-seed-offset must be >= 0")
    validate_range("alpha range", alpha_range)
    validate_range("guidance range", guidance_range)

    seed_slots = select_seed_slots(
        seed_root,
        target_slots=target_seed_slots,
        pool_size=replay_seed_pool_size,
    )
    sobol_points = draw_sobol_points(
        sobol_start_index + sobol_samples_per_seed,
        scramble_seed=sobol_scramble_seed,
    )[sobol_start_index:]

    rows: list[dict[str, Any]] = []
    for offset, point in enumerate(sobol_points):
        sobol_index = sobol_start_index + offset
        for seed_slot in seed_slots:
            rows.append(
                sobol_prompt_search_row(
                    sobol_index=sobol_index,
                    point=point,
                    seed_slot=seed_slot,
                    alpha_range=alpha_range,
                    guidance_range=guidance_range,
                    noise_seed_offset=noise_seed_offset,
                )
            )

    return {
        "schema_version": 1,
        "kind": "bo_prompt_search_trial_table",
        "seed_root": str(seed_root),
        "replay_seed_pool_size": replay_seed_pool_size,
        "target_seed_slots": [slot.to_json() for slot in seed_slots],
        "sobol_samples_per_seed": sobol_samples_per_seed,
        "sobol_start_index": sobol_start_index,
        "sobol_scramble_seed": sobol_scramble_seed,
        "noise_seed_offset": noise_seed_offset,
        "alpha_range": list(alpha_range),
        "guidance_range": list(guidance_range),
        "n_sobol_prompt_search_trials": len(rows),
        "all_meta": rows,
    }


def parse_range(raw: str) -> tuple[float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be LOW,HIGH")
    try:
        values = (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("range bounds must be numeric") from exc
    validate_range("range", values)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, default=DEFAULT_SEED_ROOT)
    parser.add_argument("--target-seed-slot", type=int, action="append")
    parser.add_argument(
        "--replay-seed-pool-size",
        type=int,
        default=DEFAULT_REPLAY_SEED_POOL_SIZE,
    )
    parser.add_argument("--sobol-samples-per-seed", type=int, default=8)
    parser.add_argument("--sobol-start-index", type=int, default=512)
    parser.add_argument("--sobol-scramble-seed", type=int, default=42)
    parser.add_argument(
        "--noise-seed-offset",
        type=int,
        default=0,
        help=(
            "Offset added to each Sobol index when setting the SVD generation "
            "noise_seed. Use this to replay an accepted recipe neighborhood "
            "with fresh stochastic seeds while preserving recipe indices."
        ),
    )
    parser.add_argument(
        "--alpha-range",
        type=parse_range,
        default=DEFAULT_ALPHA_RANGE,
        help="alpha search range as LOW,HIGH",
    )
    parser.add_argument(
        "--guidance-range",
        type=parse_range,
        default=DEFAULT_GUIDANCE_RANGE,
        help="guidance search range as LOW,HIGH",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/reports/bo_prompt_search_trial_table.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_prompt_search_manifest(
        seed_root=args.seed_root,
        target_seed_slots=args.target_seed_slot,
        replay_seed_pool_size=args.replay_seed_pool_size,
        sobol_samples_per_seed=args.sobol_samples_per_seed,
        sobol_start_index=args.sobol_start_index,
        sobol_scramble_seed=args.sobol_scramble_seed,
        noise_seed_offset=args.noise_seed_offset,
        alpha_range=args.alpha_range,
        guidance_range=args.guidance_range,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "report_path": str(args.report_path),
                "target_seed_slots": [
                    slot["bmd_name"] for slot in manifest["target_seed_slots"]
                ],
                "sobol_samples_per_seed": manifest["sobol_samples_per_seed"],
                "n_trials": len(manifest["all_meta"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
