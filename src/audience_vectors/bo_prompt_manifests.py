"""Shared manifest helpers for BO memorability replay panels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


def build_replay_seed_pool(seed_root: Path, *, pool_size: int) -> list[SeedSlot]:
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
    target_slots: list[int] | None,
    pool_size: int,
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


def draw_sobol_points(count: int, *, scramble_seed: int) -> np.ndarray:
    """Draw 2D Sobol points for alpha/guidance controls."""
    from torch.quasirandom import SobolEngine  # noqa: PLC0415

    if count < 0:
        raise ValueError("count must be >= 0")
    engine = SobolEngine(dimension=2, scramble=True, seed=scramble_seed)
    return np.asarray(engine.draw(count), dtype=np.float64)
