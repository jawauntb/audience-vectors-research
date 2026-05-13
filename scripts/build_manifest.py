"""Build the unified manifest from configured dataset adapters.

Walks every configured dataset root, emits one `CanonicalVideo` per source
video, and writes the combined manifest to `data/training/manifest.parquet`.

Usage:
    uv run python scripts/build_manifest.py
    uv run python scripts/build_manifest.py --datasets Memento10k --limit 20
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from pathlib import Path

from audience_vectors.config import get_config
from audience_vectors.datasets import (
    BoldMomentsAdapter,
    DatasetAdapter,
    Memento10kAdapter,
)
from audience_vectors.schemas import CanonicalVideo


# (name, env var holding root, adapter class). Add new entries here when
# new adapters land; the rest of this script is generic.
ADAPTERS: tuple[tuple[str, str, type[DatasetAdapter]], ...] = (
    ("BOLDMoments", "BOLD_MOMENTS_ROOT", BoldMomentsAdapter),
    ("Memento10k", "MEMENTO10K_ROOT", Memento10kAdapter),
)


def _iter_configured(selected: set[str] | None) -> Iterator[CanonicalVideo]:
    for name, env_var, cls in ADAPTERS:
        if selected and name not in selected:
            continue
        root_str = os.environ.get(env_var, "").strip()
        if not root_str:
            print(f"[skip] {name}: ${env_var} not set in .env")
            continue
        root = Path(root_str)
        if not root.exists():
            print(f"[skip] {name}: root path does not exist: {root}")
            continue
        print(f"[scan] {name}: {root}")
        adapter = cls(root)
        for video in adapter:
            yield video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Restrict to specific adapters by name (default: all configured).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total videos emitted across all datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path (default: data/training/manifest.parquet).",
    )
    args = parser.parse_args()

    cfg = get_config()
    cfg.paths.ensure()

    out_path = args.output or (cfg.paths.training / "manifest.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected = set(args.datasets) if args.datasets else None
    rows: list[dict[str, object]] = []
    for video in _iter_configured(selected):
        rows.append(video.model_dump(mode="json"))
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        print("[done] no videos found — check your dataset roots in .env")
        return

    # Lazy-import polars so `--help` and the dry-run path don't pay
    # for the import (and don't fail before `uv sync --extra ml`).
    import polars as pl  # noqa: PLC0415

    df = pl.DataFrame(rows)
    df.write_parquet(out_path)
    print(f"[done] wrote {len(rows)} videos -> {out_path}")


if __name__ == "__main__":
    main()
