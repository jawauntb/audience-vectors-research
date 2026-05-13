"""Run the Gemini synthetic labeler over segmented clips.

Reads `data/training/segments.parquet`, scores each segment with Gemini,
writes `data/labels/synthetic_gemini.parquet`. Idempotent — segments
already present in the output file are skipped.

Usage:
    uv run python scripts/label_segments.py
    uv run python scripts/label_segments.py --limit 20 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from audience_vectors.config import get_config
from audience_vectors.labeling import GeminiLabeler
from audience_vectors.schemas import Segment


def _read_segments(parquet_path: Path) -> list[Segment]:
    import polars as pl  # noqa: PLC0415

    df = pl.read_parquet(parquet_path)
    rows = df.to_dicts()
    return [Segment.model_validate(r) for r in rows]


def _read_already_labeled(parquet_path: Path) -> set[str]:
    if not parquet_path.exists():
        return set()
    import polars as pl  # noqa: PLC0415

    df = pl.read_parquet(parquet_path)
    if "segment_id" not in df.columns:
        return set()
    return set(df.get_column("segment_id").to_list())


async def _run(args: argparse.Namespace) -> int:
    cfg = get_config()
    cfg.paths.ensure()

    segments_path = args.segments or (cfg.paths.training / "segments.parquet")
    if not segments_path.exists():
        print(f"[fail] no segments file at {segments_path} — run scripts/segment_dataset.py first")
        return 1

    output_path = args.output or (cfg.paths.labels / "synthetic_gemini.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cfg.api_keys.google:
        print("[fail] GOOGLE_API_KEY missing in .env")
        return 1

    segments = _read_segments(segments_path)
    already = _read_already_labeled(output_path)
    pending = [s for s in segments if s.sample_id not in already]
    if args.limit:
        pending = pending[: args.limit]

    print(
        f"[plan] total={len(segments)} already_labeled={len(already)} "
        f"pending={len(pending)} model={cfg.models.gemini_video}"
    )
    if not pending:
        print("[done] nothing to label.")
        return 0

    labeler = GeminiLabeler(
        api_key=cfg.api_keys.google,
        model=cfg.models.gemini_video,
        max_concurrency=args.concurrency,
    )
    results = await labeler.label_many(pending)
    print(f"[result] labeled={len(results)} / pending={len(pending)}")

    if not results:
        print("[warn] no labels produced; nothing written.")
        return 1

    import polars as pl  # noqa: PLC0415

    new_rows = [r.model_dump(mode="json") for r in results]
    new_df = pl.DataFrame(new_rows)
    if output_path.exists():
        old_df = pl.read_parquet(output_path)
        combined = pl.concat([old_df, new_df], how="vertical_relaxed")
    else:
        combined = new_df
    combined.write_parquet(output_path)
    print(f"[done] wrote {len(results)} new labels -> {output_path} (total rows: {len(combined)})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
