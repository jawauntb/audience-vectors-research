"""Segment every video in the manifest into fixed-length windows.

Reads the manifest at `data/training/manifest.parquet` (or runs adapters
on the fly if no manifest exists), produces segment clips under
`data/processed/clips_3s/`, and writes a `segments.parquet` file.

Usage:
    uv run python scripts/segment_dataset.py
    uv run python scripts/segment_dataset.py --limit 20 --datasets BOLDMoments
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
from audience_vectors.media import Segmenter
from audience_vectors.schemas import CanonicalVideo

ADAPTERS: tuple[tuple[str, str, type[DatasetAdapter]], ...] = (
    ("BOLDMoments", "BOLD_MOMENTS_ROOT", BoldMomentsAdapter),
    ("Memento10k", "MEMENTO10K_ROOT", Memento10kAdapter),
)


def _iter_videos(selected: set[str] | None) -> Iterator[CanonicalVideo]:
    for name, env_var, cls in ADAPTERS:
        if selected and name not in selected:
            continue
        root_str = os.environ.get(env_var, "").strip()
        if not root_str:
            continue
        root = Path(root_str)
        if not root.exists():
            continue
        print(f"[scan] {name}: {root}")
        yield from cls(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap videos processed.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = get_config()
    cfg.paths.ensure()

    out_path = args.output or (cfg.paths.training / "segments.parquet")
    segmenter = Segmenter(
        segment_length_s=cfg.pipeline.segment_length_s,
        segment_stride_s=cfg.pipeline.segment_stride_s,
        output_dir=cfg.paths.clips,
    )

    selected = set(args.datasets) if args.datasets else None
    video_iter = _iter_videos(selected)
    if args.limit:
        video_iter = (v for i, v in enumerate(video_iter) if i < args.limit)

    rows: list[dict[str, object]] = []
    n_videos = n_segments = n_passthrough = n_extracted = 0
    skipped: list[str] = []

    for video in video_iter:
        n_videos += 1
        jobs = segmenter.plan_jobs(video)
        if not jobs:
            skipped.append(video.video_id)
            continue
        for job in jobs:
            try:
                segmenter.run_job(job)
            except Exception as exc:  # noqa: BLE001
                print(f"  [fail] {job.sample_id}: {exc}")
                continue
            seg = segmenter._segment_from_job(video, job)  # noqa: SLF001
            rows.append(seg.model_dump(mode="json"))
            n_segments += 1
            if job.is_passthrough:
                n_passthrough += 1
            else:
                n_extracted += 1

    print(
        f"\n[summary] videos={n_videos} segments={n_segments} "
        f"passthrough={n_passthrough} extracted={n_extracted} skipped={len(skipped)}"
    )
    if skipped[:5]:
        print(f"  skipped sample: {skipped[:5]}")

    if not rows:
        print("[done] no segments produced — check dataset roots + local video paths")
        return

    import polars as pl  # noqa: PLC0415

    # Drop `labels` and `captions` — both are populated by downstream jobs
    # (gemini_labeler / persona_labeler) and writing empty structs/lists
    # here trips polars' parquet schema inference.
    for row in rows:
        row.pop("labels", None)
        row.pop("captions", None)

    df = pl.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"[done] wrote {len(rows)} segments -> {out_path}")


if __name__ == "__main__":
    main()
