"""Extract TRIBE NPZ features for attention-capture Phase 1 samples."""

from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audience_vectors.services.tribe_service import TribeService, TribeValidationError


@dataclass(frozen=True)
class VideoFeatureJob:
    sample_id: str
    media_path: str
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--media-path-column", default="video_path")
    parser.add_argument("--app-name", default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--transport",
        choices=("bytes", "path"),
        default="bytes",
        help=(
            "bytes sends local file bytes through the Modal RPC; path passes "
            "the media path/URL directly to TRIBE."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(run(args)))


async def run(args: argparse.Namespace) -> int:
    jobs = load_jobs_from_csv(
        source_csv=args.source_csv,
        output_dir=args.output_dir,
        sample_id_column=args.sample_id_column,
        media_path_column=args.media_path_column,
        limit=args.limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cached = [job for job in jobs if job.output_path.exists() and job.output_path.stat().st_size > 0]
    pending = [job for job in jobs if job not in set(cached)]
    print(
        f"[plan] jobs={len(jobs)} cached={len(cached)} "
        f"to_extract={len(pending)} output={args.output_dir}",
        flush=True,
    )
    if not pending:
        return 0

    service = TribeService(app_name=args.app_name)
    sem = asyncio.Semaphore(max(1, args.concurrency))
    tasks = [
        extract_one(
            service=service,
            sem=sem,
            job=job,
            transport=args.transport,
        )
        for job in pending
    ]
    results = await asyncio.gather(*tasks)
    written = [path for path in results if path is not None]
    print(f"[done] extracted {len(written)}/{len(pending)} missing TRIBE features")
    return 0 if len(written) == len(pending) else 1


def load_jobs_from_csv(
    *,
    source_csv: Path,
    output_dir: Path,
    sample_id_column: str = "sample_id",
    media_path_column: str = "video_path",
    limit: int | None = None,
) -> list[VideoFeatureJob]:
    with source_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    jobs: list[VideoFeatureJob] = []
    for row in rows:
        sample_id = required_cell(row, sample_id_column, source_csv)
        media_path = required_cell(row, media_path_column, source_csv)
        jobs.append(
            VideoFeatureJob(
                sample_id=sample_id,
                media_path=media_path,
                output_path=output_dir / f"{sample_id}.npz",
            )
        )
        if limit is not None and len(jobs) >= limit:
            break
    return jobs


async def extract_one(
    *,
    service: TribeService,
    sem: asyncio.Semaphore,
    job: VideoFeatureJob,
    transport: str,
) -> Path | None:
    if job.output_path.exists() and job.output_path.stat().st_size > 0:
        return job.output_path

    async with sem:
        try:
            if transport == "bytes":
                local_path = Path(job.media_path)
                result = await service.predict_video_bytes(
                    local_path.read_bytes(),
                    suffix=local_path.suffix or ".mp4",
                )
            else:
                result = await service.predict_video(job.media_path)
        except (FileNotFoundError, TribeValidationError) as exc:
            print(f"[skip] {job.sample_id}: {exc}", flush=True)
            return None

    if result is None:
        return None
    frames, duration = result_to_arrays(result)
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        job.output_path,
        frames=frames,
        duration_seconds=np.array(duration, dtype=np.float32),
        sample_id=np.array(job.sample_id),
        media_path=np.array(job.media_path),
        transport=np.array(transport),
    )
    print(f"[tribe] wrote {job.output_path} frames={frames.shape}", flush=True)
    return job.output_path


def result_to_arrays(result: Any) -> tuple[np.ndarray, float]:
    if hasattr(result, "frames"):
        frames = np.asarray(result.frames, dtype=np.float32)
        duration = float(result.duration_seconds)
    else:
        frames = np.asarray(result["frames"], dtype=np.float32)
        duration = float(result["duration_seconds"])
    if frames.ndim == 1:
        frames = frames.reshape(1, -1)
    if frames.ndim != 2:
        raise ValueError(f"expected TRIBE frames to be 1D or 2D, got {frames.shape}")
    return frames, duration


def required_cell(row: dict[str, str], column: str, path: Path) -> str:
    value = row.get(column)
    if value is None or value == "":
        raise ValueError(f"{path} has a row missing required column {column!r}")
    return value


if __name__ == "__main__":
    main()
