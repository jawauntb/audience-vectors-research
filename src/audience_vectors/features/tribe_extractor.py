"""Extract TRIBE v2 per-vertex activations for segments via Modal.

For each segment, calls `TribeService.predict_video(media_path_or_url)`
and saves the resulting `(time, vertices)` tensor to
`data/features/tribe/{sample_id}.npz` as compressed numpy.

Output format:
    np.savez_compressed(path,
        frames=np.array(shape=(T, V), dtype=float32),
        duration_seconds=np.array(duration),
        sample_id=np.array(sample_id),
    )

Hot-path optimizations:

  - **Idempotent.** Skip segments whose `.npz` already exists.
  - **URL fallback.** If `segment.media_path` is a URL (e.g. BMD's MiT_url),
    pass it straight to Modal — TRIBE downloads inline and streams.
  - **Concurrent dispatch.** Multiple in-flight `predict_video.remote.aio`
    calls share the warm Modal container.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from audience_vectors.schemas import Segment
from audience_vectors.services.tribe_service import TribeService, TribeValidationError

logger = logging.getLogger(__name__)


def _feature_path(output_dir: Path, sample_id: str) -> Path:
    return output_dir / f"{sample_id}.npz"


def load_tribe_features(path: Path) -> dict[str, np.ndarray]:
    """Reverse of save_tribe_features. Returns the dict of arrays."""
    return dict(np.load(path, allow_pickle=False))


class TribeFeatureExtractor:
    """Wraps `TribeService` with per-segment caching + concurrent fan-out."""

    def __init__(
        self,
        *,
        output_dir: Path,
        max_concurrency: int = 4,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.service = TribeService()
        self.max_concurrency = max(1, max_concurrency)
        self._sem = asyncio.Semaphore(self.max_concurrency)

    # -- public --------------------------------------------------------------

    async def extract_segment(self, segment: Segment) -> Path | None:
        """Compute + persist features for one segment. Returns the output
        path on success, None on soft failure."""
        out = _feature_path(self.output_dir, segment.sample_id)
        if out.exists() and out.stat().st_size > 0:
            return out
        media = segment.media_path
        if not media:
            logger.warning("segment %s has no media_path", segment.sample_id)
            return None

        async with self._sem:
            try:
                result = await self.service.predict_video(media)
            except TribeValidationError as exc:
                logger.warning("TRIBE rejected %s: %s", segment.sample_id, exc)
                return None

        if result is None:
            return None

        # `result` is the Modal-side VideoPredictionResult Pydantic model.
        # Modal pickles/JSONs it through; we accept either an object or a
        # dict-shaped payload.
        if hasattr(result, "frames"):
            frames = np.asarray(result.frames, dtype=np.float32)
            duration = float(result.duration_seconds)
        else:
            frames = np.asarray(result["frames"], dtype=np.float32)
            duration = float(result["duration_seconds"])

        np.savez_compressed(
            out,
            frames=frames,
            duration_seconds=np.array(duration, dtype=np.float32),
            sample_id=np.array(segment.sample_id),
        )
        logger.info("wrote %s frames=%s", out, frames.shape)
        return out

    async def extract_many(
        self,
        segments: Iterable[Segment],
    ) -> list[Path]:
        """Run extract_segment concurrently; return the list of paths written
        or already-cached. Skipped segments (soft failures, no media) drop
        out of the result."""
        segs = list(segments)
        coros = [self.extract_segment(s) for s in segs]
        results = await asyncio.gather(*coros)
        return [p for p in results if p is not None]
