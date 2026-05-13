"""Extract V-JEPA 2 per-segment embeddings via Modal.

Lighter sibling of `tribe_extractor` — V-JEPA gives one mean-pooled
embedding per clip, not per-vertex brain activations. Saved to
`data/features/vjepa/{sample_id}.npz` with key `embedding`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from audience_vectors.schemas import Segment
from audience_vectors.services.vjepa_service import VjepaService, VjepaValidationError

logger = logging.getLogger(__name__)


class VjepaFeatureExtractor:
    """Per-segment V-JEPA feature extractor with caching + bounded concurrency."""

    def __init__(
        self,
        *,
        output_dir: Path,
        max_concurrency: int = 4,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.service = VjepaService()
        self.max_concurrency = max(1, max_concurrency)
        self._sem = asyncio.Semaphore(self.max_concurrency)

    async def extract_segment(self, segment: Segment) -> Path | None:
        out = self.output_dir / f"{segment.sample_id}.npz"
        if out.exists() and out.stat().st_size > 0:
            return out
        if not segment.media_path:
            logger.warning("segment %s has no media_path", segment.sample_id)
            return None
        async with self._sem:
            try:
                result = await self.service.predict_video(segment.media_path)
            except VjepaValidationError as exc:
                logger.warning("V-JEPA rejected %s: %s", segment.sample_id, exc)
                return None

        if result is None:
            return None

        if hasattr(result, "embedding"):
            embedding = np.asarray(result.embedding, dtype=np.float32)
            duration = float(result.duration_seconds)
            n_frames = int(result.n_frames)
        else:
            embedding = np.asarray(result["embedding"], dtype=np.float32)
            duration = float(result["duration_seconds"])
            n_frames = int(result["n_frames"])

        np.savez_compressed(
            out,
            embedding=embedding,
            duration_seconds=np.array(duration, dtype=np.float32),
            n_frames=np.array(n_frames, dtype=np.int32),
            sample_id=np.array(segment.sample_id),
        )
        logger.info("wrote %s dim=%d", out, embedding.shape[0])
        return out

    async def extract_many(self, segments: Iterable[Segment]) -> list[Path]:
        segs = list(segments)
        coros = [self.extract_segment(s) for s in segs]
        results = await asyncio.gather(*coros)
        return [p for p in results if p is not None]
