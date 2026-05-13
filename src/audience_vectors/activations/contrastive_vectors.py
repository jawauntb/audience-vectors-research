"""Compute contrastive activation directions from TRIBE features.

For a target label axis (e.g. `memorability`), pick the top-K segments
(positive set) and bottom-K segments (negative set) on that axis, mean
their TRIBE activations, and return the difference as a unit-norm
direction vector.

Score a new segment by projecting its activation onto the direction:

    project_features(features, vector) = features @ vector

That projection — interpreted as "how much this segment's activation
points along the memorability direction" — is the audience-vector score.

The same machinery works for persona-conditioned axes: pass the persona's
labels and you get a persona-specific direction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audience_vectors.schemas import AudienceVector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Pair:
    sample_id: str
    score: float
    features_path: Path


class ContrastiveVectorTrainer:
    """Extract contrastive directions from per-segment TRIBE features.

    Workflow:
        trainer = ContrastiveVectorTrainer(features_dir, vectors_dir)
        vector = trainer.train(
            target="memorability",
            scored_segments=[(sample_id, score), ...],
            model_id="facebook/tribev2",
            layer="cortical_output_mean",
            top_k_frac=0.30,
        )
    """

    def __init__(
        self,
        *,
        features_dir: Path,
        vectors_dir: Path,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.vectors_dir = Path(vectors_dir)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        *,
        target: str,
        scored_segments: list[tuple[str, float]],
        model_id: str,
        layer: str = "cortical_output_mean",
        top_k_frac: float = 0.30,
        min_set_size: int = 3,
        normalize_direction: bool = True,
        notes: str = "",
    ) -> AudienceVector:
        """Compute and persist one contrastive direction.

        Args:
            target: name of the axis (e.g. "memorability", "fast_scroll_attention").
            scored_segments: (sample_id, score) pairs. Higher score = positive set.
            model_id: identifier of the source model the features came from.
            layer: short tag describing which layer/component these features represent.
            top_k_frac: take this fraction at each tail. 0.30 = top 30% vs bottom 30%.
            min_set_size: error out if either tail has fewer than this many segments.
            normalize_direction: if True, return a unit-norm direction. The norm
                is preserved in metadata for debugging.
            notes: free-form note saved alongside the vector.
        """
        pairs = self._collect_pairs(scored_segments)
        if len(pairs) < 2 * min_set_size:
            raise ValueError(
                f"need at least {2 * min_set_size} scored segments with features, "
                f"got {len(pairs)}"
            )

        pairs.sort(key=lambda p: p.score, reverse=True)
        n_each = max(min_set_size, int(len(pairs) * top_k_frac))
        positive = pairs[:n_each]
        negative = pairs[-n_each:]
        if positive[-1].score <= negative[0].score:
            logger.warning(
                "target=%s: positive/negative sets overlap by score "
                "(pos_min=%.3f, neg_max=%.3f)",
                target, positive[-1].score, negative[0].score,
            )

        pos_mean = self._mean_features(positive)
        neg_mean = self._mean_features(negative)
        direction = pos_mean - neg_mean
        raw_norm = float(np.linalg.norm(direction))
        if normalize_direction and raw_norm > 1e-12:
            direction = direction / raw_norm

        vector_id = f"{model_id.replace('/', '__')}__{layer}__{target}"
        direction_path = self.vectors_dir / f"{vector_id}.npz"
        np.savez_compressed(
            direction_path,
            direction=direction.astype(np.float32),
            pos_mean=pos_mean.astype(np.float32),
            neg_mean=neg_mean.astype(np.float32),
            pos_ids=np.array([p.sample_id for p in positive]),
            neg_ids=np.array([p.sample_id for p in negative]),
            raw_norm=np.array(raw_norm, dtype=np.float32),
        )

        meta_path = direction_path.with_suffix(".json")
        meta_path.write_text(json.dumps({
            "vector_id": vector_id,
            "target": target,
            "model_id": model_id,
            "layer": layer,
            "dim": int(direction.shape[0]),
            "positive_set_size": len(positive),
            "negative_set_size": len(negative),
            "raw_norm": raw_norm,
            "top_k_frac": top_k_frac,
            "normalized": normalize_direction,
            "positive_ids": [p.sample_id for p in positive],
            "negative_ids": [p.sample_id for p in negative],
            "notes": notes,
        }, indent=2))

        return AudienceVector(
            vector_id=vector_id,
            target=target,
            model_id=model_id,
            layer=layer,
            direction_uri=str(direction_path),
            dim=int(direction.shape[0]),
            positive_set_size=len(positive),
            negative_set_size=len(negative),
            notes=notes,
        )

    # -- helpers ----------------------------------------------------------

    def _collect_pairs(
        self,
        scored_segments: list[tuple[str, float]],
    ) -> list[_Pair]:
        pairs: list[_Pair] = []
        for sample_id, score in scored_segments:
            path = self.features_dir / f"{sample_id}.npz"
            if not path.exists():
                logger.debug("skipping %s: features file missing at %s", sample_id, path)
                continue
            pairs.append(_Pair(sample_id=sample_id, score=float(score), features_path=path))
        return pairs

    def _mean_features(self, group: list[_Pair]) -> np.ndarray:
        accum: np.ndarray | None = None
        n = 0
        for p in group:
            payload = np.load(p.features_path, allow_pickle=False)
            # TRIBE saves under `frames` (T, V); V-JEPA under `embedding` (V,).
            if "frames" in payload.files:
                arr = np.asarray(payload["frames"], dtype=np.float32)
                vec = arr.mean(axis=0) if arr.ndim == 2 else arr
            elif "embedding" in payload.files:
                vec = np.asarray(payload["embedding"], dtype=np.float32)
            else:
                raise ValueError(
                    f"feature file {p.features_path} has neither 'frames' nor 'embedding'"
                )
            if accum is None:
                accum = np.zeros_like(vec)
            accum += vec
            n += 1
        if accum is None or n == 0:
            raise ValueError("no usable features in group")
        return accum / n


def project_features(
    frames: np.ndarray,
    direction: np.ndarray,
    *,
    time_reduce: str = "mean",
) -> float:
    """Project a per-frame feature tensor onto a contrastive direction.

    `frames` shape (T, V): per-frame activations
    `direction` shape (V,): contrastive vector

    Returns a scalar projection score (the segment-level score along this axis).
    """
    if frames.ndim == 1:
        vec = frames
    elif time_reduce == "mean":
        vec = frames.mean(axis=0)
    elif time_reduce == "max":
        vec = frames.max(axis=0)
    else:
        raise ValueError(f"unknown time_reduce: {time_reduce!r}")
    return float(np.dot(vec, direction))
