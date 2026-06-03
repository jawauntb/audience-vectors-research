"""Best-of-N baseline (Condition C1 from the proposal).

Generate N random variants per seed, score each with TRIBE on v_mem,
return the highest-scoring one. This is the strong baseline that the
multi-objective BO must outperform.

This also handles Condition C2 (best-of-N + alpha-steering at α=+10)
to confirm the null result from §6.9.1 of Brown (2026).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from rich.console import Console
from torch import Tensor

from bo_mem.objectives.clip_scorer import CLIPScorer
from bo_mem.objectives.tribe_scorer import TribeScorer

console = Console()


@dataclass
class BestOfNResult:
    seed_id: str
    winner_path: Path
    winner_mem_score: float
    winner_clip_score: float
    median_mem_score: float
    lift: float  # winner - median memorability
    all_scores: list[float]
    n: int


class BestOfNBaseline:
    """Replicates the §6.7 best-of-N experiment from Brown (2026)."""

    def __init__(
        self,
        tribe_scorer: TribeScorer,
        clip_scorer: CLIPScorer,
        prompt: str,
    ) -> None:
        self.tribe_scorer = tribe_scorer
        self.clip_scorer = clip_scorer
        self.prompt = prompt

    def run_from_existing(
        self,
        seed_dir: Path,
        seed_id: str,
        n_variants: int = 10,
    ) -> BestOfNResult:
        """Score pre-generated variants in a directory.

        Expects files named: {seed_id}_n00.mp4, ..., {seed_id}_n{N-1:02d}.mp4
        This matches the format of audience_vectors_share/generated/svd_best_of_n/.
        """
        paths = sorted(seed_dir.glob(f"{seed_id}_n*.mp4"))[:n_variants]
        if not paths:
            raise FileNotFoundError(f"No variants found in {seed_dir} for seed {seed_id}")

        mem_scores = [self.tribe_scorer.score(p) for p in paths]
        best_idx = int(torch.tensor(mem_scores).argmax().item())
        median_score = float(torch.tensor(mem_scores).median().item())
        winner_path = paths[best_idx]
        clip_score = self.clip_scorer.score(winner_path, self.prompt)

        return BestOfNResult(
            seed_id=seed_id,
            winner_path=winner_path,
            winner_mem_score=mem_scores[best_idx],
            winner_clip_score=clip_score,
            median_mem_score=median_score,
            lift=mem_scores[best_idx] - median_score,
            all_scores=mem_scores,
            n=len(paths),
        )

    def run_all_seeds(
        self,
        generated_dir: Path,
        n_variants: int = 10,
    ) -> list[BestOfNResult]:
        """Run best-of-N for all seeds found in generated_dir."""
        # Discover seeds from filenames like vid_idx0001_n00.mp4
        seed_ids = sorted({
            "_".join(p.stem.split("_")[:-1])
            for p in generated_dir.glob("*_n*.mp4")
        })

        results: list[BestOfNResult] = []
        for sid in seed_ids:
            console.log(f"Scoring seed: {sid}")
            r = self.run_from_existing(generated_dir, sid, n_variants)
            console.log(f"  lift={r.lift:+.3f} winner={r.winner_mem_score:.3f} median={r.median_mem_score:.3f}")
            results.append(r)

        return results

    def summarize(self, results: list[BestOfNResult]) -> dict[str, float]:
        lifts = [r.lift for r in results]
        mem_scores = [r.winner_mem_score for r in results]
        return {
            "mean_lift": float(torch.tensor(lifts).mean().item()),
            "std_lift": float(torch.tensor(lifts).std().item()),
            "mean_winner_mem": float(torch.tensor(mem_scores).mean().item()),
            "n_seeds": len(results),
        }
