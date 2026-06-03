"""Search space definition for the BO loop.

x = (α_1, ..., α_k, guidance_scale, seed_id)
    ├── α_i ∈ [alpha_min, alpha_max]   continuous — steering coefficients on TRIBE directions
    ├── guidance_scale ∈ [1, 15]       continuous — classifier-free guidance
    └── seed_id ∈ {0, ..., n_seeds-1}  discrete categorical — generator noise seed
"""

from __future__ import annotations

import torch
from botorch.utils.sampling import manual_seed
from torch import Tensor

from bo_mem.config import BOConfig


class SearchSpace:
    """Defines bounds and utilities for the mixed continuous+categorical search space."""

    def __init__(self, cfg: BOConfig) -> None:
        self.cfg = cfg
        self.k = cfg.k_directions

        # Continuous bounds: shape (2, k+1)
        # rows = [lower, upper]; cols = [α_1..α_k, guidance_scale]
        lowers = [cfg.alpha_min] * cfg.k_directions + [cfg.guidance_min]
        uppers = [cfg.alpha_max] * cfg.k_directions + [cfg.guidance_max]
        self.bounds_continuous: Tensor = torch.tensor([lowers, uppers], dtype=torch.double)

        # Categorical: seed_id is a one-hot index ∈ {0, ..., n_seeds-1}
        self.n_seeds = cfg.n_seeds

    @property
    def dim_continuous(self) -> int:
        return self.cfg.dim_continuous

    @property
    def dim_total(self) -> int:
        return self.cfg.dim_total

    def sobol_initial_points(self, n: int, seed: int = 42) -> Tensor:
        """Generate n quasi-random initial points via Sobol sequence."""
        from botorch.utils.sampling import draw_sobol_samples

        with manual_seed(seed):
            # Sobol over continuous dims
            raw = draw_sobol_samples(
                bounds=self.bounds_continuous, n=n, q=1, seed=seed
            ).squeeze(1)  # (n, k+1)

        # Sample seed_id uniformly
        seed_ids = torch.randint(0, self.n_seeds, (n, 1), dtype=torch.double)
        return torch.cat([raw, seed_ids], dim=1)  # (n, k+2)

    def unnormalize(self, x_unit: Tensor) -> Tensor:
        """Map from [0,1]^(k+1) to actual continuous bounds (no-op for seed_id)."""
        lo = self.bounds_continuous[0]
        hi = self.bounds_continuous[1]
        continuous = x_unit[..., : self.dim_continuous] * (hi - lo) + lo
        if x_unit.shape[-1] > self.dim_continuous:
            seed_id = x_unit[..., self.dim_continuous :]
            return torch.cat([continuous, seed_id], dim=-1)
        return continuous

    def normalize(self, x: Tensor) -> Tensor:
        """Map continuous dims to [0,1]; pass seed_id through."""
        lo = self.bounds_continuous[0]
        hi = self.bounds_continuous[1]
        continuous_norm = (x[..., : self.dim_continuous] - lo) / (hi - lo)
        if x.shape[-1] > self.dim_continuous:
            seed_id = x[..., self.dim_continuous :]
            return torch.cat([continuous_norm, seed_id], dim=-1)
        return continuous_norm

    def split(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Split x into (alphas, guidance_scale, seed_id)."""
        alphas = x[..., : self.k]
        guidance = x[..., self.k : self.k + 1]
        seed_id = x[..., self.k + 1 :].long()
        return alphas, guidance, seed_id

    def seed_id_to_int(self, x: Tensor) -> int:
        return int(x[..., -1].item()) % self.n_seeds
