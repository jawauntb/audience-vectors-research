"""qEHVI acquisition function — the core of multi-objective BO.

q-Expected Hypervolume Improvement selects the next batch of q candidates
by maximizing the expected improvement in the dominated hypervolume of the
Pareto frontier.

References:
  Daulton et al. (2020) NeurIPS — differentiable qEHVI
  Daulton et al. (2021) NeurIPS — noisy qEHVI (qNEHVI)
"""

from __future__ import annotations

import torch
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models.model import Model
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor


def get_reference_point(train_y: Tensor, slack: float = 0.1) -> Tensor:
    """Compute a reference point slightly below the current Pareto nadir.

    The reference point must be dominated by all Pareto-optimal points.
    Slack of 0.1 = 10% below the minimum observed value per objective.
    """
    mins = train_y.min(dim=0).values
    ranges = train_y.max(dim=0).values - mins
    return mins - slack * ranges.clamp_min(1e-6)


def optimize_qehvi(
    model: Model,
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    batch_size: int = 4,
    num_restarts: int = 10,
    raw_samples: int = 512,
    mc_samples: int = 128,
    cat_dims: list[int] | None = None,
    n_seeds: int | None = None,
) -> Tensor:
    """Optimize qEHVI to select the next batch of candidates.

    Args:
        model:       Fitted ModelListGP (one GP per objective)
        train_y:     (N, M) observed objective values — used to build Pareto partitioning
        bounds:      (2, D) search space bounds in original scale
        batch_size:  q — number of candidates per iteration
        cat_dims:    indices of categorical dimensions (seed_id) if any
        n_seeds:     number of discrete seed values (needed for mixed optimization)

    Returns:
        candidates: (batch_size, D) tensor of next points to evaluate
    """
    ref_point = get_reference_point(train_y)

    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=train_x,
        prune_baseline=True,
    )

    if cat_dims and n_seeds is not None:
        # Mixed optimization: continuous + discrete seed_id
        fixed_features_list = [{dim: float(v) for dim in cat_dims for v in range(n_seeds)}]
        candidates, _ = optimize_acqf_mixed(
            acq_function=acqf,
            bounds=bounds,
            fixed_features_list=[{cat_dims[0]: float(s)} for s in range(n_seeds)],
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    else:
        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return candidates.detach()


def compute_hypervolume(train_y: Tensor, ref_point: Tensor | None = None) -> float:
    """Compute dominated hypervolume of the current Pareto frontier."""
    if ref_point is None:
        ref_point = get_reference_point(train_y)

    pareto_mask = is_non_dominated(train_y)
    pareto_y = train_y[pareto_mask]

    if len(pareto_y) == 0:
        return 0.0

    bd = FastNondominatedPartitioning(ref_point=ref_point, Y=pareto_y)
    return float(bd.compute_hypervolume().item())


def get_pareto_front(train_y: Tensor) -> Tensor:
    """Return the non-dominated subset of observed objective values."""
    mask = is_non_dominated(train_y)
    return train_y[mask]
