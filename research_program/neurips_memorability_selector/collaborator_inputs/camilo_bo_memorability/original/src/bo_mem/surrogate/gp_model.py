"""GP surrogate models for the multi-objective BO.

Three variants, selected by the loop depending on available data:
  - SingleTaskGP: clean baseline, homoscedastic noise
  - HeteroskedasticGP: models noise as a function of x (§4.6 of proposal)
  - MixedGP: handles continuous + categorical (seed_id) inputs (§4.5)
"""

from __future__ import annotations

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import SumMarginalLogLikelihood
from torch import Tensor


def build_single_task_gp(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
) -> SingleTaskGP:
    """Vanilla GP for homoscedastic, continuous-only inputs.

    Args:
        train_x: (N, D) normalized to [0,1]^D
        train_y: (N, M) — M objectives
        bounds:  (2, D) original bounds (used for Normalize transform)
    """
    d = train_x.shape[-1]
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=Normalize(d=d, bounds=bounds),
        outcome_transform=Standardize(m=train_y.shape[-1]),
    )
    return model


def build_mixed_gp(
    train_x: Tensor,
    train_y: Tensor,
    cat_dims: list[int],
    bounds: Tensor,
) -> MixedSingleTaskGP:
    """GP with product kernel for mixed continuous + categorical inputs.

    cat_dims: list of column indices that are categorical (e.g., [k+1] for seed_id)
    Uses a Hamming kernel for categorical dims and Matern-5/2 for continuous.
    """
    cont_dims = [i for i in range(train_x.shape[-1]) if i not in cat_dims]
    model = MixedSingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        cat_dims=cat_dims,
        outcome_transform=Standardize(m=train_y.shape[-1]),
    )
    return model


def build_heteroscedastic_gp(
    train_x: Tensor,
    train_y: Tensor,
    train_yvar: Tensor,
    bounds: Tensor,
) -> SingleTaskGP:
    """GP with known observation noise (train_Yvar).

    HeteroskedasticSingleTaskGP was removed in botorch>=0.12.
    Uses SingleTaskGP with train_Yvar, which has equivalent behavior.
    """
    d = train_x.shape[-1]
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=train_yvar,
        input_transform=Normalize(d=d, bounds=bounds),
    )
    return model


def build_model_list(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    cat_dims: list[int] | None = None,
) -> ModelListGP:
    """Build a ModelListGP with one GP per objective.

    BoTorch's qEHVI requires a ModelListGP (one model per output).
    """
    m = train_y.shape[-1]
    models = []
    for j in range(m):
        y_j = train_y[:, j : j + 1]
        if cat_dims:
            gp = build_mixed_gp(train_x, y_j, cat_dims, bounds)
        else:
            gp = build_single_task_gp(train_x, y_j, bounds)
        models.append(gp)
    return ModelListGP(*models)


def fit_model(model: ModelListGP | SingleTaskGP | MixedSingleTaskGP) -> None:
    """Fit hyperparameters via marginal log-likelihood maximization."""
    if isinstance(model, ModelListGP):
        mll = SumMarginalLogLikelihood(model.likelihood, model)
    else:
        from gpytorch.mlls import ExactMarginalLogLikelihood

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)


def compute_holdout_metrics(
    model: SingleTaskGP,
    test_x: Tensor,
    test_y: Tensor,
) -> dict[str, float]:
    """Compute R² and mean negative log-likelihood on a hold-out set."""
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(test_x)
        pred_mean = posterior.mean  # (N, M)
        pred_var = posterior.variance.clamp_min(1e-9)

    y_mean = test_y.mean(dim=0)
    ss_res = ((test_y - pred_mean) ** 2).sum(dim=0)
    ss_tot = ((test_y - y_mean) ** 2).sum(dim=0)
    r2 = (1 - ss_res / ss_tot.clamp_min(1e-9)).mean().item()

    # Gaussian NLL
    nll = (0.5 * ((test_y - pred_mean) ** 2 / pred_var + pred_var.log() + torch.log(torch.tensor(2 * 3.14159)))).mean().item()

    return {"r2": r2, "nll": nll}
