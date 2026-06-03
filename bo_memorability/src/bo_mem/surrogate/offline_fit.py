"""Offline GP validation using persona_scores.csv (F2 of the proposal).

Pipeline:
  1. Load 1022-clip dataset with 12 persona scores + human memorability
  2. PCA-compress persona scores to 4 components (effective rank = 3.56/12)
  3. Use (PC1, PC2, PC3, PC4) as x — the latent audience-response space
  4. Use bmd_human_memorability as y1 (objective 1 proxy)
  5. Train SingleTaskGP on 80% / evaluate on 20%
  6. Report R², NLL and Pareto frontier visualization

This validates the GP can learn the memorability landscape BEFORE we run
the expensive closed-loop with GPU inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from bo_mem.surrogate.gp_model import (
    build_model_list,
    build_single_task_gp,
    compute_holdout_metrics,
    fit_model,
)

PERSONA_COLS = [
    "ad-blocker-priya", "bass-drop-reyna", "deep-dive-felix", "drama-thread-nico",
    "frame-poet-cleo", "giggle-loop-mara", "golden-hour-vance", "highlight-hunter-dex",
    "lore-keeper-syd", "spec-sheet-sam", "swipe-king-zara", "tearjerker-theo",
]


def load_persona_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load persona_scores.csv → (X_raw [1022, 12], y [1022, 2]).

    y[:, 0] = bmd_human_memorability (objective 1)
    y[:, 1] = BMD_human_global (objective 2 proxy — raw engagement score)
    """
    df = pd.read_csv(csv_path)
    X = df[PERSONA_COLS].values.astype(np.float32)
    y = df[["bmd_human_memorability", "BMD_human_global"]].values.astype(np.float32)
    return X, y


def pca_compress(
    X: np.ndarray,
    n_components: int = 4,
    scaler: StandardScaler | None = None,
    pca: PCA | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, StandardScaler, PCA]:
    """Compress 12-dim persona space to n_components via PCA."""
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
    else:
        assert scaler is not None and pca is not None
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
    return X_pca, scaler, pca  # type: ignore[return-value]


def run_offline_validation(
    csv_path: Path,
    n_pca_components: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Full offline GP validation pipeline.

    Returns dict with keys: r2_obj1, nll_obj1, r2_obj2, nll_obj2,
    pca_variance_explained, n_train, n_test.
    """
    X_raw, y = load_persona_dataset(csv_path)

    X_pca, scaler, pca = pca_compress(X_raw, n_components=n_pca_components)
    var_explained = float(pca.explained_variance_ratio_.sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=test_size, random_state=random_state
    )

    train_x = torch.tensor(X_train, dtype=torch.double)
    test_x = torch.tensor(X_test, dtype=torch.double)
    train_y = torch.tensor(y_train, dtype=torch.double)
    test_y = torch.tensor(y_test, dtype=torch.double)

    # Bounds from training data (used for normalization transform)
    bounds = torch.stack([train_x.min(dim=0).values, train_x.max(dim=0).values])

    results: dict[str, float] = {
        "pca_variance_explained": var_explained,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Fit one GP per objective
    for j, obj_name in enumerate(["memorability", "global_engagement"]):
        y_j_train = train_y[:, j : j + 1]
        y_j_test = test_y[:, j : j + 1]

        gp = build_single_task_gp(train_x, y_j_train, bounds)
        fit_model(gp)

        metrics = compute_holdout_metrics(gp, test_x, y_j_test)
        results[f"r2_{obj_name}"] = metrics["r2"]
        results[f"nll_{obj_name}"] = metrics["nll"]

    return results, scaler, pca  # type: ignore[return-value]


def build_offline_surrogate(
    csv_path: Path,
    n_pca_components: int = 4,
) -> tuple:
    """Build a fitted ModelListGP on all data (no split) for use in the loop.

    Returns (model, scaler, pca, bounds) ready for warm-starting the BO.
    """
    X_raw, y = load_persona_dataset(csv_path)
    X_pca, scaler, pca = pca_compress(X_raw, n_components=n_pca_components)

    train_x = torch.tensor(X_pca, dtype=torch.double)
    train_y = torch.tensor(y, dtype=torch.double)
    bounds = torch.stack([train_x.min(dim=0).values, train_x.max(dim=0).values])

    model = build_model_list(train_x, train_y, bounds)
    fit_model(model)

    return model, scaler, pca, bounds
