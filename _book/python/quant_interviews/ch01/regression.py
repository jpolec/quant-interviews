"""Regression uncertainty calculations for Chapter 1."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike


def newey_west_covariance(
    design: ArrayLike,
    residuals: ArrayLike,
    max_lag: int,
) -> np.ndarray:
    """Return the Bartlett-kernel Newey-West sandwich covariance."""

    regressors = np.asarray(design, dtype=float)
    errors = np.asarray(residuals, dtype=float)
    shape_invalid = (
        regressors.ndim != 2
        or errors.ndim != 1
        or errors.size != regressors.shape[0]
        or regressors.shape[0] == 0
        or regressors.shape[1] == 0
        or isinstance(max_lag, bool)
        or not isinstance(max_lag, Integral)
        or max_lag < 0
    )
    if shape_invalid:
        raise ValueError(
            "aligned design/residuals and nonnegative lag are required"
        )
    if not np.all(np.isfinite(regressors)) or not np.all(np.isfinite(errors)):
        raise ValueError("design and residuals must contain only finite values")
    if np.linalg.matrix_rank(regressors) < regressors.shape[1]:
        raise ValueError("design matrix must have full column rank")

    observations = regressors.shape[0]
    score = regressors * errors[:, None]
    meat = score.T @ score
    for lag in range(1, min(max_lag, observations - 1) + 1):
        bartlett_weight = 1.0 - lag / (max_lag + 1.0)
        lag_covariance = score[lag:].T @ score[:-lag]
        meat += bartlett_weight * (
            lag_covariance + lag_covariance.T
        )
    cross_product = regressors.T @ regressors
    bread = np.linalg.solve(
        cross_product,
        np.eye(cross_product.shape[0]),
    )
    return bread @ meat @ bread
