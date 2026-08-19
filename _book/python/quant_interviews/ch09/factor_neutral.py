from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NeutralizationResult:
    weights: FloatArray
    constraint_residual: FloatArray
    singular_values: FloatArray
    effective_rank: int
    dropped_directions: int


def euclidean_residualize(
    alpha: FloatArray,
    exposures: FloatArray,
    *,
    rcond: float = 1e-10,
) -> FloatArray:
    signal = np.asarray(alpha, dtype=float)
    matrix = np.asarray(exposures, dtype=float)
    if (
        signal.ndim != 1
        or matrix.ndim != 2
        or matrix.shape[0] != signal.size
        or not np.isfinite(signal).all()
        or not np.isfinite(matrix).all()
        or not np.isfinite(rcond)
        or rcond <= 0.0
    ):
        raise ValueError("invalid alpha or exposure matrix")
    return signal - matrix @ (np.linalg.pinv(matrix, rcond=rcond) @ signal)


def constrained_neutral_portfolio(
    alpha: FloatArray,
    cov: FloatArray,
    constraints: FloatArray,
    targets: FloatArray,
    *,
    rcond: float = 1e-10,
) -> NeutralizationResult:
    signal = np.asarray(alpha, dtype=float)
    covariance = np.asarray(cov, dtype=float)
    mapping = np.asarray(constraints, dtype=float)
    target = np.asarray(targets, dtype=float)
    if (
        signal.ndim != 1
        or covariance.shape != (signal.size, signal.size)
        or mapping.ndim != 2
        or mapping.shape[0] != signal.size
        or target.shape != (mapping.shape[1],)
    ):
        raise ValueError("constraint dimensions are inconsistent")
    if not (
        np.isfinite(signal).all()
        and np.isfinite(covariance).all()
        and np.isfinite(mapping).all()
        and np.isfinite(target).all()
    ):
        raise ValueError("inputs must be finite")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
        raise ValueError("covariance must be symmetric")
    if float(np.linalg.eigvalsh(covariance).min()) <= 0.0:
        raise ValueError("covariance must be positive definite")

    inv_alpha = np.linalg.solve(covariance, signal)
    inv_constraints = np.linalg.solve(covariance, mapping)
    schur = mapping.T @ inv_constraints
    singular_values = np.linalg.svd(schur, compute_uv=False)
    threshold = rcond * (singular_values[0] if singular_values.size else 1.0)
    rank = int(np.sum(singular_values > threshold))
    multipliers = np.linalg.pinv(schur, rcond=rcond) @ (
        mapping.T @ inv_alpha - target
    )
    weights = inv_alpha - inv_constraints @ multipliers
    residual = mapping.T @ weights - target
    if float(np.max(np.abs(residual), initial=0.0)) > 1e-8:
        raise ValueError("constraint system is infeasible or rank deficient")
    return NeutralizationResult(
        weights=weights,
        constraint_residual=residual,
        singular_values=singular_values,
        effective_rank=rank,
        dropped_directions=int(mapping.shape[1] - rank),
    )


def covariance_aware_neutral(
    alpha: FloatArray,
    cov: FloatArray,
    exposures: FloatArray,
) -> FloatArray:
    mapping = np.asarray(exposures, dtype=float)
    if mapping.ndim != 2:
        raise ValueError("exposures must be a two-dimensional matrix")
    result = constrained_neutral_portfolio(
        alpha,
        cov,
        mapping,
        np.zeros(mapping.shape[1]),
    )
    return result.weights


def exposure_residual(
    weights: FloatArray,
    exposures: FloatArray,
) -> FloatArray:
    mapping = np.asarray(exposures, dtype=float)
    portfolio = np.asarray(weights, dtype=float)
    if (
        mapping.ndim != 2
        or portfolio.ndim != 1
        or mapping.shape[0] != portfolio.size
        or not np.isfinite(mapping).all()
        or not np.isfinite(portfolio).all()
    ):
        raise ValueError("invalid exposure-residual inputs")
    return mapping.T @ portfolio
