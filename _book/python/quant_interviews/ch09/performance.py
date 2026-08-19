from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _long_run_variance(returns: FloatArray, hac_lags: int) -> float:
    values = np.asarray(returns, dtype=float)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or not 0 <= hac_lags < values.size
    ):
        raise ValueError("invalid HAC inputs")
    centered = values - values.mean()
    variance = float(centered @ centered / values.size)
    for lag in range(1, hac_lags + 1):
        covariance = float(
            centered[lag:] @ centered[:-lag] / values.size
        )
        variance += 2.0 * (1.0 - lag / (hac_lags + 1)) * covariance
    if variance <= 0.0:
        raise ValueError("long-run variance must be positive")
    return variance


def sharpe_ratio(
    returns: FloatArray,
    periods: int = 252,
    hac_lags: int = 0,
) -> float:
    values = np.asarray(returns, dtype=float)
    if periods <= 0:
        raise ValueError("periods must be positive")
    variance = _long_run_variance(values, hac_lags)
    return float(values.mean() * periods / np.sqrt(variance * periods))


def information_ratio(
    active_returns: FloatArray,
    periods: int = 252,
    hac_lags: int = 0,
) -> float:
    return sharpe_ratio(active_returns, periods, hac_lags)


def spectral_effective_rank(correlation: FloatArray) -> float:
    matrix = _validate_correlation(correlation)
    return float(np.trace(matrix) ** 2 / np.trace(matrix @ matrix))


def effective_breadth(correlation: FloatArray) -> float:
    """Compatibility alias for spectral participation-ratio breadth."""
    return spectral_effective_rank(correlation)


def directional_effective_breadth(
    allocations: FloatArray,
    correlation: FloatArray,
) -> float:
    weights = np.asarray(allocations, dtype=float)
    matrix = _validate_correlation(correlation)
    if (
        weights.shape != (matrix.shape[0],)
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError("allocation shape or values are invalid")
    denominator = float(weights @ matrix @ weights)
    if denominator <= 0.0:
        raise ValueError("directional breadth denominator is zero")
    return float(weights.sum() ** 2 / denominator)


def _validate_correlation(correlation: FloatArray) -> FloatArray:
    matrix = np.asarray(correlation, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-10)
        or not np.allclose(np.diag(matrix), 1.0, atol=1e-10, rtol=1e-10)
        or float(np.linalg.eigvalsh(matrix).min()) < -1e-10
    ):
        raise ValueError("correlation must be finite, symmetric, PSD and unit diagonal")
    return matrix


def transfer_coefficient(
    constrained: FloatArray,
    unconstrained: FloatArray,
    cov: FloatArray,
) -> float:
    first = np.asarray(constrained, dtype=float)
    second = np.asarray(unconstrained, dtype=float)
    matrix = np.asarray(cov, dtype=float)
    if (
        first.shape != second.shape
        or matrix.shape != (first.size, first.size)
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-10)
        or float(np.linalg.eigvalsh(matrix).min()) < -1e-10
    ):
        raise ValueError("invalid transfer-coefficient inputs")
    first_norm = float(first @ matrix @ first)
    second_norm = float(second @ matrix @ second)
    if first_norm <= 0.0 or second_norm <= 0.0:
        raise ValueError("transfer coefficient has zero denominator")
    value = float(
        first @ matrix @ second / np.sqrt(first_norm * second_norm)
    )
    return float(np.clip(value, -1.0, 1.0))
