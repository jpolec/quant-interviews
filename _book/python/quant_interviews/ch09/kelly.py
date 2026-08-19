from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _validated_predictive_second_moment(
    mean: FloatArray,
    covariance: FloatArray,
    posterior_mean_covariance: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    expected = np.asarray(mean, dtype=float)
    return_covariance = np.asarray(covariance, dtype=float)
    uncertainty = (
        np.zeros_like(return_covariance)
        if posterior_mean_covariance is None
        else np.asarray(posterior_mean_covariance, dtype=float)
    )
    if (
        expected.ndim != 1
        or return_covariance.shape != (expected.size, expected.size)
        or uncertainty.shape != return_covariance.shape
        or not np.isfinite(expected).all()
        or not np.isfinite(return_covariance).all()
        or not np.isfinite(uncertainty).all()
    ):
        raise ValueError("invalid predictive second-moment inputs")
    for matrix, name, strict in (
        (return_covariance, "return covariance", True),
        (uncertainty, "posterior mean covariance", False),
    ):
        scale = max(float(np.linalg.norm(matrix, ord=2)), 1.0)
        if not np.allclose(
            matrix,
            matrix.T,
            atol=1e-12 * scale,
            rtol=1e-10,
        ):
            raise ValueError(f"{name} must be symmetric")
        minimum = float(np.linalg.eigvalsh(matrix).min())
        if minimum < -1e-12 * scale or (strict and minimum <= 0.0):
            raise ValueError(f"{name} has invalid spectrum")
    second_moment = (
        return_covariance
        + uncertainty
        + np.outer(expected, expected)
    )
    minimum = float(np.linalg.eigvalsh(second_moment).min())
    if minimum <= 0.0:
        raise ValueError("predictive second moment must be positive definite")
    return expected, second_moment


def scalar_kelly(
    mean: float,
    variance: float,
    posterior_mean_variance: float = 0.0,
    fraction: float = 1.0,
) -> float:
    if (
        not np.isfinite(mean)
        or not np.isfinite(variance)
        or not np.isfinite(posterior_mean_variance)
        or variance <= 0.0
        or posterior_mean_variance < 0.0
        or not 0.0 < fraction <= 1.0
    ):
        raise ValueError("invalid Kelly inputs")
    # E[R^2] = Var(R) + Var(E[R | data]) + E[R]^2.
    second_moment = variance + posterior_mean_variance + mean * mean
    return fraction * mean / second_moment


def multivariate_kelly(
    mean: FloatArray,
    cov: FloatArray,
    posterior_mean_cov: FloatArray | None = None,
    fraction: float = 1.0,
) -> FloatArray:
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("invalid multivariate Kelly inputs")
    expected, second_moment = _validated_predictive_second_moment(
        mean,
        cov,
        posterior_mean_cov,
    )
    return fraction * np.linalg.solve(second_moment, expected)


def risk_adjusted_certainty_equivalent(
    mean: FloatArray,
    predictive_covariance: FloatArray,
) -> float:
    expected, second_moment = _validated_predictive_second_moment(
        mean,
        predictive_covariance,
    )
    weights = np.linalg.solve(second_moment, expected)
    # Maximum of w'm - 0.5 w'Mw at w=M^{-1}m.
    return float(0.5 * expected @ weights)


def simulate_wealth(
    weights: FloatArray,
    returns: FloatArray,
    initial: float = 1.0,
) -> FloatArray:
    portfolio_weights = np.asarray(weights, dtype=float)
    scenarios = np.asarray(returns, dtype=float)
    if (
        portfolio_weights.ndim != 1
        or scenarios.ndim != 2
        or scenarios.shape[1] != portfolio_weights.size
        or not np.isfinite(portfolio_weights).all()
        or not np.isfinite(scenarios).all()
        or not np.isfinite(initial)
        or initial <= 0.0
    ):
        raise ValueError("invalid wealth simulation inputs")
    growth = scenarios @ portfolio_weights
    if np.any(1.0 + growth <= 0.0):
        raise ValueError("nonpositive wealth state")
    return initial * np.cumprod(1.0 + growth)
