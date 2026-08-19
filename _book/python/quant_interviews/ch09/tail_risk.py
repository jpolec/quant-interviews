from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.stats import norm

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CVaROptimizationResult:
    weights: FloatArray
    var_threshold: float
    expected_shortfall: float
    budget_residual: float
    status: str


def _tail_weights(losses: FloatArray, alpha: float) -> tuple[float, FloatArray]:
    values = np.asarray(losses, dtype=float)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("invalid loss sample or alpha")
    # `inverted_cdf` implements inf{x: F_n(x) >= alpha}, matching the text.
    var = float(np.quantile(values, alpha, method="inverted_cdf"))
    tail_observations = values.size * (1.0 - alpha)
    conditional_weights = np.zeros(values.size)
    tolerance = 1e-12 * max(abs(var), float(np.max(np.abs(values))), 1.0)
    strictly_above = values > var + tolerance
    at_threshold = np.abs(values - var) <= tolerance
    above_count = int(strictly_above.sum())
    remaining_mass = tail_observations - above_count
    if remaining_mass < -1e-10 or not np.any(at_threshold):
        raise RuntimeError("empirical tail allocation is inconsistent")
    conditional_weights[strictly_above] = 1.0 / tail_observations
    if remaining_mass > 1e-14:
        # Equal allocation across a VaR atom is permutation invariant.
        conditional_weights[at_threshold] = (
            remaining_mass / int(at_threshold.sum()) / tail_observations
        )
    if not np.isclose(
        conditional_weights.sum(),
        1.0,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("empirical ES weights do not sum to one")
    return var, conditional_weights


def historical_var_es(
    losses: FloatArray,
    alpha: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(losses, dtype=float)
    var, weights = _tail_weights(values, alpha)
    return var, float(weights @ values)


def normal_var_es(
    mean: float,
    sigma: float,
    alpha: float = 0.95,
) -> tuple[float, float]:
    if (
        not np.isfinite(mean)
        or not np.isfinite(sigma)
        or sigma < 0.0
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("invalid normal-loss parameters")
    z = float(norm.ppf(alpha))
    return (
        mean + sigma * z,
        mean + sigma * float(norm.pdf(z)) / (1.0 - alpha),
    )


def cvar_objective(
    losses: FloatArray,
    z: float,
    alpha: float = 0.95,
) -> float:
    values = np.asarray(losses, dtype=float)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or not np.isfinite(z)
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("invalid CVaR objective inputs")
    return float(z + np.maximum(values - z, 0.0).mean() / (1.0 - alpha))


def es_contributions(
    asset_losses: FloatArray,
    weights: FloatArray,
    alpha: float = 0.95,
) -> FloatArray:
    scenarios = np.asarray(asset_losses, dtype=float)
    portfolio_weights = np.asarray(weights, dtype=float)
    if (
        scenarios.ndim != 2
        or portfolio_weights.shape != (scenarios.shape[1],)
        or not np.isfinite(scenarios).all()
        or not np.isfinite(portfolio_weights).all()
    ):
        raise ValueError("scenario or weight shape is invalid")
    total = scenarios @ portfolio_weights
    _, tail_weights = _tail_weights(total, alpha)
    return portfolio_weights * (tail_weights @ scenarios)


def optimize_cvar(
    asset_losses: FloatArray,
    alpha: float = 0.95,
) -> CVaROptimizationResult:
    scenarios = np.asarray(asset_losses, dtype=float)
    if (
        scenarios.ndim != 2
        or scenarios.shape[0] < 2
        or not np.isfinite(scenarios).all()
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("invalid CVaR optimization inputs")
    sample_count, asset_count = scenarios.shape
    objective = np.concatenate(
        (
            np.zeros(asset_count),
            np.ones(1),
            np.full(sample_count, 1.0 / (sample_count * (1.0 - alpha))),
        )
    )
    # scenario_loss @ w - z - u <= 0
    inequality = np.hstack(
        (
            scenarios,
            -np.ones((sample_count, 1)),
            -np.eye(sample_count),
        )
    )
    equality = np.zeros((1, asset_count + 1 + sample_count))
    equality[0, :asset_count] = 1.0
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.zeros(sample_count),
        A_eq=equality,
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * asset_count
        + [(None, None)]
        + [(0.0, None)] * sample_count,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.asarray(result.x[:asset_count])
    portfolio_losses = scenarios @ weights
    _, expected_shortfall = historical_var_es(portfolio_losses, alpha)
    return CVaROptimizationResult(
        weights=weights,
        var_threshold=float(result.x[asset_count]),
        expected_shortfall=expected_shortfall,
        budget_residual=abs(float(weights.sum()) - 1.0),
        status="PASS",
    )
