"""Sparse-allocation primitives and portfolio-specific L1 audits."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize


def _positive_definite_covariance(
    covariance: ArrayLike,
    asset_count: int,
) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if (
        matrix.shape != (asset_count, asset_count)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.T, atol=1e-14, rtol=1e-10)
    ):
        raise ValueError("finite symmetric covariance required")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(
        float(np.max(np.abs(eigenvalues))),
        np.finfo(float).tiny,
    )
    if eigenvalues[0] <= 1e-10 * scale:
        raise ValueError("positive-definite covariance required")
    return matrix


def soft_threshold(values: ArrayLike, penalty: float) -> np.ndarray:
    """Apply the proximal operator of a nonnegative L1 penalty."""

    vector = np.asarray(values, dtype=float)
    if (
        not np.all(np.isfinite(vector))
        or not np.isfinite(penalty)
        or penalty < 0.0
    ):
        raise ValueError("finite values and nonnegative penalty required")
    return np.sign(vector) * np.maximum(np.abs(vector) - penalty, 0.0)


def project_l2_ball(values: ArrayLike, radius: float) -> np.ndarray:
    """Project a finite vector radially onto an L2 ball."""

    vector = np.asarray(values, dtype=float)
    if (
        not np.all(np.isfinite(vector))
        or not np.isfinite(radius)
        or radius < 0.0
    ):
        raise ValueError("finite values and nonnegative radius required")
    norm = float(np.linalg.norm(vector))
    return vector if norm <= radius or norm == 0.0 else vector * radius / norm


def long_only_l1_norm(weights: ArrayLike) -> float:
    """Return the constant holdings L1 norm on the long-only simplex."""

    vector = np.asarray(weights, dtype=float)
    if (
        vector.ndim != 1
        or not np.all(np.isfinite(vector))
        or np.any(vector < 0.0)
        or not np.isclose(vector.sum(), 1.0)
    ):
        raise ValueError("long-only fully invested weights required")
    return float(np.abs(vector).sum())


def long_only_l1_constant_audit(
    covariance: ArrayLike,
    expected_returns: ArrayLike,
    penalties: ArrayLike,
    *,
    risk_aversion: float,
) -> dict[str, float | np.ndarray]:
    """Verify that holdings L1 is constant under long-only full investment."""

    expected = np.asarray(expected_returns, dtype=float)
    penalty_grid = np.asarray(penalties, dtype=float)
    asset_count = expected.size
    matrix = _positive_definite_covariance(covariance, asset_count)
    if (
        not np.all(np.isfinite(expected))
        or penalty_grid.ndim != 1
        or penalty_grid.size < 2
        or not np.all(np.isfinite(penalty_grid))
        or np.any(penalty_grid < 0.0)
        or not np.isfinite(risk_aversion)
        or risk_aversion <= 0.0
    ):
        raise ValueError("valid covariance, returns, penalties and risk required")

    solutions = []
    initial = np.full(asset_count, 1.0 / asset_count)
    for penalty in penalty_grid:
        result = minimize(
            lambda weights: float(
                0.5 * risk_aversion * weights @ matrix @ weights
                - expected @ weights
                + penalty * np.abs(weights).sum()
            ),
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * asset_count,
            constraints={
                "type": "eq",
                "fun": lambda weights: weights.sum() - 1.0,
            },
            options={"ftol": 1e-12, "maxiter": 2_000},
        )
        if not result.success:
            raise RuntimeError(result.message)
        initial = np.asarray(result.x)
        solutions.append(initial.copy())
    solution_matrix = np.vstack(solutions)
    return {
        "weights": solution_matrix,
        "l1_norms": np.array(
            [long_only_l1_norm(weights) for weights in solution_matrix]
        ),
        "maximum_weight_difference": float(
            np.max(np.abs(solution_matrix - solution_matrix[0]))
        ),
    }


def gross_exposure_constrained_portfolio(
    covariance: ArrayLike,
    expected_returns: ArrayLike,
    *,
    risk_aversion: float,
    gross_limit: float,
) -> dict[str, float | np.ndarray]:
    """Solve a long-short portfolio with an explicit gross-exposure limit."""

    expected = np.asarray(expected_returns, dtype=float)
    asset_count = expected.size
    matrix = _positive_definite_covariance(covariance, asset_count)
    if (
        not np.all(np.isfinite(expected))
        or not np.isfinite(risk_aversion)
        or risk_aversion <= 0.0
        or not np.isfinite(gross_limit)
        or gross_limit < 1.0
    ):
        raise ValueError("valid portfolio inputs and gross limit >= 1 required")
    initial_weights = np.full(asset_count, 1.0 / asset_count)
    initial = np.concatenate([initial_weights, np.abs(initial_weights)])
    result = minimize(
        lambda variables: float(
            0.5
            * risk_aversion
            * variables[:asset_count]
            @ matrix
            @ variables[:asset_count]
            - expected @ variables[:asset_count]
        ),
        initial,
        method="SLSQP",
        bounds=[(None, None)] * asset_count + [(0.0, None)] * asset_count,
        constraints=[
            {
                "type": "eq",
                "fun": lambda variables: variables[:asset_count].sum() - 1.0,
            },
            {
                "type": "ineq",
                "fun": lambda variables: (
                    variables[asset_count:] - variables[:asset_count]
                ),
            },
            {
                "type": "ineq",
                "fun": lambda variables: (
                    variables[asset_count:] + variables[:asset_count]
                ),
            },
            {
                "type": "ineq",
                "fun": lambda variables: (
                    gross_limit - variables[asset_count:].sum()
                ),
            },
        ],
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.asarray(result.x[:asset_count])
    return {
        "weights": weights,
        "gross_exposure": float(np.abs(weights).sum()),
        "budget_residual": float(abs(weights.sum() - 1.0)),
    }


def _l2_constrained_portfolio(
    covariance: np.ndarray,
    expected_returns: np.ndarray,
    *,
    risk_aversion: float,
    l2_radius: float,
) -> np.ndarray:
    asset_count = expected_returns.size
    minimum_radius = 1.0 / np.sqrt(asset_count)
    if not np.isfinite(l2_radius) or l2_radius < minimum_radius:
        raise ValueError("L2 radius is infeasible with full investment")
    result = minimize(
        lambda weights: float(
            0.5 * risk_aversion * weights @ covariance @ weights
            - expected_returns @ weights
        ),
        np.full(asset_count, 1.0 / asset_count),
        method="SLSQP",
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: weights.sum() - 1.0,
            },
            {
                "type": "ineq",
                "fun": lambda weights: l2_radius - np.linalg.norm(weights),
            },
        ],
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return np.asarray(result.x)


def allocation_stability_metrics(
    weights: np.ndarray,
    perturbed_weights: np.ndarray,
) -> dict[str, float | int]:
    """Summarize concentration and active-set stability for one norm."""

    active = np.abs(weights) > 1e-7
    perturbed_active = np.abs(perturbed_weights) > 1e-7
    active_union = np.logical_or(active, perturbed_active)
    stability = (
        1.0
        if not np.any(active_union)
        else float(
            np.sum(np.logical_and(active, perturbed_active))
            / np.sum(active_union)
        )
    )
    return {
        "gross_exposure": float(np.abs(weights).sum()),
        "concentration": float(np.sum(weights**2)),
        "near_zero_weights": int(np.sum(~active)),
        "perturbation_turnover": float(
            np.sum(np.abs(perturbed_weights - weights))
        ),
        "active_set_stability": stability,
    }


def norm_constrained_portfolio_comparison(
    covariance: ArrayLike,
    expected_returns: ArrayLike,
    *,
    risk_aversion: float,
    gross_limit: float,
    l2_radius: float,
    perturbation: float = 1e-4,
) -> dict[str, float | int | np.ndarray]:
    """Compare portfolio-level L1 and L2 geometry under one perturbation."""

    expected = np.asarray(expected_returns, dtype=float)
    asset_count = expected.size
    matrix = _positive_definite_covariance(covariance, asset_count)
    if (
        not np.all(np.isfinite(expected))
        or not np.isfinite(risk_aversion)
        or risk_aversion <= 0.0
        or not np.isfinite(gross_limit)
        or not np.isfinite(l2_radius)
        or not np.isfinite(perturbation)
        or perturbation < 0.0
    ):
        raise ValueError("finite returns and positive risk controls required")
    shock = perturbation * np.where(
        np.arange(asset_count) % 2 == 0,
        1.0,
        -1.0,
    )
    l1 = gross_exposure_constrained_portfolio(
        matrix,
        expected,
        risk_aversion=risk_aversion,
        gross_limit=gross_limit,
    )
    l1_perturbed = gross_exposure_constrained_portfolio(
        matrix,
        expected + shock,
        risk_aversion=risk_aversion,
        gross_limit=gross_limit,
    )
    l1_weights = np.asarray(l1["weights"])
    l1_perturbed_weights = np.asarray(l1_perturbed["weights"])
    l2_weights = _l2_constrained_portfolio(
        matrix,
        expected,
        risk_aversion=risk_aversion,
        l2_radius=l2_radius,
    )
    l2_perturbed_weights = _l2_constrained_portfolio(
        matrix,
        expected + shock,
        risk_aversion=risk_aversion,
        l2_radius=l2_radius,
    )

    l1_metrics = allocation_stability_metrics(
        l1_weights,
        l1_perturbed_weights,
    )
    l2_metrics = allocation_stability_metrics(
        l2_weights,
        l2_perturbed_weights,
    )
    return {
        "l1_weights": l1_weights,
        "l2_weights": l2_weights,
        **{f"l1_{key}": value for key, value in l1_metrics.items()},
        **{f"l2_{key}": value for key, value in l2_metrics.items()},
    }
