"""Risk contributions and convex risk budgeting for Chapter 2."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize


def _finite_positive_scalar(value: float, *, name: str) -> float:
    candidate = np.asarray(value)
    if candidate.ndim != 0 or candidate.dtype.kind not in "iuf":
        raise ValueError(f"positive {name} must be a finite scalar")
    scalar = float(candidate)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"positive {name} must be a finite scalar")
    return scalar


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a positive integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def risk_contributions(
    weights: ArrayLike,
    covariance: ArrayLike,
) -> np.ndarray:
    """Return signed volatility contributions that reconcile to volatility."""

    portfolio = np.asarray(weights, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] == 0
        or portfolio.shape != (matrix.shape[0],)
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(portfolio))
    ):
        raise ValueError("finite aligned weights and covariance required")
    scale = max(
        float(np.linalg.norm(matrix, ord=np.inf)),
        np.finfo(float).tiny,
    )
    symmetry_tolerance = 1e-14 + 1e-10 * scale
    if float(np.max(np.abs(matrix - matrix.T))) > symmetry_tolerance:
        raise ValueError("symmetric covariance required")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    psd_scale = max(
        float(np.max(np.abs(eigenvalues))),
        np.finfo(float).tiny,
    )
    psd_tolerance = 1e-14 + 1e-10 * psd_scale
    if eigenvalues[0] < -psd_tolerance:
        raise ValueError("positive-semidefinite covariance required")
    variance = float(portfolio @ matrix @ portfolio)
    variance_floor = 100.0 * np.finfo(float).eps * max(
        scale * float(portfolio @ portfolio),
        np.finfo(float).tiny,
    )
    if not np.isfinite(variance) or variance <= variance_floor:
        raise ValueError("positive portfolio variance required")
    volatility = np.sqrt(variance)
    return portfolio * (matrix @ portfolio) / volatility


def risk_parity_core(
    covariance: np.ndarray,
    budget: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, bool, float, bool, str, str, float]:
    """Solve the validated convex log-barrier objective."""

    tolerance = _finite_positive_scalar(tolerance, name="tolerance")
    max_iterations = _positive_integer(
        max_iterations,
        name="max_iterations",
    )

    def objective(raw: np.ndarray) -> float:
        return float(0.5 * raw @ covariance @ raw - budget @ np.log(raw))

    def gradient(raw: np.ndarray) -> np.ndarray:
        return covariance @ raw - budget / raw

    result = minimize(
        objective,
        np.ones(covariance.shape[0]),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(1e-12, None)] * covariance.shape[0],
        options={"ftol": 1e-15, "gtol": tolerance, "maxiter": max_iterations},
    )
    if not np.all(np.isfinite(result.x)) or np.any(result.x <= 0.0):
        raise RuntimeError("risk-parity solver returned an invalid solution")
    gradient_residual = float(np.linalg.norm(gradient(result.x), ord=np.inf))
    fallback_residual_limit = max(
        10.0 * tolerance,
        100.0 * np.finfo(float).eps,
    )
    if (
        not np.isfinite(gradient_residual)
    ):
        raise RuntimeError("risk-parity solver returned an invalid solution")
    if not result.success and gradient_residual > fallback_residual_limit:
        raise RuntimeError(result.message)
    lower_bound_touched = bool(np.any(result.x <= 10.0e-12))
    termination_status = (
        "SOLVER_SUCCESS" if result.success else "RESIDUAL_ACCEPTED"
    )
    return (
        np.asarray(result.x),
        int(result.nit),
        bool(result.success),
        gradient_residual,
        lower_bound_touched,
        termination_status,
        str(result.message),
        fallback_residual_limit,
    )


def risk_parity(
    covariance: ArrayLike,
    budget: ArrayLike | None = None,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 4_000,
) -> dict[str, float | int | bool | str | np.ndarray]:
    """Solve the exact convex log-barrier risk-budgeting formulation."""

    matrix = np.asarray(covariance, dtype=float)
    tolerance = _finite_positive_scalar(tolerance, name="tolerance")
    max_iterations = _positive_integer(
        max_iterations,
        name="max_iterations",
    )
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] == 0
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("positive-definite covariance required")
    symmetry_floor = (
        1e-14
        + 1e-10
        * max(float(np.linalg.norm(matrix, ord=2)), np.finfo(float).tiny)
    )
    if not np.allclose(
        matrix,
        matrix.T,
        atol=symmetry_floor,
        rtol=0.0,
    ):
        raise ValueError("positive-definite covariance required")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    spectral_scale = max(
        float(np.max(np.abs(eigenvalues))),
        np.finfo(float).tiny,
    )
    if eigenvalues[0] <= tolerance * spectral_scale:
        raise ValueError("positive-definite covariance required")
    asset_count = matrix.shape[0]
    target = (
        np.full(asset_count, 1.0 / asset_count)
        if budget is None
        else np.asarray(budget, dtype=float)
    )
    if (
        target.shape != (asset_count,)
        or not np.all(np.isfinite(target))
        or np.any(target <= 0.0)
    ):
        raise ValueError("strictly positive aligned risk budget required")
    target_sum = float(target.sum())
    if not np.isfinite(target_sum) or target_sum <= 0.0:
        raise ValueError("strictly positive aligned risk budget required")
    target = target / target_sum

    (
        raw,
        iterations,
        success,
        gradient_residual,
        lower_bound_touched,
        termination_status,
        termination_message,
        fallback_residual_limit,
    ) = risk_parity_core(
        matrix,
        target,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    weights = raw / raw.sum()
    contributions = risk_contributions(weights, matrix)
    volatility = float(contributions.sum())
    shares = contributions / volatility
    risk_budget_residual = float(np.max(np.abs(shares - target)))
    euler_residual = float(
        abs(contributions.sum() - np.sqrt(weights @ matrix @ weights))
    )
    maximum_residual = max(
        gradient_residual,
        risk_budget_residual,
        euler_residual,
    )
    pass_tolerance = max(100.0 * tolerance, 1e-8)
    conditional_tolerance = max(10_000.0 * tolerance, 1e-6)
    status = (
        "PASS"
        if maximum_residual <= pass_tolerance
        and not lower_bound_touched
        and termination_status == "SOLVER_SUCCESS"
        else "CONDITIONAL_PASS"
        if maximum_residual <= conditional_tolerance
        else "REJECT"
    )
    return {
        "weights": weights,
        "raw_solution": raw,
        "risk_contributions": contributions,
        "risk_budget_shares": shares,
        "gradient_residual": gradient_residual,
        "risk_budget_residual": risk_budget_residual,
        "euler_residual": euler_residual,
        "maximum_residual": maximum_residual,
        "iterations": iterations,
        "success": success,
        "termination_status": termination_status,
        "termination_message": termination_message,
        "fallback_residual_limit": fallback_residual_limit,
        "lower_bound_touched": lower_bound_touched,
        "status": status,
    }
