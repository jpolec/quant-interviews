from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RiskBudgetResult:
    weights: FloatArray
    target_budgets: FloatArray
    realized_budgets: FloatArray
    maximum_budget_residual: float
    euler_reconciliation_error: float
    status: str
    iterations: int


def _validate_covariance(cov: FloatArray) -> FloatArray:
    matrix = np.asarray(cov, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(matrix).all():
        raise ValueError("covariance must be finite")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-10):
        raise ValueError("covariance must be symmetric")
    if float(np.linalg.eigvalsh(matrix).min()) <= 0.0:
        raise ValueError("covariance must be positive definite")
    return matrix


def risk_contributions(weights: FloatArray, cov: FloatArray) -> FloatArray:
    vector = np.asarray(weights, dtype=float)
    matrix = _validate_covariance(cov)
    if vector.shape != (matrix.shape[0],) or not np.isfinite(vector).all():
        raise ValueError("weight shape or values are invalid")
    variance = float(vector @ matrix @ vector)
    if variance <= 0.0:
        raise ValueError("portfolio volatility must be positive")
    return vector * (matrix @ vector) / np.sqrt(variance)


def solve_risk_budget(
    cov: FloatArray,
    budgets: FloatArray | None = None,
    *,
    tolerance: float = 1e-8,
    maximum_iterations: int = 2000,
) -> RiskBudgetResult:
    matrix = _validate_covariance(cov)
    if tolerance <= 0.0 or maximum_iterations < 1:
        raise ValueError("invalid solver policy")
    raw_budgets = (
        np.ones(matrix.shape[0])
        if budgets is None
        else np.asarray(budgets, dtype=float)
    )
    if (
        raw_budgets.shape != (matrix.shape[0],)
        or not np.isfinite(raw_budgets).all()
        or np.any(raw_budgets <= 0.0)
    ):
        raise ValueError("risk budgets must be finite and positive")
    target = raw_budgets / raw_budgets.sum()

    def objective(log_x: FloatArray) -> float:
        x = np.exp(log_x)
        return float(0.5 * x @ matrix @ x - target @ log_x)

    def gradient(log_x: FloatArray) -> FloatArray:
        x = np.exp(log_x)
        return x * (matrix @ x) - target

    result = minimize(
        objective,
        np.zeros(matrix.shape[0]),
        jac=gradient,
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": tolerance / 10.0, "maxiter": maximum_iterations},
    )
    raw = np.exp(result.x)
    weights = raw / raw.sum()
    contributions = risk_contributions(weights, matrix)
    portfolio_volatility = float(np.sqrt(weights @ matrix @ weights))
    realized = contributions / portfolio_volatility
    maximum_residual = float(np.max(np.abs(realized - target)))
    euler_error = abs(float(contributions.sum()) - portfolio_volatility)
    if maximum_residual <= tolerance and euler_error <= tolerance:
        status = "PASS"
    elif maximum_residual <= 10.0 * tolerance:
        status = "CONDITIONAL_PASS"
    else:
        status = "REJECT"
    if status == "REJECT":
        raise RuntimeError(
            f"risk-budget solver failed: {result.message}; residual={maximum_residual}"
        )
    return RiskBudgetResult(
        weights=weights,
        target_budgets=target,
        realized_budgets=realized,
        maximum_budget_residual=maximum_residual,
        euler_reconciliation_error=euler_error,
        status=status,
        iterations=int(result.nit),
    )
