from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FrontierPoint:
    weights: FloatArray
    expected_return: float
    volatility: float


@dataclass(frozen=True)
class PortfolioCertificate:
    portfolio_type: str
    weights: FloatArray
    expected_return: float
    volatility: float
    budget_residual: float
    return_residual: float | None
    condition_number: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    optimizer_weight_residual_l2: float
    optimizer_success: bool
    input_sha256: str
    condition_number_limit: float
    tolerance: float
    status: str


def _validate_covariance(cov: FloatArray) -> FloatArray:
    matrix = np.asarray(cov, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    if not np.isfinite(matrix).all():
        raise ValueError("covariance must be finite")
    scale = max(float(np.linalg.norm(matrix, ord=2)), 1.0)
    if not np.allclose(matrix, matrix.T, atol=1e-12 * scale, rtol=1e-10):
        raise ValueError("covariance must be symmetric")
    minimum = float(np.linalg.eigvalsh(matrix).min())
    if minimum <= 100.0 * np.finfo(float).eps * scale:
        raise ValueError("covariance must be well-conditioned positive definite")
    return matrix


def _validate(
    mu: FloatArray,
    cov: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    mean = np.asarray(mu, dtype=float)
    matrix = _validate_covariance(cov)
    if mean.ndim != 1 or matrix.shape != (mean.size, mean.size):
        raise ValueError("shape mismatch")
    if not np.isfinite(mean).all():
        raise ValueError("expected returns must be finite")
    return mean, matrix


def gmv_weights(cov: FloatArray) -> FloatArray:
    matrix = _validate_covariance(cov)
    one = np.ones(matrix.shape[0])
    solution = np.linalg.solve(matrix, one)
    denominator = float(one @ solution)
    scale = max(float(np.linalg.norm(solution)), 1.0)
    if abs(denominator) <= 100.0 * np.finfo(float).eps * scale:
        raise ValueError("GMV budget normalization is ill-conditioned")
    return solution / denominator


def target_return_weights(
    mu: FloatArray,
    cov: FloatArray,
    target: float,
) -> FrontierPoint:
    mean, matrix = _validate(mu, cov)
    if not np.isfinite(target):
        raise ValueError("target must be finite")
    one = np.ones(mean.size)
    inv_one = np.linalg.solve(matrix, one)
    inv_mean = np.linalg.solve(matrix, mean)
    a = float(one @ inv_one)
    b = float(one @ inv_mean)
    c = float(mean @ inv_mean)
    determinant = a * c - b * b
    scale = max(abs(a * c), b * b, 1.0)
    threshold = 100.0 * np.finfo(float).eps * scale
    if determinant <= threshold:
        raise ValueError("degenerate or ill-conditioned frontier")
    weights = (
        ((c - b * target) / determinant) * inv_one
        + ((a * target - b) / determinant) * inv_mean
    )
    variance = float(weights @ matrix @ weights)
    return FrontierPoint(
        weights=weights,
        expected_return=float(weights @ mean),
        volatility=float(np.sqrt(variance)),
    )


def tangency_direction(
    mu: FloatArray,
    cov: FloatArray,
    risk_free: float = 0.0,
) -> FloatArray:
    mean, matrix = _validate(mu, cov)
    if not np.isfinite(risk_free):
        raise ValueError("risk-free rate must be finite")
    return np.linalg.solve(matrix, mean - risk_free)


def tangency_weights(
    mu: FloatArray,
    cov: FloatArray,
    risk_free: float = 0.0,
) -> FloatArray:
    direction = tangency_direction(mu, cov, risk_free)
    denominator = float(direction.sum())
    threshold = 100.0 * np.finfo(float).eps * max(
        float(np.linalg.norm(direction)),
        1.0,
    )
    if abs(denominator) <= threshold:
        raise ValueError(
            "risky-fund normalization is zero; retain the raw tangency direction"
        )
    return direction / denominator


def portfolio_certificate(
    mu: FloatArray,
    cov: FloatArray,
    portfolio_type: str,
    *,
    target: float | None = None,
    risk_free: float = 0.0,
    condition_number_limit: float = 1.0e8,
    tolerance: float = 1.0e-8,
) -> PortfolioCertificate:
    mean, matrix = _validate(mu, cov)
    if (
        portfolio_type not in {"GMV", "TARGET_RETURN", "TANGENCY"}
        or not np.isfinite(condition_number_limit)
        or condition_number_limit <= 1.0
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
        or not np.isfinite(risk_free)
    ):
        raise ValueError("invalid portfolio-certificate policy")
    if portfolio_type == "GMV":
        weights = gmv_weights(matrix)
        constraints = [
            {"type": "eq", "fun": lambda value: value.sum() - 1.0}
        ]
        objective = lambda value: float(value @ matrix @ value)
        return_residual: float | None = None
    elif portfolio_type == "TARGET_RETURN":
        if target is None or not np.isfinite(target):
            raise ValueError("a finite target return is required")
        weights = target_return_weights(mean, matrix, target).weights
        constraints = [
            {"type": "eq", "fun": lambda value: value.sum() - 1.0},
            {"type": "eq", "fun": lambda value: value @ mean - target},
        ]
        objective = lambda value: float(value @ matrix @ value)
        return_residual = abs(float(weights @ mean) - target)
    else:
        weights = tangency_weights(mean, matrix, risk_free)
        constraints = [
            {"type": "eq", "fun": lambda value: value.sum() - 1.0}
        ]
        objective = lambda value: -float(
            (value @ mean - risk_free)
            / np.sqrt(value @ matrix @ value)
        )
        return_residual = None
    numerical = minimize(
        objective,
        np.ones(mean.size) / mean.size,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": tolerance / 100.0, "maxiter": 2000},
    )
    eigenvalues = np.linalg.eigvalsh(matrix)
    condition_number = float(eigenvalues[-1] / eigenvalues[0])
    budget_residual = abs(float(weights.sum()) - 1.0)
    optimizer_residual = float(np.linalg.norm(weights - numerical.x))
    payload = {
        "mean": mean.tolist(),
        "covariance": matrix.tolist(),
        "portfolio_type": portfolio_type,
        "target": target,
        "risk_free": risk_free,
        "condition_number_limit": condition_number_limit,
        "tolerance": tolerance,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    status = (
        "PASS"
        if (
            numerical.success
            and budget_residual <= tolerance
            and (return_residual is None or return_residual <= tolerance)
            and optimizer_residual <= 10.0 * tolerance
            and condition_number <= condition_number_limit
        )
        else "REJECT"
    )
    return PortfolioCertificate(
        portfolio_type=portfolio_type,
        weights=weights,
        expected_return=float(weights @ mean),
        volatility=float(np.sqrt(weights @ matrix @ weights)),
        budget_residual=budget_residual,
        return_residual=return_residual,
        condition_number=condition_number,
        minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]),
        optimizer_weight_residual_l2=optimizer_residual,
        optimizer_success=bool(numerical.success),
        input_sha256=digest,
        condition_number_limit=float(condition_number_limit),
        tolerance=float(tolerance),
        status=status,
    )
