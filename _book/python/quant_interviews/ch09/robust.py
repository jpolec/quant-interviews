from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import __version__ as scipy_version
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RobustResult:
    weights: FloatArray
    worst_case_latent_direction: FloatArray
    worst_case_mean_shock: FloatArray
    worst_case_expected_return: float
    support_function_value: float
    support_function_residual: float
    budget_residual: float
    minimum_weight: float
    objective: float
    solver_name: str
    solver_version: str
    solver_success: bool
    solver_message: str
    iterations: int
    stationarity_residual: float
    bound_residual: float
    uncertainty_set_version: str
    worst_case_direction_unique: bool
    status: str

    @property
    def worst_case_direction(self) -> FloatArray:
        """Deprecated compatibility alias for latent uncertainty direction."""
        return self.worst_case_latent_direction


def _validated_inputs(
    mean: FloatArray,
    cov: FloatArray,
    uncertainty: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    expected = np.asarray(mean, dtype=float)
    covariance = np.asarray(cov, dtype=float)
    mapping = np.asarray(uncertainty, dtype=float)
    if (
        expected.ndim != 1
        or covariance.shape != (expected.size, expected.size)
        or mapping.ndim != 2
        or mapping.shape[0] != expected.size
        or not np.isfinite(expected).all()
        or not np.isfinite(covariance).all()
        or not np.isfinite(mapping).all()
    ):
        raise ValueError("robust-optimization shapes or values are invalid")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
        raise ValueError("covariance must be symmetric")
    if float(np.linalg.eigvalsh(covariance).min()) < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    return expected, covariance, mapping


def robust_mean_variance(
    mean: FloatArray,
    cov: FloatArray,
    uncertainty: FloatArray,
    radius: float,
    risk_aversion: float,
    long_only: bool = True,
    *,
    uncertainty_set_version: str = "ellipsoid-v1",
    optimality_tolerance: float = 1e-7,
) -> RobustResult:
    expected, covariance, mapping = _validated_inputs(
        mean,
        cov,
        uncertainty,
    )
    if (
        not np.isfinite(radius)
        or not np.isfinite(risk_aversion)
        or radius < 0.0
        or risk_aversion <= 0.0
        or not uncertainty_set_version
        or not np.isfinite(optimality_tolerance)
        or optimality_tolerance <= 0.0
    ):
        raise ValueError("invalid robust-optimization parameters")

    def objective(weights: FloatArray) -> float:
        penalty = radius * np.linalg.norm(mapping.T @ weights)
        utility = (
            expected @ weights
            - penalty
            - 0.5 * risk_aversion * weights @ covariance @ weights
        )
        return -float(utility)

    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}]
    bounds = [(0.0, None)] * expected.size if long_only else None
    result = minimize(
        objective,
        np.ones(expected.size) / expected.size,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.asarray(result.x)
    projected = mapping.T @ weights
    norm = float(np.linalg.norm(projected))
    latent = (
        np.zeros_like(projected)
        if norm == 0.0
        else -radius * projected / norm
    )
    mean_shock = mapping @ latent
    explicit_worst_case = float((expected + mean_shock) @ weights)
    support_value = float(expected @ weights - radius * norm)
    residual = abs(explicit_worst_case - support_value)
    budget_residual = abs(float(weights.sum()) - 1.0)
    minimum_weight = float(weights.min())
    penalty_gradient = (
        np.zeros_like(weights)
        if norm == 0.0
        else radius * mapping @ projected / norm
    )
    gradient = (
        -expected
        + penalty_gradient
        + risk_aversion * covariance @ weights
    )
    active = np.ones(weights.size, dtype=bool)
    if long_only:
        active = weights > optimality_tolerance
    lagrange = -float(np.mean(gradient[active]))
    reduced_gradient = gradient + lagrange
    stationarity = float(
        max(
            np.max(np.abs(reduced_gradient[active]), initial=0.0),
            np.max(
                np.maximum(-reduced_gradient[~active], 0.0),
                initial=0.0,
            ),
        )
    )
    bound_residual = (
        max(0.0, -minimum_weight) if long_only else 0.0
    )
    status = (
        "PASS"
        if (
            result.success
            and residual <= 1e-10
            and budget_residual <= 1e-10
            and bound_residual <= optimality_tolerance
            and stationarity <= optimality_tolerance
        )
        else "REJECT"
    )
    return RobustResult(
        weights=weights,
        worst_case_latent_direction=latent,
        worst_case_mean_shock=mean_shock,
        worst_case_expected_return=explicit_worst_case,
        support_function_value=support_value,
        support_function_residual=residual,
        budget_residual=budget_residual,
        minimum_weight=minimum_weight,
        objective=-float(result.fun),
        solver_name="scipy.optimize.SLSQP",
        solver_version=scipy_version,
        solver_success=bool(result.success),
        solver_message=str(result.message),
        iterations=int(result.nit),
        stationarity_residual=stationarity,
        bound_residual=bound_residual,
        uncertainty_set_version=uncertainty_set_version,
        worst_case_direction_unique=norm > optimality_tolerance,
        status=status,
    )


def worst_case_mean(
    mean: FloatArray,
    uncertainty: FloatArray,
    radius: float,
    weights: FloatArray,
) -> float:
    expected = np.asarray(mean, dtype=float)
    mapping = np.asarray(uncertainty, dtype=float)
    portfolio = np.asarray(weights, dtype=float)
    if (
        expected.ndim != 1
        or portfolio.shape != expected.shape
        or mapping.ndim != 2
        or mapping.shape[0] != expected.size
        or not np.isfinite(radius)
        or radius < 0.0
        or not np.isfinite(expected).all()
        or not np.isfinite(mapping).all()
        or not np.isfinite(portfolio).all()
    ):
        raise ValueError("invalid worst-case-mean inputs")
    return float(
        expected @ portfolio - radius * np.linalg.norm(mapping.T @ portfolio)
    )
