"""Conditioned nonlinear calibration records for Chapter 3."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, least_squares


@dataclass(frozen=True)
class CalibrationPolicy:
    """Signed numerical limits used to classify a calibration run."""

    version: str = "ch03-calibration-policy-v0.11"
    weighted_rmse_limit: float = 1e-6
    heldout_rmse_limit: float = 1e-6
    restart_dispersion_limit: float = 1e-4
    condition_limit: float = 1e6
    regularization_fraction_limit: float = 0.25
    maximum_active_bounds: int = 0
    minimum_successful_restarts: int = 1
    require_all_restarts_successful: bool = True
    require_heldout: bool = False
    rank_relative_tolerance: float | None = None

    def validate(self) -> None:
        """Reject malformed policy values before optimization starts."""

        positive_limits = (
            self.weighted_rmse_limit,
            self.heldout_rmse_limit,
            self.restart_dispersion_limit,
            self.condition_limit,
        )
        if (
            not self.version
            or not all(math.isfinite(value) and value > 0.0 for value in positive_limits)
            or not math.isfinite(self.regularization_fraction_limit)
            or not 0.0 <= self.regularization_fraction_limit <= 1.0
            or isinstance(self.maximum_active_bounds, bool)
            or not isinstance(self.maximum_active_bounds, int)
            or self.maximum_active_bounds < 0
            or isinstance(self.minimum_successful_restarts, bool)
            or not isinstance(self.minimum_successful_restarts, int)
            or self.minimum_successful_restarts < 1
            or not isinstance(self.require_all_restarts_successful, bool)
            or not isinstance(self.require_heldout, bool)
            or (
                self.rank_relative_tolerance is not None
                and (
                    not math.isfinite(self.rank_relative_tolerance)
                    or self.rank_relative_tolerance <= 0.0
                )
            )
        ):
            raise ValueError("finite positive calibration-policy controls required")


@dataclass(frozen=True)
class CalibrationRecord:
    """Evidence required to distinguish fit from identification."""

    parameters: np.ndarray
    weighted_rmse: float
    regularization_penalty: float
    regularization_fraction: float
    singular_values: np.ndarray
    scaled_jacobian_condition: float
    observation_count: int
    parameter_count: int
    numerical_rank: int
    nullity: int
    is_locally_identified: bool
    active_bounds: int
    heldout_rmse: float | None
    restart_dispersion: float
    restart_statuses: tuple[str, ...]
    successful_restarts: int
    objective: float
    policy_version: str
    status_reasons: tuple[str, ...]
    status: str


def weighted_residuals(
    parameters: np.ndarray,
    model: Callable[[np.ndarray], ArrayLike],
    observations: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return residuals after an explicit observation-noise weighting."""

    modeled = np.asarray(model(parameters), dtype=float)
    if modeled.shape != observations.shape or not np.all(np.isfinite(modeled)):
        raise ValueError("model output must align with finite observations")
    return np.sqrt(weights) * (modeled - observations)


def _scaled_jacobian_diagnostics(
    jacobian: ArrayLike,
    parameter_scale: ArrayLike | None = None,
    *,
    rank_relative_tolerance: float | None = None,
) -> tuple[np.ndarray, int, float]:
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("finite nonempty two-dimensional Jacobian required")
    scale = (
        np.ones(matrix.shape[1], dtype=float)
        if parameter_scale is None
        else np.asarray(parameter_scale, dtype=float)
    )
    if (
        scale.shape != (matrix.shape[1],)
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
        or (
            rank_relative_tolerance is not None
            and (
                not math.isfinite(rank_relative_tolerance)
                or rank_relative_tolerance <= 0.0
            )
        )
    ):
        raise ValueError("finite Jacobian and positive parameter scale required")
    singular_values = np.linalg.svd(
        matrix * scale[np.newaxis, :],
        compute_uv=False,
    )
    largest = float(singular_values[0]) if singular_values.size else 0.0
    relative_cutoff = (
        max(matrix.shape) * np.finfo(float).eps
        if rank_relative_tolerance is None
        else rank_relative_tolerance
    )
    cutoff = relative_cutoff * largest
    rank = int(np.count_nonzero(singular_values > cutoff))
    parameter_count = matrix.shape[1]
    condition = (
        float("inf")
        if rank < parameter_count or singular_values.size < parameter_count
        else largest / float(singular_values[-1])
    )
    return singular_values, rank, condition


def jacobian_condition(
    jacobian: ArrayLike,
    parameter_scale: ArrayLike | None = None,
    *,
    rank_relative_tolerance: float | None = None,
) -> float:
    """Return infinity unless every parameter direction is identified."""

    _, _, condition = _scaled_jacobian_diagnostics(
        jacobian,
        parameter_scale,
        rank_relative_tolerance=rank_relative_tolerance,
    )
    return condition


def _is_successful_result(
    result: OptimizeResult,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    parameters = np.asarray(result.x, dtype=float)
    residuals = np.asarray(result.fun, dtype=float)
    feasibility_scale = 10.0 * np.finfo(float).eps * np.maximum(
        1.0,
        np.maximum(np.abs(lower), np.abs(upper)),
    )
    finite_bounds_scale = np.where(np.isfinite(feasibility_scale), feasibility_scale, 0.0)
    return bool(
        result.success
        and parameters.shape == lower.shape
        and np.all(np.isfinite(parameters))
        and np.all(np.isfinite(residuals))
        and math.isfinite(float(result.cost))
        and np.all(parameters >= lower - finite_bounds_scale)
        and np.all(parameters <= upper + finite_bounds_scale)
    )


def solve_calibration_restarts(
    objective_residuals: Callable[[np.ndarray], np.ndarray],
    starts: Sequence[np.ndarray],
    bounds: tuple[ArrayLike, ArrayLike],
    parameter_scale: np.ndarray,
) -> tuple[OptimizeResult, list[OptimizeResult]]:
    """Solve every governed start and select only among successful runs."""

    governed_starts = [np.asarray(start, dtype=float) for start in starts]
    scale = np.asarray(parameter_scale, dtype=float)
    if (
        not governed_starts
        or scale.ndim != 1
        or scale.size == 0
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
        or any(
            start.shape != scale.shape or not np.all(np.isfinite(start))
            for start in governed_starts
        )
    ):
        raise ValueError("finite aligned starts and positive parameter scale required")
    lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), scale.shape).copy()
    upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), scale.shape).copy()
    if (
        np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower >= upper)
        or any(np.any(start < lower) or np.any(start > upper) for start in governed_starts)
    ):
        raise ValueError("ordered bounds must contain every governed start")

    results = [
        least_squares(
            objective_residuals,
            start,
            bounds=(lower, upper),
            x_scale=scale,
        )
        for start in governed_starts
    ]
    successful = [
        result
        for result in results
        if _is_successful_result(result, lower, upper)
    ]
    if not successful:
        raise RuntimeError("no successful feasible calibration restart")
    best = min(successful, key=lambda result: float(2.0 * result.cost))
    return best, results


def calibrate(
    initial: ArrayLike,
    model: Callable[[np.ndarray], ArrayLike],
    observations: ArrayLike,
    weights: ArrayLike,
    *,
    bounds: tuple[ArrayLike, ArrayLike] = (-np.inf, np.inf),
    prior: ArrayLike | None = None,
    regularization: float = 0.0,
    parameter_scale: ArrayLike | None = None,
    restart_points: Sequence[ArrayLike] = (),
    heldout_model: Callable[[np.ndarray], ArrayLike] | None = None,
    heldout_observations: ArrayLike | None = None,
    policy: CalibrationPolicy | None = None,
) -> CalibrationRecord:
    """Fit a model and classify fit, identification and stability evidence."""

    governed_policy = CalibrationPolicy() if policy is None else policy
    governed_policy.validate()
    start = np.asarray(initial, dtype=float)
    observed = np.asarray(observations, dtype=float)
    observation_weights = np.asarray(weights, dtype=float)
    scale = (
        np.ones_like(start)
        if parameter_scale is None
        else np.asarray(parameter_scale, dtype=float)
    )
    prior_values = (
        np.zeros_like(start)
        if prior is None
        else np.asarray(prior, dtype=float)
    )
    if (
        start.ndim != 1
        or start.size == 0
        or observed.ndim != 1
        or observed.size == 0
        or observation_weights.shape != observed.shape
        or scale.shape != start.shape
        or prior_values.shape != start.shape
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(observed))
        or not np.all(np.isfinite(observation_weights))
        or not np.all(np.isfinite(scale))
        or not np.all(np.isfinite(prior_values))
        or np.any(observation_weights <= 0.0)
        or np.any(scale <= 0.0)
        or not math.isfinite(regularization)
        or regularization < 0.0
        or (heldout_model is None) != (heldout_observations is None)
    ):
        raise ValueError("finite aligned calibration inputs and positive scales required")

    def objective_residuals(parameters: np.ndarray) -> np.ndarray:
        market = weighted_residuals(
            parameters,
            model,
            observed,
            observation_weights,
        )
        if regularization == 0.0:
            return market
        regularizer = np.sqrt(regularization) * (parameters - prior_values) / scale
        return np.concatenate([market, regularizer])

    starts = [start] + [np.asarray(point, dtype=float) for point in restart_points]
    best, results = solve_calibration_restarts(
        objective_residuals,
        starts,
        bounds,
        scale,
    )
    lower = np.broadcast_to(np.asarray(bounds[0], dtype=float), start.shape)
    upper = np.broadcast_to(np.asarray(bounds[1], dtype=float), start.shape)
    successful_results = [
        result for result in results if _is_successful_result(result, lower, upper)
    ]
    market_residual = weighted_residuals(
        best.x,
        model,
        observed,
        observation_weights,
    )
    market_jacobian = np.asarray(best.jac, dtype=float)[: observed.size]
    singular_values, numerical_rank, condition = _scaled_jacobian_diagnostics(
        market_jacobian,
        scale,
        rank_relative_tolerance=governed_policy.rank_relative_tolerance,
    )
    parameter_count = start.size
    nullity = parameter_count - numerical_rank
    locally_identified = nullity == 0

    heldout_rmse: float | None = None
    if heldout_model is not None and heldout_observations is not None:
        heldout = np.asarray(heldout_observations, dtype=float)
        prediction = np.asarray(heldout_model(best.x), dtype=float)
        if (
            heldout.ndim != 1
            or heldout.size == 0
            or prediction.shape != heldout.shape
            or not np.all(np.isfinite(heldout))
            or not np.all(np.isfinite(prediction))
        ):
            raise ValueError("finite held-out model and observations must align")
        heldout_rmse = float(np.sqrt(np.mean((prediction - heldout) ** 2)))

    fitted = np.vstack([np.asarray(result.x, dtype=float) for result in successful_results])
    dispersion = float(
        np.max(np.linalg.norm((fitted - best.x) / scale, axis=1))
    )
    penalty = float(
        regularization * np.sum(((best.x - prior_values) / scale) ** 2)
    )
    objective = float(2.0 * best.cost)
    regularization_fraction = penalty / objective if objective > 0.0 else 0.0
    weighted_rmse = float(np.sqrt(np.mean(market_residual**2)))
    active_bounds = int(np.count_nonzero(best.active_mask))
    restart_statuses = tuple(
        "SUCCESS" if _is_successful_result(result, lower, upper) else "FAILED"
        for result in results
    )

    reasons: list[str] = []
    if weighted_rmse > governed_policy.weighted_rmse_limit:
        reasons.append("FIT_RMSE_LIMIT_EXCEEDED")
    if governed_policy.require_heldout and heldout_rmse is None:
        reasons.append("HELDOUT_EVIDENCE_MISSING")
    if heldout_rmse is not None and heldout_rmse > governed_policy.heldout_rmse_limit:
        reasons.append("HELDOUT_RMSE_LIMIT_EXCEEDED")
    if dispersion > governed_policy.restart_dispersion_limit:
        reasons.append("RESTART_DISPERSION_LIMIT_EXCEEDED")
    if len(successful_results) < governed_policy.minimum_successful_restarts:
        reasons.append("INSUFFICIENT_SUCCESSFUL_RESTARTS")
    if governed_policy.require_all_restarts_successful and len(successful_results) != len(results):
        reasons.append("FAILED_RESTART_PRESENT")
    if active_bounds > governed_policy.maximum_active_bounds:
        reasons.append("ACTIVE_BOUND_LIMIT_EXCEEDED")
    if regularization_fraction > governed_policy.regularization_fraction_limit:
        reasons.append("REGULARIZATION_DOMINANCE_LIMIT_EXCEEDED")
    if condition > governed_policy.condition_limit:
        reasons.append("CONDITION_LIMIT_EXCEEDED")

    if not locally_identified:
        identification_reason = (
            "PRIOR_DRIVEN_NOT_IDENTIFIED"
            if regularization > 0.0
            else "NOT_IDENTIFIED"
        )
        reasons.insert(0, identification_reason)
        status = "NOT_IDENTIFIED"
    elif any(reason != "CONDITION_LIMIT_EXCEEDED" for reason in reasons):
        status = "REJECT"
    elif reasons:
        status = "CONDITIONAL_PASS"
    else:
        status = "PASS"

    return CalibrationRecord(
        parameters=np.asarray(best.x),
        weighted_rmse=weighted_rmse,
        regularization_penalty=penalty,
        regularization_fraction=regularization_fraction,
        singular_values=singular_values,
        scaled_jacobian_condition=condition,
        observation_count=observed.size,
        parameter_count=parameter_count,
        numerical_rank=numerical_rank,
        nullity=nullity,
        is_locally_identified=locally_identified,
        active_bounds=active_bounds,
        heldout_rmse=heldout_rmse,
        restart_dispersion=dispersion,
        restart_statuses=restart_statuses,
        successful_restarts=len(successful_results),
        objective=objective,
        policy_version=governed_policy.version,
        status_reasons=tuple(reasons),
        status=status,
    )
