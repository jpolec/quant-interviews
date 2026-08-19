from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TailRiskCertificate:
    confidence: float
    observations: int
    quantile_convention: str
    value_at_risk: float
    expected_shortfall: float
    exception_count: int
    mean_excess_severity: float | None
    status: Literal["METRIC_COMPUTED"]


@dataclass(frozen=True)
class TailRiskDecision:
    utilization: float
    coverage: float
    minimum_coverage: float
    tail_model_adequate: bool
    decision: Literal["PASS", "REJECT"]


@dataclass(frozen=True)
class ReverseStressCertificate:
    scale: float
    repriced_loss: float
    residual: float
    iterations: int
    monotonicity_points: int
    status: Literal[
        "PASS",
        "BOUNDARY_HIT",
        "ALREADY_EXCEEDS",
        "NO_BRACKET",
        "NON_MONOTONE",
        "BOUND_ONLY",
        "MAX_ITERATIONS",
    ]


def _finite_vector(
    values: NDArray[np.float64],
    name: str,
    *,
    nonnegative: bool = False,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with at least two values")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must use a non-negative loss convention")
    return array


def historical_var_es(
    losses: NDArray[np.float64],
    confidence: float = 0.975,
) -> TailRiskCertificate:
    # Standard historical loss is signed: profitable observations have L < 0.
    sample = _finite_vector(losses, "losses", nonnegative=False)
    if not isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    ordered = np.sort(sample)
    order_index = ceil(confidence * ordered.size) - 1
    value_at_risk = float(ordered[order_index])

    tail_mass = (1.0 - confidence) * ordered.size
    whole = int(np.floor(tail_mass))
    fractional = tail_mass - whole
    total = float(np.sum(ordered[-whole:])) if whole > 0 else 0.0
    if fractional > 0.0:
        total += fractional * float(ordered[-whole - 1])
    expected_shortfall = total / tail_mass

    excess = sample[sample > value_at_risk] - value_at_risk
    severity = float(np.mean(excess)) if excess.size else None
    if not isfinite(value_at_risk) or not isfinite(expected_shortfall):
        raise FloatingPointError("tail-risk metric is non-finite")
    return TailRiskCertificate(
        confidence=confidence,
        observations=int(sample.size),
        quantile_convention="inverse_empirical_cdf_ceil_alpha_n",
        value_at_risk=value_at_risk,
        expected_shortfall=expected_shortfall,
        exception_count=int(excess.size),
        mean_excess_severity=severity,
        status="METRIC_COMPUTED",
    )


def tail_risk_gate(
    certificate: TailRiskCertificate,
    *,
    limit: float,
    coverage: float,
    minimum_coverage: float,
    tail_model_adequate: bool,
) -> TailRiskDecision:
    if not all(isfinite(value) for value in (limit, coverage, minimum_coverage)):
        raise ValueError("tail-risk gate values must be finite")
    if limit <= 0.0 or not 0.0 <= coverage <= 1.0 or not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("limit and coverage domains are invalid")
    utilization = certificate.expected_shortfall / limit
    decision: Literal["PASS", "REJECT"] = (
        "PASS"
        if utilization <= 1.0
        and coverage >= minimum_coverage
        and tail_model_adequate
        else "REJECT"
    )
    return TailRiskDecision(
        utilization=utilization,
        coverage=coverage,
        minimum_coverage=minimum_coverage,
        tail_model_adequate=tail_model_adequate,
        decision=decision,
    )


def scenario_pnl(
    exposures: NDArray[np.float64],
    shocks: NDArray[np.float64],
    gamma: NDArray[np.float64] | None = None,
) -> float:
    delta = np.asarray(exposures, dtype=float)
    move = np.asarray(shocks, dtype=float)
    if (
        delta.ndim != 1
        or move.shape != delta.shape
        or not np.all(np.isfinite(delta))
        or not np.all(np.isfinite(move))
    ):
        raise ValueError("exposures and shocks must be aligned finite vectors")
    try:
        with np.errstate(over="raise", invalid="raise"):
            pnl = float(delta @ move)
            if gamma is not None:
                matrix = np.asarray(gamma, dtype=float)
                if not np.all(np.isfinite(matrix)):
                    raise ValueError("gamma must be finite")
                if matrix.shape == delta.shape:
                    pnl += 0.5 * float(np.sum(matrix * move * move))
                elif matrix.shape == (delta.size, delta.size):
                    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
                        raise ValueError("gamma matrix must be symmetric")
                    pnl += 0.5 * float(move @ matrix @ move)
                else:
                    raise ValueError(
                        "gamma must be a vector or square matrix aligned with shocks"
                    )
    except FloatingPointError as error:
        raise FloatingPointError("scenario P&L overflow or invalid arithmetic") from error
    if not isfinite(pnl):
        raise FloatingPointError("scenario P&L is non-finite")
    return pnl


def limit_utilization(nonnegative_risk_measure: float, limit: float) -> float:
    if not all(isfinite(value) for value in (nonnegative_risk_measure, limit)):
        raise ValueError("risk measure and limit must be finite")
    if nonnegative_risk_measure < 0.0 or limit <= 0.0:
        raise ValueError("risk measure must be non-negative and limit positive")
    return nonnegative_risk_measure / limit


def _loss(loss_at_scale: Callable[[float], float], scale: float) -> float:
    value = float(loss_at_scale(scale))
    if not isfinite(value):
        raise FloatingPointError("non-finite repriced loss")
    return value


def reverse_stress_scale(
    loss_at_scale: Callable[[float], float],
    target_loss: float,
    *,
    initial_upper: float = 1.0,
    maximum_upper: float = 1024.0,
    atol: float = 1e-10,
    rtol: float = 1e-10,
    maximum_iterations: int = 200,
    monotonicity_grid: int = 257,
) -> ReverseStressCertificate:
    if not all(
        isfinite(value)
        for value in (target_loss, initial_upper, maximum_upper, atol, rtol)
    ):
        raise ValueError("reverse-stress inputs must be finite")
    if target_loss <= 0.0 or initial_upper <= 0.0 or maximum_upper < initial_upper:
        raise ValueError("target and bracketing limits must be positive")
    if (
        atol < 0.0
        or rtol < 0.0
        or maximum_iterations <= 0
        or monotonicity_grid < 17
    ):
        raise ValueError("invalid tolerance, iterations or monotonicity grid")
    tolerance = atol + rtol * target_loss
    zero_loss = _loss(loss_at_scale, 0.0)
    if zero_loss >= target_loss:
        residual = zero_loss - target_loss
        return ReverseStressCertificate(
            0.0,
            zero_loss,
            residual,
            0,
            1,
            "BOUNDARY_HIT" if abs(residual) <= tolerance else "ALREADY_EXCEEDS",
        )

    upper = initial_upper
    upper_loss = _loss(loss_at_scale, upper)
    while upper_loss < target_loss and upper < maximum_upper:
        upper = min(2.0 * upper, maximum_upper)
        upper_loss = _loss(loss_at_scale, upper)
    if upper_loss < target_loss:
        return ReverseStressCertificate(
            upper,
            upper_loss,
            upper_loss - target_loss,
            0,
            2,
            "NO_BRACKET",
        )

    grid = np.linspace(0.0, upper, monotonicity_grid)
    grid_losses = np.array([_loss(loss_at_scale, float(scale)) for scale in grid])
    monotone_tolerance = atol + rtol * max(1.0, float(np.max(np.abs(grid_losses))))
    if np.any(np.diff(grid_losses) < -monotone_tolerance):
        return ReverseStressCertificate(
            0.0,
            zero_loss,
            zero_loss - target_loss,
            0,
            monotonicity_grid,
            "NON_MONOTONE",
        )
    hits = np.flatnonzero(grid_losses >= target_loss)
    if hits.size == 0:
        return ReverseStressCertificate(
            upper,
            upper_loss,
            upper_loss - target_loss,
            0,
            monotonicity_grid,
            "NO_BRACKET",
        )
    hit = int(hits[0])
    lower = float(grid[hit - 1])
    upper = float(grid[hit])
    lower_loss = float(grid_losses[hit - 1])
    upper_loss = float(grid_losses[hit])
    if abs(upper_loss - target_loss) <= tolerance:
        return ReverseStressCertificate(
            upper,
            upper_loss,
            upper_loss - target_loss,
            0,
            monotonicity_grid,
            "PASS",
        )

    for iteration in range(1, maximum_iterations + 1):
        midpoint = 0.5 * (lower + upper)
        loss = _loss(loss_at_scale, midpoint)
        if loss < lower_loss - monotone_tolerance or loss > upper_loss + monotone_tolerance:
            return ReverseStressCertificate(
                midpoint,
                loss,
                loss - target_loss,
                iteration,
                monotonicity_grid,
                "NON_MONOTONE",
            )
        residual = loss - target_loss
        if abs(residual) <= tolerance:
            return ReverseStressCertificate(
                midpoint,
                loss,
                residual,
                iteration,
                monotonicity_grid,
                "PASS",
            )
        if loss < target_loss:
            lower, lower_loss = midpoint, loss
        else:
            upper, upper_loss = midpoint, loss
        if upper - lower <= atol + rtol * max(1.0, midpoint):
            best_scale, best_loss = min(
                ((lower, lower_loss), (upper, upper_loss)),
                key=lambda item: abs(item[1] - target_loss),
            )
            return ReverseStressCertificate(
                best_scale,
                best_loss,
                best_loss - target_loss,
                iteration,
                monotonicity_grid,
                "BOUND_ONLY",
            )
    midpoint = 0.5 * (lower + upper)
    loss = _loss(loss_at_scale, midpoint)
    return ReverseStressCertificate(
        midpoint,
        loss,
        loss - target_loss,
        maximum_iterations,
        monotonicity_grid,
        "MAX_ITERATIONS",
    )


def liquidity_adjusted_capital(
    expected_shortfall_of_liquidity_adjusted_loss: float,
    model_addon: float,
) -> float:
    if not all(
        isfinite(value)
        for value in (expected_shortfall_of_liquidity_adjusted_loss, model_addon)
    ):
        raise ValueError("capital components must be finite")
    if expected_shortfall_of_liquidity_adjusted_loss < 0.0 or model_addon < 0.0:
        raise ValueError("capital components must be non-negative")
    result = expected_shortfall_of_liquidity_adjusted_loss + model_addon
    if not isfinite(result):
        raise FloatingPointError("liquidity-adjusted capital is non-finite")
    return result
