from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StressResult:
    linear_pnl: float
    delta_gamma_pnl: float
    full_pnl: float | None
    approximation_error: float | None
    status: str
    reason_code: str


@dataclass(frozen=True)
class ReverseStressCertificate:
    shock: FloatArray
    objective: float
    loss: float
    constraint_residual: float
    solver_name: str
    solver_version: str
    success: bool
    status: str


@dataclass(frozen=True)
class RiskCoverageProfile:
    market_value: float
    absolute_delta: float
    vega: float
    stress_loss: float
    notional: float
    governing_coverage: float
    status: str


def stress_pnl(
    delta: FloatArray,
    gamma: FloatArray,
    shock: FloatArray,
    full_reprice: Callable[[FloatArray], float] | None = None,
) -> StressResult:
    first_order = np.asarray(delta, dtype=float)
    second_order = np.asarray(gamma, dtype=float)
    scenario = np.asarray(shock, dtype=float)
    if (
        first_order.ndim != 1
        or second_order.shape != (first_order.size, first_order.size)
        or scenario.shape != first_order.shape
        or not np.isfinite(first_order).all()
        or not np.isfinite(second_order).all()
        or not np.isfinite(scenario).all()
    ):
        raise ValueError("invalid stress dimensions or values")
    if not np.allclose(second_order, second_order.T, atol=1e-12, rtol=1e-10):
        raise ValueError("gamma must be symmetric")
    linear = float(first_order @ scenario)
    delta_gamma = float(linear + 0.5 * scenario @ second_order @ scenario)
    if full_reprice is None:
        full = None
        status = "CONDITIONAL_PASS"
        reason = "FULL_REPRICE_NOT_REQUESTED"
    else:
        try:
            full = float(full_reprice(scenario))
        except Exception:
            return StressResult(
                linear,
                delta_gamma,
                None,
                None,
                "REJECT",
                "FULL_REPRICE_EXCEPTION",
            )
        if not np.isfinite(full):
            return StressResult(
                linear,
                delta_gamma,
                None,
                None,
                "REJECT",
                "FULL_REPRICE_NONFINITE",
            )
        status = "PASS"
        reason = "FULL_REPRICE_FINITE"
    return StressResult(
        linear,
        delta_gamma,
        full,
        None if full is None else full - delta_gamma,
        status,
        reason,
    )


def reverse_stress(
    delta: FloatArray,
    metric: FloatArray,
    loss_threshold: float,
) -> ReverseStressCertificate:
    """Closed-form minimum-metric linear reverse-stress certificate."""
    exposure = np.asarray(delta, dtype=float)
    geometry = np.asarray(metric, dtype=float)
    if (
        exposure.ndim != 1
        or geometry.shape != (exposure.size, exposure.size)
        or not np.isfinite(exposure).all()
        or not np.isfinite(geometry).all()
    ):
        raise ValueError("invalid reverse-stress inputs")
    if not np.allclose(geometry, geometry.T, atol=1e-12, rtol=1e-10):
        raise ValueError("metric must be symmetric")
    if float(np.linalg.eigvalsh(geometry).min()) <= 0.0:
        raise ValueError("metric must be positive definite")
    if not np.isfinite(loss_threshold) or loss_threshold <= 0.0:
        raise ValueError("threshold must be positive")
    denominator = float(exposure @ geometry @ exposure)
    scale = max(
        float(np.linalg.norm(exposure)) ** 2
        * float(np.linalg.norm(geometry, ord=2)),
        1.0,
    )
    if denominator <= 1e-14 * scale:
        raise ValueError("reverse stress requires a nonzero exposure direction")
    shock = -loss_threshold * (geometry @ exposure) / denominator
    loss = -float(exposure @ shock)
    residual = abs(loss - loss_threshold)
    objective = float(shock @ np.linalg.solve(geometry, shock))
    tolerance = 1e-10 * max(loss_threshold, 1.0)
    status = "PASS" if residual <= tolerance else "REJECT"
    return ReverseStressCertificate(
        shock=shock,
        objective=objective,
        loss=loss,
        constraint_residual=residual,
        solver_name="ANALYTIC_LINEAR_REVERSE_STRESS",
        solver_version="1.0",
        success=status == "PASS",
        status=status,
    )


def coverage(market_values: FloatArray, covered: FloatArray) -> float:
    values = np.asarray(market_values, dtype=float)
    fractions = np.asarray(covered, dtype=float)
    if (
        values.shape != fractions.shape
        or values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or not np.isfinite(fractions).all()
        or np.any(fractions < 0.0)
        or np.any(fractions > 1.0)
    ):
        raise ValueError("coverage inputs are invalid")
    gross = float(np.abs(values).sum())
    if gross <= 0.0:
        raise ValueError("gross market value must be positive")
    return float((np.abs(values) * fractions).sum() / gross)


def risk_coverage_profile(
    market_values: FloatArray,
    deltas: FloatArray,
    vegas: FloatArray,
    stress_losses: FloatArray,
    notionals: FloatArray,
    covered: FloatArray,
    *,
    minimum_coverage: float,
) -> RiskCoverageProfile:
    """Report parallel frozen-universe coverages for a nonlinear book."""

    arrays = [
        np.asarray(values, dtype=float)
        for values in (market_values, deltas, vegas, stress_losses, notionals)
    ]
    fractions = np.asarray(covered, dtype=float)
    if (
        fractions.ndim != 1
        or fractions.size == 0
        or any(values.shape != fractions.shape for values in arrays)
        or any(not np.isfinite(values).all() for values in arrays)
        or not np.isfinite(fractions).all()
        or np.any((fractions < 0.0) | (fractions > 1.0))
        or not np.isfinite(minimum_coverage)
        or not 0.0 <= minimum_coverage <= 1.0
    ):
        raise ValueError("risk coverage inputs are invalid")

    def weighted_coverage(values: FloatArray) -> float:
        weights = np.abs(values)
        denominator = float(weights.sum())
        # No exposure in a declared dimension is fully covered, not missing.
        return 1.0 if denominator == 0.0 else float(weights @ fractions / denominator)

    coverages = tuple(weighted_coverage(values) for values in arrays)
    governing = min(coverages)
    return RiskCoverageProfile(
        market_value=coverages[0],
        absolute_delta=coverages[1],
        vega=coverages[2],
        stress_loss=coverages[3],
        notional=coverages[4],
        governing_coverage=governing,
        status="PASS" if governing >= minimum_coverage else "REJECT",
    )


def liquidation_cost(
    quantities: FloatArray,
    impact_coefficients: FloatArray,
    daily_volumes: FloatArray,
) -> float:
    quantity = np.abs(np.asarray(quantities, dtype=float))
    eta = np.asarray(impact_coefficients, dtype=float)
    volume = np.asarray(daily_volumes, dtype=float)
    if (
        quantity.shape != eta.shape
        or quantity.shape != volume.shape
        or not np.isfinite(quantity).all()
        or not np.isfinite(eta).all()
        or not np.isfinite(volume).all()
        or np.any(eta < 0.0)
        or np.any(volume <= 0.0)
    ):
        raise ValueError("invalid liquidation inputs")
    return float(np.sum(quantity * eta * np.sqrt(quantity / volume)))
