from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ImpactFit:
    coefficient: float
    intercept: float
    r_squared: float
    observations: int
    input_rows: int
    dropped_rows: int
    residual_standard_error: float
    coefficient_standard_error: float
    coefficient_confidence_interval: tuple[float, float]
    condition_number: float
    support_minimum: float
    support_maximum: float
    status: str


def _validate(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    if xv.ndim != 1 or yv.ndim != 1 or len(xv) != len(yv) or len(xv) < 5:
        raise ValueError("x and y must be one-dimensional, aligned and length >= 5")
    if not np.isfinite(xv).all() or not np.isfinite(yv).all():
        raise ValueError("impact fits fail closed on non-finite observations")
    if float(np.ptp(xv)) <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(xv)))):
        raise ValueError("impact regressor has no identifiable variation")
    return xv, yv


def _fit_linear(
    x: FloatArray,
    y: FloatArray,
    *,
    confidence_level: float,
    minimum_r_squared: float,
    maximum_condition_number: float,
) -> ImpactFit:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0,1)")
    if not np.isfinite(minimum_r_squared) or not -1.0 <= minimum_r_squared <= 1.0:
        raise ValueError("minimum_r_squared must lie in [-1,1]")
    if not np.isfinite(maximum_condition_number) or maximum_condition_number <= 1.0:
        raise ValueError("maximum_condition_number must be finite and above one")
    design = np.column_stack([np.ones_like(x), x])
    condition_number = float(np.linalg.cond(design))
    if not np.isfinite(condition_number):
        raise ValueError("impact design matrix is singular")
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    degrees_of_freedom = len(x) - 2
    residual_se = float(np.sqrt(ss_res / degrees_of_freedom))
    centered_sum = float(np.sum((x - x.mean()) ** 2))
    coefficient_se = residual_se / np.sqrt(centered_sum)
    critical = float(
        student_t.ppf(0.5 + confidence_level / 2.0, degrees_of_freedom)
    )
    interval = (
        float(beta[1] - critical * coefficient_se),
        float(beta[1] + critical * coefficient_se),
    )
    if beta[1] < 0.0 or condition_number > maximum_condition_number:
        status = "REJECT"
    elif interval[0] >= 0.0 and r2 >= minimum_r_squared:
        status = "PASS"
    else:
        status = "CONDITIONAL_PASS"
    return ImpactFit(
        coefficient=float(beta[1]),
        intercept=float(beta[0]),
        r_squared=r2,
        observations=len(x),
        input_rows=len(x),
        dropped_rows=0,
        residual_standard_error=residual_se,
        coefficient_standard_error=float(coefficient_se),
        coefficient_confidence_interval=interval,
        condition_number=condition_number,
        support_minimum=float(np.min(x)),
        support_maximum=float(np.max(x)),
        status=status,
    )


def fit_kyle_lambda(
    signed_volume: FloatArray,
    price_change: FloatArray,
    *,
    confidence_level: float = 0.95,
    minimum_r_squared: float = 0.0,
    maximum_condition_number: float = 1e8,
) -> ImpactFit:
    x, y = _validate(signed_volume, price_change)
    return _fit_linear(
        x,
        y,
        confidence_level=confidence_level,
        minimum_r_squared=minimum_r_squared,
        maximum_condition_number=maximum_condition_number,
    )


def fit_square_root_coefficient(
    participation_fraction: FloatArray,
    impact_in_volatility_units: FloatArray,
    *,
    confidence_level: float = 0.95,
    minimum_r_squared: float = 0.0,
    maximum_condition_number: float = 1e8,
) -> ImpactFit:
    q, y = _validate(participation_fraction, impact_in_volatility_units)
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("participation fractions must lie in [0,1]")
    x = np.sqrt(q)
    return _fit_linear(
        x,
        y,
        confidence_level=confidence_level,
        minimum_r_squared=minimum_r_squared,
        maximum_condition_number=maximum_condition_number,
    )


@dataclass(frozen=True)
class CapacityDiagnostics:
    optimal_quantity: float
    break_even_quantity: float
    expected_profit_at_optimum: float
    average_impact_cost_at_optimum: float
    participation_at_optimum: float
    participation_at_break_even: float
    optimal_extrapolation_ratio: float
    break_even_extrapolation_ratio: float
    optimal_status: str
    break_even_status: str
    status: str

    @property
    def extrapolation_ratio(self) -> float:
        """Compatibility alias for the optimal operating-point diagnostic."""

        return self.optimal_extrapolation_ratio


def square_root_capacity(
    *,
    alpha_per_unit: float,
    fixed_cost_per_unit: float,
    impact_coefficient: float,
    volatility: float,
    daily_volume: float,
    maximum_calibration_participation: float,
) -> CapacityDiagnostics:
    values = np.asarray(
        [
            alpha_per_unit,
            fixed_cost_per_unit,
            impact_coefficient,
            volatility,
            daily_volume,
            maximum_calibration_participation,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("capacity inputs must be finite")
    if (
        impact_coefficient <= 0.0
        or volatility <= 0.0
        or daily_volume <= 0.0
        or maximum_calibration_participation <= 0.0
    ):
        raise ValueError("impact, volatility, volume and calibration range must be positive")
    net_alpha = alpha_per_unit - fixed_cost_per_unit
    if net_alpha <= 0.0:
        return CapacityDiagnostics(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "REJECT",
            "REJECT",
            "REJECT",
        )
    # Convention: I(Q)=Y sigma sqrt(Q/V) is average execution cost per unit.
    optimal = daily_volume * (2.0 * net_alpha / (3.0 * impact_coefficient * volatility)) ** 2
    break_even = daily_volume * (net_alpha / (impact_coefficient * volatility)) ** 2
    average_impact = impact_coefficient * volatility * np.sqrt(optimal / daily_volume)
    profit = optimal * (net_alpha - average_impact)
    participation = optimal / daily_volume
    break_even_participation = break_even / daily_volume
    optimal_extrapolation = participation / maximum_calibration_participation
    break_even_extrapolation = (
        break_even_participation / maximum_calibration_participation
    )
    optimal_status = "PASS" if optimal_extrapolation <= 1.0 else "OUTSIDE_SUPPORT"
    break_even_status = (
        "PASS" if break_even_extrapolation <= 1.0 else "OUTSIDE_SUPPORT"
    )
    status = (
        "PASS"
        if optimal_status == break_even_status == "PASS"
        else "CONDITIONAL_PASS"
    )
    outputs = np.asarray(
        [optimal, break_even, average_impact, profit, optimal_extrapolation, break_even_extrapolation]
    )
    if not np.isfinite(outputs).all():
        raise ValueError("capacity calculation produced non-finite output")
    return CapacityDiagnostics(
        float(optimal),
        float(break_even),
        float(profit),
        float(average_impact),
        float(participation),
        float(break_even_participation),
        float(optimal_extrapolation),
        float(break_even_extrapolation),
        optimal_status,
        break_even_status,
        status,
    )
