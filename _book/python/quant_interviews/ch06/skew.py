from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class SkewFit:
    intercept: float
    slope: float
    slope_standard_error: float
    slope_confidence_interval: tuple[float, float]
    residual_rmse: float
    coordinate: str
    degrees_of_freedom: int
    confidence_level: float
    confidence_interval_method: str
    maximum_slope: float
    status: str


def skew_fit(
    *,
    strikes: np.ndarray,
    forward: float,
    vols: np.ndarray,
    confidence_level: float = 0.95,
    maximum_slope: float = 0.0,
) -> SkewFit:
    strike_array = np.asarray(strikes, dtype=float)
    vol_array = np.asarray(vols, dtype=float)
    if (
        forward <= 0.0
        or strike_array.ndim != 1
        or vol_array.shape != strike_array.shape
        or strike_array.size < 3
        or not np.isfinite(strike_array).all()
        or not np.isfinite(vol_array).all()
        or np.any(strike_array <= 0.0)
        or np.any(np.diff(strike_array) <= 0.0)
        or np.any(vol_array <= 0.0)
        or not np.isfinite(confidence_level)
        or not 0.0 < confidence_level < 1.0
        or not np.isfinite(maximum_slope)
    ):
        raise ValueError("invalid skew-fit inputs")
    x = np.log(strike_array / forward)
    design = np.c_[np.ones(x.size), x]
    beta = np.linalg.lstsq(design, vol_array, rcond=None)[0]
    residual = vol_array - design @ beta
    degrees = x.size - 2
    residual_variance = float(residual @ residual / degrees)
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    slope_standard_error = float(np.sqrt(covariance[1, 1]))
    slope = float(beta[1])
    critical_value = float(
        student_t.ppf(0.5 + 0.5 * confidence_level, degrees)
    )
    confidence_interval = (
        slope - critical_value * slope_standard_error,
        slope + critical_value * slope_standard_error,
    )
    if not np.isfinite((slope_standard_error, critical_value)).all():
        status = "REJECT"
    elif confidence_interval[1] <= maximum_slope:
        status = "PASS"
    elif slope <= maximum_slope:
        status = "CONDITIONAL_PASS"
    else:
        status = "REJECT"
    return SkewFit(
        intercept=float(beta[0]),
        slope=slope,
        slope_standard_error=slope_standard_error,
        slope_confidence_interval=confidence_interval,
        residual_rmse=float(np.sqrt(np.mean(residual * residual))),
        coordinate="log-forward-moneyness",
        degrees_of_freedom=degrees,
        confidence_level=confidence_level,
        confidence_interval_method="Student-t OLS",
        maximum_slope=maximum_slope,
        status=status,
    )


def linear_log_moneyness_skew(
    strikes: np.ndarray,
    forward: float,
    vols: np.ndarray,
) -> tuple[float, float]:
    result = skew_fit(strikes=strikes, forward=forward, vols=vols)
    return result.intercept, result.slope


def put_call_skew_premium(put_vol: float, atm_vol: float) -> float:
    if not np.isfinite((put_vol, atm_vol)).all():
        raise ValueError("volatilities must be finite")
    return float(put_vol - atm_vol)
