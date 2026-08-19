from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExecutionSchedule:
    times: FloatArray
    holdings: FloatArray
    trade_rates: FloatArray
    expected_temporary_cost: float
    inventory_variance: float
    risk_penalty: float
    status: str


def _stable_holding_ratio(kappa: float, horizon: float, times: FloatArray) -> FloatArray:
    z = kappa * horizon
    if z < 1e-7:
        return 1.0 - times / horizon
    remaining = horizon - times
    if z < 40.0:
        return np.sinh(kappa * remaining) / np.sinh(z)
    # Scaled form avoids sinh overflow for long horizons/high risk aversion.
    numerator = np.exp(-kappa * times) * (
        1.0 - np.exp(-2.0 * kappa * remaining)
    )
    denominator = 1.0 - np.exp(-2.0 * z)
    return numerator / denominator


def optimal_schedule(
    shares: float,
    horizon: float,
    intervals: int,
    temporary_impact: float,
    risk_aversion: float,
    volatility: float,
) -> ExecutionSchedule:
    values = np.asarray(
        [shares, horizon, temporary_impact, risk_aversion, volatility],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("schedule inputs must be finite")
    if (
        shares <= 0.0
        or horizon <= 0.0
        or isinstance(intervals, bool)
        or not isinstance(intervals, int)
        or intervals <= 0
    ):
        raise ValueError("shares, horizon and intervals must be positive")
    if temporary_impact <= 0.0 or risk_aversion < 0.0 or volatility < 0.0:
        raise ValueError("invalid impact, risk aversion or volatility")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        times = np.linspace(0.0, horizon, intervals + 1)
        if risk_aversion == 0.0 or volatility == 0.0:
            holdings = shares * (1.0 - times / horizon)
        else:
            kappa = np.sqrt(risk_aversion * volatility**2 / temporary_impact)
            holdings = shares * _stable_holding_ratio(kappa, horizon, times)
        dt = horizon / intervals
        trades = holdings[:-1] - holdings[1:]
        rates = trades / dt
        temporary = float(temporary_impact * np.sum(rates**2) * dt)
        variance = float(volatility**2 * np.sum(holdings[:-1] ** 2) * dt)
        risk_penalty = float(risk_aversion * variance)
    tolerance = 1e-10 * max(1.0, shares)
    if (
        not np.isfinite(times).all()
        or not np.isfinite(holdings).all()
        or not np.isfinite(rates).all()
        or not np.isfinite([temporary, variance, risk_penalty]).all()
    ):
        raise ValueError("execution schedule produced non-finite output")
    if (
        abs(float(holdings[0]) - shares) > tolerance
        or abs(float(holdings[-1])) > tolerance
        or np.any(np.diff(holdings) > tolerance)
        or abs(float(np.sum(trades)) - shares) > tolerance
    ):
        raise ValueError("execution schedule failed liquidation postconditions")
    return ExecutionSchedule(
        times,
        holdings,
        rates,
        temporary,
        variance,
        risk_penalty,
        "PASS",
    )
