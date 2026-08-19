from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HedgePnlReconciliation:
    option_pnl: float
    delta_hedge_pnl: float
    financing_pnl: float
    dividend_pnl: float
    execution_cost: float
    attributed_pnl: float
    accounting_pnl: float
    event_residuals: tuple[float, ...]
    max_abs_event_residual: float
    residual: float
    status: str


def gamma_theta_pnl(
    gamma: np.ndarray,
    theta: np.ndarray,
    spot: np.ndarray,
    returns: np.ndarray,
    dt: float,
) -> np.ndarray:
    arrays = tuple(np.asarray(value, dtype=float) for value in (gamma, theta, spot, returns))
    if (
        not np.isfinite(dt)
        or dt <= 0.0
        or any(not np.isfinite(value).all() for value in arrays)
    ):
        raise ValueError("gamma-theta inputs must be finite and dt positive")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("gamma-theta arrays must have matching shapes")
    gamma_array, theta_array, spot_array, return_array = arrays
    return (
        0.5 * gamma_array * spot_array**2 * return_array**2
        + theta_array * dt
    )


def variance_pnl_approx(
    gamma: float,
    spot: float,
    realized_variance: float,
    implied_variance: float,
    horizon: float,
    *,
    variance_convention: str,
    horizon_unit: str,
) -> float:
    values = (gamma, spot, realized_variance, implied_variance, horizon)
    if (
        not np.isfinite(values).all()
        or spot <= 0.0
        or realized_variance < 0.0
        or implied_variance < 0.0
        or horizon < 0.0
    ):
        raise ValueError("invalid variance-PnL inputs")
    if variance_convention != "annualized_continuously_compounded_return":
        raise ValueError("unsupported variance convention")
    if horizon_unit != "years":
        raise ValueError("horizon_unit must be years")
    return float(
        0.5
        * gamma
        * spot
        * spot
        * (realized_variance - implied_variance)
        * horizon
    )


def self_financing_hedge_reconciliation(
    *,
    option_pnl: np.ndarray,
    delta_hedge_pnl: np.ndarray,
    financing_pnl: np.ndarray,
    dividend_pnl: np.ndarray,
    execution_cost: np.ndarray,
    accounting_pnl: np.ndarray,
    tolerance: float = 1e-10,
) -> HedgePnlReconciliation:
    arrays = tuple(
        np.asarray(value, dtype=float)
        for value in (
            option_pnl,
            delta_hedge_pnl,
            financing_pnl,
            dividend_pnl,
            execution_cost,
            accounting_pnl,
        )
    )
    if (
        not np.isfinite(tolerance)
        or tolerance < 0.0
        or any(not np.isfinite(value).all() for value in arrays)
    ):
        raise ValueError("reconciliation inputs must be finite")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("reconciliation arrays must have matching shapes")
    if arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("reconciliation inputs must be non-empty one-dimensional arrays")
    option, delta, financing, dividend, costs, accounting = arrays
    if np.any(costs < 0.0):
        raise ValueError("execution costs must be non-negative")
    attributed = option + delta + financing + dividend - costs
    event_residuals = accounting - attributed
    residual = float(np.sum(event_residuals))
    event_scales = np.maximum(1.0, np.abs(accounting))
    event_limits = tolerance * event_scales
    max_abs_event_residual = float(np.max(np.abs(event_residuals)))
    scale = max(1.0, abs(float(np.sum(accounting))))
    aggregate_passes = abs(residual) <= tolerance * scale
    events_pass = bool(np.all(np.abs(event_residuals) <= event_limits))
    return HedgePnlReconciliation(
        option_pnl=float(np.sum(option)),
        delta_hedge_pnl=float(np.sum(delta)),
        financing_pnl=float(np.sum(financing)),
        dividend_pnl=float(np.sum(dividend)),
        execution_cost=float(np.sum(costs)),
        attributed_pnl=float(np.sum(attributed)),
        accounting_pnl=float(np.sum(accounting)),
        event_residuals=tuple(float(value) for value in event_residuals),
        max_abs_event_residual=max_abs_event_residual,
        residual=residual,
        status="PASS" if aggregate_passes and events_pass else "REJECT",
    )


def hedge_reconciliation(
    option_change: np.ndarray,
    delta: np.ndarray,
    spot_change: np.ndarray,
    fees: np.ndarray,
) -> dict[str, float]:
    option = np.asarray(option_change, dtype=float)
    delta_array = np.asarray(delta, dtype=float)
    spot = np.asarray(spot_change, dtype=float)
    fee_array = np.asarray(fees, dtype=float)
    if (
        len({option.shape, delta_array.shape, spot.shape, fee_array.shape}) != 1
        or any(
            not value.all()
            for value in (
                np.isfinite(option),
                np.isfinite(delta_array),
                np.isfinite(spot),
                np.isfinite(fee_array),
            )
        )
    ):
        raise ValueError("legacy reconciliation inputs must be finite and aligned")
    if option.ndim != 1 or option.size == 0:
        raise ValueError("legacy reconciliation arrays must be non-empty and one-dimensional")
    if np.any(fee_array < 0.0):
        raise ValueError("fees must be non-negative")
    hedge = -delta_array * spot
    attributed = option + hedge - fee_array
    return {
        "total": float(np.sum(option)),
        "delta_hedge": float(np.sum(hedge)),
        "fees": float(-np.sum(fee_array)),
        "residual": float(np.sum(attributed)),
    }
