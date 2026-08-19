from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm


@dataclass(frozen=True)
class SurfaceViolation:
    maturity: float
    log_forward_moneyness: float
    constraint: str
    residual: float


@dataclass(frozen=True)
class SurfaceAudit:
    violations: tuple[SurfaceViolation, ...]
    minimum_call_slope_residual: float
    minimum_convexity_residual: float
    minimum_calendar_total_variance_increment: float
    minimum_normalized_calendar_call_increment: float
    bid_ask_feasible: bool | None
    executable_prices: np.ndarray | None
    implied_vol_reprices: np.ndarray
    repair_l1_norm: float | None
    maximum_constraint_violation: float
    status: str

    @property
    def maximum_mid_violation(self) -> float:
        """Compatibility alias; the value now covers every audited constraint."""

        return self.maximum_constraint_violation


def total_variance(vol: np.ndarray, maturity: np.ndarray) -> np.ndarray:
    vol_array = np.asarray(vol, float)
    maturity_array = np.asarray(maturity, float)
    if (
        not np.isfinite(vol_array).all()
        or not np.isfinite(maturity_array).all()
        or np.any(vol_array < 0.0)
        or np.any(maturity_array < 0.0)
    ):
        raise ValueError("volatility and maturity must be finite and non-negative")
    try:
        return vol_array**2 * maturity_array
    except ValueError as error:
        raise ValueError("volatility and maturity shapes are incompatible") from error


def surface_arbitrage_audit(
    *,
    log_forward_moneyness: np.ndarray,
    maturities: np.ndarray,
    forwards: np.ndarray,
    discounts: np.ndarray,
    call_prices: np.ndarray,
    implied_vols: np.ndarray,
    bid_prices: np.ndarray | None = None,
    ask_prices: np.ndarray | None = None,
    tolerance: float = 1e-10,
) -> SurfaceAudit:
    k = np.asarray(log_forward_moneyness, dtype=float)
    t = np.asarray(maturities, dtype=float)
    f = np.asarray(forwards, dtype=float)
    d = np.asarray(discounts, dtype=float)
    calls = np.asarray(call_prices, dtype=float)
    vols = np.asarray(implied_vols, dtype=float)
    arrays = (k, t, f, d, calls, vols)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("surface inputs must be finite")
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance cannot be negative")
    if k.ndim != 1 or t.ndim != 1 or f.shape != t.shape or d.shape != t.shape:
        raise ValueError("invalid surface coordinate shapes")
    if calls.shape != (t.size, k.size) or vols.shape != calls.shape:
        raise ValueError("surface value shape mismatch")
    if k.size < 3 or t.size < 2:
        raise ValueError("surface audit needs at least three strikes and two maturities")
    if np.any(np.diff(k) <= 0.0) or np.any(np.diff(t) <= 0.0):
        raise ValueError("coordinates must be strictly increasing")
    if np.any(f <= 0.0) or np.any(d <= 0.0) or np.any(vols <= 0.0):
        raise ValueError("forwards, discounts and implied vols must be positive")

    violations: list[SurfaceViolation] = []
    minimum_slope_residual = float("inf")
    minimum_convexity_residual = float("inf")
    maximum_constraint_violation = 0.0
    midpoint_outside_box = False
    implied_vol_reprices = np.empty_like(calls)

    def add_violation(
        maturity: float,
        coordinate: float,
        constraint: str,
        residual: float,
    ) -> None:
        nonlocal maximum_constraint_violation
        violations.append(
            SurfaceViolation(maturity, coordinate, constraint, residual)
        )
        maximum_constraint_violation = max(
            maximum_constraint_violation, residual
        )
    for maturity_index, maturity in enumerate(t):
        strikes = f[maturity_index] * np.exp(k)
        row = calls[maturity_index]
        lower = d[maturity_index] * np.maximum(f[maturity_index] - strikes, 0.0)
        upper = np.full_like(strikes, d[maturity_index] * f[maturity_index])
        for strike_index, log_moneyness in enumerate(k):
            if row[strike_index] < lower[strike_index] - tolerance:
                residual = lower[strike_index] - row[strike_index]
                add_violation(
                    float(maturity),
                    float(log_moneyness),
                    "LOWER_PRICE_BOUND",
                    float(residual),
                )
            if row[strike_index] > upper[strike_index] + tolerance:
                residual = row[strike_index] - upper[strike_index]
                add_violation(
                    float(maturity),
                    float(log_moneyness),
                    "UPPER_PRICE_BOUND",
                    float(residual),
                )

            volatility = vols[maturity_index, strike_index]
            root_t = np.sqrt(maturity)
            d1 = (-log_moneyness + 0.5 * volatility * volatility * maturity) / (
                volatility * root_t
            )
            d2 = d1 - volatility * root_t
            price_from_vol = d[maturity_index] * (
                f[maturity_index] * norm.cdf(d1)
                - strikes[strike_index] * norm.cdf(d2)
            )
            implied_vol_reprices[maturity_index, strike_index] = price_from_vol
            price_vol_residual = abs(row[strike_index] - price_from_vol)
            if price_vol_residual > tolerance * max(1.0, abs(row[strike_index])):
                add_violation(
                    float(maturity),
                    float(log_moneyness),
                    "PRICE_VOL_INCONSISTENCY",
                    float(price_vol_residual),
                )

        slopes = np.diff(row) / np.diff(strikes)
        slope_upper_residual = -slopes
        slope_lower_residual = slopes + d[maturity_index]
        minimum_slope_residual = min(
            minimum_slope_residual,
            float(np.min(slope_upper_residual)),
            float(np.min(slope_lower_residual)),
        )
        for strike_index, slope in enumerate(slopes):
            coordinate = 0.5 * (k[strike_index] + k[strike_index + 1])
            if slope > tolerance:
                add_violation(
                    float(maturity),
                    float(coordinate),
                    "STRIKE_MONOTONICITY",
                    float(slope),
                )
            if slope < -d[maturity_index] - tolerance:
                add_violation(
                    float(maturity),
                    float(coordinate),
                    "VERTICAL_SPREAD_LOWER_BOUND",
                    float(-d[maturity_index] - slope),
                )
        convexity = np.diff(slopes)
        minimum_convexity_residual = min(
            minimum_convexity_residual,
            float(np.min(convexity)),
        )
        for strike_index, residual in enumerate(convexity):
            if residual < -tolerance:
                add_violation(
                    float(maturity),
                    float(k[strike_index + 1]),
                    "BUTTERFLY_CONVEXITY",
                    float(-residual),
                )

    total_variances = vols * vols * t[:, None]
    calendar_increments = np.diff(total_variances, axis=0)
    minimum_calendar_increment = float(np.min(calendar_increments))
    for maturity_index, row in enumerate(calendar_increments):
        for strike_index, increment in enumerate(row):
            if increment < -tolerance:
                add_violation(
                    float(t[maturity_index + 1]),
                    float(k[strike_index]),
                    "CALENDAR_TOTAL_VARIANCE",
                    float(-increment),
                )

    normalized_calls = calls / (d[:, None] * f[:, None])
    normalized_calendar_increments = np.diff(normalized_calls, axis=0)
    minimum_normalized_calendar_increment = float(
        np.min(normalized_calendar_increments)
    )
    for maturity_index, row in enumerate(normalized_calendar_increments):
        for strike_index, increment in enumerate(row):
            if increment < -tolerance:
                add_violation(
                    float(t[maturity_index + 1]),
                    float(k[strike_index]),
                    "NORMALIZED_CALL_CALENDAR",
                    float(-increment),
                )

    bid_ask_feasible: bool | None = None
    executable_prices: np.ndarray | None = None
    repair_l1_norm: float | None = None
    if bid_prices is not None or ask_prices is not None:
        if bid_prices is None or ask_prices is None:
            raise ValueError("bid and ask grids must be provided together")
        bid = np.asarray(bid_prices, dtype=float)
        ask = np.asarray(ask_prices, dtype=float)
        if (
            bid.shape != calls.shape
            or ask.shape != calls.shape
            or not np.isfinite(bid).all()
            or not np.isfinite(ask).all()
            or np.any(bid > ask)
            or np.any(bid < 0.0)
        ):
            raise ValueError("invalid bid/ask grids")
        for maturity_index, maturity in enumerate(t):
            for strike_index, coordinate in enumerate(k):
                if calls[maturity_index, strike_index] < bid[maturity_index, strike_index] - tolerance:
                    midpoint_outside_box = True
                    add_violation(
                        float(maturity),
                        float(coordinate),
                        "MID_BELOW_BID",
                        float(bid[maturity_index, strike_index] - calls[maturity_index, strike_index]),
                    )
                if calls[maturity_index, strike_index] > ask[maturity_index, strike_index] + tolerance:
                    midpoint_outside_box = True
                    add_violation(
                        float(maturity),
                        float(coordinate),
                        "MID_ABOVE_ASK",
                        float(calls[maturity_index, strike_index] - ask[maturity_index, strike_index]),
                    )

        price_variable_count = t.size * k.size
        variable_count = 2 * price_variable_count
        a_ub: list[np.ndarray] = []
        b_ub: list[float] = []

        def price_index(maturity_index: int, strike_index: int) -> int:
            return maturity_index * k.size + strike_index

        for maturity_index in range(t.size):
            strikes = f[maturity_index] * np.exp(k)
            for index in range(k.size - 1):
                monotonic = np.zeros(variable_count)
                monotonic[price_index(maturity_index, index + 1)] = 1.0
                monotonic[price_index(maturity_index, index)] = -1.0
                a_ub.append(monotonic)
                b_ub.append(0.0)
                lower_slope = -monotonic
                a_ub.append(lower_slope)
                b_ub.append(d[maturity_index] * (strikes[index + 1] - strikes[index]))
            for index in range(k.size - 2):
                left_width = strikes[index + 1] - strikes[index]
                right_width = strikes[index + 2] - strikes[index + 1]
                convex = np.zeros(variable_count)
                convex[price_index(maturity_index, index)] = -1.0 / left_width
                convex[price_index(maturity_index, index + 1)] = 1.0 / left_width + 1.0 / right_width
                convex[price_index(maturity_index, index + 2)] = -1.0 / right_width
                a_ub.append(convex)
                b_ub.append(0.0)
        for maturity_index in range(t.size - 1):
            for strike_index in range(k.size):
                calendar = np.zeros(variable_count)
                calendar[price_index(maturity_index, strike_index)] = 1.0 / (
                    d[maturity_index] * f[maturity_index]
                )
                calendar[price_index(maturity_index + 1, strike_index)] = -1.0 / (
                    d[maturity_index + 1] * f[maturity_index + 1]
                )
                a_ub.append(calendar)
                b_ub.append(0.0)

        flat_mid = calls.ravel()
        for index, midpoint in enumerate(flat_mid):
            positive = np.zeros(variable_count)
            positive[index] = 1.0
            positive[price_variable_count + index] = -1.0
            a_ub.append(positive)
            b_ub.append(float(midpoint))
            negative = np.zeros(variable_count)
            negative[index] = -1.0
            negative[price_variable_count + index] = -1.0
            a_ub.append(negative)
            b_ub.append(float(-midpoint))

        price_bounds: list[tuple[float, float]] = []
        bounds_valid = True
        for maturity_index in range(t.size):
            strikes = f[maturity_index] * np.exp(k)
            no_arbitrage_lower = d[maturity_index] * np.maximum(
                f[maturity_index] - strikes, 0.0
            )
            no_arbitrage_upper = np.full_like(
                strikes, d[maturity_index] * f[maturity_index]
            )
            for strike_index in range(k.size):
                lower_bound = max(
                    float(bid[maturity_index, strike_index]),
                    float(no_arbitrage_lower[strike_index]),
                    float(
                        implied_vol_reprices[maturity_index, strike_index]
                        - tolerance
                        * max(
                            1.0,
                            abs(implied_vol_reprices[maturity_index, strike_index]),
                        )
                    ),
                )
                upper_bound = min(
                    float(ask[maturity_index, strike_index]),
                    float(no_arbitrage_upper[strike_index]),
                    float(
                        implied_vol_reprices[maturity_index, strike_index]
                        + tolerance
                        * max(
                            1.0,
                            abs(implied_vol_reprices[maturity_index, strike_index]),
                        )
                    ),
                )
                bounds_valid &= lower_bound <= upper_bound
                price_bounds.append((lower_bound, upper_bound))
        if bounds_valid:
            result = linprog(
                c=np.r_[np.zeros(price_variable_count), np.ones(price_variable_count)],
                A_ub=np.asarray(a_ub),
                b_ub=np.asarray(b_ub),
                bounds=[*price_bounds, *[(0.0, None)] * price_variable_count],
                method="highs",
            )
            bid_ask_feasible = bool(result.success)
            if result.success:
                executable_prices = result.x[:price_variable_count].reshape(calls.shape)
                repair_l1_norm = float(np.sum(np.abs(executable_prices - calls)))
        else:
            bid_ask_feasible = False

    if bid_ask_feasible is False or midpoint_outside_box:
        status = "REJECT"
    elif not violations:
        status = "PASS"
    elif bid_ask_feasible is True:
        status = "CONDITIONAL_PASS"
    else:
        status = "REJECT"
    return SurfaceAudit(
        violations=tuple(violations),
        minimum_call_slope_residual=minimum_slope_residual,
        minimum_convexity_residual=minimum_convexity_residual,
        minimum_calendar_total_variance_increment=minimum_calendar_increment,
        minimum_normalized_calendar_call_increment=minimum_normalized_calendar_increment,
        bid_ask_feasible=bid_ask_feasible,
        executable_prices=executable_prices,
        implied_vol_reprices=implied_vol_reprices,
        repair_l1_norm=repair_l1_norm,
        maximum_constraint_violation=maximum_constraint_violation,
        status=status,
    )


def calendar_total_variance_violations(
    vol_grid: np.ndarray,
    maturities: np.ndarray,
    tol: float = 1e-12,
) -> list[tuple[int, int, float]]:
    vol_array = np.asarray(vol_grid, float)
    maturity_array = np.asarray(maturities, float)
    if (
        vol_array.ndim != 2
        or maturity_array.ndim != 1
        or vol_array.shape[0] != maturity_array.size
        or maturity_array.size < 2
        or not np.isfinite(vol_array).all()
        or not np.isfinite(maturity_array).all()
        or np.any(vol_array < 0.0)
        or np.any(maturity_array <= 0.0)
        or not isfinite(tol)
        or tol < 0.0
    ):
        raise ValueError("invalid total-variance calendar inputs")
    w = vol_array**2 * maturity_array[:, None]
    return [
        (i, j, float(value))
        for i in range(w.shape[0] - 1)
        for j, value in enumerate(w[i] - w[i + 1])
        if value > tol
    ]


def convex_call_violations(
    strikes: np.ndarray,
    calls: np.ndarray,
    tol: float = 1e-12,
) -> list[int]:
    strike_array = np.asarray(strikes, float)
    call_array = np.asarray(calls, float)
    if (
        strike_array.ndim != 1
        or call_array.shape != strike_array.shape
        or np.any(np.diff(strike_array) <= 0.0)
        or not np.isfinite(strike_array).all()
        or not np.isfinite(call_array).all()
        or not isfinite(tol)
        or tol < 0.0
    ):
        raise ValueError("strikes must be unique, sorted and shape-consistent")
    slopes = np.diff(call_array) / np.diff(strike_array)
    return [i for i, value in enumerate(np.diff(slopes)) if value < -tol]
