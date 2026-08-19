"""Curve interpolation and executable no-arbitrage audits."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _strict_nodes(
    coordinates: ArrayLike,
    values: ArrayLike,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(coordinates, dtype=float)
    observations = np.asarray(values, dtype=float)
    if (
        nodes.ndim != 1
        or observations.shape != nodes.shape
        or nodes.size < 2
        or not np.all(np.isfinite(nodes))
        or not np.all(np.isfinite(observations))
        or np.any(np.diff(nodes) <= 0.0)
    ):
        raise ValueError(f"{name} nodes must be finite, aligned and increasing")
    return nodes, observations


def interpolate_log_discount(
    times: ArrayLike,
    discounts: ArrayLike,
    query: ArrayLike,
) -> np.ndarray:
    """Interpolate positive discount factors without silent extrapolation."""

    nodes, values = _strict_nodes(times, discounts, "discount")
    requested = np.asarray(query, dtype=float)
    if (
        not np.all(np.isfinite(requested))
        or np.any(values <= 0.0)
        or np.any(requested < nodes[0])
        or np.any(requested > nodes[-1])
    ):
        raise ValueError("positive discounts and in-range finite query required")
    return np.exp(np.interp(requested, nodes, np.log(values)))


def monotone_total_variance(
    times: ArrayLike,
    total_variance: ArrayLike,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Check calendar monotonicity at a fixed forward-moneyness coordinate."""

    _, variance = _strict_nodes(times, total_variance, "total-variance")
    if (
        np.any(variance < 0.0)
        or not np.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("nonnegative total variance and tolerance required")
    return bool(np.all(np.diff(variance) >= -tolerance))


def call_price_arbitrage_audit(
    strikes: ArrayLike,
    calls: ArrayLike,
    *,
    forward: float,
    discount: float,
    tolerance: float | None = None,
    price_tolerance: float = 1e-10,
    slope_tolerance: float = 1e-10,
    convexity_tolerance: float = 1e-10,
) -> dict[str, float | int | str]:
    """Audit bounds, decreasing strike prices and discrete convexity."""

    strike, prices = _strict_nodes(strikes, calls, "call-price")
    if strike.size < 3:
        raise ValueError("at least three strikes are required for convexity evidence")
    if tolerance is not None:
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("legacy tolerance must be finite and nonnegative")
        price_tolerance = slope_tolerance = convexity_tolerance = tolerance
    if (
        not np.isfinite(forward)
        or not np.isfinite(discount)
        or forward <= 0.0
        or discount <= 0.0
        or not all(
            np.isfinite(value) and value >= 0.0
            for value in (
                price_tolerance,
                slope_tolerance,
                convexity_tolerance,
            )
        )
    ):
        raise ValueError("finite positive market inputs and tolerances required")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lower = discount * np.maximum(forward - strike, 0.0)
        upper = np.full_like(strike, discount * forward)
        first_slopes = np.diff(prices) / np.diff(strike)
        slope_changes = np.diff(first_slopes)
        density_proxy = 2.0 * slope_changes / (strike[2:] - strike[:-2])
    if not all(
        np.all(np.isfinite(values))
        for values in (lower, upper, first_slopes, slope_changes, density_proxy)
    ):
        raise ValueError("derived arbitrage evidence must be finite")
    bound_violations = int(
        np.sum(
            (prices < lower - price_tolerance)
            | (prices > upper + price_tolerance)
        )
    )
    slope_upper_violations = int(np.sum(first_slopes > slope_tolerance))
    slope_lower_violations = int(
        np.sum(first_slopes < -discount - slope_tolerance)
    )
    convexity_violations = int(
        np.sum(slope_changes < -convexity_tolerance)
    )
    minimum_density_proxy = float(np.min(density_proxy))
    violations = (
        bound_violations
        + slope_upper_violations
        + slope_lower_violations
        + convexity_violations
    )
    return {
        "bound_violations": bound_violations,
        "monotonicity_violations": slope_upper_violations,
        "slope_upper_violations": slope_upper_violations,
        "slope_lower_violations": slope_lower_violations,
        "convexity_violations": convexity_violations,
        "minimum_density_proxy": minimum_density_proxy,
        "status": "PASS" if violations == 0 else "REJECT",
    }


def curve_surface_audit(
    discount_times: ArrayLike,
    discounts: ArrayLike,
    variance_times: ArrayLike,
    total_variance: ArrayLike,
    strikes: ArrayLike,
    calls: ArrayLike,
    *,
    forward: float,
    discount: float,
    price_tolerance: float = 1e-10,
    slope_tolerance: float = 1e-10,
    convexity_tolerance: float = 1e-10,
) -> dict[str, float | int | str | bool]:
    """Combine curve, calendar and strike no-arbitrage evidence."""

    time_nodes, discount_nodes = _strict_nodes(
        discount_times,
        discounts,
        "discount",
    )
    if np.any(discount_nodes <= 0.0):
        raise ValueError("discount factors must be positive")
    forwards = -np.diff(np.log(discount_nodes)) / np.diff(time_nodes)
    call_audit = call_price_arbitrage_audit(
        strikes,
        calls,
        forward=forward,
        discount=discount,
        price_tolerance=price_tolerance,
        slope_tolerance=slope_tolerance,
        convexity_tolerance=convexity_tolerance,
    )
    calendar_pass = monotone_total_variance(
        variance_times,
        total_variance,
    )
    status = (
        "PASS"
        if calendar_pass and call_audit["status"] == "PASS"
        else "REJECT"
    )
    return {
        **call_audit,
        "calendar_pass": calendar_pass,
        "maximum_forward_jump": float(
            np.max(np.abs(np.diff(forwards))) if forwards.size > 1 else 0.0
        ),
        "extrapolation_status": "NOT_USED",
        "status": status,
    }
