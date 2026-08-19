from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from typing import Any

from .black_scholes import normal_cdf


@dataclass(frozen=True)
class DigitalReplicationStudy:
    rows: tuple[dict[str, Any], ...]
    errors_monotone: bool
    execution_costs_monotone: bool
    final_observed_order: float | None
    minimum_order: float
    maximum_order: float
    status: str


def central_call_spread_digital(
    call_price: Callable[[float], float],
    *,
    strike: float,
    width: float,
) -> float:
    if (
        not isfinite(strike)
        or not isfinite(width)
        or strike <= 0.0
        or width <= 0.0
        or width >= 2.0 * strike
    ):
        raise ValueError("invalid call-spread controls")
    lower = call_price(strike - 0.5 * width)
    upper = call_price(strike + 0.5 * width)
    if not isfinite(lower) or not isfinite(upper):
        raise ValueError("call pricer returned a non-finite value")
    return (lower - upper) / width


def analytic_cash_digital_call(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
) -> float:
    values = (spot, strike, rate, dividend_yield, volatility, maturity)
    if not all(isfinite(value) for value in values):
        raise ValueError("digital inputs must be finite")
    if spot <= 0.0 or strike <= 0.0 or volatility <= 0.0 or maturity <= 0.0:
        raise ValueError("digital benchmark requires an interior contract")
    root_t = sqrt(maturity)
    d2 = (
        log(spot / strike)
        + (rate - dividend_yield - 0.5 * volatility**2) * maturity
    ) / (volatility * root_t)
    return exp(-rate * maturity) * normal_cdf(d2)


def digital_replication_convergence(
    call_price: Callable[[float], float],
    *,
    strike: float,
    widths: Sequence[float],
    analytic_digital: float,
    call_half_spread: float,
    contract_notional: float = 1.0,
    minimum_order: float = 1.5,
    maximum_order: float = 2.5,
) -> DigitalReplicationStudy:
    """Measure approximation error against execution and pin-risk proxies."""

    widths = tuple(float(width) for width in widths)
    if (
        len(widths) < 3
        or not isfinite(strike)
        or strike <= 0.0
        or any(
            not isfinite(width) or width <= 0.0 or width >= 2.0 * strike
            for width in widths
        )
        or any(widths[index + 1] >= widths[index] for index in range(len(widths) - 1))
        or not isfinite(analytic_digital)
        or analytic_digital < 0.0
        or not isfinite(call_half_spread)
        or call_half_spread < 0.0
        or not isfinite(contract_notional)
        or contract_notional <= 0.0
        or not isfinite(minimum_order)
        or not isfinite(maximum_order)
        or minimum_order < 0.0
        or maximum_order < minimum_order
    ):
        raise ValueError("invalid convergence-study controls")

    rows: list[dict[str, Any]] = []
    previous_error: float | None = None
    previous_width: float | None = None
    for width in widths:
        estimate = central_call_spread_digital(
            call_price,
            strike=strike,
            width=width,
        )
        error = abs(estimate - analytic_digital)
        observed_order = None
        if (
            previous_error is not None
            and previous_width is not None
            and error > 0.0
            and previous_error > 0.0
        ):
            observed_order = log(previous_error / error) / log(
                previous_width / width
            )
        scaled_notional = contract_notional / width
        rows.append(
            {
                "width": width,
                "spread_estimate": estimate,
                "analytic_digital": analytic_digital,
                "absolute_error": error,
                "observed_order": observed_order,
                "scaled_call_notional": scaled_notional,
                "estimated_crossing_cost": (
                    2.0 * call_half_spread * scaled_notional
                ),
                "pin_risk_proxy": scaled_notional,
            }
        )
        previous_error = error
        previous_width = width
    errors = tuple(float(row["absolute_error"]) for row in rows)
    costs = tuple(float(row["estimated_crossing_cost"]) for row in rows)
    errors_monotone = all(
        errors[index + 1] < errors[index]
        for index in range(len(errors) - 1)
    )
    exact = all(error == 0.0 for error in errors)
    costs_monotone = all(
        costs[index + 1] > costs[index]
        for index in range(len(costs) - 1)
    )
    final_order = rows[-1]["observed_order"]
    order_pass = exact or (
        final_order is not None
        and minimum_order <= float(final_order) <= maximum_order
    )
    passed = (errors_monotone or exact) and costs_monotone and order_pass
    return DigitalReplicationStudy(
        rows=tuple(rows),
        errors_monotone=errors_monotone or exact,
        execution_costs_monotone=costs_monotone,
        final_observed_order=(
            None if final_order is None else float(final_order)
        ),
        minimum_order=minimum_order,
        maximum_order=maximum_order,
        status="PASS" if passed else "REJECT",
    )
