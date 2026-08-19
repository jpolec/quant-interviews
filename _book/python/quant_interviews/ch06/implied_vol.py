from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite

from .black_scholes_greeks import call_greeks, call_price


@dataclass(frozen=True)
class ImpliedVolResult:
    status: str
    implied_vol: float | None
    repricing_residual: float
    lower_vol: float
    upper_vol: float
    bracket_width: float
    vega: float
    price_to_vol_condition: float
    iterations: int


@dataclass(frozen=True)
class ImpliedVolInterval:
    bid: ImpliedVolResult
    ask: ImpliedVolResult
    numerical_status: str
    economic_status: str
    status: str


def call_price_bounds(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    horizon: float,
) -> tuple[float, float]:
    values = (spot, strike, rate, dividend, horizon)
    if not all(isfinite(value) for value in values):
        raise ValueError("price-bound inputs must be finite")
    if spot <= 0.0 or strike <= 0.0 or horizon <= 0.0:
        raise ValueError("spot, strike and horizon must be positive")
    discounted_spot = spot * exp(-dividend * horizon)
    discounted_strike = strike * exp(-rate * horizon)
    return max(discounted_spot - discounted_strike, 0.0), discounted_spot


def implied_vol_call(
    price: float,
    s: float,
    k: float,
    r: float,
    q: float,
    t: float,
    *,
    initial_vol: float = 0.20,
    initial_upper_vol: float = 0.50,
    maximum_vol: float = 20.0,
    price_tolerance: float = 1e-12,
    volatility_tolerance: float = 1e-10,
    vega_floor: float = 1e-8,
    max_iterations: int = 200,
) -> ImpliedVolResult:
    numeric_controls = (
        price,
        initial_vol,
        initial_upper_vol,
        maximum_vol,
        price_tolerance,
        volatility_tolerance,
        vega_floor,
    )
    if not all(isfinite(value) for value in numeric_controls):
        raise ValueError("price and solver controls must be finite")
    if price < 0.0:
        raise ValueError("price cannot be negative")
    if price_tolerance <= 0.0 or volatility_tolerance <= 0.0:
        raise ValueError("solver tolerances must be positive")
    if (
        vega_floor <= 0.0
        or isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise ValueError("vega floor and max_iterations must be positive")
    if not 0.0 < initial_vol <= maximum_vol:
        raise ValueError("initial_vol is outside the search domain")
    if not 0.0 < initial_upper_vol <= maximum_vol:
        raise ValueError("invalid volatility search domain")
    lower_price, upper_price = call_price_bounds(s, k, r, q, t)
    bound_scale = max(1.0, abs(price), abs(lower_price), abs(upper_price))
    price_scale = max(1.0, abs(price))
    bound_tolerance = price_tolerance * bound_scale
    if price < lower_price - bound_tolerance or price > upper_price + bound_tolerance:
        return ImpliedVolResult(
            "INVALID_QUOTE",
            None,
            min(abs(price - lower_price), abs(price - upper_price)),
            0.0,
            0.0,
            0.0,
            0.0,
            float("inf"),
            0,
        )
    if abs(price - lower_price) <= bound_tolerance:
        return ImpliedVolResult(
            "LOWER_BOUND",
            0.0,
            price - lower_price,
            0.0,
            0.0,
            0.0,
            0.0,
            float("inf"),
            0,
        )
    if upper_price - price <= bound_tolerance:
        return ImpliedVolResult(
            "UPPER_BOUND_LIMIT",
            None,
            upper_price - price,
            maximum_vol,
            float("inf"),
            float("inf"),
            0.0,
            float("inf"),
            0,
        )

    lower_vol = 0.0
    upper_vol = initial_upper_vol
    upper_model_price = call_price(s, k, r, q, upper_vol, t)
    while upper_model_price < price and upper_vol < maximum_vol:
        upper_vol = min(2.0 * upper_vol, maximum_vol)
        upper_model_price = call_price(s, k, r, q, upper_vol, t)
    if upper_model_price < price:
        return ImpliedVolResult(
            "NO_BRACKET",
            None,
            upper_model_price - price,
            lower_vol,
            upper_vol,
            upper_vol - lower_vol,
            0.0,
            float("inf"),
            0,
        )

    current = min(max(initial_vol, 0.5 * upper_vol), upper_vol)
    residual = float("inf")
    vega = 0.0
    for iteration in range(1, max_iterations + 1):
        model_price = call_price(s, k, r, q, current, t)
        residual = model_price - price
        vega = call_greeks(s, k, r, q, current, t).raw_vega_per_unit_vol
        if residual > 0.0:
            upper_vol = current
        else:
            lower_vol = current
        bracket_width = upper_vol - lower_vol
        if (
            abs(residual) <= price_tolerance * price_scale
            and bracket_width <= volatility_tolerance
        ):
            status = "CONVERGED" if vega >= vega_floor else "ILL_CONDITIONED_VEGA"
            return ImpliedVolResult(
                status,
                current,
                residual,
                lower_vol,
                upper_vol,
                bracket_width,
                vega,
                1.0 / max(vega, vega_floor),
                iteration,
            )
        newton = current - residual / vega if vega >= vega_floor else float("nan")
        if isfinite(newton) and lower_vol < newton < upper_vol:
            current = newton
        else:
            current = 0.5 * (lower_vol + upper_vol)

    final_model_price = call_price(s, k, r, q, current, t)
    final_residual = final_model_price - price
    final_vega = call_greeks(
        s, k, r, q, current, t
    ).raw_vega_per_unit_vol
    return ImpliedVolResult(
        "MAX_ITERATIONS",
        current,
        final_residual,
        lower_vol,
        upper_vol,
        upper_vol - lower_vol,
        final_vega,
        1.0 / max(final_vega, vega_floor),
        max_iterations,
    )


def implied_vol_bid_ask(
    bid_price: float,
    ask_price: float,
    s: float,
    k: float,
    r: float,
    q: float,
    t: float,
    **solver_options: float | int,
) -> ImpliedVolInterval:
    if not isfinite(bid_price) or not isfinite(ask_price):
        raise ValueError("bid and ask prices must be finite")
    if bid_price < 0.0 or ask_price < 0.0:
        raise ValueError("bid and ask prices cannot be negative")
    if bid_price > ask_price:
        raise ValueError("bid price cannot exceed ask price")
    bid = implied_vol_call(bid_price, s, k, r, q, t, **solver_options)
    ask = implied_vol_call(ask_price, s, k, r, q, t, **solver_options)
    conditional_states = {"ILL_CONDITIONED_VEGA", "LOWER_BOUND"}
    numeric_states = {"CONVERGED", *conditional_states}
    numerical_status = (
        "PASS"
        if bid.status == "CONVERGED" and ask.status == "CONVERGED"
        else (
            "CONDITIONAL_PASS"
            if bid.status in numeric_states and ask.status in numeric_states
            else "REJECT"
        )
    )
    ordered_vols = (
        bid.implied_vol is not None
        and ask.implied_vol is not None
        and bid.implied_vol <= ask.implied_vol
    )
    economic_status = "PASS" if ordered_vols else "REJECT"
    status = (
        "REJECT"
        if "REJECT" in {numerical_status, economic_status}
        else numerical_status
    )
    return ImpliedVolInterval(
        bid=bid,
        ask=ask,
        numerical_status=numerical_status,
        economic_status=economic_status,
        status=status,
    )
