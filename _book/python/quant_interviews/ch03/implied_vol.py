"""Black-Scholes call pricing and governed implied-volatility inversion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import erf, exp, isfinite, log, pi, sqrt
from sys import float_info

from .root_finding import safeguarded_newton


def normal_cdf(value: float) -> float:
    """Return the standard-normal cumulative distribution function."""

    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _validate_option_inputs(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    maturity: float,
) -> None:
    if (
        not all(isfinite(value) for value in (spot, strike, rate, dividend, maturity))
        or spot <= 0.0
        or strike <= 0.0
        or maturity <= 0.0
    ):
        raise ValueError("finite positive spot, strike and maturity required")


def call_price(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
) -> float:
    """Price a European call, including its zero-volatility boundary."""

    _validate_option_inputs(spot, strike, rate, dividend, maturity)
    if not isfinite(volatility) or volatility < 0.0:
        raise ValueError("finite nonnegative volatility required")
    discounted_spot = spot * exp(-dividend * maturity)
    discounted_strike = strike * exp(-rate * maturity)
    if volatility == 0.0:
        return max(discounted_spot - discounted_strike, 0.0)
    total_volatility = volatility * sqrt(maturity)
    d1 = (
        log(spot / strike)
        + (rate - dividend) * maturity
    ) / total_volatility + 0.5 * total_volatility
    d2 = d1 - total_volatility
    return (
        discounted_spot * normal_cdf(d1)
        - discounted_strike * normal_cdf(d2)
    )


def call_vega(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
) -> float:
    """Return call vega in price units per unit of volatility."""

    _validate_option_inputs(spot, strike, rate, dividend, maturity)
    if not isfinite(volatility) or volatility <= 0.0:
        return 0.0
    total_volatility = volatility * sqrt(maturity)
    d1 = (
        log(spot / strike)
        + (rate - dividend) * maturity
    ) / total_volatility + 0.5 * total_volatility
    density = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)
    return spot * exp(-dividend * maturity) * density * sqrt(maturity)


@dataclass(frozen=True)
class ImpliedVolResult:
    """Structured status for an implied-volatility quote inversion."""

    volatility: float | None
    status: str
    price_residual: float
    volatility_bracket_width: float | None
    iterations: int
    vega: float

    @property
    def root(self) -> float | None:
        """Compatibility alias for scalar root-finding clients."""

        return self.volatility


def call_price_bounds(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    maturity: float,
) -> tuple[float, float]:
    """Return the finite lower and limiting upper call-price bounds."""

    _validate_option_inputs(spot, strike, rate, dividend, maturity)
    discounted_spot = spot * exp(-dividend * maturity)
    discounted_strike = strike * exp(-rate * maturity)
    return max(discounted_spot - discounted_strike, 0.0), discounted_spot


def classify_call_quote(
    price: float,
    lower_bound: float,
    upper_bound: float,
    price_tolerance: float,
) -> ImpliedVolResult | None:
    """Return a boundary/invalid state, or None for a finite interior root."""

    if price < lower_bound - price_tolerance or price > upper_bound + price_tolerance:
        bound_violation = max(lower_bound - price, price - upper_bound, 0.0)
        return ImpliedVolResult(
            None,
            "INVALID_QUOTE",
            bound_violation,
            0.0,
            0,
            0.0,
        )
    if abs(price - lower_bound) <= price_tolerance:
        return ImpliedVolResult(
            0.0,
            "LOWER_BOUND",
            abs(price - lower_bound),
            0.0,
            0,
            0.0,
        )
    if abs(price - upper_bound) <= price_tolerance:
        return ImpliedVolResult(
            None,
            "UPPER_BOUND_LIMIT",
            abs(price - upper_bound),
            None,
            0,
            0.0,
        )
    return None


def implied_vol_core(
    objective: Callable[[float], float],
    derivative: Callable[[float], float],
    *,
    price_tolerance: float,
    volatility_tolerance: float,
    price_relative_tolerance: float,
    volatility_relative_tolerance: float,
    price_scale: float,
    volatility_scale: float,
    vega_floor: float,
    maximum_volatility: float,
) -> ImpliedVolResult:
    """Expand a volatility bracket and solve the finite-root state."""

    lower_volatility = 0.0
    upper_volatility = min(0.5, maximum_volatility)
    while (
        objective(upper_volatility) < 0.0
        and upper_volatility < maximum_volatility
    ):
        upper_volatility = min(
            2.0 * upper_volatility,
            maximum_volatility,
        )
    terminal_residual = float(objective(upper_volatility))
    if terminal_residual < 0.0:
        return ImpliedVolResult(
            None,
            "NO_BRACKET",
            abs(terminal_residual),
            upper_volatility,
            0,
            float(derivative(upper_volatility)),
        )

    root = safeguarded_newton(
        objective,
        derivative,
        lower_volatility,
        upper_volatility,
        min(0.2, upper_volatility),
        residual_tolerance=(
            price_tolerance
            + price_relative_tolerance * abs(price_scale)
        ),
        bracket_tolerance=volatility_tolerance,
        bracket_threshold=lambda volatility: (
            volatility_tolerance
            + volatility_relative_tolerance
            * max(volatility_scale, abs(volatility))
        ),
        max_iterations=200,
    )
    solved_vega = derivative(root.root)
    status = "CONVERGED" if root.converged else "NO_CONVERGENCE"
    if root.converged and solved_vega < vega_floor:
        status = "ILL_CONDITIONED_VEGA"
    return ImpliedVolResult(
        root.root,
        status,
        root.residual,
        root.bracket_width,
        root.iterations,
        solved_vega,
    )


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    maturity: float,
    *,
    price_tolerance: float = 1e-10,
    volatility_tolerance: float = 1e-10,
    price_relative_tolerance: float = 1e-12,
    volatility_relative_tolerance: float = 1e-12,
    volatility_scale: float = 0.01,
    vega_floor: float = 1e-8,
    maximum_volatility: float = 20.0,
) -> ImpliedVolResult:
    """Invert an admissible call quote with explicit boundary states."""

    _validate_option_inputs(spot, strike, rate, dividend, maturity)
    if (
        not all(
            isfinite(value)
            for value in (
                price,
                price_tolerance,
                volatility_tolerance,
                price_relative_tolerance,
                volatility_relative_tolerance,
                volatility_scale,
                vega_floor,
                maximum_volatility,
            )
        )
        or price_tolerance <= 0.0
        or volatility_tolerance <= 0.0
        or price_relative_tolerance < 0.0
        or volatility_relative_tolerance < 0.0
        or volatility_scale <= 0.0
        or vega_floor < 0.0
        or maximum_volatility <= 0.0
    ):
        raise ValueError("finite quote and positive solver controls required")
    lower_bound, upper_bound = call_price_bounds(
        spot,
        strike,
        rate,
        dividend,
        maturity,
    )
    price_scale = max(
        abs(price),
        abs(lower_bound),
        abs(upper_bound),
        float_info.min,
    )
    price_threshold = price_tolerance + price_relative_tolerance * price_scale
    boundary_state = classify_call_quote(
        price,
        lower_bound,
        upper_bound,
        price_threshold,
    )
    if boundary_state is not None:
        return boundary_state

    objective = lambda volatility: (
        call_price(
            spot,
            strike,
            rate,
            dividend,
            volatility,
            maturity,
        )
        - price
    )
    derivative = lambda volatility: call_vega(
        spot,
        strike,
        rate,
        dividend,
        volatility,
        maturity,
    )
    return implied_vol_core(
        objective,
        derivative,
        price_tolerance=price_tolerance,
        volatility_tolerance=volatility_tolerance,
        price_relative_tolerance=price_relative_tolerance,
        volatility_relative_tolerance=volatility_relative_tolerance,
        price_scale=price_scale,
        volatility_scale=volatility_scale,
        vega_floor=vega_floor,
        maximum_volatility=maximum_volatility,
    )
