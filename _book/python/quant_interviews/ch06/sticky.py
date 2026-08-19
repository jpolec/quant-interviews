from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from typing import Callable

from scipy.optimize import brentq
from scipy.stats import norm

from .black_scholes_greeks import call_price


@dataclass(frozen=True)
class StickyDeltaScenario:
    delta_convention: str
    initial_delta: float
    initial_strike: float
    initial_log_forward_moneyness: float
    moved_forward: float
    moved_log_forward_moneyness: float
    moved_delta_at_fixed_strike: float
    sticky_strike_vol: float
    sticky_delta_vol: float
    sticky_strike_pnl: float
    sticky_delta_pnl: float
    finite_move_average_delta: float
    local_sticky_delta: float
    iterations: int
    status: str

    @property
    def sticky_delta_effective_delta(self) -> float:
        """Deprecated alias for the finite-move average delta."""

        return self.finite_move_average_delta


def _call_delta(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    horizon: float,
    convention: str,
) -> float:
    forward = spot * exp((rate - dividend) * horizon)
    d1 = (
        log(forward / strike) + 0.5 * vol * vol * horizon
    ) / (vol * sqrt(horizon))
    forward_call = float(norm.cdf(d1))
    if convention == "forward_call":
        return forward_call
    if convention == "spot_call":
        return exp(-dividend * horizon) * forward_call
    if convention == "premium_adjusted_spot_call":
        premium = call_price(spot, strike, rate, dividend, vol, horizon)
        return exp(-dividend * horizon) * forward_call - premium / spot
    raise ValueError(f"unsupported delta convention: {convention}")


def strike_for_delta(
    *,
    spot: float,
    target_delta: float,
    rate: float,
    dividend: float,
    vol: float,
    horizon: float,
    convention: str,
) -> float:
    values = (spot, target_delta, rate, dividend, vol, horizon)
    if not all(isfinite(value) for value in values):
        raise ValueError("delta-to-strike inputs must be finite")
    if spot <= 0.0 or vol <= 0.0 or horizon <= 0.0:
        raise ValueError("spot, volatility and horizon must be positive")
    if convention == "forward_call":
        delta_upper_bound = 1.0
    elif convention == "spot_call":
        delta_upper_bound = exp(-dividend * horizon)
    elif convention == "premium_adjusted_spot_call":
        delta_upper_bound = _call_delta(
            spot=spot,
            strike=spot * exp((rate - dividend) * horizon),
            rate=rate,
            dividend=dividend,
            vol=vol,
            horizon=horizon,
            convention=convention,
        )
    else:
        raise ValueError(f"unsupported delta convention: {convention}")
    if not 0.0 < target_delta < delta_upper_bound:
        raise ValueError(
            "target delta is outside the supported call-delta branch"
        )
    forward = spot * exp((rate - dividend) * horizon)

    def objective(log_moneyness: float) -> float:
        strike = forward * exp(log_moneyness)
        return (
            _call_delta(
                spot=spot,
                strike=strike,
                rate=rate,
                dividend=dividend,
                vol=vol,
                horizon=horizon,
                convention=convention,
            )
            - target_delta
        )

    lower_log_moneyness = (
        0.0 if convention == "premium_adjusted_spot_call" else -6.0
    )
    lower_value = objective(lower_log_moneyness)
    upper_value = objective(6.0)
    if lower_value * upper_value > 0.0:
        raise ValueError("target delta is not bracketed on the supported branch")
    return forward * exp(
        brentq(objective, lower_log_moneyness, 6.0, xtol=1e-13)
    )


def sticky_delta_scenario(
    *,
    spot: float,
    spot_move: float,
    target_delta: float,
    rate: float,
    dividend: float,
    horizon: float,
    smile_by_delta: Callable[[float], float],
    delta_convention: str = "forward_call",
    maximum_iterations: int = 100,
    tolerance: float = 1e-12,
) -> StickyDeltaScenario:
    scalar_inputs = (
        spot,
        spot_move,
        target_delta,
        rate,
        dividend,
        horizon,
        tolerance,
    )
    if not all(isfinite(value) for value in scalar_inputs):
        raise ValueError("sticky-delta inputs must be finite")
    if spot <= 0.0 or horizon <= 0.0:
        raise ValueError("spot and horizon must be positive")
    if spot + spot_move <= 0.0:
        raise ValueError("moved spot must remain positive")
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, int)
        or maximum_iterations <= 0
        or tolerance <= 0.0
    ):
        raise ValueError("iteration controls must be positive")
    initial_vol = float(smile_by_delta(target_delta))
    if not isfinite(initial_vol) or initial_vol <= 0.0:
        raise ValueError("smile must return a finite positive volatility")
    initial_strike = strike_for_delta(
        spot=spot,
        target_delta=target_delta,
        rate=rate,
        dividend=dividend,
        vol=initial_vol,
        horizon=horizon,
        convention=delta_convention,
    )
    moved_spot = spot + spot_move
    initial_forward = spot * exp((rate - dividend) * horizon)
    moved_forward = moved_spot * exp((rate - dividend) * horizon)
    def solve_smile(test_spot: float) -> tuple[float, float, int, bool]:
        test_vol = initial_vol
        test_delta = target_delta
        for iteration_number in range(1, maximum_iterations + 1):
            test_delta = _call_delta(
                spot=test_spot,
                strike=initial_strike,
                rate=rate,
                dividend=dividend,
                vol=test_vol,
                horizon=horizon,
                convention=delta_convention,
            )
            next_vol = float(smile_by_delta(test_delta))
            if not isfinite(next_vol) or next_vol <= 0.0:
                raise ValueError("smile returned an invalid moved volatility")
            if abs(next_vol - test_vol) <= tolerance:
                final_delta = _call_delta(
                    spot=test_spot,
                    strike=initial_strike,
                    rate=rate,
                    dividend=dividend,
                    vol=next_vol,
                    horizon=horizon,
                    convention=delta_convention,
                )
                fixed_point_error = abs(
                    float(smile_by_delta(final_delta)) - next_vol
                )
                return next_vol, final_delta, iteration_number, (
                    fixed_point_error <= tolerance
                )
            test_vol = next_vol
        return test_vol, test_delta, maximum_iterations, False

    moved_vol, moved_delta, iteration, converged = solve_smile(moved_spot)

    initial_price = call_price(
        spot, initial_strike, rate, dividend, initial_vol, horizon
    )
    sticky_strike_price = call_price(
        moved_spot, initial_strike, rate, dividend, initial_vol, horizon
    )
    sticky_delta_price = call_price(
        moved_spot, initial_strike, rate, dividend, moved_vol, horizon
    )
    local_bump = max(1e-6, moved_spot * 1e-5)
    lower_spot = moved_spot - local_bump
    if lower_spot <= 0.0:
        raise ValueError("local-delta bump would cross zero spot")
    lower_vol, _, _, lower_converged = solve_smile(lower_spot)
    upper_vol, _, _, upper_converged = solve_smile(moved_spot + local_bump)
    lower_price = call_price(
        lower_spot, initial_strike, rate, dividend, lower_vol, horizon
    )
    upper_price = call_price(
        moved_spot + local_bump,
        initial_strike,
        rate,
        dividend,
        upper_vol,
        horizon,
    )
    local_sticky_delta = (upper_price - lower_price) / (2.0 * local_bump)
    finite_move_average_delta = (
        (sticky_delta_price - initial_price) / spot_move
        if spot_move != 0.0
        else local_sticky_delta
    )
    return StickyDeltaScenario(
        delta_convention=delta_convention,
        initial_delta=target_delta,
        initial_strike=initial_strike,
        initial_log_forward_moneyness=log(initial_strike / initial_forward),
        moved_forward=moved_forward,
        moved_log_forward_moneyness=log(initial_strike / moved_forward),
        moved_delta_at_fixed_strike=moved_delta,
        sticky_strike_vol=initial_vol,
        sticky_delta_vol=moved_vol,
        sticky_strike_pnl=sticky_strike_price - initial_price,
        sticky_delta_pnl=sticky_delta_price - initial_price,
        finite_move_average_delta=finite_move_average_delta,
        local_sticky_delta=local_sticky_delta,
        iterations=iteration,
        status="PASS" if converged and lower_converged and upper_converged else "REJECT",
    )


def strike_for_delta_forward(
    forward: float,
    delta: float,
    vol: float,
    horizon: float,
    norm_ppf: Callable[[float], float],
) -> float:
    """Closed-form forward-call delta conversion retained for compatibility."""

    if not all(isfinite(value) for value in (forward, delta, vol, horizon)):
        raise ValueError("delta conversion inputs must be finite")
    if forward <= 0.0 or vol <= 0.0 or horizon <= 0.0:
        raise ValueError("forward, volatility and horizon must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("forward call delta must lie in (0, 1)")
    d1 = norm_ppf(delta)
    if not isfinite(d1):
        raise ValueError("inverse-normal callback returned a non-finite value")
    return forward * exp(
        -d1 * vol * sqrt(horizon) + 0.5 * vol * vol * horizon
    )


def sticky_strike_vol(
    strike: float,
    smile: Callable[[float], float],
) -> float:
    if not isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive")
    result = float(smile(strike))
    if not isfinite(result) or result <= 0.0:
        raise ValueError("smile returned an invalid volatility")
    return result


def sticky_delta_vol(
    spot: float,
    delta: float,
    horizon: float,
    smile_delta: Callable[[float, float, float], float],
) -> float:
    if (
        not all(isfinite(value) for value in (spot, delta, horizon))
        or spot <= 0.0
        or horizon <= 0.0
        or not 0.0 < delta < 1.0
    ):
        raise ValueError("sticky-delta helper inputs are invalid")
    result = float(smile_delta(delta, horizon, spot))
    if not isfinite(result) or result <= 0.0:
        raise ValueError("smile returned an invalid volatility")
    return result
