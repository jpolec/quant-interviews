from __future__ import annotations

from math import exp, isfinite

from .black_scholes import black_scholes_call


def forward_start_call_ratio(
    *,
    alpha: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    start: float,
    maturity: float,
) -> float:
    if not all(
        isfinite(value)
        for value in (
            alpha,
            rate,
            dividend_yield,
            volatility,
            start,
            maturity,
        )
    ):
        raise ValueError("forward-start inputs must be finite")
    if not 0.0 < alpha or not 0.0 <= start < maturity or volatility < 0.0:
        raise ValueError("invalid forward-start inputs")
    return black_scholes_call(
        spot=1.0,
        strike=alpha,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity - start,
    )


def forward_start_call_price(
    *,
    spot: float,
    alpha: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    start: float,
    maturity: float,
) -> float:
    """Return the complete time-zero forward-start call value."""

    if not isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and positive")
    ratio = forward_start_call_ratio(
        alpha=alpha,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        start=start,
        maturity=maturity,
    )
    return spot * exp(-dividend_yield * start) * ratio
