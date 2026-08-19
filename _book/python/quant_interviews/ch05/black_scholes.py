from __future__ import annotations

from math import erf, exp, isfinite, log, pi, sqrt


def normal_cdf(x: float) -> float:
    if not isfinite(x):
        if x == float("inf"):
            return 1.0
        if x == float("-inf"):
            return 0.0
        raise ValueError("normal input must be finite or signed infinity")
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normal_pdf(x: float) -> float:
    if not isfinite(x):
        if abs(x) == float("inf"):
            return 0.0
        raise ValueError("normal input must be finite or signed infinity")
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _validate_contract(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
) -> None:
    values = (spot, strike, rate, dividend_yield, volatility, maturity)
    if not all(isfinite(value) for value in values):
        raise ValueError("Black-Scholes inputs must be finite")
    if spot <= 0.0 or strike <= 0.0 or volatility < 0.0 or maturity < 0.0:
        raise ValueError("invalid Black-Scholes contract")


def black_scholes_call(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
) -> float:
    _validate_contract(
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity,
    )
    if maturity == 0.0:
        return max(spot - strike, 0.0)
    if volatility == 0.0:
        return max(
            spot * exp(-dividend_yield * maturity) - strike * exp(-rate * maturity), 0.0
        )
    root_t = sqrt(maturity)
    d1 = (
        log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    return spot * exp(-dividend_yield * maturity) * normal_cdf(d1) - strike * exp(
        -rate * maturity
    ) * normal_cdf(d2)


def black_scholes_delta(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
) -> float:
    _validate_contract(
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity,
    )
    if maturity == 0.0:
        if spot > strike:
            return 1.0
        if spot < strike:
            return 0.0
        return 0.5
    if volatility == 0.0:
        discounted_spot = spot * exp(-dividend_yield * maturity)
        discounted_strike = strike * exp(-rate * maturity)
        if discounted_spot > discounted_strike:
            return exp(-dividend_yield * maturity)
        if discounted_spot < discounted_strike:
            return 0.0
        return 0.5 * exp(-dividend_yield * maturity)
    root_t = sqrt(maturity)
    d1 = (
        log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * root_t)
    return exp(-dividend_yield * maturity) * normal_cdf(d1)
