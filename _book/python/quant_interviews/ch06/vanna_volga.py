from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from typing import Callable

from .black_scholes_greeks import call_greeks


@dataclass(frozen=True)
class VannaVolgaAudit:
    analytic_vanna: float
    bump_vanna: float
    analytic_volga: float
    bump_volga: float
    vanna_scaled_error: float
    volga_scaled_error: float
    volga_sign: str
    status: str


def finite_difference_vanna(
    price_fn: Callable[[float, float], float],
    spot: float,
    vol: float,
    ds: float = 1e-2,
    dv: float = 1e-4,
) -> float:
    if (
        not all(isfinite(value) for value in (spot, vol, ds, dv))
        or ds <= 0.0
        or dv <= 0.0
        or spot <= ds
        or vol <= dv
    ):
        raise ValueError("bump sizes must be positive")
    result = (
        price_fn(spot + ds, vol + dv)
        - price_fn(spot + ds, vol - dv)
        - price_fn(spot - ds, vol + dv)
        + price_fn(spot - ds, vol - dv)
    ) / (4.0 * ds * dv)
    if not isfinite(result):
        raise ValueError("price function produced a non-finite vanna bump")
    return result


def finite_difference_volga(
    price_fn: Callable[[float, float], float],
    spot: float,
    vol: float,
    dv: float = 1e-4,
) -> float:
    if (
        not all(isfinite(value) for value in (spot, vol, dv))
        or spot <= 0.0
        or dv <= 0.0
        or vol <= dv
    ):
        raise ValueError("volatility bump must be positive")
    result = (
        price_fn(spot, vol + dv)
        - 2.0 * price_fn(spot, vol)
        + price_fn(spot, vol - dv)
    ) / (dv * dv)
    if not isfinite(result):
        raise ValueError("price function produced a non-finite volga bump")
    return result


def vanna_volga_audit(
    *,
    price_fn: Callable[[float, float], float],
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    horizon: float,
    spot_bump: float = 0.02,
    vol_bump: float = 0.0002,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 5e-4,
) -> VannaVolgaAudit:
    values = (
        spot,
        strike,
        rate,
        dividend,
        vol,
        horizon,
        spot_bump,
        vol_bump,
        absolute_tolerance,
        relative_tolerance,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("vanna/volga inputs must be finite")
    if (
        spot_bump <= 0.0
        or vol_bump <= 0.0
        or spot <= spot_bump
        or vol <= vol_bump
    ):
        raise ValueError("bump sizes must be positive")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("tolerances cannot be negative")

    greeks = call_greeks(spot, strike, rate, dividend, vol, horizon)
    root_t = sqrt(horizon)
    d1 = (
        log(spot / strike)
        + (rate - dividend + 0.5 * vol * vol) * horizon
    ) / (vol * root_t)
    d2 = d1 - vol * root_t
    analytic_vanna = (
        -greeks.raw_vega_per_unit_vol * d2 / (spot * vol * root_t)
    )
    analytic_volga = greeks.raw_vega_per_unit_vol * d1 * d2 / vol
    bump_vanna = finite_difference_vanna(
        price_fn,
        spot,
        vol,
        ds=spot_bump,
        dv=vol_bump,
    )
    bump_volga = finite_difference_volga(
        price_fn,
        spot,
        vol,
        dv=vol_bump,
    )

    def scaled_error(actual: float, reference: float) -> float:
        scale = absolute_tolerance + relative_tolerance * max(
            abs(actual), abs(reference)
        )
        return abs(actual - reference) / max(scale, 1e-16)

    vanna_error = scaled_error(analytic_vanna, bump_vanna)
    volga_error = scaled_error(analytic_volga, bump_volga)
    return VannaVolgaAudit(
        analytic_vanna=analytic_vanna,
        bump_vanna=bump_vanna,
        analytic_volga=analytic_volga,
        bump_volga=bump_volga,
        vanna_scaled_error=vanna_error,
        volga_scaled_error=volga_error,
        volga_sign=(
            "POSITIVE"
            if analytic_volga > 0.0
            else "NEGATIVE"
            if analytic_volga < 0.0
            else "ZERO"
        ),
        status="PASS" if max(vanna_error, volga_error) <= 1.0 else "REJECT",
    )
