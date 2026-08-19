from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt

from .black_scholes_greeks import call_greeks


@dataclass(frozen=True)
class CalendarTradeCertificate:
    short_strike: float
    long_strike: float
    minimum_short_long_weight: float
    maximum_short_long_weight: float
    selected_short_long_weight: float
    net_delta: float
    delta_hedge: float
    net_gamma: float
    net_vega_per_vol_point: float
    net_theta_per_day: float
    net_volga_per_vol_point_squared: float
    scenario_pnl: float
    estimated_crossing_cost: float
    greek_feasibility_status: str
    economic_scenario_status: str
    status: str


def _d1_d2(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    horizon: float,
) -> tuple[float, float]:
    root_t = sqrt(horizon)
    d1 = (
        log(spot / strike)
        + (rate - dividend + 0.5 * vol * vol) * horizon
    ) / (vol * root_t)
    return d1, d1 - vol * root_t


def long_gamma_short_vega_calendar(
    *,
    spot: float,
    rate: float,
    dividend: float,
    vol: float,
    short_horizon: float,
    long_horizon: float,
    log_forward_moneyness: float = 0.0,
    selected_weight: float | None = None,
    spot_return_scenario: float = 0.02,
    volatility_point_scenario: float = 1.0,
    holding_days: float = 1.0,
    short_option_half_spread: float = 0.02,
    long_option_half_spread: float = 0.04,
) -> CalendarTradeCertificate:
    inputs = (
        spot,
        rate,
        dividend,
        vol,
        short_horizon,
        long_horizon,
        log_forward_moneyness,
        spot_return_scenario,
        volatility_point_scenario,
        holding_days,
        short_option_half_spread,
        long_option_half_spread,
    )
    if not all(isfinite(value) for value in inputs):
        raise ValueError("calendar inputs must be finite")
    if spot <= 0.0 or vol <= 0.0:
        raise ValueError("spot and volatility must be positive")
    if not 0.0 < short_horizon < long_horizon:
        raise ValueError("require 0 < short_horizon < long_horizon")
    if holding_days < 0.0:
        raise ValueError("holding_days cannot be negative")
    if short_option_half_spread < 0.0 or long_option_half_spread < 0.0:
        raise ValueError("half spreads cannot be negative")

    short_forward = spot * exp((rate - dividend) * short_horizon)
    long_forward = spot * exp((rate - dividend) * long_horizon)
    short_strike = short_forward * exp(log_forward_moneyness)
    long_strike = long_forward * exp(log_forward_moneyness)
    short = call_greeks(
        spot,
        short_strike,
        rate,
        dividend,
        vol,
        short_horizon,
    )
    long = call_greeks(
        spot,
        long_strike,
        rate,
        dividend,
        vol,
        long_horizon,
    )

    lower_weight = short.raw_vega_per_unit_vol / long.raw_vega_per_unit_vol
    upper_weight = short.gamma / long.gamma
    feasible = lower_weight < upper_weight
    if selected_weight is None:
        selected_weight = 0.5 * (lower_weight + upper_weight)
    if not isfinite(selected_weight) or selected_weight <= 0.0:
        raise ValueError("selected_weight must be finite and positive")

    short_d1, short_d2 = _d1_d2(
        spot,
        short_strike,
        rate,
        dividend,
        vol,
        short_horizon,
    )
    long_d1, long_d2 = _d1_d2(
        spot,
        long_strike,
        rate,
        dividend,
        vol,
        long_horizon,
    )
    short_volga = short.raw_vega_per_unit_vol * short_d1 * short_d2 / vol
    long_volga = long.raw_vega_per_unit_vol * long_d1 * long_d2 / vol

    net_delta = short.delta - selected_weight * long.delta
    net_gamma = short.gamma - selected_weight * long.gamma
    net_vega = (
        short.vega_per_vol_point - selected_weight * long.vega_per_vol_point
    )
    net_theta = (
        short.theta_per_calendar_day
        - selected_weight * long.theta_per_calendar_day
    )
    net_volga = (
        short_volga - selected_weight * long_volga
    ) * 0.01 * 0.01
    spot_change = spot * spot_return_scenario
    scenario_pnl = (
        0.5 * net_gamma * spot_change * spot_change
        + net_vega * volatility_point_scenario
        + net_theta * holding_days
        + 0.5
        * net_volga
        * volatility_point_scenario
        * volatility_point_scenario
    )
    crossing_cost = (
        short_option_half_spread
        + selected_weight * long_option_half_spread
    )
    selected_is_feasible = (
        feasible
        and selected_weight > lower_weight
        and selected_weight < upper_weight
        and net_gamma > 0.0
        and net_vega < 0.0
    )
    economic_scenario_passes = scenario_pnl > crossing_cost
    greek_status = "PASS" if selected_is_feasible else "REJECT"
    economic_status = "PASS" if economic_scenario_passes else "REJECT"
    return CalendarTradeCertificate(
        short_strike=short_strike,
        long_strike=long_strike,
        minimum_short_long_weight=lower_weight,
        maximum_short_long_weight=upper_weight,
        selected_short_long_weight=selected_weight,
        net_delta=net_delta,
        delta_hedge=-net_delta,
        net_gamma=net_gamma,
        net_vega_per_vol_point=net_vega,
        net_theta_per_day=net_theta,
        net_volga_per_vol_point_squared=net_volga,
        scenario_pnl=scenario_pnl,
        estimated_crossing_cost=crossing_cost,
        greek_feasibility_status=greek_status,
        economic_scenario_status=economic_status,
        status=(
            "PASS"
            if greek_status == "PASS" and economic_status == "PASS"
            else "REJECT"
        ),
    )


def calendar_weights(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    short_t: float,
    long_t: float,
) -> tuple[float, float, float]:
    """Compatibility wrapper returning a strict interior calendar weight."""

    if (
        not all(
            isfinite(value)
            for value in (spot, strike, rate, dividend, vol, short_t, long_t)
        )
        or spot <= 0.0
        or strike <= 0.0
        or vol <= 0.0
        or not 0.0 < short_t < long_t
    ):
        raise ValueError("invalid calendar-weight inputs")
    log_moneyness = log(
        strike / (spot * exp((rate - dividend) * short_t))
    )
    result = long_gamma_short_vega_calendar(
        spot=spot,
        rate=rate,
        dividend=dividend,
        vol=vol,
        short_horizon=short_t,
        long_horizon=long_t,
        log_forward_moneyness=log_moneyness,
    )
    return (
        result.selected_short_long_weight,
        result.net_gamma,
        result.net_vega_per_vol_point / 0.01,
    )
