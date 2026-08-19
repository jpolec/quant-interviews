from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, isfinite, log, pi, sqrt


def cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


@dataclass(frozen=True)
class GreekConventions:
    currency: str = "USD"
    delta_convention: str = "spot_call"
    theta_convention: str = "calendar_time_decay"
    day_count_convention: str = "ACT/365F"
    calendar_days_per_year: int = 365
    volatility_point: float = 0.01
    rate_basis_point: float = 0.0001

    def __post_init__(self) -> None:
        if len(self.currency) != 3 or not all(
            "A" <= character <= "Z" for character in self.currency
        ):
            raise ValueError("currency must be a three-letter uppercase ISO code")
        if self.delta_convention != "spot_call":
            raise ValueError("only the spot_call delta convention is supported")
        if self.theta_convention != "calendar_time_decay":
            raise ValueError(
                "only the calendar_time_decay theta convention is supported"
            )
        if self.day_count_convention != "ACT/365F":
            raise ValueError("only the ACT/365F day-count convention is supported")
        if (
            isinstance(self.calendar_days_per_year, bool)
            or not isinstance(self.calendar_days_per_year, int)
            or self.calendar_days_per_year <= 0
        ):
            raise ValueError("calendar_days_per_year must be a positive integer")
        if not isfinite(self.volatility_point) or self.volatility_point <= 0.0:
            raise ValueError("volatility_point must be finite and positive")
        if not isfinite(self.rate_basis_point) or self.rate_basis_point <= 0.0:
            raise ValueError("rate_basis_point must be finite and positive")


@dataclass(frozen=True)
class Greeks:
    delta: float
    delta_upper_bound: float
    gamma: float
    cash_gamma: float
    raw_vega_per_unit_vol: float
    vega_per_vol_point: float
    theta_per_year: float
    theta_per_calendar_day: float
    raw_rho_per_unit_rate: float
    rho_per_basis_point: float
    conventions: GreekConventions

    @property
    def vega(self) -> float:
        """Compatibility alias; callers should prefer the labelled fields."""

        return self.raw_vega_per_unit_vol

    @property
    def theta(self) -> float:
        return self.theta_per_year

    @property
    def rho(self) -> float:
        return self.raw_rho_per_unit_rate


def _validate_inputs(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    horizon: float,
) -> None:
    values = (spot, strike, rate, dividend, vol, horizon)
    if not all(isfinite(value) for value in values):
        raise ValueError("Black-Scholes inputs must be finite")
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot and strike must be positive")
    if vol <= 0.0:
        raise ValueError("volatility must be positive for differential Greeks")
    if horizon <= 0.0:
        raise ValueError("horizon must be positive for differential Greeks")


def call_price(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    horizon: float,
) -> float:
    _validate_inputs(spot, strike, rate, dividend, vol, horizon)
    root_t = sqrt(horizon)
    d1 = (
        log(spot / strike)
        + (rate - dividend + 0.5 * vol * vol) * horizon
    ) / (vol * root_t)
    d2 = d1 - vol * root_t
    return (
        spot * exp(-dividend * horizon) * cdf(d1)
        - strike * exp(-rate * horizon) * cdf(d2)
    )


def call_greeks(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    horizon: float,
    *,
    conventions: GreekConventions | None = None,
) -> Greeks:
    _validate_inputs(spot, strike, rate, dividend, vol, horizon)
    conventions = conventions or GreekConventions()
    if not isinstance(conventions, GreekConventions):
        raise TypeError("conventions must be a GreekConventions instance")

    root_t = sqrt(horizon)
    d1 = (
        log(spot / strike)
        + (rate - dividend + 0.5 * vol * vol) * horizon
    ) / (vol * root_t)
    d2 = d1 - vol * root_t
    dividend_discount = exp(-dividend * horizon)
    strike_discount = exp(-rate * horizon)

    delta = dividend_discount * cdf(d1)
    gamma = dividend_discount * pdf(d1) / (spot * vol * root_t)
    raw_vega = spot * dividend_discount * pdf(d1) * root_t
    theta_per_year = (
        -(spot * dividend_discount * pdf(d1) * vol) / (2.0 * root_t)
        - rate * strike * strike_discount * cdf(d2)
        + dividend * spot * dividend_discount * cdf(d1)
    )
    raw_rho = strike * horizon * strike_discount * cdf(d2)

    return Greeks(
        delta=delta,
        delta_upper_bound=dividend_discount,
        gamma=gamma,
        cash_gamma=gamma * spot * spot,
        raw_vega_per_unit_vol=raw_vega,
        vega_per_vol_point=raw_vega * conventions.volatility_point,
        theta_per_year=theta_per_year,
        theta_per_calendar_day=(
            theta_per_year / conventions.calendar_days_per_year
        ),
        raw_rho_per_unit_rate=raw_rho,
        rho_per_basis_point=raw_rho * conventions.rate_basis_point,
        conventions=conventions,
    )
