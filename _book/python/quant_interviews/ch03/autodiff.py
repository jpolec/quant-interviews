"""Forward and reverse automatic-differentiation examples for Chapter 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, exp, isfinite, log, pi, sqrt

import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class Dual:
    """Scalar forward-mode value and directional derivative."""

    value: float
    derivative: float

    @staticmethod
    def lift(other: float | Dual) -> Dual:
        return other if isinstance(other, Dual) else Dual(float(other), 0.0)

    def __add__(self, other: float | Dual) -> Dual:
        lifted = self.lift(other)
        return Dual(self.value + lifted.value, self.derivative + lifted.derivative)

    __radd__ = __add__

    def __neg__(self) -> Dual:
        return Dual(-self.value, -self.derivative)

    def __sub__(self, other: float | Dual) -> Dual:
        return self + (-self.lift(other))

    def __rsub__(self, other: float | Dual) -> Dual:
        return self.lift(other) - self

    def __mul__(self, other: float | Dual) -> Dual:
        lifted = self.lift(other)
        return Dual(
            self.value * lifted.value,
            self.derivative * lifted.value + self.value * lifted.derivative,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: float | Dual) -> Dual:
        lifted = self.lift(other)
        return Dual(
            self.value / lifted.value,
            (
                self.derivative * lifted.value
                - self.value * lifted.derivative
            )
            / lifted.value**2,
        )

    def __rtruediv__(self, other: float | Dual) -> Dual:
        return self.lift(other) / self


def dual_log(value: Dual) -> Dual:
    return Dual(log(value.value), value.derivative / value.value)


def dual_exp(value: Dual) -> Dual:
    result = exp(value.value)
    return Dual(result, result * value.derivative)


def dual_cdf(value: Dual) -> Dual:
    density = exp(-0.5 * value.value**2) / sqrt(2.0 * pi)
    return Dual(
        0.5 * (1.0 + erf(value.value / sqrt(2.0))),
        density * value.derivative,
    )


def _validate_black_scholes_domain(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
) -> None:
    """Enforce the economic domain before comparing derivatives."""

    if (
        not all(
            isfinite(value)
            for value in (spot, strike, rate, dividend, volatility, maturity)
        )
        or spot <= 0.0
        or strike <= 0.0
        or volatility <= 0.0
        or maturity <= 0.0
    ):
        raise ValueError(
            "finite positive spot, strike, volatility and maturity required"
        )


def black_scholes_call_dual(
    spot: Dual,
    strike: float,
    rate: float,
    dividend: float,
    volatility: Dual,
    maturity: float,
) -> Dual:
    """Execute Black-Scholes through scalar dual arithmetic."""

    if not all(
        isfinite(value)
        for value in (
            spot.value,
            spot.derivative,
            volatility.value,
            volatility.derivative,
        )
    ):
        raise ValueError("dual values and derivatives must be finite")
    _validate_black_scholes_domain(
        spot.value,
        strike,
        rate,
        dividend,
        volatility.value,
        maturity,
    )

    root_time = sqrt(maturity)
    total_volatility = volatility * root_time
    d1 = (
        dual_log(spot / strike)
        + (rate - dividend) * maturity
        + 0.5 * volatility * volatility * maturity
    ) / total_volatility
    d2 = d1 - total_volatility
    return (
        spot * exp(-dividend * maturity) * dual_cdf(d1)
        - strike * exp(-rate * maturity) * dual_cdf(d2)
    )


def black_scholes_greek_crosscheck(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    *,
    spot_bump: float | None = None,
) -> dict[str, float]:
    """Compare AD delta/vega with analytic, complex-step and bump references."""

    _validate_black_scholes_domain(
        spot,
        strike,
        rate,
        dividend,
        volatility,
        maturity,
    )
    bump = (
        min(1e-5 * max(1.0, spot), 0.25 * spot)
        if spot_bump is None
        else spot_bump
    )
    if not isfinite(bump) or bump <= 0.0 or spot - bump <= 0.0:
        raise ValueError("finite positive bump must keep both bumped spots positive")

    delta_ad = black_scholes_call_dual(
        Dual(spot, 1.0),
        strike,
        rate,
        dividend,
        Dual(volatility, 0.0),
        maturity,
    )
    vega_ad = black_scholes_call_dual(
        Dual(spot, 0.0),
        strike,
        rate,
        dividend,
        Dual(volatility, 1.0),
        maturity,
    )

    def complex_price(complex_spot: complex, complex_vol: complex) -> complex:
        root_time = sqrt(maturity)
        total_volatility = complex_vol * root_time
        d1 = (
            np.log(complex_spot / strike)
            + (rate - dividend + 0.5 * complex_vol**2) * maturity
        ) / total_volatility
        d2 = d1 - total_volatility
        return (
            complex_spot * exp(-dividend * maturity) * ndtr(d1)
            - strike * exp(-rate * maturity) * ndtr(d2)
        )

    def independent_real_price(real_spot: float) -> float:
        root_time = sqrt(maturity)
        total_volatility = volatility * root_time
        d1_real = (
            log(real_spot / strike)
            + (rate - dividend + 0.5 * volatility**2) * maturity
        ) / total_volatility
        d2_real = d1_real - total_volatility
        cdf_d1 = 0.5 * (1.0 + erf(d1_real / sqrt(2.0)))
        cdf_d2 = 0.5 * (1.0 + erf(d2_real / sqrt(2.0)))
        return (
            real_spot * exp(-dividend * maturity) * cdf_d1
            - strike * exp(-rate * maturity) * cdf_d2
        )

    total_volatility = volatility * sqrt(maturity)
    d1 = (
        log(spot / strike)
        + (rate - dividend + 0.5 * volatility**2) * maturity
    ) / total_volatility
    analytic_delta = exp(-dividend * maturity) * float(ndtr(d1))
    analytic_vega = (
        spot
        * exp(-dividend * maturity)
        * exp(-0.5 * d1**2)
        / sqrt(2.0 * pi)
        * sqrt(maturity)
    )
    complex_step = 1e-20
    return {
        "price": delta_ad.value,
        "ad_delta": delta_ad.derivative,
        "ad_vega": vega_ad.derivative,
        "analytic_delta": analytic_delta,
        "analytic_vega": analytic_vega,
        "complex_delta": float(
            complex_price(spot + 1j * complex_step, volatility).imag
            / complex_step
        ),
        "complex_vega": float(
            complex_price(spot, volatility + 1j * complex_step).imag
            / complex_step
        ),
        "bump_delta": float(
            (
                independent_real_price(spot + bump)
                - independent_real_price(spot - bump)
            )
            / (2.0 * bump)
        ),
    }


def polynomial_value_gradient(value: float) -> Dual:
    """Retain the smallest forward-mode teaching fixture."""

    variable = Dual(value, 1.0)
    return 3.0 * variable * variable + 2.0 * variable + 1.0


@dataclass
class ReverseScalar:
    """Minimal reverse-mode scalar node."""

    value: float
    parents: list[tuple[ReverseScalar, float]] = field(default_factory=list)
    gradient: float = 0.0

    def backward(self, seed: float = 1.0) -> None:
        self.gradient += seed
        for parent, local_derivative in self.parents:
            parent.backward(seed * local_derivative)


def reverse_polynomial_gradient(value: float) -> float:
    """Differentiate 3x²+2x+1 through a reverse computational graph."""

    variable = ReverseScalar(value)
    square = ReverseScalar(value * value, [(variable, 2.0 * value)])
    output = ReverseScalar(
        3.0 * square.value + 2.0 * variable.value + 1.0,
        [(square, 3.0), (variable, 2.0)],
    )
    output.backward()
    return variable.gradient
