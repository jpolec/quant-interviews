from __future__ import annotations

from collections.abc import Sequence
from math import exp, isfinite
from typing import Any


def put_call_parity_residual(
    *,
    call: float,
    put: float,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    dividend_yield: float = 0.0,
    borrow_adjustment: float = 0.0,
) -> float:
    """Frictionless diagnostic retained for analytic round-trip checks."""

    values = (
        call,
        put,
        spot,
        strike,
        rate,
        maturity,
        dividend_yield,
        borrow_adjustment,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("parity inputs must be finite")
    if spot <= 0.0 or strike < 0.0 or maturity < 0.0:
        raise ValueError("invalid parity inputs")
    prepaid_forward = spot * exp(
        -(dividend_yield + borrow_adjustment) * maturity
    )
    discounted_strike = strike * exp(-rate * maturity)
    return (call - put) - (prepaid_forward - discounted_strike)


def _present_value_dividends(
    dividends: Sequence[tuple[float, float]],
    *,
    maturity: float,
    discount_rate: float,
) -> float:
    present_value = 0.0
    for payment_time, amount in dividends:
        if (
            not isfinite(payment_time)
            or not isfinite(amount)
            or payment_time <= 0.0
            or payment_time > maturity
            or amount < 0.0
        ):
            raise ValueError("invalid cash-dividend schedule")
        present_value += amount * exp(-discount_rate * payment_time)
    return present_value


def executable_put_call_parity_band(
    *,
    call_bid: float,
    call_ask: float,
    put_bid: float,
    put_ask: float,
    spot_bid: float,
    spot_ask: float,
    strike: float,
    maturity: float,
    long_funding_rate: float,
    short_funding_rate: float,
    borrow_rate: float = 0.0,
    dividend_yield: float = 0.0,
    cash_dividends: Sequence[tuple[float, float]] = (),
    settlement_convention: str = "premium_paid_upfront",
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Audit whether executable option and carry intervals intersect.

    The band is a governed continuous-carry approximation.  The lower edge uses
    the stock bid, stock-borrow charge and short-cash funding; the upper edge
    uses the stock ask and long-cash funding.  Both option legs cross their
    executable sides.  Futures-style or deferred-premium settlement must use a
    different contract implementation.
    """

    numeric = (
        call_bid,
        call_ask,
        put_bid,
        put_ask,
        spot_bid,
        spot_ask,
        strike,
        maturity,
        long_funding_rate,
        short_funding_rate,
        borrow_rate,
        dividend_yield,
        tolerance,
    )
    if not all(isfinite(value) for value in numeric):
        raise ValueError("parity-band inputs must be finite")
    if (
        call_bid < 0.0
        or call_ask < call_bid
        or put_bid < 0.0
        or put_ask < put_bid
        or spot_bid <= 0.0
        or spot_ask < spot_bid
        or strike < 0.0
        or maturity < 0.0
        or borrow_rate < 0.0
        or tolerance < 0.0
    ):
        raise ValueError("invalid executable parity inputs")
    if settlement_convention != "premium_paid_upfront":
        raise ValueError("unsupported settlement convention")

    dividend_pv = _present_value_dividends(
        cash_dividends,
        maturity=maturity,
        discount_rate=long_funding_rate,
    )
    lower_prepaid_stock = max(spot_bid - dividend_pv, 0.0) * exp(
        -(dividend_yield + borrow_rate) * maturity
    )
    upper_prepaid_stock = max(spot_ask - dividend_pv, 0.0) * exp(
        -dividend_yield * maturity
    )
    conversion_lower = lower_prepaid_stock - strike * exp(
        -short_funding_rate * maturity
    )
    reverse_conversion_upper = upper_prepaid_stock - strike * exp(
        -long_funding_rate * maturity
    )
    if conversion_lower > reverse_conversion_upper + tolerance:
        raise ValueError("carry inputs produce an inverted parity band")

    observed_lower = call_bid - put_ask
    observed_upper = call_ask - put_bid
    lower_residual = observed_upper - conversion_lower
    upper_residual = reverse_conversion_upper - observed_lower
    executable = lower_residual >= -tolerance and upper_residual >= -tolerance

    if executable:
        state = "PASS"
        trade = "none"
    elif lower_residual < -tolerance:
        state = "REJECT"
        trade = (
            "buy call at ask; sell put at bid; short the executable "
            "conversion package"
        )
    else:
        state = "REJECT"
        trade = (
            "sell call at bid; buy put at ask; buy the executable "
            "reverse-conversion package"
        )

    return {
        "conversion_lower": conversion_lower,
        "reverse_conversion_upper": reverse_conversion_upper,
        "observed_call_minus_put_lower": observed_lower,
        "observed_call_minus_put_upper": observed_upper,
        "lower_residual": lower_residual,
        "upper_residual": upper_residual,
        "dividend_present_value": dividend_pv,
        "settlement_convention": settlement_convention,
        "executable_trade": trade,
        "status": state,
    }
