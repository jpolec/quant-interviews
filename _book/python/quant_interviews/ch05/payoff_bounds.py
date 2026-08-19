from __future__ import annotations

from math import exp, isfinite
from typing import Any


def european_bounds(
    *,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    dividend_yield: float = 0.0,
) -> dict[str, tuple[float, float]]:
    values = (spot, strike, rate, maturity, dividend_yield)
    if not all(isfinite(value) for value in values):
        raise ValueError("option inputs must be finite")
    if spot <= 0.0 or strike < 0.0 or maturity < 0.0:
        raise ValueError("invalid option inputs")
    discounted_spot = spot * exp(-dividend_yield * maturity)
    discounted_strike = strike * exp(-rate * maturity)
    return {
        "call": (max(discounted_spot - discounted_strike, 0.0), discounted_spot),
        "put": (max(discounted_strike - discounted_spot, 0.0), discounted_strike),
    }


def bound_violations(
    price: float, lower: float, upper: float, tolerance: float = 1e-10
) -> list[str]:
    if not all(isfinite(value) for value in (price, lower, upper, tolerance)):
        raise ValueError("bound inputs must be finite")
    if tolerance < 0.0 or lower > upper:
        raise ValueError("invalid bound interval")
    violations: list[str] = []
    if price < lower - tolerance:
        violations.append("below_lower_bound")
    if price > upper + tolerance:
        violations.append("above_upper_bound")
    return violations


def executable_call_bound_certificate(
    *,
    call_bid: float,
    call_ask: float,
    spot_bid: float,
    spot_ask: float,
    strike: float,
    discount_factor: float,
    continuous_dividend_discount_factor: float = 1.0,
    execution_cost: float = 0.0,
    stock_borrow_available: bool = True,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Turn a European-call bound violation into an executable trade record.

    The lower-bound trade crosses the option ask and prepaid-stock bid.  The
    upper-bound trade crosses the option bid and prepaid-stock ask.  The
    dividend input is explicitly a multiplicative continuous-carry factor;
    discrete dividends require a different, cash-flow-level certificate.
    """

    numeric = (
        call_bid,
        call_ask,
        spot_bid,
        spot_ask,
        strike,
        discount_factor,
        continuous_dividend_discount_factor,
        execution_cost,
        tolerance,
    )
    if not all(isfinite(value) for value in numeric):
        raise ValueError("bound-certificate inputs must be finite")
    if (
        call_bid < 0.0
        or call_ask < call_bid
        or spot_bid <= 0.0
        or spot_ask < spot_bid
        or strike < 0.0
        or discount_factor <= 0.0
        or continuous_dividend_discount_factor <= 0.0
        or execution_cost < 0.0
        or tolerance < 0.0
    ):
        raise ValueError("invalid executable-bound inputs")

    lower = max(
        spot_bid * continuous_dividend_discount_factor
        - strike * discount_factor,
        0.0,
    )
    upper = spot_ask * continuous_dividend_discount_factor
    lower_profit = lower - call_ask - execution_cost
    upper_profit = call_bid - upper - execution_cost

    if lower_profit > tolerance and stock_borrow_available:
        status = "REJECT"
        violation = "below_executable_lower_bound"
        terminal_payoff = "max(strike - terminal_spot, 0)"
        legs = (
            {
                "instrument": "european_call",
                "quantity": 1.0,
                "side": "BUY",
                "execution_price": call_ask,
                "cashflow_time": 0.0,
            },
            {
                "instrument": "continuous-carry_prepaid_stock",
                "quantity": -1.0,
                "side": "SELL",
                "execution_price": spot_bid
                * continuous_dividend_discount_factor,
                "cashflow_time": 0.0,
            },
            {
                "instrument": "zero_coupon_bond_face_strike",
                "quantity": 1.0,
                "side": "BUY",
                "execution_price": strike * discount_factor,
                "cashflow_time": 0.0,
            },
        )
        initial_cash_flow = lower_profit
    elif upper_profit > tolerance:
        status = "REJECT"
        violation = "above_executable_upper_bound"
        terminal_payoff = "min(terminal_spot, strike)"
        legs = (
            {
                "instrument": "european_call",
                "quantity": -1.0,
                "side": "SELL",
                "execution_price": call_bid,
                "cashflow_time": 0.0,
            },
            {
                "instrument": "continuous-carry_prepaid_stock",
                "quantity": 1.0,
                "side": "BUY",
                "execution_price": spot_ask
                * continuous_dividend_discount_factor,
                "cashflow_time": 0.0,
            },
        )
        initial_cash_flow = upper_profit
    elif lower_profit > tolerance and not stock_borrow_available:
        status = "CONDITIONAL_PASS"
        violation = "theoretical_lower_bound_only"
        terminal_payoff = None
        legs = ()
        initial_cash_flow = 0.0
    else:
        status = "PASS"
        violation = None
        terminal_payoff = None
        legs = ()
        initial_cash_flow = 0.0

    if violation == "below_executable_lower_bound" and stock_borrow_available:
        terminal_payoff_floor = 0.0
        terminal_payoff_at_breakpoints = {
            "terminal_spot_0": strike,
            "terminal_spot_strike": 0.0,
            "terminal_spot_large_limit": 0.0,
        }
    elif violation == "above_executable_upper_bound":
        terminal_payoff_floor = 0.0
        terminal_payoff_at_breakpoints = {
            "terminal_spot_0": 0.0,
            "terminal_spot_strike": strike,
            "terminal_spot_large_limit": strike,
        }
    else:
        terminal_payoff_floor = None
        terminal_payoff_at_breakpoints = {}

    return {
        "lower_bound": lower,
        "upper_bound": upper,
        "call_bid": call_bid,
        "call_ask": call_ask,
        "lower_profit_after_cost": lower_profit,
        "upper_profit_after_cost": upper_profit,
        "violation": violation,
        "legs": legs,
        "terminal_payoff": terminal_payoff,
        "carry_convention": "continuous_dividend_discount_factor",
        "initial_cash_flow": initial_cash_flow,
        "terminal_payoff_floor": terminal_payoff_floor,
        "terminal_payoff_at_breakpoints": terminal_payoff_at_breakpoints,
        "status": status,
    }
