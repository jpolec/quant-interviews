from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, isfinite, log, pi, sqrt
from typing import Literal


@dataclass(frozen=True)
class CallGreeks:
    price: float
    delta: float
    gamma: float
    raw_vega_per_unit_volatility: float
    vega_per_vol_point: float
    theta_per_year: float
    dividend_yield: float
    boundary_convention: str | None
    status: Literal["PASS", "BOUNDARY_CONVENTION", "BOUNDARY"]


@dataclass(frozen=True)
class HedgePnL:
    option_position: float
    stock_position: float
    initial_cash: float
    option_change: float
    stock_hedge_change: float
    cash_financing: float
    dividend_cash_flow: float
    transaction_cost: float
    hedged_pnl: float
    delta_gamma_vega_theta_approximation: float
    approximation_residual: float
    tolerance: float
    status: Literal["PASS", "REJECT"]


def _finite(*values: float) -> bool:
    return all(isfinite(value) for value in values)


def normal_cdf(value: float) -> float:
    result = 0.5 * (1.0 + erf(value / sqrt(2.0)))
    if not isfinite(result):
        raise FloatingPointError("normal CDF is non-finite")
    return result


def normal_pdf(value: float) -> float:
    result = exp(-0.5 * value * value) / sqrt(2.0 * pi)
    if not isfinite(result):
        raise FloatingPointError("normal PDF is non-finite")
    return result


def _checked_greeks(
    price: float,
    delta: float,
    gamma: float,
    raw_vega: float,
    theta: float,
) -> None:
    if not _finite(price, delta, gamma, raw_vega, theta):
        raise FloatingPointError("Black-Scholes output is non-finite")
    if price < 0.0 or gamma < 0.0 or raw_vega < 0.0:
        raise FloatingPointError("Black-Scholes output violates sign constraints")


def black_scholes_call(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> CallGreeks:
    if not _finite(spot, strike, maturity, rate, volatility, dividend_yield):
        raise ValueError("Black-Scholes inputs must be finite")
    if spot <= 0.0 or strike <= 0.0 or maturity < 0.0 or volatility < 0.0:
        raise ValueError("spot/strike must be positive; maturity/volatility non-negative")
    if maturity == 0.0:
        intrinsic = max(spot - strike, 0.0)
        if spot > strike:
            delta = 1.0
            convention = None
            status: Literal["PASS", "BOUNDARY_CONVENTION", "BOUNDARY"] = "BOUNDARY"
        elif spot < strike:
            delta = 0.0
            convention = None
            status = "BOUNDARY"
        else:
            delta = 0.5
            convention = "symmetric_subgradient_at_expiry_kink"
            status = "BOUNDARY_CONVENTION"
        return CallGreeks(
            intrinsic,
            delta,
            0.0,
            0.0,
            0.0,
            0.0,
            dividend_yield,
            convention,
            status,
        )
    try:
        if volatility == 0.0:
            discounted_spot = spot * exp(-dividend_yield * maturity)
            discounted_strike = strike * exp(-rate * maturity)
            forward_intrinsic = max(discounted_spot - discounted_strike, 0.0)
            delta = (
                exp(-dividend_yield * maturity)
                if discounted_spot > discounted_strike
                else 0.0
            )
            _checked_greeks(forward_intrinsic, delta, 0.0, 0.0, 0.0)
            return CallGreeks(
                forward_intrinsic,
                delta,
                0.0,
                0.0,
                0.0,
                0.0,
                dividend_yield,
                "zero_volatility_forward_intrinsic",
                "BOUNDARY",
            )

        root_t = sqrt(maturity)
        d1 = (
            log(spot / strike)
            + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
        ) / (volatility * root_t)
        d2 = d1 - volatility * root_t
        discount_r = exp(-rate * maturity)
        discount_q = exp(-dividend_yield * maturity)
        density = normal_pdf(d1)
        price = (
            spot * discount_q * normal_cdf(d1)
            - strike * discount_r * normal_cdf(d2)
        )
        delta = discount_q * normal_cdf(d1)
        gamma = discount_q * density / (spot * volatility * root_t)
        raw_vega = spot * discount_q * density * root_t
        theta = (
            -spot * discount_q * density * volatility / (2.0 * root_t)
            - rate * strike * discount_r * normal_cdf(d2)
            + dividend_yield * spot * discount_q * normal_cdf(d1)
        )
    except (OverflowError, ZeroDivisionError) as error:
        raise FloatingPointError("Black-Scholes numerical range failure") from error
    _checked_greeks(price, delta, gamma, raw_vega, theta)
    return CallGreeks(
        price=price,
        delta=delta,
        gamma=gamma,
        raw_vega_per_unit_volatility=raw_vega,
        vega_per_vol_point=raw_vega * 0.01,
        theta_per_year=theta,
        dividend_yield=dividend_yield,
        boundary_convention=None,
        status="PASS",
    )


def black_scholes_pde_residual(
    greeks: CallGreeks,
    *,
    spot: float,
    rate: float,
    volatility: float,
) -> float:
    if not _finite(spot, rate, volatility) or spot <= 0.0 or volatility < 0.0:
        raise ValueError("PDE inputs must be finite with positive spot")
    residual = (
        greeks.theta_per_year
        + (rate - greeks.dividend_yield) * spot * greeks.delta
        + 0.5 * volatility * volatility * spot * spot * greeks.gamma
        - rate * greeks.price
    )
    if not isfinite(residual):
        raise FloatingPointError("PDE residual is non-finite")
    return residual


def delta_hedged_pnl(
    *,
    price_before: float,
    price_after: float,
    spot_before: float,
    spot_after: float,
    delta_before: float,
    gamma_before: float,
    raw_vega_before: float,
    theta_per_year: float,
    volatility_move: float,
    elapsed_years: float,
    rate: float,
    dividend_yield: float,
    option_position: float = 1.0,
    transaction_cost: float = 0.0,
    tolerance: float = 1.0e-2,
) -> HedgePnL:
    values = (
        price_before,
        price_after,
        spot_before,
        spot_after,
        delta_before,
        gamma_before,
        raw_vega_before,
        theta_per_year,
        volatility_move,
        elapsed_years,
        rate,
        dividend_yield,
        option_position,
        transaction_cost,
        tolerance,
    )
    if not _finite(*values) or elapsed_years < 0.0:
        raise ValueError("hedge P&L inputs must be finite and elapsed time non-negative")
    if spot_before <= 0.0 or spot_after <= 0.0:
        raise ValueError("spot values must be positive")
    if transaction_cost < 0.0 or tolerance < 0.0:
        raise ValueError("cost and tolerance must be non-negative")

    stock_position = -option_position * delta_before
    initial_cash = -(option_position * price_before + stock_position * spot_before)
    try:
        cash_financing = initial_cash * (exp(rate * elapsed_years) - 1.0)
        dividend_cash_flow = stock_position * spot_before * (
            exp(dividend_yield * elapsed_years) - 1.0
        )
    except OverflowError as error:
        raise FloatingPointError("financing or dividend cash flow overflow") from error
    spot_move = spot_after - spot_before
    option_change = option_position * (price_after - price_before)
    stock_change = stock_position * spot_move
    hedged = (
        option_change
        + stock_change
        + cash_financing
        + dividend_cash_flow
        - transaction_cost
    )
    approximation = (
        option_position
        * (
            0.5 * gamma_before * spot_move * spot_move
            + raw_vega_before * volatility_move
            + theta_per_year * elapsed_years
        )
        + cash_financing
        + dividend_cash_flow
        - transaction_cost
    )
    residual = hedged - approximation
    if not _finite(
        stock_position,
        initial_cash,
        cash_financing,
        dividend_cash_flow,
        option_change,
        stock_change,
        hedged,
        approximation,
        residual,
    ):
        raise FloatingPointError("self-financing hedge ledger is non-finite")
    return HedgePnL(
        option_position=option_position,
        stock_position=stock_position,
        initial_cash=initial_cash,
        option_change=option_change,
        stock_hedge_change=stock_change,
        cash_financing=cash_financing,
        dividend_cash_flow=dividend_cash_flow,
        transaction_cost=transaction_cost,
        hedged_pnl=hedged,
        delta_gamma_vega_theta_approximation=approximation,
        approximation_residual=residual,
        tolerance=tolerance,
        status="PASS" if abs(residual) <= tolerance else "REJECT",
    )


def weighted_rmse(residuals: list[float], weights: list[float]) -> float:
    if len(residuals) != len(weights) or not residuals:
        raise ValueError("residuals and weights must be aligned and non-empty")
    if not _finite(*residuals, *weights) or any(weight < 0.0 for weight in weights):
        raise ValueError("residuals/weights must be finite and weights non-negative")
    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("positive total weight required")
    result = sqrt(
        sum(
            weight * residual * residual
            for residual, weight in zip(residuals, weights)
        )
        / total_weight
    )
    if not isfinite(result):
        raise FloatingPointError("weighted RMSE is non-finite")
    return result
