from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Any

import numpy as np

from .american_binomial import american_lattice_convergence
from .barrier_mc import barrier_monitoring_experiment
from .binomial_state_prices import two_state_replication_certificate
from .dupire import dupire_local_variance_diagnostic
from .model_free_bounds import discrete_model_free_bounds
from .parity_audit import executable_put_call_parity_band
from .payoff_bounds import executable_call_bound_certificate
from .static_arbitrage import SurfaceQuote, bid_ask_static_arbitrage_lp


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(nested) for nested in value]
    return value


def _hash(value: Any) -> str:
    return sha256(
        json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _validate_execution(execution: dict[str, Any]) -> None:
    if (
        set(execution) != {"request_id", "data_as_of", "code_version", "input_sha256"}
        or not execution["request_id"]
        or not execution["data_as_of"]
        or execution["code_version"] != "v0.10"
        or not isinstance(execution["input_sha256"], str)
        or len(execution["input_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in execution["input_sha256"])
    ):
        raise ValueError("invalid execution envelope")


def _response(result: dict[str, Any]) -> dict[str, Any]:
    response = _jsonable(result)
    response["evidence_sha256"] = _hash(response)
    return response


def audit_option_bounds(
    *, execution: dict[str, Any], call_bid: float, call_ask: float,
    spot_bid: float, spot_ask: float, strike: float, discount_factor: float,
    continuous_dividend_discount_factor: float, execution_cost: float,
    stock_borrow_available: bool, tolerance: float,
) -> dict[str, Any]:
    _validate_execution(execution)
    result = executable_call_bound_certificate(
        call_bid=call_bid, call_ask=call_ask, spot_bid=spot_bid,
        spot_ask=spot_ask, strike=strike, discount_factor=discount_factor,
        continuous_dividend_discount_factor=continuous_dividend_discount_factor,
        execution_cost=execution_cost,
        stock_borrow_available=stock_borrow_available, tolerance=tolerance,
    )
    return _response({
        "lower_bound": result["lower_bound"],
        "upper_bound": result["upper_bound"],
        "initial_cash_flow": result["initial_cash_flow"],
        "terminal_payoff_floor": result["terminal_payoff_floor"],
        "status": result["status"],
    })


def compute_state_prices(
    *, execution: dict[str, Any], spot: float, up: float, down: float,
    payoff_up: float, payoff_down: float, gross_rate: float,
) -> dict[str, Any]:
    _validate_execution(execution)
    result = two_state_replication_certificate(
        spot=spot, up=up, down=down, payoff_up=payoff_up,
        payoff_down=payoff_down, gross_rate=gross_rate,
    )
    return _response({
        "state_prices": result["state_prices"],
        "replication_price": result["replication_price"],
        "maximum_residual": max(
            abs(result["up_state_residual"]),
            abs(result["down_state_residual"]),
            abs(result["pricing_residual"]),
        ),
        "status": result["status"],
    })


def audit_put_call_parity(
    *, execution: dict[str, Any], call_bid: float, call_ask: float,
    put_bid: float, put_ask: float, spot_bid: float, spot_ask: float,
    strike: float, maturity: float, long_funding_rate: float,
    short_funding_rate: float, borrow_rate: float, dividend_yield: float,
    tolerance: float,
) -> dict[str, Any]:
    _validate_execution(execution)
    result = executable_put_call_parity_band(
        call_bid=call_bid, call_ask=call_ask, put_bid=put_bid,
        put_ask=put_ask, spot_bid=spot_bid, spot_ask=spot_ask,
        strike=strike, maturity=maturity, long_funding_rate=long_funding_rate,
        short_funding_rate=short_funding_rate, borrow_rate=borrow_rate,
        dividend_yield=dividend_yield, tolerance=tolerance,
    )
    return _response({
        "lower_residual": result["lower_residual"],
        "upper_residual": result["upper_residual"],
        "status": result["status"],
    })


def validate_static_surface(
    *, execution: dict[str, Any], quotes: list[dict[str, Any]], tolerance: float,
) -> dict[str, Any]:
    _validate_execution(execution)
    result = bid_ask_static_arbitrage_lp(
        tuple(SurfaceQuote(**quote) for quote in quotes), tolerance=tolerance
    )
    return _response({
        "minimum_quote_repair": result["minimum_quote_repair"],
        "moved_quote_count": len(result["moved_quotes"]),
        "status": result["status"],
    })


def price_american_lattice(
    *, execution: dict[str, Any], spot: float, strike: float, rate: float,
    dividend_yield: float, volatility: float, maturity: float,
    base_steps: list[int], price_absolute_tolerance: float,
    price_relative_tolerance: float, boundary_tolerance: float,
) -> dict[str, Any]:
    _validate_execution(execution)
    result = american_lattice_convergence(
        spot=spot, strike=strike, rate=rate, dividend_yield=dividend_yield,
        volatility=volatility, maturity=maturity,
        base_steps=tuple(base_steps),
        price_absolute_tolerance=price_absolute_tolerance,
        price_relative_tolerance=price_relative_tolerance,
        boundary_tolerance=boundary_tolerance,
    )
    return _response({
        "price": result.finest_price,
        "finest_pair_spread": result.pair_spreads[-1],
        "boundary_stability_error": result.boundary_stability_error,
        "status": result.status,
    })


def price_barrier_option(
    *, execution: dict[str, Any], spot: float, strike: float, barrier: float,
    rate: float, dividend_yield: float, volatility: float, maturity: float,
    fine_steps: int, paths: int, monitoring_schedules: list[dict[str, Any]],
    seed: int, batch_size: int, reference_sigma_multiple: float,
    reference_absolute_tolerance: float,
) -> dict[str, Any]:
    _validate_execution(execution)
    result = barrier_monitoring_experiment(
        spot=spot, strike=strike, barrier=barrier, rate=rate,
        dividend_yield=dividend_yield, volatility=volatility,
        maturity=maturity, fine_steps=fine_steps, paths=paths,
        monitoring_indices={
            str(item["label"]): tuple(item["indices"])
            for item in monitoring_schedules
        }, seed=seed, batch_size=batch_size,
        reference_sigma_multiple=reference_sigma_multiple,
        reference_absolute_tolerance=reference_absolute_tolerance,
    )
    return _response({
        "bridge_value": result["bridge_continuous_value"],
        "standard_error": result["bridge_standard_error"],
        "reference_error": result["bridge_reference_error"],
        "reference_limit": result["bridge_reference_limit"],
        "status": result["status"],
    })


def compute_dupire_local_vol(
    *, execution: dict[str, Any], strike: list[float], maturity: list[float],
    call_surface: list[list[float]], rate: float, dividend_yield: float,
    forwards: list[float], discount_factors: list[float],
    upstream_surface_sha256: str, upstream_static_arbitrage_status: str,
    expected_local_variance: float, density_absolute_tolerance: float,
    density_relative_tolerance: float, median_recovery_tolerance: float,
    maximum_recovery_tolerance: float,
) -> dict[str, Any]:
    _validate_execution(execution)
    result = dupire_local_variance_diagnostic(
        strike=np.asarray(strike), maturity=np.asarray(maturity),
        call_surface=np.asarray(call_surface), rate=rate,
        dividend_yield=dividend_yield, forwards=np.asarray(forwards),
        discount_factors=np.asarray(discount_factors),
        upstream_surface_sha256=upstream_surface_sha256,
        upstream_static_arbitrage_status=upstream_static_arbitrage_status,
        expected_local_variance=expected_local_variance,
        density_absolute_tolerance=density_absolute_tolerance,
        density_relative_tolerance=density_relative_tolerance,
        median_recovery_tolerance=median_recovery_tolerance,
        maximum_recovery_tolerance=maximum_recovery_tolerance,
    )
    return _response({
        "median_local_variance": result.median_interior_local_variance,
        "invalid_fraction": max(result.negative_fraction, result.nonfinite_fraction),
        "call_bound_violation_count": result.call_bound_violation_count,
        "strike_slope_violation_count": result.strike_slope_violation_count,
        "status": result.status,
    })


def solve_model_free_bound(
    *, execution: dict[str, Any], state_coordinates: list[float],
    payoff: list[float], hedge_payoffs: list[list[float]],
    hedge_prices: list[float], target_breakpoints: list[float],
    tolerance: float,
) -> dict[str, Any]:
    _validate_execution(execution)
    result = discrete_model_free_bounds(
        np.asarray(state_coordinates), np.asarray(payoff),
        np.asarray(hedge_payoffs), np.asarray(hedge_prices),
        target_breakpoints=np.asarray(target_breakpoints), tolerance=tolerance,
    )
    return _response({
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "maximum_duality_gap": max(
            abs(result.lower_duality_gap), abs(result.upper_duality_gap)
        ),
        "scope": result.scope,
        "continuum_certified": result.continuum_certified,
        "status": result.solver_status,
    })
