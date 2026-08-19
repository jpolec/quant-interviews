from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from .almgren_chriss import optimal_schedule
from .execution_state import EventType, OrderEvent, reconstruct_order
from .hawkes import HawkesParameters, hawkes_diagnostics as _hawkes_diagnostics
from .iceberg import infer_iceberg as _infer_iceberg
from .impact_models import fit_kyle_lambda, fit_square_root_coefficient
from .market_maker import avellaneda_stoikov_quotes as _avellaneda_stoikov_quotes
from .order_book import BookSnapshot
from .order_choice import OrderChoiceInputs, choose_order_type
from .pretrade import OrderIntent, RiskLimits, pretrade_check as _pretrade_check
from .queue_fill import QueueModel
from .smart_router import VenueQuote, route_order
from .tca import tca_decomposition


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def compute_order_book_metrics(
    *,
    bid_prices: Sequence[float],
    bid_sizes: Sequence[float],
    ask_prices: Sequence[float],
    ask_sizes: Sequence[float],
    tick_size: float,
    lot_size: float,
    sweep_side: str | None,
    sweep_quantity: float | None,
) -> dict[str, Any]:
    inputs = {
        "bid_prices": bid_prices,
        "bid_sizes": bid_sizes,
        "ask_prices": ask_prices,
        "ask_sizes": ask_sizes,
        "tick_size": tick_size,
        "lot_size": lot_size,
        "sweep_side": sweep_side,
        "sweep_quantity": sweep_quantity,
    }
    book = BookSnapshot(
        np.asarray(bid_prices, dtype=float),
        np.asarray(bid_sizes, dtype=float),
        np.asarray(ask_prices, dtype=float),
        np.asarray(ask_sizes, dtype=float),
        tick_size,
        lot_size,
    )
    if (sweep_side is None) != (sweep_quantity is None):
        raise ValueError("sweep side and quantity must be provided together")
    metrics = asdict(book.metrics())
    metrics["sweep_vwap"] = (
        None
        if sweep_side is None
        else book.market_order_vwap(sweep_side, float(sweep_quantity))
    )
    metrics["snapshot_hash"] = _digest(inputs)
    metrics["status"] = "PASS"
    return _json_value(metrics)


def estimate_queue_fill_probability(
    *,
    queue_ahead_lots: int,
    marketable_lots_per_second: float,
    cancellation_ahead_lots_per_second: float,
    horizon_seconds: float,
) -> dict[str, Any]:
    probability = QueueModel(
        queue_ahead_lots,
        marketable_lots_per_second,
        cancellation_ahead_lots_per_second,
    ).fill_probability(horizon_seconds)
    return {
        "fill_probability": probability,
        "model": "ORDERED_FINITE_STATE_CTMC",
        "status": "PASS",
    }


def compare_order_types(
    *,
    half_spread_bps: float,
    market_fee_bps: float,
    passive_fee_bps: float,
    passive_rebate_bps: float,
    market_impact_bps: float,
    fill_probability: float,
    adverse_selection_bps_if_filled: float,
    nonfill_opportunity_cost_bps: float,
    replacement_cost_bps: float,
    tie_tolerance_bps: float,
    threshold_tolerance: float,
) -> dict[str, Any]:
    result = choose_order_type(
        OrderChoiceInputs(
            half_spread_bps,
            market_fee_bps,
            passive_fee_bps,
            passive_rebate_bps,
            market_impact_bps,
            fill_probability,
            adverse_selection_bps_if_filled,
            nonfill_opportunity_cost_bps,
            replacement_cost_bps,
        ),
        tie_tolerance_bps=tie_tolerance_bps,
        threshold_tolerance=threshold_tolerance,
    )
    return _json_value(result)


def fit_price_impact_model(
    *,
    model: str,
    regressor: Sequence[float],
    response: Sequence[float],
    confidence_level: float,
    minimum_r_squared: float,
    maximum_condition_number: float,
) -> dict[str, Any]:
    function = {
        "KYLE_LINEAR": fit_kyle_lambda,
        "SQUARE_ROOT": fit_square_root_coefficient,
    }.get(model)
    if function is None:
        raise ValueError("model must be KYLE_LINEAR or SQUARE_ROOT")
    result = function(
        np.asarray(regressor, dtype=float),
        np.asarray(response, dtype=float),
        confidence_level=confidence_level,
        minimum_r_squared=minimum_r_squared,
        maximum_condition_number=maximum_condition_number,
    )
    return _json_value(result)


def optimize_execution_schedule(
    *,
    shares: float,
    horizon: float,
    intervals: int,
    temporary_impact: float,
    risk_aversion: float,
    volatility: float,
) -> dict[str, Any]:
    return _json_value(
        optimal_schedule(
            shares,
            horizon,
            intervals,
            temporary_impact,
            risk_aversion,
            volatility,
        )
    )


def route_child_order(
    *,
    quantity: int,
    venues: Sequence[Mapping[str, Any]],
    non_completion_penalty_bps: float,
    minimum_expected_completion_fraction: float,
    release_mode: str,
) -> dict[str, Any]:
    result = route_order(
        quantity,
        [VenueQuote(**dict(venue)) for venue in venues],
        non_completion_penalty_bps=non_completion_penalty_bps,
        minimum_expected_completion_fraction=minimum_expected_completion_fraction,
        release_mode=release_mode,
    )
    return _json_value(result)


def compute_execution_tca(
    *,
    side: str,
    arrival_price: float,
    order_quantity: float,
    market_volume: float,
    fill_times: Sequence[float],
    fill_quantities: Sequence[float],
    fill_prices: Sequence[float],
    venues: Sequence[str],
    fee_bps: Sequence[float],
    midpoint_at_fills: Sequence[float],
    counterfactual_midpoint_at_fills: Sequence[float],
    ledger_all_in_cost_bps: float,
    reconciliation_tolerance_bps: float,
    post_trade_midpoints: Sequence[float] | None,
    markout_horizon_seconds: float | None,
) -> dict[str, Any]:
    result = tca_decomposition(
        side=side,
        arrival_price=arrival_price,
        order_quantity=order_quantity,
        market_volume=market_volume,
        fill_times=np.asarray(fill_times, dtype=float),
        fill_quantities=np.asarray(fill_quantities, dtype=float),
        fill_prices=np.asarray(fill_prices, dtype=float),
        venues=tuple(venues),
        fee_bps=np.asarray(fee_bps, dtype=float),
        midpoint_at_fills=np.asarray(midpoint_at_fills, dtype=float),
        counterfactual_midpoint_at_fills=np.asarray(
            counterfactual_midpoint_at_fills, dtype=float
        ),
        ledger_all_in_cost_bps=ledger_all_in_cost_bps,
        reconciliation_tolerance_bps=reconciliation_tolerance_bps,
        post_trade_midpoints=(
            None
            if post_trade_midpoints is None
            else np.asarray(post_trade_midpoints, dtype=float)
        ),
        markout_horizon_seconds=markout_horizon_seconds,
    )
    return _json_value(result)


def reconstruct_execution_state(
    *, events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    parsed = [
        OrderEvent(
            event_id=str(event["event_id"]),
            sequence=event["sequence"],
            event_type=EventType(event["event_type"]),
            quantity=event.get("quantity", 0),
            price=event.get("price"),
        )
        for event in events
    ]
    return _json_value(reconstruct_order(parsed))


def infer_iceberg(
    *,
    displayed_after_events: Sequence[float],
    executed_at_level: Sequence[float],
    tolerance: float,
) -> dict[str, Any]:
    return _json_value(
        _infer_iceberg(
            np.asarray(displayed_after_events, dtype=float),
            np.asarray(executed_at_level, dtype=float),
            tolerance,
        )
    )


def hawkes_diagnostics(
    *,
    events: Sequence[float],
    history_before_zero: Sequence[float],
    horizon: float,
    baseline: float,
    excitation: float,
    decay: float,
    minimum_event_count: int,
    minimum_ks_pvalue: float,
    minimum_independence_pvalue: float,
    maximum_absolute_mean_error: float,
    maximum_absolute_variance_error: float,
) -> dict[str, Any]:
    result = _hawkes_diagnostics(
        np.asarray(events, dtype=float),
        HawkesParameters(baseline, excitation, decay),
        horizon,
        history_before_zero=np.asarray(history_before_zero, dtype=float),
        minimum_event_count=minimum_event_count,
        minimum_ks_pvalue=minimum_ks_pvalue,
        minimum_independence_pvalue=minimum_independence_pvalue,
        maximum_absolute_mean_error=maximum_absolute_mean_error,
        maximum_absolute_variance_error=maximum_absolute_variance_error,
    )
    return _json_value(result)


def avellaneda_stoikov_quotes(
    *,
    mid: float,
    inventory: float,
    risk_aversion: float,
    volatility: float,
    time_remaining: float,
    order_arrival_decay: float,
    tick_size: float,
    max_absolute_inventory: float,
    maker_fee: float,
    order_size: float,
    working_bid_quantity: float,
    working_ask_quantity: float,
) -> dict[str, Any]:
    return _json_value(
        _avellaneda_stoikov_quotes(
            mid,
            inventory,
            risk_aversion,
            volatility,
            time_remaining,
            order_arrival_decay,
            tick_size=tick_size,
            max_absolute_inventory=max_absolute_inventory,
            maker_fee=maker_fee,
            order_size=order_size,
            working_bid_quantity=working_bid_quantity,
            working_ask_quantity=working_ask_quantity,
        )
    )


def pretrade_check(
    *, order: Mapping[str, Any], limits: Mapping[str, Any]
) -> dict[str, Any]:
    breaches = _pretrade_check(OrderIntent(**dict(order)), RiskLimits(**dict(limits)))
    return {
        "breaches": breaches,
        "breach_count": len(breaches),
        "status": "PASS" if not breaches else "REJECT",
    }
