from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TCAResult:
    executed_quantity: float
    coverage_fraction: float
    price_slippage_bps: float
    all_in_cost_bps: float
    model_attribution_total_bps: float
    spread_component_bps: float
    impact_component_bps: float
    timing_component_bps: float
    fee_component_bps: float
    residual_bps: float
    post_trade_drift_bps: float | None
    adverse_selection_cost_bps: float | None
    participation_rate: float
    reconciliation_tolerance_bps: float
    status: str


def tca_decomposition(
    *,
    side: str,
    arrival_price: float,
    order_quantity: float,
    market_volume: float,
    fill_times: FloatArray,
    fill_quantities: FloatArray,
    fill_prices: FloatArray,
    venues: tuple[str, ...],
    fee_bps: FloatArray,
    midpoint_at_fills: FloatArray,
    counterfactual_midpoint_at_fills: FloatArray,
    ledger_all_in_cost_bps: float,
    reconciliation_tolerance_bps: float = 1e-8,
    post_trade_midpoints: FloatArray | None = None,
    markout_horizon_seconds: float | None = None,
) -> TCAResult:
    times = np.asarray(fill_times, dtype=float)
    quantities = np.asarray(fill_quantities, dtype=float)
    prices = np.asarray(fill_prices, dtype=float)
    fees = np.asarray(fee_bps, dtype=float)
    midpoints = np.asarray(midpoint_at_fills, dtype=float)
    counterfactuals = np.asarray(counterfactual_midpoint_at_fills, dtype=float)
    arrays = (times, quantities, prices, fees, midpoints, counterfactuals)
    if (
        any(array.ndim != 1 for array in arrays)
        or len(times) == 0
        or len({len(array) for array in arrays}) != 1
        or len(venues) != len(times)
    ):
        raise ValueError("fill-level TCA arrays and venue IDs must align")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("fill-level TCA inputs must be finite")
    if any(not venue for venue in venues):
        raise ValueError("venue IDs must be non-empty")
    if np.any(np.diff(times) < 0.0) or np.any(times < 0.0):
        raise ValueError("fill times must be chronological and non-negative")
    if np.any(quantities <= 0.0):
        raise ValueError("fill quantities must be strictly positive")
    if np.any(prices <= 0.0) or np.any(midpoints <= 0.0) or np.any(counterfactuals <= 0.0):
        raise ValueError("fill and benchmark prices must be strictly positive")
    scalars = np.asarray(
        [
            arrival_price,
            order_quantity,
            market_volume,
            ledger_all_in_cost_bps,
            reconciliation_tolerance_bps,
        ],
        dtype=float,
    )
    if not np.isfinite(scalars).all():
        raise ValueError("TCA controls must be finite")
    if arrival_price <= 0.0 or order_quantity <= 0.0 or market_volume <= 0.0:
        raise ValueError("arrival price and quantities must be positive")
    if reconciliation_tolerance_bps < 0.0:
        raise ValueError("reconciliation tolerance cannot be negative")
    executed = float(np.sum(quantities))
    if executed > order_quantity + 1e-12:
        raise ValueError("fill records exceed the parent quantity")
    participation = order_quantity / market_volume
    if not 0.0 <= participation <= 1.0:
        raise ValueError("participation must lie in [0,1]")
    sign = 1.0 if side.lower() == "buy" else -1.0 if side.lower() == "sell" else None
    if sign is None:
        raise ValueError("side must be buy or sell")

    scale = 10_000.0 / arrival_price
    weights = quantities / executed
    price_slippage = float(
        np.dot(weights, sign * (prices - arrival_price) * scale)
    )
    spread = float(np.dot(weights, sign * (prices - midpoints) * scale))
    impact = float(
        np.dot(weights, sign * (midpoints - counterfactuals) * scale)
    )
    timing = float(
        np.dot(weights, sign * (counterfactuals - arrival_price) * scale)
    )
    fee_component = float(np.dot(weights, fees))
    model_total = spread + impact + timing + fee_component
    residual = ledger_all_in_cost_bps - model_total
    if (post_trade_midpoints is None) != (markout_horizon_seconds is None):
        raise ValueError("post-trade midpoints and markout horizon must be provided together")
    if post_trade_midpoints is None:
        signed_markout = None
    else:
        post = np.asarray(post_trade_midpoints, dtype=float)
        if (
            post.shape != prices.shape
            or not np.isfinite(post).all()
            or np.any(post <= 0.0)
            or not np.isfinite(markout_horizon_seconds)
            or markout_horizon_seconds <= 0.0
        ):
            raise ValueError("invalid post-trade markout inputs")
        signed_markout = float(
            np.dot(weights, sign * (post - midpoints) * scale)
        )
    adverse_selection_cost = None if signed_markout is None else -signed_markout
    return TCAResult(
        executed,
        executed / order_quantity,
        price_slippage,
        ledger_all_in_cost_bps,
        model_total,
        spread,
        impact,
        timing,
        fee_component,
        residual,
        signed_markout,
        adverse_selection_cost,
        participation,
        reconciliation_tolerance_bps,
        "PASS" if abs(residual) <= reconciliation_tolerance_bps else "REJECT",
    )


def moving_block_bootstrap_mean(
    observations: FloatArray,
    block_length: int,
    simulations: int = 5_000,
    seed: int = 11,
) -> tuple[float, float, float]:
    x = np.asarray(observations, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("observations must be a finite one-dimensional array")
    if (
        isinstance(block_length, bool)
        or not isinstance(block_length, int)
        or not 1 <= block_length <= len(x)
    ):
        raise ValueError("invalid block length")
    if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations <= 0:
        raise ValueError("simulations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(x) - block_length + 1)
    means = np.empty(simulations)
    blocks_needed = int(np.ceil(len(x) / block_length))
    for i in range(simulations):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([x[s : s + block_length] for s in chosen])[: len(x)]
        means[i] = sample.mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)
