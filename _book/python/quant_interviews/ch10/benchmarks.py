from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExecutionBenchmarks:
    realized_average_price: float | None
    executed_quantity: float
    market_vwap: float
    market_twap: float
    terminal_market_price: float
    implementation_shortfall: float
    delay_cost: float
    trading_cost: float
    opportunity_cost: float


def execution_benchmarks(
    side: str,
    decision_price: float,
    arrival_price: float,
    child_prices: FloatArray,
    child_quantities: FloatArray,
    market_prices: FloatArray,
    market_volumes: FloatArray,
    target_quantity: float,
    *,
    market_timestamps: FloatArray,
    decision_time: float,
    arrival_time: float,
    end_time: float,
) -> ExecutionBenchmarks:
    p = np.asarray(child_prices, dtype=float)
    q = np.asarray(child_quantities, dtype=float)
    mp = np.asarray(market_prices, dtype=float)
    mv = np.asarray(market_volumes, dtype=float)
    mt = np.asarray(market_timestamps, dtype=float)
    if (
        p.ndim != 1
        or q.ndim != 1
        or mp.ndim != 1
        or mv.ndim != 1
        or mt.ndim != 1
        or len(p) != len(q)
        or len(mp) != len(mv)
        or len(mp) != len(mt)
        or len(mp) < 2
    ):
        raise ValueError("prices and quantities must align")
    arrays = (p, q, mp, mv, mt)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("price and quantity arrays must be finite")
    economic_scalars = np.asarray(
        [decision_price, arrival_price, target_quantity],
        dtype=float,
    )
    time_scalars = np.asarray([decision_time, arrival_time, end_time], dtype=float)
    if (
        not np.all(np.isfinite(economic_scalars))
        or np.any(economic_scalars <= 0.0)
        or not np.all(np.isfinite(time_scalars))
    ):
        raise ValueError("prices and target quantity must be finite and positive")
    if (p.size and np.any(p <= 0.0)) or np.any(mp <= 0.0):
        raise ValueError("prices must be positive")
    if np.any(q < 0.0) or np.any(mv < 0.0) or mv.sum() <= 0.0:
        raise ValueError("quantities must be non-negative and market volume positive")
    if (
        np.any(np.diff(mt) <= 0.0)
        or decision_time > arrival_time
        or arrival_time > mt[0]
        or not np.isclose(mt[-1], end_time, rtol=0.0, atol=1e-12)
        or end_time <= mt[0]
    ):
        raise ValueError("market timestamps must be ordered and end at the PIT cutoff")
    sign = 1.0 if side.lower() == "buy" else -1.0 if side.lower() == "sell" else None
    if sign is None:
        raise ValueError("side must be buy or sell")
    executed = float(q.sum())
    if executed > target_quantity:
        raise ValueError("executed quantity cannot exceed target quantity")
    avg = None if executed == 0.0 else float(np.dot(p, q) / executed)
    vwap = float(np.dot(mp, mv) / mv.sum())
    widths = np.diff(mt)
    twap = float(np.sum(0.5 * (mp[:-1] + mp[1:]) * widths) / (end_time - mt[0]))
    final_price = float(mp[-1])
    delay = sign * target_quantity * (arrival_price - decision_price)
    trading = 0.0 if avg is None else sign * executed * (avg - arrival_price)
    opportunity = sign * (target_quantity - executed) * (final_price - arrival_price)
    shortfall = delay + trading + opportunity
    return ExecutionBenchmarks(
        avg,
        executed,
        vwap,
        twap,
        final_price,
        delay + trading + opportunity,
        delay,
        trading,
        opportunity,
    )
