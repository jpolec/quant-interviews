from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog


@dataclass(frozen=True)
class SurfaceQuote:
    quote_id: str
    underlying: str
    maturity: float
    strike: float
    bid: float
    ask: float
    forward: float
    discount_factor: float
    exercise_style: str = "EUROPEAN"
    settlement_convention: str = "CASH"


def _validated_slice(
    strikes: NDArray[np.float64],
    calls: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    strikes = np.asarray(strikes, dtype=float)
    calls = np.asarray(calls, dtype=float)
    if strikes.ndim != 1 or calls.shape != strikes.shape or strikes.size < 2:
        raise ValueError("strike slice must contain aligned one-dimensional data")
    if not np.all(np.isfinite(strikes)) or not np.all(np.isfinite(calls)):
        raise ValueError("strike slice must be finite")
    order = np.argsort(strikes, kind="stable")
    sorted_strikes = strikes[order]
    if np.any(np.diff(sorted_strikes) <= 0.0):
        raise ValueError("strikes must be unique")
    return sorted_strikes, calls[order], order


def vertical_spread_audit(
    strikes: NDArray[np.float64],
    calls: NDArray[np.float64],
    discount_factor: float,
    *,
    quote_ids: tuple[str, ...] | None = None,
    tolerance: float = 1e-10,
) -> list[dict[str, Any]]:
    if (
        not isfinite(discount_factor)
        or discount_factor <= 0.0
        or not isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("invalid vertical-spread controls")
    k, c, order = _validated_slice(strikes, calls)
    if quote_ids is not None and len(quote_ids) != k.size:
        raise ValueError("quote-id alignment failure")
    slopes = np.diff(c) / np.diff(k)
    violations: list[dict[str, Any]] = []
    for index, slope in enumerate(slopes):
        if slope > tolerance:
            constraint = "call_price_nonincreasing"
            residual = slope
        elif slope < -discount_factor - tolerance:
            constraint = "vertical_spread_discount_bound"
            residual = -discount_factor - slope
        else:
            continue
        left_original = int(order[index])
        right_original = int(order[index + 1])
        violations.append(
            {
                "left_strike": float(k[index]),
                "right_strike": float(k[index + 1]),
                "slope": float(slope),
                "threshold": (
                    0.0
                    if constraint == "call_price_nonincreasing"
                    else -discount_factor
                ),
                "residual": float(residual),
                "constraint_type": constraint,
                "left_quote_id": (
                    quote_ids[left_original] if quote_ids is not None else None
                ),
                "right_quote_id": (
                    quote_ids[right_original] if quote_ids is not None else None
                ),
            }
        )
    return violations


def vertical_spread_violations(
    strikes: NDArray[np.float64],
    calls: NDArray[np.float64],
    discount_factor: float,
    tolerance: float = 1e-10,
) -> list[int]:
    """Compatibility wrapper returning sorted-interval indices."""

    k, c, _ = _validated_slice(strikes, calls)
    slopes = np.diff(c) / np.diff(k)
    return [
        int(index)
        for index, slope in enumerate(slopes)
        if slope > tolerance or slope < -discount_factor - tolerance
    ]


def butterfly_audit(
    strikes: NDArray[np.float64],
    calls: NDArray[np.float64],
    *,
    quote_ids: tuple[str, ...] | None = None,
    tolerance: float = 1e-10,
) -> list[dict[str, Any]]:
    k, c, order = _validated_slice(strikes, calls)
    if k.size < 3:
        raise ValueError("butterfly audit requires at least three strikes")
    if quote_ids is not None and len(quote_ids) != k.size:
        raise ValueError("quote-id alignment failure")
    left = np.diff(c[:-1]) / np.diff(k[:-1])
    right = np.diff(c[1:]) / np.diff(k[1:])
    violations: list[dict[str, Any]] = []
    for index, (left_slope, right_slope) in enumerate(zip(left, right)):
        residual = left_slope - right_slope
        if residual <= tolerance:
            continue
        original = [int(order[index + offset]) for offset in range(3)]
        violations.append(
            {
                "left_strike": float(k[index]),
                "middle_strike": float(k[index + 1]),
                "right_strike": float(k[index + 2]),
                "left_slope": float(left_slope),
                "right_slope": float(right_slope),
                "curvature_residual": float(residual),
                "constraint_type": "call_price_convexity",
                "quote_ids": (
                    tuple(quote_ids[item] for item in original)
                    if quote_ids is not None
                    else None
                ),
            }
        )
    return violations


def butterfly_violations(
    strikes: NDArray[np.float64],
    calls: NDArray[np.float64],
    tolerance: float = 1e-10,
) -> list[int]:
    k, c, _ = _validated_slice(strikes, calls)
    if k.size < 3:
        raise ValueError("butterfly audit requires at least three strikes")
    left = np.diff(c[:-1]) / np.diff(k[:-1])
    right = np.diff(c[1:]) / np.diff(k[1:])
    return [
        int(index + 1)
        for index, (a, b) in enumerate(zip(left, right))
        if b < a - tolerance
    ]


def calendar_surface_audit(
    *,
    log_moneyness: NDArray[np.float64],
    maturities: NDArray[np.float64],
    total_variance: NDArray[np.float64],
    forwards: NDArray[np.float64],
    discount_factors: NDArray[np.float64],
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Audit total variance on an explicitly common forward-moneyness grid."""

    k = np.asarray(log_moneyness, dtype=float)
    t = np.asarray(maturities, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    f = np.asarray(forwards, dtype=float)
    d = np.asarray(discount_factors, dtype=float)
    if (
        k.ndim != 1
        or t.ndim != 1
        or w.shape != (t.size, k.size)
        or f.shape != t.shape
        or d.shape != t.shape
        or k.size == 0
        or t.size < 2
    ):
        raise ValueError("calendar surface has inconsistent shape")
    if not all(
        np.all(np.isfinite(array)) for array in (k, t, w, f, d)
    ):
        raise ValueError("calendar surface must be finite")
    if (
        np.any(np.diff(k) <= 0.0)
        or np.any(np.diff(t) <= 0.0)
        or np.any(f <= 0.0)
        or np.any(d <= 0.0)
        or np.any(w < 0.0)
        or not isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("invalid calendar-surface coordinates")

    violations: list[dict[str, Any]] = []
    for maturity_index in range(t.size - 1):
        change = w[maturity_index + 1] - w[maturity_index]
        for strike_index in np.flatnonzero(change < -tolerance):
            price_scale = (
                d[maturity_index + 1]
                * f[maturity_index + 1]
                * max(abs(change[strike_index]), 1e-16)
            )
            violations.append(
                {
                    "log_moneyness": float(k[strike_index]),
                    "earlier_maturity": float(t[maturity_index]),
                    "later_maturity": float(t[maturity_index + 1]),
                    "earlier_total_variance": float(
                        w[maturity_index, strike_index]
                    ),
                    "later_total_variance": float(
                        w[maturity_index + 1, strike_index]
                    ),
                    "variance_residual": float(-change[strike_index]),
                    "forward_variance_scale_proxy": float(price_scale),
                    "assumption": (
                        "common forward log-moneyness with reconciled forward "
                        "and discount curves"
                    ),
                }
            )
    return {
        "violations": violations,
        "violation_count": len(violations),
        "status": "PASS" if not violations else "REJECT",
        "common_log_moneyness": k,
    }


def calendar_violations(
    maturities: NDArray[np.float64],
    total_variances: NDArray[np.float64],
    tolerance: float = 1e-10,
) -> list[int]:
    """Compatibility helper for one fixed forward-moneyness coordinate."""

    maturities = np.asarray(maturities, dtype=float)
    total_variances = np.asarray(total_variances, dtype=float)
    if (
        maturities.ndim != 1
        or total_variances.shape != maturities.shape
        or maturities.size < 2
        or not np.all(np.isfinite(maturities))
        or not np.all(np.isfinite(total_variances))
        or not isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError("invalid calendar inputs")
    order = np.argsort(maturities, kind="stable")
    sorted_maturities = maturities[order]
    if np.any(np.diff(sorted_maturities) <= 0.0):
        raise ValueError("maturities must be unique")
    w = total_variances[order]
    return [
        int(index)
        for index, value in enumerate(np.diff(w))
        if value < -tolerance
    ]


def _arbitrage_constraints(
    quotes: tuple[SurfaceQuote, ...],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    list[dict[str, Any]],
    NDArray[np.int64],
]:
    order = np.array(
        sorted(
            range(len(quotes)),
            key=lambda index: (
                quotes[index].underlying,
                quotes[index].maturity,
                quotes[index].exercise_style,
                quotes[index].settlement_convention,
                quotes[index].strike,
            ),
        ),
        dtype=int,
    )
    sorted_quotes = tuple(quotes[index] for index in order)
    rows: list[NDArray[np.float64]] = []
    rhs: list[float] = []
    metadata: list[dict[str, Any]] = []
    slice_keys = sorted(
        {
            (
                quote.underlying,
                quote.maturity,
                quote.exercise_style,
                quote.settlement_convention,
            )
            for quote in sorted_quotes
        }
    )
    for underlying, maturity, exercise_style, settlement in slice_keys:
        indices = [
            index
            for index, quote in enumerate(sorted_quotes)
            if (
                quote.underlying,
                quote.maturity,
                quote.exercise_style,
                quote.settlement_convention,
            )
            == (underlying, maturity, exercise_style, settlement)
        ]
        if len(indices) < 3:
            raise ValueError("each maturity requires at least three strikes")
        maturity_quotes = [sorted_quotes[index] for index in indices]
        strikes = np.array([quote.strike for quote in maturity_quotes])
        if np.any(np.diff(strikes) <= 0.0):
            raise ValueError("duplicate strikes within maturity")
        discount = maturity_quotes[0].discount_factor
        forward = maturity_quotes[0].forward
        if any(
            not np.isclose(quote.discount_factor, discount, rtol=1e-12, atol=1e-14)
            for quote in maturity_quotes
        ):
            raise ValueError("discount factor differs within maturity")
        if any(
            not np.isclose(quote.forward, forward, rtol=1e-12, atol=1e-12)
            for quote in maturity_quotes
        ):
            raise ValueError("forward differs within strike slice")

        for index in indices:
            quote = sorted_quotes[index]
            lower_bound = max(
                quote.discount_factor * (quote.forward - quote.strike),
                0.0,
            )
            upper_bound = quote.discount_factor * quote.forward

            lower_row = np.zeros(len(sorted_quotes))
            lower_row[index] = -1.0
            rows.append(lower_row)
            rhs.append(-lower_bound)
            metadata.append(
                {
                    "type": "call_lower_bound",
                    "underlying": underlying,
                    "maturity": maturity,
                    "quote_ids": (quote.quote_id,),
                    "signed_limit": lower_bound,
                }
            )

            upper_row = np.zeros(len(sorted_quotes))
            upper_row[index] = 1.0
            rows.append(upper_row)
            rhs.append(upper_bound)
            metadata.append(
                {
                    "type": "call_upper_bound",
                    "underlying": underlying,
                    "maturity": maturity,
                    "quote_ids": (quote.quote_id,),
                    "signed_limit": upper_bound,
                }
            )

        for local in range(len(indices) - 1):
            left = indices[local]
            right = indices[local + 1]
            width = strikes[local + 1] - strikes[local]
            monotone = np.zeros(len(sorted_quotes))
            monotone[right] = 1.0
            monotone[left] = -1.0
            rows.append(monotone)
            rhs.append(0.0)
            metadata.append(
                {
                    "type": "monotonicity",
                    "underlying": underlying,
                    "maturity": maturity,
                    "quote_ids": (
                        sorted_quotes[left].quote_id,
                        sorted_quotes[right].quote_id,
                    ),
                }
            )
            slope_floor = np.zeros(len(sorted_quotes))
            slope_floor[left] = 1.0
            slope_floor[right] = -1.0
            rows.append(slope_floor)
            rhs.append(discount * width)
            metadata.append(
                {
                    "type": "vertical_spread_floor",
                    "underlying": underlying,
                    "maturity": maturity,
                    "quote_ids": (
                        sorted_quotes[left].quote_id,
                        sorted_quotes[right].quote_id,
                    ),
                }
            )

        for local in range(1, len(indices) - 1):
            left = indices[local - 1]
            middle = indices[local]
            right = indices[local + 1]
            left_width = strikes[local] - strikes[local - 1]
            right_width = strikes[local + 1] - strikes[local]
            convexity = np.zeros(len(sorted_quotes))
            convexity[left] = -1.0 / left_width
            convexity[middle] = 1.0 / left_width + 1.0 / right_width
            convexity[right] = -1.0 / right_width
            rows.append(convexity)
            rhs.append(0.0)
            metadata.append(
                {
                    "type": "convexity",
                    "underlying": underlying,
                    "maturity": maturity,
                    "quote_ids": (
                        sorted_quotes[left].quote_id,
                        sorted_quotes[middle].quote_id,
                        sorted_quotes[right].quote_id,
                    ),
                }
            )
    return np.vstack(rows), np.asarray(rhs), metadata, order


def bid_ask_static_arbitrage_lp(
    quotes: tuple[SurfaceQuote, ...],
    *,
    repair_weights: NDArray[np.float64] | None = None,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Audit strike-slice bid/ask feasibility or compute spread expansion.

    Calendar consistency is deliberately a separate certificate because it
    requires reconciled coordinates across maturities.  This LP groups quotes
    by underlying, maturity, exercise style and settlement convention and
    enforces call bounds, vertical slopes and convexity within each slice.
    """

    if len(quotes) < 3 or not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("invalid sparse-surface controls")
    for quote in quotes:
        numeric = (
            quote.maturity,
            quote.strike,
            quote.bid,
            quote.ask,
            quote.forward,
            quote.discount_factor,
        )
        if (
            not quote.quote_id
            or not quote.underlying
            or not quote.exercise_style
            or not quote.settlement_convention
            or not all(isfinite(value) for value in numeric)
            or quote.maturity <= 0.0
            or quote.strike <= 0.0
            or quote.bid < 0.0
            or quote.ask < quote.bid
            or quote.forward <= 0.0
            or quote.discount_factor <= 0.0
        ):
            raise ValueError("invalid surface quote")
    if len({quote.quote_id for quote in quotes}) != len(quotes):
        raise ValueError("surface quote IDs must be globally unique")

    constraints, rhs, metadata, order = _arbitrage_constraints(quotes)
    sorted_quotes = tuple(quotes[index] for index in order)
    mid = np.array([(quote.bid + quote.ask) / 2.0 for quote in sorted_quotes])
    bounds = [(quote.bid, quote.ask) for quote in sorted_quotes]
    feasibility = linprog(
        np.zeros(len(sorted_quotes)),
        A_ub=constraints,
        b_ub=rhs,
        bounds=bounds,
        method="highs",
    )
    mid_residuals = constraints @ mid - rhs
    mid_violations = [
        {
            **metadata[index],
            "residual": float(value),
        }
        for index, value in enumerate(mid_residuals)
        if value > tolerance
    ]
    if feasibility.success:
        selected = feasibility.x
        binding = [
            {
                **metadata[index],
                "slack": float(rhs[index] - constraints[index] @ selected),
            }
            for index in range(len(metadata))
            if abs(rhs[index] - constraints[index] @ selected) <= tolerance
        ]
        return {
            "status": "PASS",
            "selected_prices": {
                quote.quote_id: float(selected[index])
                for index, quote in enumerate(sorted_quotes)
            },
            "mid_violations": mid_violations,
            "executable_violations": [],
            "minimum_quote_repair": 0.0,
            "repair_objective_unit": "price units of total bid/ask expansion",
            "moved_quotes": [],
            "binding_constraints": binding,
            "dual_prices": [],
        }

    count = len(sorted_quotes)
    if repair_weights is None:
        weights = np.ones(count)
    else:
        raw_weights = np.asarray(repair_weights, dtype=float)
        if raw_weights.shape != (count,):
            raise ValueError("invalid repair weights")
        weights = raw_weights[order]
    if (
        weights.shape != (count,)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("invalid repair weights")
    objective = np.concatenate((np.zeros(count), weights, weights))
    repair_rows: list[NDArray[np.float64]] = []
    repair_rhs: list[float] = []
    repair_metadata: list[dict[str, Any]] = []
    for row, value, item in zip(constraints, rhs, metadata):
        repair_rows.append(np.concatenate((row, np.zeros(2 * count))))
        repair_rhs.append(float(value))
        repair_metadata.append(item)
    for index, quote in enumerate(sorted_quotes):
        lower = np.zeros(3 * count)
        lower[index] = -1.0
        lower[count + index] = -1.0
        repair_rows.append(lower)
        repair_rhs.append(-quote.bid)
        repair_metadata.append(
            {"type": "bid_bound", "quote_ids": (quote.quote_id,)}
        )
        upper = np.zeros(3 * count)
        upper[index] = 1.0
        upper[2 * count + index] = -1.0
        repair_rows.append(upper)
        repair_rhs.append(quote.ask)
        repair_metadata.append(
            {"type": "ask_bound", "quote_ids": (quote.quote_id,)}
        )
    repair = linprog(
        objective,
        A_ub=np.vstack(repair_rows),
        b_ub=np.asarray(repair_rhs),
        bounds=[(None, None)] * count + [(0.0, None)] * (2 * count),
        method="highs",
    )
    if not repair.success:
        raise RuntimeError(f"minimum-repair LP failed: {repair.message}")
    lower_slack = repair.x[count : 2 * count]
    upper_slack = repair.x[2 * count :]
    moved = []
    for index, quote in enumerate(sorted_quotes):
        if lower_slack[index] > tolerance or upper_slack[index] > tolerance:
            moved.append(
                {
                    "quote_id": quote.quote_id,
                    "underlying": quote.underlying,
                    "maturity": quote.maturity,
                    "strike": quote.strike,
                    "bid_repair": float(lower_slack[index]),
                    "ask_repair": float(upper_slack[index]),
                    "log_moneyness": log(quote.strike / quote.forward),
                }
            )
    marginals = np.asarray(repair.ineqlin.marginals)
    dual_certificate = [
        {
            **repair_metadata[index],
            "dual": float(marginals[index]),
        }
        for index in np.flatnonzero(np.abs(marginals) > tolerance)
    ]
    return {
        "status": "REJECT",
        "selected_prices": {
            quote.quote_id: float(repair.x[index])
            for index, quote in enumerate(sorted_quotes)
        },
        "mid_violations": mid_violations,
        "executable_violations": dual_certificate,
        "minimum_quote_repair": float(repair.fun),
        "repair_objective_unit": "price units of total bid/ask expansion",
        "moved_quotes": moved,
        "binding_constraints": [],
        "dual_prices": dual_certificate,
    }
