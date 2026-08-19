from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Brinson:
    allocation: FloatArray
    selection: FloatArray
    interaction: FloatArray
    active_return: float
    explained_return: float
    residual: float
    status: str


@dataclass(frozen=True)
class ReconciliationResult:
    total: float
    explained: float
    residual: float
    residual_to_nav: float
    residual_to_gross_attribution: float
    status: str


@dataclass(frozen=True)
class GeometricLinkResult:
    linked_contributions: FloatArray
    compounded_return: float
    explained_return: float
    residual: float
    maximum_period_residual: float
    status: str


def brinson(
    portfolio_weights: FloatArray,
    benchmark_weights: FloatArray,
    portfolio_returns: FloatArray,
    benchmark_returns: FloatArray,
) -> Brinson:
    wp = np.asarray(portfolio_weights, dtype=float)
    wb = np.asarray(benchmark_weights, dtype=float)
    rp = np.asarray(portfolio_returns, dtype=float)
    rb = np.asarray(benchmark_returns, dtype=float)
    if not (
        wp.shape == wb.shape == rp.shape == rb.shape
        and wp.ndim == 1
        and np.isfinite(np.concatenate((wp, wb, rp, rb))).all()
    ):
        raise ValueError("invalid Brinson inputs")
    tolerance = 1e-10
    if (
        abs(float(wp.sum()) - 1.0) > tolerance
        or abs(float(wb.sum()) - 1.0) > tolerance
    ):
        raise ValueError("portfolio and benchmark weights must each sum to one")
    total_benchmark = float(wb @ rb)
    allocation = (wp - wb) * (rb - total_benchmark)
    selection = wb * (rp - rb)
    interaction = (wp - wb) * (rp - rb)
    active_return = float(wp @ rp - wb @ rb)
    explained = float((allocation + selection + interaction).sum())
    residual = active_return - explained
    return Brinson(
        allocation=allocation,
        selection=selection,
        interaction=interaction,
        active_return=active_return,
        explained_return=explained,
        residual=residual,
        status="PASS" if abs(residual) <= tolerance else "REJECT",
    )


def execution_cost(
    signed_quantity: FloatArray,
    fill_price: FloatArray,
    decision_price: FloatArray,
) -> float:
    quantity = np.asarray(signed_quantity, dtype=float)
    fill = np.asarray(fill_price, dtype=float)
    decision = np.asarray(decision_price, dtype=float)
    if (
        quantity.shape != fill.shape
        or quantity.shape != decision.shape
        or not np.isfinite(quantity).all()
        or not np.isfinite(fill).all()
        or not np.isfinite(decision).all()
    ):
        raise ValueError("execution arrays must be finite and shape-consistent")
    return float(np.sum(quantity * (fill - decision)))


def reconcile(
    total: float,
    components: dict[str, float],
    *,
    nav: float,
    absolute_tolerance: float = 1e-8,
    nav_tolerance: float = 1e-8,
    gross_tolerance: float = 1e-8,
) -> ReconciliationResult:
    values = np.asarray(tuple(components.values()), dtype=float)
    if (
        not np.isfinite(total)
        or not np.isfinite(nav)
        or nav <= 0.0
        or values.size == 0
        or not np.isfinite(values).all()
        or not np.isfinite(
            np.asarray(
                (absolute_tolerance, nav_tolerance, gross_tolerance),
                dtype=float,
            )
        ).all()
        or min(absolute_tolerance, nav_tolerance, gross_tolerance) < 0.0
    ):
        raise ValueError("invalid reconciliation inputs")
    explained = float(values.sum())
    residual = float(total - explained)
    residual_to_nav = abs(residual) / nav
    gross = float(np.abs(values).sum())
    residual_to_gross = (
        0.0 if gross == 0.0 and residual == 0.0
        else float("inf") if gross == 0.0
        else abs(residual) / gross
    )
    status = (
        "PASS"
        if (
            abs(residual) <= absolute_tolerance
            and residual_to_nav <= nav_tolerance
            and residual_to_gross <= gross_tolerance
        )
        else "REJECT"
    )
    return ReconciliationResult(
        total=float(total),
        explained=explained,
        residual=residual,
        residual_to_nav=residual_to_nav,
        residual_to_gross_attribution=residual_to_gross,
        status=status,
    )


def geometric_link(
    returns: FloatArray,
    contributions: FloatArray,
) -> GeometricLinkResult:
    period_returns = np.asarray(returns, dtype=float)
    component_returns = np.asarray(contributions, dtype=float)
    if (
        period_returns.ndim != 1
        or component_returns.ndim != 2
        or component_returns.shape[0] != period_returns.size
        or not np.isfinite(period_returns).all()
        or not np.isfinite(component_returns).all()
    ):
        raise ValueError("invalid geometric-link inputs")
    period_residuals = period_returns - component_returns.sum(axis=1)
    tolerance = 1e-12
    maximum_period_residual = float(
        np.max(np.abs(period_residuals), initial=0.0)
    )
    if maximum_period_residual > tolerance:
        raise ValueError("period contributions do not reconcile to returns")
    linked = np.zeros(component_returns.shape[1])
    for period in range(len(period_returns)):
        linked += component_returns[period] * np.prod(
            1.0 + period_returns[period + 1 :]
        )
    compounded = float(np.prod(1.0 + period_returns) - 1.0)
    explained = float(linked.sum())
    residual = compounded - explained
    return GeometricLinkResult(
        linked_contributions=linked,
        compounded_return=compounded,
        explained_return=explained,
        residual=residual,
        maximum_period_residual=maximum_period_residual,
        status="PASS" if abs(residual) <= tolerance else "REJECT",
    )
