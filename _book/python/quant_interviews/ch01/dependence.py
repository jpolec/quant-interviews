"""Dependence and aggregation diagnostics for Chapter 1."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def empirical_lower_tail_dependence(
    x: ArrayLike,
    y: ArrayLike,
    quantile: float = 0.05,
) -> dict[str, float]:
    """Estimate joint downside dependence at a pre-specified quantile."""

    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    if (
        left.ndim != 1
        or right.ndim != 1
        or left.shape != right.shape
        or left.size < 20
    ):
        raise ValueError("aligned one-dimensional samples are required")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("samples must be finite")
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        raise ValueError("both samples must have positive variation")
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must lie in (0, 0.5)")

    left_threshold = float(np.quantile(left, quantile))
    right_threshold = float(np.quantile(right, quantile))
    left_breach = left <= left_threshold
    right_breach = right <= right_threshold
    joint_breach = left_breach & right_breach
    return {
        "pearson_correlation": float(np.corrcoef(left, right)[0, 1]),
        "joint_breach_probability": float(joint_breach.mean()),
        "lower_tail_dependence": float(
            joint_breach.sum() / left_breach.sum()
        ),
    }


def correlation_and_quadratic_dependence(
    x: ArrayLike,
    y: ArrayLike,
) -> dict[str, float]:
    """Return linear and quadratic diagnostics for a nonlinear extension."""

    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    if (
        left.ndim != 1
        or right.ndim != 1
        or left.shape != right.shape
        or left.size < 3
    ):
        raise ValueError("aligned one-dimensional samples are required")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("samples must be finite")
    squared_left = left * left
    if not np.all(np.isfinite(squared_left)):
        raise ValueError("squared sample must remain finite")
    if (
        np.ptp(left) == 0.0
        or np.ptp(right) == 0.0
        or np.ptp(squared_left) == 0.0
    ):
        raise ValueError("diagnostics require positive variation")
    return {
        "linear_correlation": float(np.corrcoef(left, right)[0, 1]),
        "quadratic_dependence": float(
            np.corrcoef(squared_left, right)[0, 1]
        ),
    }


def simpson_weighted_means(
    return_sums: ArrayLike,
    observation_counts: ArrayLike,
) -> dict[str, np.ndarray]:
    """Return regime and aggregate means without discarding observation weights."""

    sums = np.asarray(return_sums, dtype=float)
    counts = np.asarray(observation_counts, dtype=float)
    invalid = (
        sums.shape != counts.shape
        or sums.ndim != 2
        or sums.shape[0] < 2
        or sums.shape[1] < 2
        or np.any(counts <= 0.0)
        or not np.all(np.isfinite(sums))
        or not np.all(np.isfinite(counts))
    )
    if invalid:
        raise ValueError("aligned finite return sums and positive counts required")
    return {
        "regime_means": sums / counts,
        "aggregate_means": (
            sums.sum(axis=1) / counts.sum(axis=1)
        ),
    }


def simpson_group_rates(
    successes: ArrayLike,
    totals: ArrayLike,
) -> dict[str, np.ndarray]:
    """Compatibility wrapper for binary-rate examples."""

    result = simpson_weighted_means(successes, totals)
    return {
        "group_rates": result["regime_means"],
        "aggregate_rates": result["aggregate_means"],
    }
