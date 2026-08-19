from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PSICertificate:
    value: float
    bins_requested: int
    bins_used: int
    floor: float
    smoothing_policy: str
    reference_observations: int
    current_observations: int


@dataclass(frozen=True)
class EconomicValueCertificate:
    threshold: float
    gross_value: float
    turnover: float
    total_cost: float
    net_value: float
    terminal_exit_turnover: float
    interpretation: Literal["PER_LABEL_DESCRIPTIVE", "NONOVERLAPPING_TRADABLE"]
    status: Literal["PER_LABEL_ONLY", "PASS", "REJECT"]


def purged_train_indices(
    interval_starts: NDArray[np.datetime64],
    interval_ends: NDArray[np.datetime64],
    test_indices: NDArray[np.int64],
    *,
    candidate_train_indices: NDArray[np.int64] | None = None,
    embargo: np.timedelta64 = np.timedelta64(0, "D"),
    walk_forward: bool = True,
) -> NDArray[np.int64]:
    starts = np.asarray(interval_starts)
    ends = np.asarray(interval_ends)
    test = np.asarray(test_indices, dtype=int)
    if starts.ndim != 1 or ends.shape != starts.shape or starts.size == 0:
        raise ValueError("interval starts and ends must be aligned non-empty vectors")
    if np.any(np.isnat(starts)) or np.any(np.isnat(ends)) or np.any(starts >= ends):
        raise ValueError("label intervals must be finite and half-open with start < end")
    if (
        test.ndim != 1
        or test.size == 0
        or np.any(test < 0)
        or np.any(test >= starts.size)
    ):
        raise ValueError("test_indices must be a non-empty in-range vector")
    if np.unique(test).size != test.size or embargo < np.timedelta64(0, "ns"):
        raise ValueError("test indices must be unique and embargo non-negative")
    if candidate_train_indices is None:
        candidates = np.asarray(
            [index for index in range(starts.size) if index not in set(test)],
            dtype=int,
        )
    else:
        candidates = np.asarray(candidate_train_indices, dtype=int)
    if (
        candidates.ndim != 1
        or np.unique(candidates).size != candidates.size
        or np.any(candidates < 0)
        or np.any(candidates >= starts.size)
    ):
        raise ValueError("candidate_train_indices must be unique and in range")
    if np.intersect1d(candidates, test).size:
        raise ValueError("candidate train and test indices must be disjoint")

    test_starts = starts[test]
    protected_ends = ends[test] + embargo
    first_test_start = np.min(test_starts)
    train: list[int] = []
    for index in candidates:
        start = starts[index]
        end = ends[index]
        if walk_forward and start >= first_test_start:
            continue
        overlaps_protected_union = bool(
            np.any((start < protected_ends) & (end > test_starts))
        )
        if not overlaps_protected_union:
            train.append(int(index))
    return np.asarray(train, dtype=int)


def brier_score(
    probabilities: NDArray[np.float64],
    outcomes: NDArray[np.int64],
) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes)
    if p.ndim != 1 or y.shape != p.shape or p.size == 0:
        raise ValueError("probabilities and outcomes must be aligned non-empty vectors")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("outcomes must be binary")
    return float(np.mean((p - y) ** 2))


def log_loss(
    probabilities: NDArray[np.float64],
    outcomes: NDArray[np.int64],
    *,
    epsilon: float,
) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes)
    if p.ndim != 1 or y.shape != p.shape or p.size == 0:
        raise ValueError("probabilities and outcomes must align")
    if (
        not np.all(np.isfinite(p))
        or np.any((p < 0.0) | (p > 1.0))
        or np.any((y != 0) & (y != 1))
    ):
        raise ValueError("probabilities must be in [0,1] and outcomes binary")
    if not isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5)")
    clipped = np.clip(p, epsilon, 1.0 - epsilon)
    result = -float(np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))
    if not isfinite(result):
        raise FloatingPointError("log loss is non-finite")
    return result


def population_stability_index(
    reference: NDArray[np.float64],
    current: NDArray[np.float64],
    bins: int = 10,
    floor: float = 1e-6,
) -> PSICertificate:
    reference_values = np.asarray(reference, dtype=float)
    current_values = np.asarray(current, dtype=float)
    if (
        reference_values.ndim != 1
        or current_values.ndim != 1
        or reference_values.size < bins
        or current_values.size < 2
        or bins < 2
        or not isfinite(floor)
        or floor <= 0.0
    ):
        raise ValueError("PSI requires finite non-empty samples and at least two bins")
    if not np.all(np.isfinite(reference_values)) or not np.all(
        np.isfinite(current_values)
    ):
        raise ValueError("PSI samples must be finite")
    quantiles = np.quantile(reference_values, np.linspace(0.0, 1.0, bins + 1))
    interior = np.unique(quantiles[1:-1])
    if interior.size == 0:
        raise ValueError("reference sample has no usable quantile separation")
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)
    reference_share = np.maximum(reference_counts / reference_values.size, floor)
    current_share = np.maximum(current_counts / current_values.size, floor)
    reference_share /= np.sum(reference_share)
    current_share /= np.sum(current_share)
    value = float(
        np.sum(
            (current_share - reference_share)
            * np.log(current_share / reference_share)
        )
    )
    if not isfinite(value) or value < 0.0:
        raise FloatingPointError("PSI is invalid")
    return PSICertificate(
        value=value,
        bins_requested=bins,
        bins_used=int(len(edges) - 1),
        floor=floor,
        smoothing_policy="floor_then_renormalize",
        reference_observations=int(reference_values.size),
        current_observations=int(current_values.size),
    )


def _economic_value(
    probabilities: NDArray[np.float64],
    returns: NDArray[np.float64],
    *,
    threshold: float,
    cost_per_unit_turnover: float,
    long_only: bool,
    interpretation: Literal["PER_LABEL_DESCRIPTIVE", "NONOVERLAPPING_TRADABLE"],
) -> EconomicValueCertificate:
    p = np.asarray(probabilities, dtype=float)
    values = np.asarray(returns, dtype=float)
    if p.ndim != 1 or values.shape != p.shape or p.size == 0:
        raise ValueError("probabilities and returns must align")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(values)):
        raise ValueError("economic-value inputs must be finite")
    if (
        np.any((p < 0.0) | (p > 1.0))
        or not isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("probabilities and threshold must lie in [0, 1]")
    if (
        not isfinite(cost_per_unit_turnover)
        or cost_per_unit_turnover < 0.0
    ):
        raise ValueError("turnover cost must be finite and non-negative")
    position = (p >= threshold).astype(float) if long_only else np.sign(p - threshold)
    trades = np.diff(np.concatenate(([0.0], position)))
    terminal_exit = abs(float(position[-1]))
    turnover = float(np.sum(np.abs(trades)) + terminal_exit)
    gross = float(np.sum(position * values))
    total_cost = cost_per_unit_turnover * turnover
    net = gross - total_cost
    if not all(isfinite(value) for value in (turnover, gross, total_cost, net)):
        raise FloatingPointError("economic-value ledger is non-finite")
    if interpretation == "PER_LABEL_DESCRIPTIVE":
        status: Literal["PER_LABEL_ONLY", "PASS", "REJECT"] = "PER_LABEL_ONLY"
    else:
        status = "PASS" if net > 0.0 else "REJECT"
    return EconomicValueCertificate(
        threshold=threshold,
        gross_value=gross,
        turnover=turnover,
        total_cost=total_cost,
        net_value=net,
        terminal_exit_turnover=terminal_exit,
        interpretation=interpretation,
        status=status,
    )


def net_value(
    probabilities: NDArray[np.float64],
    forward_returns: NDArray[np.float64],
    *,
    threshold: float,
    cost_per_unit_turnover: float,
    long_only: bool = False,
) -> EconomicValueCertificate:
    """Return descriptive per-label value; never label it deployment P&L."""

    return _economic_value(
        probabilities,
        forward_returns,
        threshold=threshold,
        cost_per_unit_turnover=cost_per_unit_turnover,
        long_only=long_only,
        interpretation="PER_LABEL_DESCRIPTIVE",
    )


def nonoverlapping_net_value(
    probabilities: NDArray[np.float64],
    realized_horizon_returns: NDArray[np.float64],
    interval_starts: NDArray[np.datetime64],
    interval_ends: NDArray[np.datetime64],
    *,
    threshold: float,
    cost_per_unit_turnover: float,
    long_only: bool = False,
) -> EconomicValueCertificate:
    starts = np.asarray(interval_starts)
    ends = np.asarray(interval_ends)
    if (
        starts.ndim != 1
        or ends.shape != starts.shape
        or starts.size != len(probabilities)
        or np.any(np.isnat(starts))
        or np.any(np.isnat(ends))
        or np.any(starts >= ends)
    ):
        raise ValueError("evaluation intervals must align and be finite half-open")
    order = np.argsort(starts, kind="mergesort")
    if not np.array_equal(order, np.arange(starts.size)):
        raise ValueError("evaluation intervals must be ordered")
    if np.any(starts[1:] < ends[:-1]):
        raise ValueError("tradable evaluation requires non-overlapping horizons")
    return _economic_value(
        probabilities,
        realized_horizon_returns,
        threshold=threshold,
        cost_per_unit_turnover=cost_per_unit_turnover,
        long_only=long_only,
        interpretation="NONOVERLAPPING_TRADABLE",
    )
