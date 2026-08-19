from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd

from .validation import NestedFold


@dataclass(frozen=True)
class StrategyEvaluation:
    threshold: float
    gross_pnl: float
    cost: float
    net_pnl: float
    turnover: float
    entry_turnover: float
    terminal_exit_turnover: float
    volatility: float
    long_run_variance: float
    utility: float
    position: NDArray[np.float64]
    period_pnl: NDArray[np.float64]
    gross_period_pnl: NDArray[np.float64]
    period_cost: NDArray[np.float64]
    period_turnover: NDArray[np.float64]


@dataclass(frozen=True)
class CapacityPoint:
    assets_under_management: float
    gross_pnl: float
    linear_cost: float
    impact_cost: float
    net_pnl: float


@dataclass(frozen=True)
class FoldEconomicEvaluation:
    outer_fold: int
    selected_threshold: float
    inner_selection_observations: int
    outer_test_observations: int
    evaluation: StrategyEvaluation


@dataclass(frozen=True)
class NestedEconomicEvaluation:
    folds: tuple[FoldEconomicEvaluation, ...]
    gross_pnl: float
    linear_cost: float
    net_pnl: float
    turnover: float
    utility: float
    attribution_residual: float
    capacity_curve: tuple[CapacityPoint, ...]


def _probabilities(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 1
        or result.size < 2
        or not np.all(np.isfinite(result))
        or np.any((result < 0.0) | (result > 1.0))
    ):
        raise ValueError("probabilities must be a finite vector in [0, 1].")
    return result


def positions_from_probability(
    probabilities: ArrayLike,
    threshold: float,
) -> NDArray[np.float64]:
    probability = _probabilities(probabilities)
    if not np.isfinite(threshold) or not 0.5 <= threshold < 1.0:
        raise ValueError("threshold must lie in [0.5, 1).")
    long_signal = probability >= threshold
    short_signal = probability <= 1.0 - threshold
    return long_signal.astype(float) - short_signal.astype(float)


def newey_west_long_run_variance(
    values: ArrayLike,
    *,
    max_lag: int | None = None,
) -> float:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < 2 or not np.all(np.isfinite(result)):
        raise ValueError("P&L must be a finite vector with at least two periods.")
    centered = result - np.mean(result)
    observations = centered.size
    lag = (
        int(np.floor(4.0 * (observations / 100.0) ** (2.0 / 9.0)))
        if max_lag is None
        else max_lag
    )
    if lag < 0 or lag >= observations:
        raise ValueError("max_lag must lie between zero and n-1.")
    variance = float(np.dot(centered, centered) / observations)
    for index in range(1, lag + 1):
        covariance = float(
            np.dot(centered[index:], centered[:-index]) / observations
        )
        variance += 2.0 * (1.0 - index / (lag + 1.0)) * covariance
    return max(variance, 0.0)


def evaluate_threshold(
    probabilities: ArrayLike,
    returns: ArrayLike,
    *,
    threshold: float,
    cost_per_unit_turnover: float,
    risk_aversion: float,
    long_run_variance_lag: int | None = None,
) -> StrategyEvaluation:
    probability = _probabilities(probabilities)
    return_array = np.asarray(returns, dtype=float)
    if (
        return_array.ndim != 1
        or probability.shape != return_array.shape
        or not np.all(np.isfinite(return_array))
    ):
        raise ValueError("probabilities and finite returns must align.")
    if (
        not np.isfinite(cost_per_unit_turnover)
        or cost_per_unit_turnover < 0.0
        or not np.isfinite(risk_aversion)
        or risk_aversion < 0.0
    ):
        raise ValueError("Cost and risk aversion must be finite and nonnegative.")

    position = positions_from_probability(probability, threshold)
    trades = np.diff(np.r_[0.0, position])
    period_turnover = np.abs(trades)
    terminal_exit_turnover = abs(float(position[-1]))
    turnover = float(period_turnover.sum() + terminal_exit_turnover)
    gross_period = position * return_array
    period_cost = cost_per_unit_turnover * period_turnover
    period_pnl = gross_period - period_cost
    period_pnl[-1] -= cost_per_unit_turnover * terminal_exit_turnover
    gross = float(gross_period.sum())
    cost = float(cost_per_unit_turnover * turnover)
    net = float(period_pnl.sum())
    long_run_variance = newey_west_long_run_variance(
        period_pnl,
        max_lag=long_run_variance_lag,
    )
    horizon_variance = period_pnl.size * long_run_variance
    utility = net - 0.5 * risk_aversion * horizon_variance
    return StrategyEvaluation(
        threshold=float(threshold),
        gross_pnl=gross,
        cost=cost,
        net_pnl=net,
        turnover=turnover,
        entry_turnover=float(period_turnover[0]),
        terminal_exit_turnover=terminal_exit_turnover,
        volatility=float(np.std(period_pnl, ddof=1)),
        long_run_variance=long_run_variance,
        utility=float(utility),
        position=position,
        period_pnl=period_pnl,
        gross_period_pnl=gross_period,
        period_cost=np.r_[
            period_cost[:-1],
            period_cost[-1] + cost_per_unit_turnover * terminal_exit_turnover,
        ],
        period_turnover=np.r_[
            period_turnover[:-1],
            period_turnover[-1] + terminal_exit_turnover,
        ],
    )


def select_threshold(
    probabilities: ArrayLike,
    returns: ArrayLike,
    thresholds: ArrayLike,
    *,
    cost_per_unit_turnover: float,
    risk_aversion: float,
    selection_context: Literal["inner_validation"],
) -> StrategyEvaluation:
    if selection_context != "inner_validation":
        raise ValueError("Threshold selection is permitted only on inner_validation data.")
    threshold_array = np.asarray(thresholds, dtype=float)
    if (
        threshold_array.ndim != 1
        or threshold_array.size == 0
        or not np.all(np.isfinite(threshold_array))
        or np.unique(threshold_array).size != threshold_array.size
    ):
        raise ValueError("thresholds must be a finite, unique nonempty vector.")
    results = [
        evaluate_threshold(
            probabilities,
            returns,
            threshold=float(threshold),
            cost_per_unit_turnover=cost_per_unit_turnover,
            risk_aversion=risk_aversion,
        )
        for threshold in threshold_array
    ]
    return max(
        results,
        key=lambda result: (
            result.utility,
            -result.turnover,
            result.threshold,
        ),
    )


def nested_economic_evaluation(
    probabilities: ArrayLike,
    returns: ArrayLike,
    *,
    timestamps: ArrayLike,
    return_known_timestamps: ArrayLike,
    nested_folds: Sequence[NestedFold],
    thresholds: ArrayLike,
    cost_per_unit_turnover: float,
    risk_aversion: float,
    assets_under_management: ArrayLike = (1.0,),
    impact_coefficient: float = 0.0,
) -> NestedEconomicEvaluation:
    """Select only on inner validation rows, then score each outer test once."""

    probability = _probabilities(probabilities)
    return_array = np.asarray(returns, dtype=float)
    if return_array.shape != probability.shape or not np.all(np.isfinite(return_array)):
        raise ValueError("probabilities and returns must be aligned and finite.")
    score_times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True, errors="raise"))
    known_times = pd.DatetimeIndex(
        pd.to_datetime(return_known_timestamps, utc=True, errors="raise")
    )
    if len(score_times) != probability.size or len(known_times) != probability.size:
        raise ValueError("Economic timestamps must align with observations.")
    if score_times.has_duplicates or not score_times.is_monotonic_increasing:
        raise ValueError("Economic timestamps must be unique and increasing.")
    aum = np.asarray(assets_under_management, dtype=float)
    if (
        aum.ndim != 1
        or aum.size == 0
        or not np.all(np.isfinite(aum))
        or np.any(aum <= 0.0)
        or np.unique(aum).size != aum.size
        or not np.isfinite(impact_coefficient)
        or impact_coefficient < 0.0
    ):
        raise ValueError("Capacity scenarios and impact coefficient are invalid.")
    fold_results: list[FoldEconomicEvaluation] = []
    for fold_number, nested in enumerate(nested_folds):
        outer_test = np.asarray(nested.outer.test_index, dtype=int)
        if outer_test.size == 0:
            raise ValueError("Every outer fold requires test observations.")
        inner_rows = np.asarray(
            sorted(
                {
                    int(index)
                    for inner in nested.inner
                    for index in inner.validation_index
                }
            ),
            dtype=int,
        )
        if inner_rows.size == 0 or np.intersect1d(inner_rows, outer_test).size:
            raise ValueError("Inner selection rows must exist and exclude outer test.")
        if score_times[inner_rows].max() >= score_times[outer_test].min():
            raise ValueError("Inner selection must precede the outer test.")
        if known_times[inner_rows].max() > score_times[outer_test].min():
            raise ValueError("Inner returns were not known before the outer test.")
        selected = select_threshold(
            probability[inner_rows],
            return_array[inner_rows],
            thresholds,
            cost_per_unit_turnover=cost_per_unit_turnover,
            risk_aversion=risk_aversion,
            selection_context="inner_validation",
        )
        evaluated = evaluate_threshold(
            probability[outer_test],
            return_array[outer_test],
            threshold=selected.threshold,
            cost_per_unit_turnover=cost_per_unit_turnover,
            risk_aversion=risk_aversion,
        )
        fold_results.append(
            FoldEconomicEvaluation(
                outer_fold=fold_number,
                selected_threshold=selected.threshold,
                inner_selection_observations=int(inner_rows.size),
                outer_test_observations=int(outer_test.size),
                evaluation=evaluated,
            )
        )
    if not fold_results:
        raise ValueError("At least one nested fold is required.")
    gross = float(sum(result.evaluation.gross_pnl for result in fold_results))
    linear_cost = float(sum(result.evaluation.cost for result in fold_results))
    net = float(sum(result.evaluation.net_pnl for result in fold_results))
    turnover = float(sum(result.evaluation.turnover for result in fold_results))
    utility = float(sum(result.evaluation.utility for result in fold_results))
    capacity = tuple(
        CapacityPoint(
            assets_under_management=float(level),
            gross_pnl=float(level * gross),
            linear_cost=float(level * linear_cost),
            impact_cost=float(impact_coefficient * turnover * level**1.5),
            net_pnl=float(
                level * gross
                - level * linear_cost
                - impact_coefficient * turnover * level**1.5
            ),
        )
        for level in sorted(aum)
    )
    return NestedEconomicEvaluation(
        folds=tuple(fold_results),
        gross_pnl=gross,
        linear_cost=linear_cost,
        net_pnl=net,
        turnover=turnover,
        utility=utility,
        attribution_residual=float(net - (gross - linear_cost)),
        capacity_curve=capacity,
    )
