from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm

FloatArray = NDArray[np.float64]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _utc(value: str, name: str) -> datetime:
    if not value or not value.endswith("Z"):
        raise ValueError(f"{name} must be a UTC ISO-8601 timestamp")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} must be a UTC ISO-8601 timestamp") from error


@dataclass(frozen=True)
class VaRForecastRecord:
    forecast_id: str
    forecast_origin: str
    target_start: str
    target_end: str
    knowledge_time: str
    model_version: str
    position_snapshot_sha256: str
    forecast_loss: float
    realized_loss: float
    realized_loss_knowledge_time: str


@dataclass(frozen=True)
class BacktestCertificate:
    statistics: dict[str, float]
    used_count: int
    excluded_count: int
    model_versions: tuple[str, ...]
    position_snapshot_hashes: tuple[str, ...]
    alignment_status: str
    status: str


def bernoulli_var_counterexample(
    alpha: float = 0.95,
    p_loss: float = 0.04,
) -> dict[str, float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    lower = 1.0 - math.sqrt(alpha)
    upper = 1.0 - alpha
    if not lower < p_loss <= upper:
        raise ValueError(
            "require 1-sqrt(alpha) < p_loss <= 1-alpha for the counterexample"
        )
    combined_positive = 1.0 - (1.0 - p_loss) ** 2
    return {
        "standalone_var": 0.0,
        "combined_var": 1.0,
        "combined_positive_probability": combined_positive,
    }


def horizon_variance(gamma: FloatArray, horizon: int) -> float:
    autocovariance = np.asarray(gamma, dtype=float)
    if (
        autocovariance.ndim != 1
        or not np.isfinite(autocovariance).all()
        or horizon < 1
        or autocovariance.size < horizon
    ):
        raise ValueError("insufficient or invalid autocovariances")
    if autocovariance[0] < 0.0:
        raise ValueError("lag-zero autocovariance must be nonnegative")
    toeplitz = np.fromfunction(
        lambda row, column: autocovariance[
            np.abs(row - column).astype(int)
        ],
        (horizon, horizon),
        dtype=int,
    )
    scale = max(float(np.linalg.norm(toeplitz, ord=2)), 1.0)
    if float(np.linalg.eigvalsh(toeplitz).min()) < -1e-12 * scale:
        raise ValueError("autocovariances do not define a PSD horizon matrix")
    value = float(
        horizon * autocovariance[0]
        + 2.0
        * sum(
            (horizon - lag) * autocovariance[lag]
            for lag in range(1, horizon)
        )
    )
    if value < -1e-12 * scale:
        raise ValueError("computed horizon variance is materially negative")
    return max(value, 0.0)


def iid_normal_horizon_var(
    daily_mean_loss: float,
    daily_volatility: float,
    alpha: float,
    horizon: int,
) -> float:
    if (
        not np.isfinite(daily_mean_loss)
        or not np.isfinite(daily_volatility)
        or daily_volatility < 0.0
        or not 0.0 < alpha < 1.0
        or horizon < 1
    ):
        raise ValueError("invalid horizon-VaR inputs")
    return float(
        horizon * daily_mean_loss
        + np.sqrt(horizon) * daily_volatility * norm.ppf(alpha)
    )


def _xlogy(count: int, ratio: float) -> float:
    return 0.0 if count == 0 else count * math.log(ratio)


def exceedance_summary(
    losses: FloatArray,
    forecasts: FloatArray,
    alpha: float = 0.95,
) -> dict[str, float]:
    values = np.asarray(losses, dtype=float)
    limits = np.asarray(forecasts, dtype=float)
    if (
        values.shape != limits.shape
        or values.ndim != 1
        or values.size < 3
        or not np.isfinite(values).all()
        or not np.isfinite(limits).all()
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("invalid exceedance inputs")
    indicators = (values > limits).astype(int)
    count = int(indicators.sum())
    sample_size = indicators.size
    expected_rate = 1.0 - alpha
    observed_rate = count / sample_size
    likelihood_null = _xlogy(count, expected_rate) + _xlogy(
        sample_size - count,
        1.0 - expected_rate,
    )
    if 0 < count < sample_size:
        likelihood_fit = _xlogy(count, observed_rate) + _xlogy(
            sample_size - count,
            1.0 - observed_rate,
        )
        kupiec = 2.0 * (likelihood_fit - likelihood_null)
    else:
        kupiec = -2.0 * likelihood_null

    transitions = np.zeros((2, 2), dtype=int)
    for previous, current in zip(indicators[:-1], indicators[1:]):
        transitions[previous, current] += 1
    row_totals = transitions.sum(axis=1)
    probabilities = np.divide(
        transitions[:, 1],
        row_totals,
        out=np.zeros(2),
        where=row_totals > 0,
    )
    total_transition_rate = transitions[:, 1].sum() / transitions.sum()
    independent_log_likelihood = sum(
        _xlogy(int(transitions[row, 1]), float(probabilities[row]))
        + _xlogy(
            int(transitions[row, 0]),
            float(1.0 - probabilities[row]),
        )
        for row in (0, 1)
        if row_totals[row] > 0
    )
    pooled_log_likelihood = _xlogy(
        int(transitions[:, 1].sum()),
        float(total_transition_rate),
    ) + _xlogy(
        int(transitions[:, 0].sum()),
        float(1.0 - total_transition_rate),
    )
    independence = max(
        0.0,
        2.0 * (independent_log_likelihood - pooled_log_likelihood),
    )
    return {
        "rate": float(observed_rate),
        "expected_rate": expected_rate,
        "count": float(count),
        "kupiec_statistic": float(kupiec),
        "kupiec_pvalue": float(chi2.sf(kupiec, 1)),
        "independence_statistic": float(independence),
        "independence_pvalue": float(chi2.sf(independence, 1)),
        "n00": float(transitions[0, 0]),
        "n01": float(transitions[0, 1]),
        "n10": float(transitions[1, 0]),
        "n11": float(transitions[1, 1]),
    }


def point_in_time_exceedance_summary(
    records: tuple[VaRForecastRecord, ...],
    *,
    evaluation_time: str,
    alpha: float = 0.95,
) -> BacktestCertificate:
    """Backtest immutable forecasts only after a deterministic PIT alignment."""
    cutoff = _utc(evaluation_time, "evaluation_time")
    if len(records) < 3:
        raise ValueError("at least three forecast records are required")
    used: list[VaRForecastRecord] = []
    excluded = 0
    identifiers: set[str] = set()
    target_intervals: list[tuple[datetime, datetime]] = []
    for record in records:
        if not record.forecast_id or record.forecast_id in identifiers:
            raise ValueError("forecast IDs must be nonempty and unique")
        identifiers.add(record.forecast_id)
        origin = _utc(record.forecast_origin, "forecast_origin")
        start = _utc(record.target_start, "target_start")
        end = _utc(record.target_end, "target_end")
        knowledge = _utc(record.knowledge_time, "knowledge_time")
        loss_knowledge = _utc(
            record.realized_loss_knowledge_time,
            "realized_loss_knowledge_time",
        )
        if knowledge > origin or not origin < start <= end:
            raise ValueError("forecast violates point-in-time target ordering")
        if not record.model_version:
            raise ValueError("model_version is required")
        if not _SHA256.fullmatch(record.position_snapshot_sha256):
            raise ValueError("position snapshot must be a lowercase SHA-256")
        if not np.isfinite(record.forecast_loss) or not np.isfinite(
            record.realized_loss
        ):
            raise ValueError("forecast and realized loss must be finite")
        if loss_knowledge > cutoff:
            excluded += 1
            continue
        target_intervals.append((start, end))
        used.append(record)
    if len(used) < 3:
        raise ValueError("fewer than three PIT-aligned observations remain")
    ordered = sorted(target_intervals)
    if any(current[0] <= previous[1] for previous, current in zip(ordered, ordered[1:])):
        raise ValueError("overlapping target intervals require an explicit policy")
    statistics = exceedance_summary(
        np.asarray([record.realized_loss for record in used], dtype=float),
        np.asarray([record.forecast_loss for record in used], dtype=float),
        alpha,
    )
    return BacktestCertificate(
        statistics=statistics,
        used_count=len(used),
        excluded_count=excluded,
        model_versions=tuple(sorted({record.model_version for record in used})),
        position_snapshot_hashes=tuple(
            sorted({record.position_snapshot_sha256 for record in used})
        ),
        alignment_status="PASS",
        status="PASS",
    )
