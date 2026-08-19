from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, kstest


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HawkesParameters:
    baseline: float
    excitation: float
    decay: float

    def __post_init__(self) -> None:
        values = np.asarray([self.baseline, self.excitation, self.decay], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Hawkes parameters must be finite")
        if self.baseline <= 0.0 or self.excitation < 0.0 or self.decay <= 0.0:
            raise ValueError("invalid Hawkes parameters")
        if self.excitation / self.decay >= 1.0:
            raise ValueError("branching ratio must be below one")

    @property
    def branching_ratio(self) -> float:
        return self.excitation / self.decay


@dataclass(frozen=True)
class HawkesSample:
    events: FloatArray
    history_before_zero: FloatArray
    horizon: float
    burn_in: float
    seed: int


def simulate_hawkes(
    params: HawkesParameters,
    horizon: float,
    seed: int = 13,
    burn_in: float = 20.0,
) -> HawkesSample:
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if not np.isfinite(burn_in) or burn_in < 0.0:
        raise ValueError("burn_in must be finite and non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    t = -burn_in
    events: list[float] = []
    while t < horizon:
        upper = params.baseline + sum(
            params.excitation * np.exp(-params.decay * (t - s)) for s in events
        )
        t += rng.exponential(1.0 / upper)
        if t >= horizon:
            break
        actual = params.baseline + sum(
            params.excitation * np.exp(-params.decay * (t - s)) for s in events
        )
        if rng.random() <= actual / upper:
            events.append(t)
    return HawkesSample(
        events=np.asarray([event for event in events if event > 0.0], dtype=float),
        history_before_zero=np.asarray(
            [event for event in events if event <= 0.0],
            dtype=float,
        ),
        horizon=horizon,
        burn_in=burn_in,
        seed=seed,
    )


def _validated_events(
    events: FloatArray,
    history_before_zero: FloatArray,
    horizon: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    retained = np.asarray(events, dtype=float)
    history = np.asarray(history_before_zero, dtype=float)
    if retained.ndim != 1 or history.ndim != 1:
        raise ValueError("Hawkes events and prehistory must be one-dimensional")
    if not np.isfinite(retained).all() or not np.isfinite(history).all():
        raise ValueError("Hawkes events and prehistory must be finite")
    if (
        np.any(np.diff(retained) <= 0.0)
        or np.any(retained <= 0.0)
        or np.any(np.diff(history) <= 0.0)
        or np.any(history > 0.0)
    ):
        raise ValueError("Hawkes event times must be ordered around time zero")
    if horizon is not None and np.any(retained >= horizon):
        raise ValueError("retained Hawkes events must lie inside the horizon")
    return retained, history


def log_likelihood(
    events: FloatArray,
    params: HawkesParameters,
    horizon: float,
    *,
    history_before_zero: FloatArray | None = None,
) -> float:
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    history_input = (
        np.empty(0, dtype=float)
        if history_before_zero is None
        else history_before_zero
    )
    t, history = _validated_events(events, history_input, horizon)
    log_intensity = 0.0
    for i, ti in enumerate(t):
        prior = np.r_[history, t[:i]]
        intensity = params.baseline + np.sum(
            params.excitation * np.exp(-params.decay * (ti - prior))
        )
        log_intensity += log(float(intensity))
    compensator = params.baseline * horizon + np.sum(
        params.excitation / params.decay * (1.0 - np.exp(-params.decay * (horizon - t)))
    )
    if history.size:
        compensator += np.sum(
            params.excitation
            / params.decay
            * (
                np.exp(params.decay * history)
                - np.exp(-params.decay * (horizon - history))
            )
        )
    return float(log_intensity - compensator)


@dataclass(frozen=True)
class HawkesDiagnostics:
    branching_ratio: float
    event_count: int
    log_likelihood: float
    rescaled_mean: float
    rescaled_variance: float
    exponential_ks_pvalue: float
    independence_pvalue: float
    minimum_event_count: int
    evidence_status: str
    status: str


def time_rescaled_increments(
    events: FloatArray,
    params: HawkesParameters,
    *,
    history_before_zero: FloatArray | None = None,
) -> FloatArray:
    history_input = (
        np.empty(0, dtype=float)
        if history_before_zero is None
        else history_before_zero
    )
    t, prehistory = _validated_events(events, history_input)
    previous = 0.0
    increments = np.empty(len(t), dtype=float)
    for index, event in enumerate(t):
        baseline = params.baseline * (event - previous)
        history = np.r_[prehistory, t[:index]]
        if history.size:
            excitation = np.sum(
                params.excitation
                / params.decay
                * (
                    np.exp(-params.decay * (previous - history))
                    - np.exp(-params.decay * (event - history))
                )
            )
        else:
            excitation = 0.0
        increments[index] = baseline + excitation
        previous = event
    return increments


def _ljung_box_pvalue(values: FloatArray, maximum_lag: int) -> float:
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(float).eps:
        return 0.0
    sample_size = len(values)
    statistic = 0.0
    for lag in range(1, maximum_lag + 1):
        autocorrelation = float(
            np.dot(centered[lag:], centered[:-lag]) / denominator
        )
        statistic += autocorrelation**2 / (sample_size - lag)
    statistic *= sample_size * (sample_size + 2)
    return float(chi2.sf(statistic, maximum_lag))


def hawkes_diagnostics(
    events: FloatArray,
    params: HawkesParameters,
    horizon: float,
    *,
    history_before_zero: FloatArray | None = None,
    minimum_event_count: int = 30,
    minimum_ks_pvalue: float = 0.05,
    minimum_independence_pvalue: float = 0.05,
    maximum_absolute_mean_error: float = 0.25,
    maximum_absolute_variance_error: float = 0.50,
) -> HawkesDiagnostics:
    controls = np.asarray(
        [
            minimum_ks_pvalue,
            minimum_independence_pvalue,
            maximum_absolute_mean_error,
            maximum_absolute_variance_error,
        ],
        dtype=float,
    )
    if (
        isinstance(minimum_event_count, bool)
        or not isinstance(minimum_event_count, int)
        or minimum_event_count < 10
        or not np.isfinite(controls).all()
        or not 0.0 < minimum_ks_pvalue < 1.0
        or not 0.0 < minimum_independence_pvalue < 1.0
        or maximum_absolute_mean_error < 0.0
        or maximum_absolute_variance_error < 0.0
    ):
        raise ValueError("invalid Hawkes diagnostic policy")
    history = (
        np.empty(0, dtype=float)
        if history_before_zero is None
        else np.asarray(history_before_zero, dtype=float)
    )
    residuals = time_rescaled_increments(
        events,
        params,
        history_before_zero=history,
    )
    likelihood = log_likelihood(
        events,
        params,
        horizon,
        history_before_zero=history,
    )
    residual_mean = float(np.mean(residuals)) if len(residuals) else float("nan")
    residual_variance = (
        float(np.var(residuals, ddof=1)) if len(residuals) > 1 else float("nan")
    )
    if len(residuals) < minimum_event_count:
        return HawkesDiagnostics(
            params.branching_ratio,
            len(residuals),
            likelihood,
            residual_mean,
            residual_variance,
            float("nan"),
            float("nan"),
            minimum_event_count,
            "INSUFFICIENT_EVIDENCE",
            "CONDITIONAL_PASS",
        )
    pvalue = float(kstest(residuals, "expon").pvalue)
    lags = min(5, max(1, len(residuals) // 10))
    independence_pvalue = _ljung_box_pvalue(residuals, lags)
    passes = (
        pvalue >= minimum_ks_pvalue
        and independence_pvalue >= minimum_independence_pvalue
        and abs(residual_mean - 1.0) <= maximum_absolute_mean_error
        and abs(residual_variance - 1.0) <= maximum_absolute_variance_error
    )
    return HawkesDiagnostics(
        params.branching_ratio,
        len(residuals),
        likelihood,
        residual_mean,
        residual_variance,
        pvalue,
        independence_pvalue,
        minimum_event_count,
        "SUFFICIENT_EVIDENCE",
        "PASS" if passes else "REJECT",
    )
