from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ItoLogCorrectionAudit:
    theoretical_log_mean: float
    empirical_log_mean: float
    theoretical_log_variance: float
    empirical_log_variance: float
    wrong_uncorrected_mean: float
    mean_standard_error: float
    status: str


@dataclass(frozen=True)
class ExactTransitionAudit:
    one_step_mean_error: float
    one_step_variance_error: float
    semigroup_mean_error: float
    semigroup_variance_error: float
    minimum_terminal_value: float
    status: str


def gbm_exact_paths(
    *,
    spot: float,
    drift: float,
    volatility: float,
    horizon: float,
    steps: int,
    paths: int,
    seed: int = 13,
) -> NDArray[np.float64]:
    values = np.array([spot, drift, volatility, horizon], dtype=float)
    if (
        not np.all(np.isfinite(values))
        or spot <= 0.0
        or volatility < 0.0
        or horizon <= 0.0
        or steps <= 0
        or paths < 2
    ):
        raise ValueError("invalid GBM parameters")
    rng = np.random.default_rng(seed)
    dt = horizon / steps
    d_w = np.sqrt(dt) * rng.standard_normal((paths, steps))
    log_increments = (drift - 0.5 * volatility**2) * dt + volatility * d_w
    return spot * np.exp(np.c_[np.zeros(paths), np.cumsum(log_increments, axis=1)])


def gbm_terminal_moments(
    *,
    spot: float,
    drift: float,
    volatility: float,
    horizon: float,
) -> tuple[float, float]:
    values = np.array([spot, drift, volatility, horizon], dtype=float)
    if (
        not np.all(np.isfinite(values))
        or spot <= 0.0
        or volatility < 0.0
        or horizon < 0.0
    ):
        raise ValueError("invalid GBM parameters")
    mean = spot * np.exp(drift * horizon)
    variance = (
        spot**2
        * np.exp(2.0 * drift * horizon)
        * (np.exp(volatility**2 * horizon) - 1.0)
    )
    return float(mean), float(variance)


def ito_log_correction_audit(
    *,
    spot: float = 100.0,
    drift: float = 0.05,
    volatility: float = 0.2,
    horizon: float = 1.0,
    paths: int = 80_000,
    seed: int = 13,
) -> ItoLogCorrectionAudit:
    """Verify the Ito correction in the GBM log transition."""

    samples = gbm_exact_paths(
        spot=spot,
        drift=drift,
        volatility=volatility,
        horizon=horizon,
        steps=1,
        paths=paths,
        seed=seed,
    )[:, -1]
    log_returns = np.log(samples / spot)
    theoretical_mean = (drift - 0.5 * volatility**2) * horizon
    theoretical_variance = volatility**2 * horizon
    empirical_mean = float(np.mean(log_returns))
    empirical_variance = float(np.var(log_returns, ddof=1))
    standard_error = float(np.sqrt(theoretical_variance / paths))
    status = (
        "PASS"
        if abs(empirical_mean - theoretical_mean) <= 5.0 * standard_error
        and abs(empirical_variance - theoretical_variance)
        <= 0.03 * max(theoretical_variance, np.finfo(float).tiny)
        else "REJECT"
    )
    return ItoLogCorrectionAudit(
        theoretical_log_mean=theoretical_mean,
        empirical_log_mean=empirical_mean,
        theoretical_log_variance=theoretical_variance,
        empirical_log_variance=empirical_variance,
        wrong_uncorrected_mean=drift * horizon,
        mean_standard_error=standard_error,
        status=status,
    )


def exact_transition_audit(
    *,
    spot: float = 100.0,
    drift: float = 0.05,
    volatility: float = 0.2,
    horizon: float = 1.0,
    paths: int = 80_000,
    seed: int = 17,
) -> ExactTransitionAudit:
    """Check exact moments, positivity and the GBM semigroup."""

    one_step = gbm_exact_paths(
        spot=spot,
        drift=drift,
        volatility=volatility,
        horizon=horizon,
        steps=1,
        paths=paths,
        seed=seed,
    )[:, -1]
    two_half_steps = gbm_exact_paths(
        spot=spot,
        drift=drift,
        volatility=volatility,
        horizon=horizon,
        steps=2,
        paths=paths,
        seed=seed + 1,
    )[:, -1]
    mean, variance = gbm_terminal_moments(
        spot=spot,
        drift=drift,
        volatility=volatility,
        horizon=horizon,
    )
    one_mean_error = abs(float(np.mean(one_step)) - mean)
    one_variance_error = abs(float(np.var(one_step, ddof=1)) - variance)
    semigroup_mean_error = abs(
        float(np.mean(one_step)) - float(np.mean(two_half_steps))
    )
    semigroup_variance_error = abs(
        float(np.var(one_step, ddof=1))
        - float(np.var(two_half_steps, ddof=1))
    )
    mean_se = np.sqrt(variance / paths)
    variance_scale = max(variance, np.finfo(float).tiny)
    status = (
        "PASS"
        if one_mean_error <= 6.0 * mean_se
        and one_variance_error <= 0.04 * variance_scale
        and semigroup_mean_error <= 8.0 * mean_se
        and semigroup_variance_error <= 0.06 * variance_scale
        and min(float(np.min(one_step)), float(np.min(two_half_steps))) > 0.0
        else "REJECT"
    )
    return ExactTransitionAudit(
        one_step_mean_error=one_mean_error,
        one_step_variance_error=one_variance_error,
        semigroup_mean_error=semigroup_mean_error,
        semigroup_variance_error=semigroup_variance_error,
        minimum_terminal_value=min(
            float(np.min(one_step)),
            float(np.min(two_half_steps)),
        ),
        status=status,
    )
