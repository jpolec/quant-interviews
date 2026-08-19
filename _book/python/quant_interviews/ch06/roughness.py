from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class RoughnessRecoveryStudy:
    true_hurst: float
    estimated_hurst_mean: float
    confidence_interval: tuple[float, float]
    estimate_standard_error: float
    confidence_level: float
    degrees_of_freedom: int
    confidence_interval_method: str
    coverage_status: str
    bias_tolerance: float
    sampling_points: int
    paths: int
    microstructure_noise_std: float
    status: str


def variogram_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    values = np.asarray(series, dtype=float)
    if (
        values.ndim != 1
        or values.size < 4 * max_lag
        or max_lag < 2
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid series or max_lag for variogram regression")
    lags = np.arange(1, max_lag + 1)
    variograms = np.array(
        [np.mean((values[lag:] - values[:-lag]) ** 2) for lag in lags]
    )
    if np.any(variograms <= 0.0):
        raise ValueError("variogram contains nonpositive values")
    slope = np.polyfit(np.log(lags), np.log(variograms), 1)[0]
    return float(slope / 2.0)


def _fractional_brownian_paths(
    hurst: float,
    points: int,
    paths: int,
    seed: int,
) -> np.ndarray:
    times = np.arange(1, points + 1, dtype=float) / points
    covariance = 0.5 * (
        times[:, None] ** (2.0 * hurst)
        + times[None, :] ** (2.0 * hurst)
        - np.abs(times[:, None] - times[None, :]) ** (2.0 * hurst)
    )
    factor = np.linalg.cholesky(covariance + 1e-12 * np.eye(points))
    rng = np.random.default_rng(seed)
    return (factor @ rng.standard_normal((points, paths))).T


def roughness_recovery_study(
    *,
    true_hurst: float,
    points: int = 384,
    paths: int = 24,
    max_lag: int = 16,
    microstructure_noise_std: float = 0.0,
    seed: int = 0,
    tolerance: float = 0.10,
    confidence_level: float = 0.95,
) -> RoughnessRecoveryStudy:
    values = (
        true_hurst,
        microstructure_noise_std,
        tolerance,
        confidence_level,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("roughness-study inputs must be finite")
    if not 0.0 < true_hurst < 1.0:
        raise ValueError("true_hurst must lie in (0, 1)")
    if (
        isinstance(points, bool)
        or not isinstance(points, int)
        or isinstance(paths, bool)
        or not isinstance(paths, int)
        or isinstance(max_lag, bool)
        or not isinstance(max_lag, int)
        or points < 4 * max_lag
        or paths < 2
        or max_lag < 2
        or microstructure_noise_std < 0.0
        or tolerance < 0.0
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValueError("invalid recovery-study dimensions")
    simulated = _fractional_brownian_paths(true_hurst, points, paths, seed)
    if microstructure_noise_std > 0.0:
        rng = np.random.default_rng(seed + 1)
        simulated += microstructure_noise_std * rng.standard_normal(simulated.shape)
    estimates = np.asarray(
        [variogram_exponent(path, max_lag=max_lag) for path in simulated]
    )
    degrees_of_freedom = paths - 1
    standard_error = float(np.std(estimates, ddof=1)) / np.sqrt(paths)
    critical_value = float(
        student_t.ppf(0.5 + 0.5 * confidence_level, degrees_of_freedom)
    )
    half_width = critical_value * standard_error
    mean = float(np.mean(estimates))
    confidence_interval = (mean - half_width, mean + half_width)
    bias_passes = abs(mean - true_hurst) <= tolerance
    coverage_passes = confidence_interval[0] <= true_hurst <= confidence_interval[1]
    if bias_passes and coverage_passes:
        status = "PASS"
    elif bias_passes or coverage_passes:
        status = "CONDITIONAL_PASS"
    else:
        status = "REJECT"
    return RoughnessRecoveryStudy(
        true_hurst=true_hurst,
        estimated_hurst_mean=mean,
        confidence_interval=confidence_interval,
        estimate_standard_error=standard_error,
        confidence_level=confidence_level,
        degrees_of_freedom=degrees_of_freedom,
        confidence_interval_method="Student-t interval over path estimates",
        coverage_status="PASS" if coverage_passes else "REJECT",
        bias_tolerance=tolerance,
        sampling_points=points,
        paths=paths,
        microstructure_noise_std=microstructure_noise_std,
        status=status,
    )


def exponential_forecast_weights(
    half_life: float,
    horizons: np.ndarray,
) -> np.ndarray:
    """Generic forecast weighting; this is not a rough-volatility kernel."""

    horizon_array = np.asarray(horizons, dtype=float)
    if (
        not isfinite(half_life)
        or half_life <= 0.0
        or horizon_array.ndim != 1
        or horizon_array.size == 0
        or np.any(horizon_array < 0.0)
        or not np.isfinite(horizon_array).all()
    ):
        raise ValueError("invalid exponential forecast-weight inputs")
    weights = np.exp(-np.log(2.0) * horizon_array / half_life)
    return weights / weights.sum()


def forecast_horizon_weight(
    half_life: float,
    horizons: np.ndarray,
) -> np.ndarray:
    return exponential_forecast_weights(half_life, horizons)
