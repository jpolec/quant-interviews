"""Sampling-distribution calculations for Chapter 1."""

from __future__ import annotations

from math import isfinite, sqrt
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike


def annualized_sharpe_standard_error(
    annualized_sharpe: float,
    observations: int,
    periods_per_year: float = 252.0,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Return the standard error of an explicitly annualized Sharpe ratio."""

    if not all(
        isfinite(value)
        for value in (
            annualized_sharpe,
            periods_per_year,
            skewness,
            excess_kurtosis,
        )
    ):
        raise ValueError("Sharpe, frequency and moments must be finite")
    if (
        isinstance(observations, bool)
        or not isinstance(observations, Integral)
        or observations <= 1
        or periods_per_year <= 0.0
    ):
        raise ValueError("observations must exceed one and frequency must be positive")
    periodic_sharpe = annualized_sharpe / sqrt(periods_per_year)
    periodic_variance = (
        1.0
        - skewness * periodic_sharpe
        + 0.25
        * (excess_kurtosis + 2.0)
        * periodic_sharpe
        * periodic_sharpe
    ) / observations
    if periodic_variance < 0.0:
        raise ValueError("moment combination implies negative variance")
    return sqrt(periods_per_year * periodic_variance)


def sharpe_standard_error(
    sharpe: float,
    n: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Compatibility helper for a Sharpe ratio at the observation frequency."""

    return annualized_sharpe_standard_error(
        annualized_sharpe=sharpe,
        observations=n,
        periods_per_year=1.0,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )


def moving_block_bootstrap_indices(
    n: int,
    block_length: int,
    draws: int,
    seed: int = 0,
) -> np.ndarray:
    """Draw fixed-length contiguous blocks for a moving-block bootstrap."""

    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (n, block_length, draws)
    ):
        raise ValueError("n, block_length and draws must be integers")
    if n < 2 or not 1 <= block_length <= n or draws <= 0:
        raise ValueError("valid n, block_length and draws are required")
    blocks_per_draw = int(np.ceil(n / block_length))
    generator = np.random.default_rng(seed)
    starts = generator.integers(
        0,
        n - block_length + 1,
        size=(draws, blocks_per_draw),
    )
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets).reshape(draws, -1)
    return indices[:, :n]


def moving_block_bootstrap_mean_ci(
    returns: ArrayLike,
    block_length: int,
    *,
    draws: int = 2_000,
    confidence: float = 0.95,
    periods_per_year: float = 252.0,
    seed: int = 0,
) -> dict[str, float | np.ndarray]:
    """Estimate an annualized-mean percentile CI with moving blocks."""

    observations = np.asarray(returns, dtype=float)
    if (
        observations.ndim != 1
        or observations.size < 2
        or not np.all(np.isfinite(observations))
        or not 0.0 < confidence < 1.0
        or not isfinite(periods_per_year)
        or periods_per_year <= 0.0
    ):
        raise ValueError("finite returns, confidence and frequency are required")
    if (
        isinstance(block_length, bool)
        or not isinstance(block_length, Integral)
        or isinstance(draws, bool)
        or not isinstance(draws, Integral)
        or not 1 <= block_length <= observations.size
        or draws <= 0
    ):
        raise ValueError("valid block_length and draws are required")
    indices = moving_block_bootstrap_indices(
        n=observations.size,
        block_length=block_length,
        draws=draws,
        seed=seed,
    )
    annualized_draws = observations[indices].mean(axis=1) * periods_per_year
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(annualized_draws, [tail, 1.0 - tail])
    return {
        "annualized_mean": float(observations.mean() * periods_per_year),
        "lower": float(lower),
        "upper": float(upper),
        "bootstrap_draws": annualized_draws,
    }


def stationary_block_bootstrap_indices(
    n: int,
    expected_block: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw stationary-bootstrap indices with circular continuation."""

    if isinstance(n, bool) or not isinstance(n, Integral):
        raise ValueError("n must be a positive integer")
    if n <= 0 or not isfinite(expected_block) or expected_block < 1.0:
        raise ValueError("valid n and expected block are required")
    restart_probability = 1.0 / expected_block
    indices = np.empty(n, dtype=int)
    indices[0] = rng.integers(n)
    for position in range(1, n):
        if rng.random() < restart_probability:
            indices[position] = rng.integers(n)
        else:
            indices[position] = (indices[position - 1] + 1) % n
    return indices


def bootstrap_sharpe(
    returns: ArrayLike,
    expected_block: float,
    draws: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """Return a deterministic stationary-bootstrap Sharpe distribution."""

    observations = np.asarray(returns, dtype=float)
    if (
        observations.ndim != 1
        or observations.size < 3
        or not np.all(np.isfinite(observations))
        or not isfinite(expected_block)
        or expected_block < 1.0
        or isinstance(draws, bool)
        or not isinstance(draws, Integral)
        or draws <= 0
    ):
        raise ValueError("valid return series and draws are required")
    generator = np.random.default_rng(seed)
    bootstrap_values = np.empty(draws, dtype=float)
    for draw in range(draws):
        indices = stationary_block_bootstrap_indices(
            observations.size,
            expected_block,
            generator,
        )
        sample = observations[indices]
        sample_volatility = sample.std(ddof=1)
        if not isfinite(sample_volatility) or sample_volatility <= 0.0:
            raise ValueError(
                f"bootstrap draw {draw} has zero or nonfinite sample volatility"
            )
        bootstrap_values[draw] = sample.mean() / sample_volatility
    return bootstrap_values


def normal_mle(values: ArrayLike) -> dict[str, float]:
    """Return the normal MLE and Fisher information for its mean.

    The information is evaluated at the MLE variance. The observed and
    expected mean-mean entries coincide for this normal model.
    """

    observations = np.asarray(values, dtype=float)
    if (
        observations.ndim != 1
        or observations.size < 2
        or not np.all(np.isfinite(observations))
    ):
        raise ValueError("a finite one-dimensional sample is required")
    mean = float(observations.mean())
    variance = float(np.mean((observations - mean) ** 2))
    if variance == 0.0:
        raise ValueError("zero variance is a boundary case, not a regular MLE")
    information = observations.size / variance
    return {
        "mean": mean,
        "variance_mle": variance,
        "fisher_information_mean": information,
    }
