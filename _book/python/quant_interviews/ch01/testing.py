"""Multiple-testing, power and selection diagnostics for Chapter 1."""

from __future__ import annotations

from math import isfinite, sqrt
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import beta as beta_distribution
from scipy.stats import norm as normal_distribution


def benjamini_hochberg(
    pvalues: ArrayLike,
    alpha: float = 0.05,
) -> np.ndarray:
    """Return the Benjamini-Hochberg rejection mask."""

    probabilities = np.asarray(pvalues, dtype=float)
    invalid = (
        probabilities.ndim != 1
        or probabilities.size == 0
        or not np.all(np.isfinite(probabilities))
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
        or not 0.0 < alpha < 1.0
    )
    if invalid:
        raise ValueError("valid p-values and alpha are required")

    order = np.argsort(probabilities)
    ranked = probabilities[order]
    ranks = np.arange(1, probabilities.size + 1)
    accepted_ranks = np.flatnonzero(
        ranked <= alpha * ranks / probabilities.size
    )
    rejected = np.zeros(probabilities.size, dtype=bool)
    if accepted_ranks.size:
        rejected[order[: accepted_ranks[-1] + 1]] = True
    return rejected


def expected_max_standard_normal(models: int) -> float:
    """Return Blom's expected-maximum approximation for standard normals."""

    if isinstance(models, bool) or not isinstance(models, Integral) or models < 2:
        raise ValueError("at least two models are required")
    plotting_position = (models - 0.375) / (models + 0.25)
    return float(normal_distribution.ppf(plotting_position))


def alpha_spending_schedule(
    total_alpha: float,
    looks: int,
) -> np.ndarray:
    """Return alpha_t = alpha / (t(t+1)) for the requested looks."""

    if (
        not isfinite(total_alpha)
        or not 0.0 < total_alpha < 1.0
        or isinstance(looks, bool)
        or not isinstance(looks, Integral)
        or looks <= 0
    ):
        raise ValueError("valid alpha and looks are required")
    look_index = np.arange(1, looks + 1, dtype=float)
    return total_alpha / (look_index * (look_index + 1.0))


def gaussian_mean_e_process(
    observations: ArrayLike,
    *,
    prior_variance: float = 1.0,
) -> np.ndarray:
    """Return a normal-mixture e-process for a unit-variance Gaussian mean."""

    values = np.asarray(observations, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or not isfinite(prior_variance)
        or prior_variance <= 0.0
    ):
        raise ValueError("finite observations and positive prior variance required")
    time = np.arange(1, values.size + 1, dtype=float)
    partial_sums = np.cumsum(values)
    denominator = 1.0 + prior_variance * time
    return np.exp(
        0.5 * prior_variance * partial_sums**2 / denominator
    ) / np.sqrt(denominator)


def gaussian_e_process_null_crossing_simulation(
    *,
    paths: int = 20_000,
    horizon: int = 250,
    threshold: float = 20.0,
    prior_variance: float = 0.5,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Estimate null crossing risk and its one-sided exact binomial bound."""

    if (
        isinstance(paths, bool)
        or not isinstance(paths, Integral)
        or paths <= 0
        or isinstance(horizon, bool)
        or not isinstance(horizon, Integral)
        or horizon <= 0
        or not isfinite(threshold)
        or threshold <= 1.0
        or not isfinite(prior_variance)
        or prior_variance <= 0.0
        or not isfinite(confidence)
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("valid simulation design is required")
    generator = np.random.default_rng(seed)
    observations = generator.standard_normal((paths, horizon))
    partial_sums = np.cumsum(observations, axis=1)
    time = np.arange(1, horizon + 1, dtype=float)
    denominator = 1.0 + prior_variance * time
    e_values = np.exp(
        0.5 * prior_variance * partial_sums**2 / denominator
    ) / np.sqrt(denominator)
    crossings = int(np.any(e_values >= threshold, axis=1).sum())
    upper_bound = (
        1.0
        if crossings == paths
        else float(
            beta_distribution.ppf(
                confidence,
                crossings + 1,
                paths - crossings,
            )
        )
    )
    return {
        "paths": paths,
        "horizon": horizon,
        "crossings": crossings,
        "crossing_rate": crossings / paths,
        "upper_confidence_bound": upper_bound,
    }


def minimum_detectable_mean(
    std: float,
    n: int,
    z_alpha: float = 1.96,
    z_power: float = 0.84,
) -> float:
    """Return the normal-approximation minimum detectable mean."""

    if not all(isfinite(value) for value in (std, z_alpha, z_power)):
        raise ValueError("std and critical values must be finite")
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise ValueError("n must be an integer greater than one")
    if std <= 0.0 or n <= 1 or z_alpha <= 0.0 or z_power <= 0.0:
        raise ValueError("positive std and n greater than one are required")
    return (z_alpha + z_power) * std / sqrt(n)


def winner_curse_gaps(
    models: int,
    observations: int,
    *,
    experiments: int = 1_000,
    true_mean: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Compatibility helper returning only the simulated selection gaps."""

    return np.asarray(
        winner_curse_distribution(
            models,
            observations,
            experiments=experiments,
            true_mean=true_mean,
            seed=seed,
        )["selection_gaps"]
    )


def winner_curse_distribution(
    models: int,
    observations: int,
    *,
    experiments: int = 1_000,
    true_mean: float = 0.0,
    seed: int = 0,
) -> dict[str, float | np.ndarray]:
    """Summarize repeated selection gaps against the null approximation."""

    if (
        any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (models, observations, experiments)
        )
        or models < 2
        or observations <= 1
        or experiments <= 1
        or not isfinite(true_mean)
    ):
        raise ValueError("valid models, observations and experiments required")
    generator = np.random.default_rng(seed)
    standard_error = 1.0 / sqrt(observations)
    training_means, test_means = generator.normal(
        true_mean,
        standard_error,
        size=(2, experiments, models),
    )
    selected_index = np.argmax(training_means, axis=1)
    row_index = np.arange(experiments)
    selection_gaps = (
        training_means[row_index, selected_index]
        - test_means[row_index, selected_index]
    )
    return {
        "mean_selection_gap": float(selection_gaps.mean()),
        "median_selection_gap": float(np.median(selection_gaps)),
        "expected_null_maximum": (
            standard_error * expected_max_standard_normal(models)
        ),
        "selection_gaps": selection_gaps,
    }


def winner_curse_simulation(
    models: int,
    observations: int,
    true_mean: float = 0.0,
    seed: int = 0,
) -> dict[str, float]:
    """Compatibility wrapper returning one deterministic selection experiment."""

    summary = winner_curse_distribution(
        models,
        observations,
        experiments=2,
        true_mean=true_mean,
        seed=seed,
    )
    gaps = np.asarray(summary["selection_gaps"])
    return {
        "selection_gap": float(gaps[0]),
        "mean_selection_gap": float(summary["mean_selection_gap"]),
    }
