"""Bayesian and mixture-model calculations for Chapter 1."""

from __future__ import annotations

from math import isfinite, log, pi, sqrt
from numbers import Integral
from typing import Iterable

import numpy as np
from scipy.stats import beta as beta_distribution


def positive_predictive_value(
    prevalence: float,
    sensitivity: float,
    specificity: float,
) -> float:
    """Return P(event | positive signal) with explicit range checks."""

    for probability in (prevalence, sensitivity, specificity):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probabilities must lie in [0, 1]")

    true_positive_mass = sensitivity * prevalence
    false_positive_mass = (1.0 - specificity) * (1.0 - prevalence)
    positive_signal_mass = true_positive_mass + false_positive_mass
    if positive_signal_mass == 0.0:
        raise ValueError("conditioning event has zero probability")
    return true_positive_mass / positive_signal_mass


def latent_mixture_moments(
    weights: Iterable[float],
    conditional_means: Iterable[float],
    conditional_variances: Iterable[float],
) -> dict[str, float]:
    """Apply total expectation and total variance to a regime mixture."""

    probabilities = np.asarray(list(weights), dtype=float)
    means = np.asarray(list(conditional_means), dtype=float)
    variances = np.asarray(list(conditional_variances), dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("at least one regime is required")
    if means.shape != probabilities.shape or variances.shape != probabilities.shape:
        raise ValueError("weights, means and variances must align")
    if not (
        np.all(np.isfinite(probabilities))
        and np.all(np.isfinite(means))
        and np.all(np.isfinite(variances))
    ):
        raise ValueError("weights, means and variances must be finite")
    if np.any(probabilities < 0.0) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    if np.any(variances < 0.0):
        raise ValueError("conditional variances must be nonnegative")

    unconditional_mean = float(probabilities @ means)
    within_regime_variance = float(probabilities @ variances)
    centered_means = means - unconditional_mean
    between_regime_variance = float(probabilities @ (centered_means**2))
    total_variance = within_regime_variance + between_regime_variance
    if total_variance == 0.0:
        raise ValueError("Sharpe is undefined when total variance is zero")
    unconditional_sharpe = unconditional_mean / sqrt(total_variance)
    return {
        "unconditional_mean": unconditional_mean,
        "within_regime_variance": within_regime_variance,
        "between_regime_variance": between_regime_variance,
        "total_variance": total_variance,
        "unconditional_sharpe": unconditional_sharpe,
    }


def latent_mixture_expectation(
    weights: Iterable[float],
    conditional_means: Iterable[float],
) -> float:
    """Compatibility helper for the tower-property expectation alone."""

    probabilities = np.asarray(list(weights), dtype=float)
    means = np.asarray(list(conditional_means), dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("at least one regime is required")
    if means.shape != probabilities.shape:
        raise ValueError("weights and means must align")
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(means)):
        raise ValueError("weights and means must be finite")
    if np.any(probabilities < 0.0) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    return float(probabilities @ means)


def beta_binomial_edge_update(
    prior_alpha: float,
    prior_beta: float,
    wins: int,
    trials: int,
    *,
    lower_quantile: float = 0.10,
    kelly_fraction: float = 0.50,
) -> dict[str, float]:
    """Return the conjugate posterior and uncertainty-aware Kelly sizing."""

    if not isfinite(prior_alpha) or not isfinite(prior_beta):
        raise ValueError("Beta prior parameters must be finite")
    if prior_alpha <= 0.0 or prior_beta <= 0.0:
        raise ValueError("Beta prior parameters must be positive")
    if (
        isinstance(wins, bool)
        or isinstance(trials, bool)
        or not isinstance(wins, Integral)
        or not isinstance(trials, Integral)
        or trials < 0
        or wins < 0
        or wins > trials
    ):
        raise ValueError("wins must lie between zero and trials")
    if not isfinite(lower_quantile) or not 0.0 < lower_quantile < 0.5:
        raise ValueError("lower_quantile must lie in (0, 0.5)")
    if not isfinite(kelly_fraction) or not 0.0 < kelly_fraction <= 1.0:
        raise ValueError("kelly_fraction must lie in (0, 1]")

    posterior_alpha = prior_alpha + wins
    posterior_beta = prior_beta + trials - wins
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    posterior_lower = float(
        beta_distribution.ppf(lower_quantile, posterior_alpha, posterior_beta)
    )
    probability_of_edge = float(
        beta_distribution.sf(0.5, posterior_alpha, posterior_beta)
    )
    conservative_kelly = kelly_fraction * max(0.0, 2.0 * posterior_lower - 1.0)
    return {
        "posterior_alpha": posterior_alpha,
        "posterior_beta": posterior_beta,
        "posterior_mean": posterior_mean,
        "posterior_lower_quantile": posterior_lower,
        "probability_of_edge": probability_of_edge,
        "conservative_kelly": conservative_kelly,
    }


def normal_normal_posterior(
    prior_mean: float,
    prior_variance: float,
    sample_mean: float,
    observation_variance: float,
    n: int,
) -> dict[str, float]:
    """Return the analogous normal-normal update used as an extension."""

    if not all(
        isfinite(value)
        for value in (
            prior_mean,
            prior_variance,
            sample_mean,
            observation_variance,
        )
    ):
        raise ValueError("means and variances must be finite")
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise ValueError("n must be a positive integer")
    if prior_variance <= 0.0 or observation_variance <= 0.0 or n <= 0:
        raise ValueError("positive variances and n required")
    prior_precision = 1.0 / prior_variance
    data_precision = n / observation_variance
    posterior_variance = 1.0 / (prior_precision + data_precision)
    posterior_mean = posterior_variance * (
        prior_mean * prior_precision + sample_mean * data_precision
    )
    return {
        "mean": posterior_mean,
        "variance": posterior_variance,
        "shrinkage_to_data": data_precision / (prior_precision + data_precision),
    }


def gaussian_log_predictive_score(
    observations: Iterable[float],
    means: Iterable[float],
    variances: Iterable[float],
) -> float:
    """Score caller-supplied Gaussian predictive distributions.

    Parameter uncertainty is represented only when it has already been
    incorporated into ``variances``; this function is not an integration
    routine and does not turn a plug-in forecast into a posterior predictive.
    """

    observed = list(observations)
    forecasts = list(means)
    forecast_variances = list(variances)
    if not observed or not (
        len(observed) == len(forecasts) == len(forecast_variances)
    ):
        raise ValueError("aligned nonempty inputs required")
    if not all(
        isfinite(value)
        for value in observed + forecasts + forecast_variances
    ):
        raise ValueError("observations, means and variances must be finite")
    if any(variance <= 0.0 for variance in forecast_variances):
        raise ValueError("positive predictive variance required")
    return sum(
        -0.5
        * (
            log(2.0 * pi * variance)
            + (value - mean) ** 2 / variance
        )
        for value, mean, variance in zip(
            observed,
            forecasts,
            forecast_variances,
            strict=True,
        )
    )
