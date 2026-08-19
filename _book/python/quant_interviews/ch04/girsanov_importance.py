from __future__ import annotations

from dataclasses import dataclass
from math import erfc, exp, isfinite, log, sqrt

import numpy as np


@dataclass(frozen=True)
class ImportanceSamplingResult:
    estimate: float
    standard_error: float
    analytic_probability: float
    confidence_interval: tuple[float, float]
    effective_sample_size_fraction: float
    maximum_normalized_weight: float
    log_weight_range: float
    plain_monte_carlo_standard_error: float
    variance_reduction: float
    status: str


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + log(float(np.sum(np.exp(values - maximum))))


def normal_tail_importance_sampling(
    threshold: float,
    *,
    shift: float,
    paths: int = 200_000,
    seed: int = 19,
    confidence_z: float = 1.96,
    minimum_ess_fraction: float = 0.005,
) -> ImportanceSamplingResult:
    """Estimate P[Z>threshold] under a shifted normal law in log space."""

    controls = (threshold, shift, confidence_z, minimum_ess_fraction)
    if (
        not all(isfinite(value) for value in controls)
        or confidence_z <= 0.0
        or not 0.0 < minimum_ess_fraction <= 1.0
        or not isinstance(paths, int)
        or isinstance(paths, bool)
        or paths < 2
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise ValueError("finite controls, paths >= 2 and integer seed required")
    if threshold >= 0.0 and shift <= 0.0:
        raise ValueError("upper-tail shift must point toward a nonnegative threshold")

    rng = np.random.default_rng(seed)
    shifted = rng.standard_normal(paths) + shift
    selected = shifted > threshold
    event_count = int(np.sum(selected))
    if event_count == 0:
        return ImportanceSamplingResult(
            estimate=0.0,
            standard_error=0.0,
            analytic_probability=0.5 * erfc(threshold / sqrt(2.0)),
            confidence_interval=(0.0, 0.0),
            effective_sample_size_fraction=0.0,
            maximum_normalized_weight=0.0,
            log_weight_range=0.0,
            plain_monte_carlo_standard_error=0.0,
            variance_reduction=0.0,
            status="REJECT",
        )
    log_weights = -shift * shifted[selected] + 0.5 * shift**2
    log_sum = _logsumexp(log_weights)
    log_second_sum = _logsumexp(2.0 * log_weights)
    estimate = exp(log_sum - log(paths))
    second_moment = exp(log_second_sum - log(paths))
    sample_variance = max((second_moment - estimate**2) * paths / (paths - 1), 0.0)
    standard_error = sqrt(sample_variance / paths)
    normalized = np.exp(log_weights - log_sum)
    ess_fraction = 1.0 / float(np.sum(normalized**2)) / paths
    analytic = 0.5 * erfc(threshold / sqrt(2.0))
    plain_se = sqrt(analytic * (1.0 - analytic) / paths)
    variance_reduction = (
        (plain_se / standard_error) ** 2 if standard_error > 0.0 else 0.0
    )
    interval = (
        max(0.0, estimate - confidence_z * standard_error),
        min(1.0, estimate + confidence_z * standard_error),
    )
    diagnostics = (
        estimate,
        standard_error,
        analytic,
        ess_fraction,
        float(np.max(normalized)),
        float(np.max(log_weights) - np.min(log_weights)),
        plain_se,
        variance_reduction,
    )
    status = (
        "PASS"
        if all(isfinite(value) for value in diagnostics)
        and interval[0] <= analytic <= interval[1]
        and ess_fraction >= minimum_ess_fraction
        and variance_reduction > 1.0
        else "REJECT"
    )
    return ImportanceSamplingResult(
        estimate=estimate,
        standard_error=standard_error,
        analytic_probability=analytic,
        confidence_interval=interval,
        effective_sample_size_fraction=ess_fraction,
        maximum_normalized_weight=float(np.max(normalized)),
        log_weight_range=float(np.max(log_weights) - np.min(log_weights)),
        plain_monte_carlo_standard_error=plain_se,
        variance_reduction=variance_reduction,
        status=status,
    )
