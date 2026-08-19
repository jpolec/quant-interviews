from __future__ import annotations

from dataclasses import dataclass
from math import e, isfinite, sqrt

import numpy as np


@dataclass(frozen=True)
class SecretarySimulation:
    success_probability: float
    standard_error: float
    wilson_lower: float
    wilson_upper: float
    analytic_probability: float | None
    residual: float | None
    tolerance: float | None
    evidence_sufficient: bool
    no_record_after_cutoff: int
    policy: str
    status: str


def success_probability(n: int, cutoff: int) -> float:
    if n < 1 or not 0 <= cutoff < n:
        raise ValueError("require n >= 1 and 0 <= cutoff < n")
    if cutoff == 0:
        return 1.0 / n
    return (cutoff / n) * sum(
        1.0 / (j - 1) for j in range(cutoff + 1, n + 1)
    )


def optimal_cutoff(n: int) -> tuple[int, float]:
    if n < 1:
        raise ValueError("n must be positive")
    candidates = [(r, success_probability(n, r)) for r in range(n)]
    return max(candidates, key=lambda item: item[1])


def simulate_classical_cutoff_under_noise(
    n: int,
    cutoff: int,
    trials: int = 100_000,
    seed: int = 11,
    noise_sigma: float = 0.0,
) -> SecretarySimulation:
    """Audit the classical cutoff heuristic, not the optimal noisy policy."""
    if (
        n < 1
        or not 0 <= cutoff < n
        or trials <= 0
        or not isfinite(noise_sigma)
        or noise_sigma < 0.0
    ):
        raise ValueError("invalid secretary simulation parameters")
    rng = np.random.default_rng(seed)
    wins = 0
    fallback_count = 0
    scale = max(n - 1, 1)
    for _ in range(trials):
        quality = rng.permutation(n).astype(float) / scale
        observed = quality + rng.normal(0.0, noise_sigma, size=n)
        threshold = np.max(observed[:cutoff]) if cutoff else -np.inf
        chosen: int | None = None
        for index in range(cutoff, n):
            if observed[index] > threshold:
                chosen = index
                break
        if chosen is None:
            chosen = n - 1
            fallback_count += 1
        wins += int(quality[chosen] == np.max(quality))
    estimate = wins / trials
    standard_error = sqrt(estimate * (1.0 - estimate) / trials)
    z = 1.959963984540054
    denominator = 1.0 + z * z / trials
    centre = (estimate + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * sqrt(
            estimate * (1.0 - estimate) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    analytic = success_probability(n, cutoff) if noise_sigma == 0.0 else None
    residual = estimate - analytic if analytic is not None else None
    tolerance = max(estimate - max(0.0, centre - half_width), min(1.0, centre + half_width) - estimate)
    evidence_sufficient = trials >= 1_000
    reconciled = (
        analytic is not None
        and max(0.0, centre - half_width) <= analytic <= min(1.0, centre + half_width)
    )
    if noise_sigma > 0.0:
        status = "CONDITIONAL_PASS" if evidence_sufficient else "REJECT"
    else:
        status = "PASS" if evidence_sufficient and reconciled else "REJECT"
    return SecretarySimulation(
        success_probability=estimate,
        standard_error=standard_error,
        wilson_lower=max(0.0, centre - half_width),
        wilson_upper=min(1.0, centre + half_width),
        analytic_probability=analytic,
        residual=residual,
        tolerance=tolerance,
        evidence_sufficient=evidence_sufficient,
        no_record_after_cutoff=fallback_count,
        policy="classical cutoff; choose last candidate if no later record appears",
        status=status,
    )


def simulate(
    n: int,
    cutoff: int,
    trials: int = 100_000,
    seed: int = 11,
    noise_sigma: float = 0.0,
) -> float:
    return simulate_classical_cutoff_under_noise(
        n,
        cutoff,
        trials,
        seed,
        noise_sigma,
    ).success_probability


def asymptotic_cutoff_fraction() -> float:
    return 1.0 / e
