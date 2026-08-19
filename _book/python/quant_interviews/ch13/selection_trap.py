from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np


@dataclass(frozen=True)
class SelectionSimulation:
    posterior: float
    reports: int
    standard_error: float
    analytic_posterior: float
    residual: float
    wilson_lower: float
    wilson_upper: float
    reconciliation_tolerance: float
    evidence_sufficient: bool
    status: str


def posterior_prespecified(
    prior_prespecified: float,
    power: float,
    alpha: float,
) -> float:
    _validate(prior_prespecified, power, alpha, 1)
    numerator = prior_prespecified * power
    return numerator / (numerator + (1.0 - prior_prespecified) * alpha)


def posterior_after_screen(
    prior_program: float,
    power: float,
    alpha: float,
    candidates: int,
) -> float:
    """Assume exactly one genuine model under H1 and conditional independence."""
    _validate(prior_program, power, alpha, candidates)
    event_given_signal = (
        1.0 - (1.0 - power) * (1.0 - alpha) ** (candidates - 1)
    )
    event_given_null = 1.0 - (1.0 - alpha) ** candidates
    numerator = prior_program * event_given_signal
    return numerator / (
        numerator + (1.0 - prior_program) * event_given_null
    )


def seeded_monte_carlo_cross_check(
    prior_program: float,
    power: float,
    alpha: float,
    candidates: int,
    trials: int = 500_000,
    seed: int = 19,
) -> SelectionSimulation:
    _validate(prior_program, power, alpha, candidates)
    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)
    signal_world = rng.random(trials) < prior_program
    selected = np.zeros(trials, dtype=bool)
    for index in range(trials):
        if signal_world[index]:
            passes = rng.random(candidates) < alpha
            passes[0] = rng.random() < power
        else:
            passes = rng.random(candidates) < alpha
        selected[index] = bool(np.any(passes))
    reports = int(np.sum(selected))
    if reports == 0:
        raise RuntimeError("no reported events; posterior is undefined")
    posterior = float(np.mean(signal_world[selected]))
    standard_error = sqrt(posterior * (1.0 - posterior) / reports)
    analytic = posterior_after_screen(prior_program, power, alpha, candidates)
    residual = posterior - analytic
    z = 4.0
    denominator = 1.0 + z * z / reports
    centre = (posterior + z * z / (2.0 * reports)) / denominator
    half_width = (
        z
        * sqrt(
            posterior * (1.0 - posterior) / reports
            + z * z / (4.0 * reports * reports)
        )
        / denominator
    )
    lower = max(0.0, centre - half_width)
    upper = min(1.0, centre + half_width)
    tolerance = max(posterior - lower, upper - posterior)
    evidence_sufficient = reports >= 30
    reconciled = evidence_sufficient and lower <= analytic <= upper
    return SelectionSimulation(
        posterior=posterior,
        reports=reports,
        standard_error=standard_error,
        analytic_posterior=analytic,
        residual=residual,
        wilson_lower=lower,
        wilson_upper=upper,
        reconciliation_tolerance=tolerance,
        evidence_sufficient=evidence_sufficient,
        status="PASS" if reconciled else "REJECT",
    )


def simulate(
    prior: float,
    power: float,
    alpha: float,
    candidates: int,
    trials: int = 500_000,
    seed: int = 19,
) -> float:
    return seeded_monte_carlo_cross_check(
        prior,
        power,
        alpha,
        candidates,
        trials,
        seed,
    ).posterior


def _validate(
    prior: float,
    power: float,
    alpha: float,
    candidates: int,
) -> None:
    if (
        not all(isfinite(value) for value in (prior, power, alpha))
        or not 0.0 < prior < 1.0
        or not 0.0 < alpha < 1.0
        or not 0.0 < power < 1.0
        or candidates < 1
    ):
        raise ValueError("invalid screening parameters")
