from __future__ import annotations

from dataclasses import dataclass
from math import exp, expm1, isfinite, log, sqrt

import numpy as np


@dataclass(frozen=True)
class RuinSimulation:
    hit_upper_probability: float | None
    probability_standard_error: float | None
    analytic_hit_upper_probability: float
    probability_residual: float | None
    probability_tolerance: float | None
    expected_duration: float | None
    duration_standard_error: float | None
    analytic_expected_duration: float
    duration_residual: float | None
    duration_tolerance: float | None
    completed_paths: int
    maximum_path_steps: int
    truncated_paths: int
    status: str


def _validate_state(i: int, capital: int, p: float) -> None:
    if (
        capital <= 0
        or not 0 <= i <= capital
        or not isfinite(p)
        or not 0.0 < p < 1.0
    ):
        raise ValueError("invalid unit-step gambler's-ruin state")


def hit_upper_probability(i: int, capital: int, p: float) -> float:
    """Stable closed form for biased unit steps."""
    _validate_state(i, capital, p)
    if i == 0:
        return 0.0
    if i == capital:
        return 1.0
    q = 1.0 - p
    log_ratio = log(q / p)
    if p == 0.5:
        return i / capital
    if log_ratio > 0.0:
        numerator = -expm1(-i * log_ratio)
        denominator = -expm1(-capital * log_ratio)
        return exp((i - capital) * log_ratio) * numerator / denominator
    return expm1(i * log_ratio) / expm1(capital * log_ratio)


def expected_duration(i: int, capital: int, p: float) -> float:
    """Solve the duration recurrence with a stable Thomas algorithm."""
    _validate_state(i, capital, p)
    if i in (0, capital):
        return 0.0
    interior = capital - 1
    q = 1.0 - p
    lower = np.full(interior - 1, -q, dtype=float)
    diagonal = np.ones(interior, dtype=float)
    upper = np.full(interior - 1, -p, dtype=float)
    rhs = np.ones(interior, dtype=float)
    for index in range(1, interior):
        factor = lower[index - 1] / diagonal[index - 1]
        diagonal[index] -= factor * upper[index - 1]
        rhs[index] -= factor * rhs[index - 1]
    solution = np.empty(interior, dtype=float)
    solution[-1] = rhs[-1] / diagonal[-1]
    for index in range(interior - 2, -1, -1):
        solution[index] = (
            rhs[index] - upper[index] * solution[index + 1]
        ) / diagonal[index]
    return float(solution[i - 1])


def simulate_certificate(
    i: int,
    capital: int,
    p: float,
    trials: int = 100_000,
    seed: int = 17,
    maximum_steps: int = 1_000_000,
) -> RuinSimulation:
    _validate_state(i, capital, p)
    if trials <= 1 or maximum_steps <= 0:
        raise ValueError("trials must exceed one and maximum_steps must be positive")
    rng = np.random.default_rng(seed)
    wins = 0
    durations: list[int] = []
    truncated = 0
    maximum_observed = 0
    for _ in range(trials):
        state = i
        steps = 0
        while 0 < state < capital and steps < maximum_steps:
            state += 1 if rng.random() < p else -1
            steps += 1
        maximum_observed = max(maximum_observed, steps)
        if 0 < state < capital:
            truncated += 1
            continue
        wins += int(state == capital)
        durations.append(steps)
    completed = trials - truncated
    analytic_probability = hit_upper_probability(i, capital, p)
    analytic_duration = expected_duration(i, capital, p)
    if truncated > 0 or completed < 2:
        return RuinSimulation(
            hit_upper_probability=None,
            probability_standard_error=None,
            analytic_hit_upper_probability=analytic_probability,
            probability_residual=None,
            probability_tolerance=None,
            expected_duration=None,
            duration_standard_error=None,
            analytic_expected_duration=analytic_duration,
            duration_residual=None,
            duration_tolerance=None,
            completed_paths=completed,
            maximum_path_steps=maximum_observed,
            truncated_paths=truncated,
            status="REJECT",
        )
    probability = wins / completed
    probability_se = sqrt(probability * (1.0 - probability) / completed)
    duration_array = np.asarray(durations, dtype=float)
    duration_mean = float(np.mean(duration_array))
    duration_se = float(np.std(duration_array, ddof=1) / sqrt(completed))
    probability_residual = probability - analytic_probability
    duration_residual = duration_mean - analytic_duration
    probability_tolerance = 4.0 * probability_se
    duration_tolerance = 4.0 * duration_se
    finite_outputs = all(
        isfinite(value)
        for value in (
            probability,
            probability_se,
            analytic_probability,
            probability_residual,
            probability_tolerance,
            duration_mean,
            duration_se,
            analytic_duration,
            duration_residual,
            duration_tolerance,
        )
    )
    reconciled = (
        finite_outputs
        and abs(probability_residual) <= probability_tolerance
        and abs(duration_residual) <= duration_tolerance
    )
    return RuinSimulation(
        hit_upper_probability=probability,
        probability_standard_error=probability_se,
        analytic_hit_upper_probability=analytic_probability,
        probability_residual=probability_residual,
        probability_tolerance=probability_tolerance,
        expected_duration=duration_mean,
        duration_standard_error=duration_se,
        analytic_expected_duration=analytic_duration,
        duration_residual=duration_residual,
        duration_tolerance=duration_tolerance,
        completed_paths=completed,
        maximum_path_steps=maximum_observed,
        truncated_paths=truncated,
        status="PASS" if reconciled else "REJECT",
    )


def simulate(
    i: int,
    capital: int,
    p: float,
    trials: int = 100_000,
    seed: int = 17,
) -> tuple[float, float]:
    result = simulate_certificate(i, capital, p, trials, seed)
    if result.hit_upper_probability is None or result.expected_duration is None:
        raise RuntimeError("simulation did not produce an uncensored estimate")
    return result.hit_upper_probability, result.expected_duration
