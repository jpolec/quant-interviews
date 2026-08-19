from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class HittingSimulationResult:
    dt: float
    total_paths: int
    estimate: float | None
    standard_error: float | None
    analytic_probability: float
    probability_lower_bound: float
    probability_upper_bound: float
    mean_exit_time: float | None
    analytic_mean_exit_time: float
    unresolved_paths: int
    unresolved_fraction: float
    steps_used: int
    status: str


@dataclass(frozen=True)
class HittingConvergenceStudy:
    rows: tuple[HittingSimulationResult, ...]
    probability_refinement_change: float
    exit_time_refinement_change: float
    probability_bias_limit: float
    exit_time_bias_limit: float
    maximum_unresolved_fraction: float
    confidence_z: float
    status: str


def _validate_barriers(x: float, lower: float, upper: float) -> None:
    if not all(math.isfinite(value) for value in (x, lower, upper)):
        raise ValueError("barriers and starting state must be finite")
    if not lower < x < upper:
        raise ValueError("require lower < x < upper")


def brownian_hit_upper_before_lower(x: float, lower: float, upper: float) -> float:
    _validate_barriers(x, lower, upper)
    return (x - lower) / (upper - lower)


def brownian_mean_exit_time(x: float, lower: float, upper: float) -> float:
    _validate_barriers(x, lower, upper)
    return (x - lower) * (upper - x)


def simulate_hit_probability(
    x0: float,
    lower: float,
    upper: float,
    *,
    dt: float = 1e-3,
    paths: int = 10_000,
    seed: int = 23,
    maximum_steps: int = 1_000_000,
) -> float:
    result = simulate_hitting_experiment(
        x0,
        lower,
        upper,
        dt=dt,
        paths=paths,
        seed=seed,
        maximum_steps=maximum_steps,
    )
    if result.estimate is None:
        raise RuntimeError("no path resolved before truncation")
    return result.estimate


def simulate_hitting_experiment(
    x0: float,
    lower: float,
    upper: float,
    *,
    dt: float = 1e-3,
    paths: int = 10_000,
    seed: int = 23,
    maximum_steps: int = 1_000_000,
) -> HittingSimulationResult:
    """Produce one endpoint-monitored row; it is evidence, not a release gate."""

    analytic = brownian_hit_upper_before_lower(x0, lower, upper)
    analytic_exit_time = brownian_mean_exit_time(x0, lower, upper)
    if (
        not math.isfinite(dt)
        or dt <= 0.0
        or not isinstance(paths, int)
        or isinstance(paths, bool)
        or paths < 2
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not isinstance(maximum_steps, int)
        or isinstance(maximum_steps, bool)
        or maximum_steps < 1
    ):
        raise ValueError("positive finite dt, integer paths/seed/maximum_steps required")
    rng = np.random.default_rng(seed)
    x = np.full(paths, x0, dtype=float)
    active = np.ones(paths, dtype=bool)
    hit_upper = np.zeros(paths, dtype=bool)
    exit_steps = np.full(paths, maximum_steps, dtype=int)
    steps_used = 0
    while np.any(active) and steps_used < maximum_steps:
        steps_used += 1
        x[active] += math.sqrt(dt) * rng.standard_normal(int(np.sum(active)))
        above = active & (x >= upper)
        below = active & (x <= lower)
        hit_upper[above] = True
        exit_steps[above | below] = steps_used
        active[above | below] = False
    resolved = ~active
    resolved_count = int(np.sum(resolved))
    unresolved = paths - resolved_count
    upper_hits = int(np.sum(hit_upper))
    lower_bound = upper_hits / paths
    upper_bound = (upper_hits + unresolved) / paths
    if resolved_count == 0:
        estimate = None
        standard_error = None
        mean_exit_time = None
    else:
        estimate = float(np.mean(hit_upper[resolved]))
        standard_error = float(
            math.sqrt(estimate * (1.0 - estimate) / resolved_count)
        )
        mean_exit_time = float(np.mean(exit_steps[resolved]) * dt)
    return HittingSimulationResult(
        dt=dt,
        total_paths=paths,
        estimate=estimate,
        standard_error=standard_error,
        analytic_probability=analytic,
        probability_lower_bound=lower_bound,
        probability_upper_bound=upper_bound,
        mean_exit_time=mean_exit_time,
        analytic_mean_exit_time=analytic_exit_time,
        unresolved_paths=unresolved,
        unresolved_fraction=unresolved / paths,
        steps_used=steps_used,
        status="EVIDENCE_ONLY" if unresolved == 0 else "CENSORED",
    )


def hitting_convergence_study(
    x0: float,
    lower: float,
    upper: float,
    *,
    step_sizes: tuple[float, ...] = (0.008, 0.004, 0.002),
    paths: int = 12_000,
    seed: int = 23,
    maximum_steps: int = 100_000,
    confidence_z: float = 4.0,
    probability_bias_limit: float = 0.025,
    exit_time_bias_limit: float = 0.055,
    maximum_unresolved_fraction: float = 0.001,
    maximum_coarsest_step: float = 0.05,
) -> HittingConvergenceStudy:
    """Gate endpoint monitoring with signed sampling, refinement and censor budgets."""

    controls = (
        confidence_z,
        probability_bias_limit,
        exit_time_bias_limit,
        maximum_unresolved_fraction,
        maximum_coarsest_step,
    )
    if (
        len(step_sizes) < 3
        or any(not math.isfinite(dt) or dt <= 0.0 for dt in step_sizes)
        or tuple(sorted(set(step_sizes), reverse=True)) != step_sizes
        or any(not math.isfinite(value) or value < 0.0 for value in controls)
        or confidence_z <= 0.0
        or paths < 2
    ):
        raise ValueError("three decreasing grids and finite signed policies required")
    rows = tuple(
        simulate_hitting_experiment(
            x0,
            lower,
            upper,
            dt=dt,
            paths=paths,
            seed=seed + index,
            maximum_steps=maximum_steps,
        )
        for index, dt in enumerate(step_sizes)
    )
    previous, finest = rows[-2], rows[-1]
    if (
        previous.estimate is None
        or finest.estimate is None
        or previous.mean_exit_time is None
        or finest.mean_exit_time is None
        or finest.standard_error is None
    ):
        probability_change = float("inf")
        exit_change = float("inf")
        status = "REJECT"
    else:
        probability_change = abs(finest.estimate - previous.estimate)
        exit_change = abs(finest.mean_exit_time - previous.mean_exit_time)
        probability_error = abs(finest.estimate - finest.analytic_probability)
        exit_error = abs(finest.mean_exit_time - finest.analytic_mean_exit_time)
        status = (
            "PASS"
            if step_sizes[0] <= maximum_coarsest_step
            and probability_error
            <= confidence_z * finest.standard_error + probability_bias_limit
            and exit_error <= exit_time_bias_limit
            and probability_change <= probability_bias_limit
            and exit_change <= exit_time_bias_limit / 2.0
            and max(row.unresolved_fraction for row in rows)
            <= maximum_unresolved_fraction
            else "REJECT"
        )
    return HittingConvergenceStudy(
        rows=rows,
        probability_refinement_change=probability_change,
        exit_time_refinement_change=exit_change,
        probability_bias_limit=probability_bias_limit,
        exit_time_bias_limit=exit_time_bias_limit,
        maximum_unresolved_fraction=maximum_unresolved_fraction,
        confidence_z=confidence_z,
        status=status,
    )
