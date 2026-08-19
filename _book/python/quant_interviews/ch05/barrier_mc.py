from __future__ import annotations

from math import exp, isfinite, log, sqrt
from typing import Any

import numpy as np

from .black_scholes import normal_cdf


def analytic_down_and_out_call(
    *,
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
) -> float:
    """Closed-form zero-rebate continuous down-and-out call for H < K."""

    numeric = (
        spot,
        strike,
        barrier,
        rate,
        dividend_yield,
        volatility,
        maturity,
    )
    if not all(isfinite(value) for value in numeric):
        raise ValueError("barrier inputs must be finite")
    if (
        not 0.0 < barrier < min(spot, strike)
        or volatility <= 0.0
        or maturity <= 0.0
    ):
        raise ValueError("closed form requires 0 < barrier < min(spot, strike)")
    root_t = sqrt(maturity)
    mu = (rate - dividend_yield - 0.5 * volatility**2) / volatility**2
    x1 = (
        log(spot / strike) / (volatility * root_t)
        + (1.0 + mu) * volatility * root_t
    )
    y1 = (
        log(barrier**2 / (spot * strike)) / (volatility * root_t)
        + (1.0 + mu) * volatility * root_t
    )
    reflected_spot = (barrier / spot) ** (2.0 * (mu + 1.0))
    reflected_strike = (barrier / spot) ** (2.0 * mu)
    return (
        spot
        * exp(-dividend_yield * maturity)
        * (
            normal_cdf(x1)
            - reflected_spot * normal_cdf(y1)
        )
        - strike
        * exp(-rate * maturity)
        * (
            normal_cdf(x1 - volatility * root_t)
            - reflected_strike
            * normal_cdf(y1 - volatility * root_t)
        )
    )


def _summary(sum_value: float, sum_square: float, paths: int) -> tuple[float, float]:
    mean = sum_value / paths
    variance = max(
        (sum_square - paths * mean**2) / max(paths - 1, 1),
        0.0,
    )
    return mean, sqrt(variance / paths)


def barrier_monitoring_experiment(
    *,
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    fine_steps: int,
    paths: int,
    monitoring_indices: dict[str, tuple[int, ...]] | None = None,
    monitoring_observations: tuple[int, ...] | None = None,
    seed: int = 43,
    batch_size: int = 4096,
    reference_sigma_multiple: float = 5.0,
    reference_absolute_tolerance: float = 0.03,
    parity_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Compare coupled discrete schedules and Brownian-bridge monitoring."""

    numeric = (
        spot,
        strike,
        barrier,
        rate,
        dividend_yield,
        volatility,
        maturity,
        reference_sigma_multiple,
        reference_absolute_tolerance,
        parity_tolerance,
    )
    if not all(isfinite(value) for value in numeric):
        raise ValueError("barrier inputs must be finite")
    if (
        not 0.0 < barrier < min(spot, strike)
        or volatility <= 0.0
        or maturity <= 0.0
        or fine_steps <= 1
        or paths <= 1
        or batch_size <= 0
        or reference_sigma_multiple <= 0.0
        or reference_absolute_tolerance < 0.0
        or parity_tolerance < 0.0
        or (monitoring_indices is None) == (monitoring_observations is None)
    ):
        raise ValueError("invalid barrier experiment")
    if monitoring_observations is not None:
        if (
            not monitoring_observations
            or len(set(monitoring_observations)) != len(monitoring_observations)
            or any(
                not isinstance(observations, int)
                or isinstance(observations, bool)
                or observations <= 0
                or observations > fine_steps
                or fine_steps % observations != 0
                for observations in monitoring_observations
            )
        ):
            raise ValueError(
                "count schedules must be unique exact divisors of fine_steps"
            )
        schedule_map = {
            str(observations): tuple(
                range(
                    fine_steps // observations,
                    fine_steps + 1,
                    fine_steps // observations,
                )
            )
            for observations in monitoring_observations
        }
    else:
        assert monitoring_indices is not None
        schedule_map = {
            str(label): tuple(indices)
            for label, indices in monitoring_indices.items()
        }
        if not schedule_map or len(schedule_map) != len(monitoring_indices):
            raise ValueError("monitoring schedule labels must be unique")
    for label, indices in schedule_map.items():
        if (
            not label
            or not indices
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index <= 0
                or index > fine_steps
                for index in indices
            )
            or any(indices[index + 1] <= indices[index] for index in range(len(indices) - 1))
            or indices[-1] != fine_steps
        ):
            raise ValueError("invalid explicit monitoring index schedule")
    observation_counts = [len(indices) for indices in schedule_map.values()]
    if len(set(observation_counts)) != len(observation_counts):
        raise ValueError("monitoring schedules must have distinct observation counts")
    schedules = tuple(
        sorted(schedule_map, key=lambda label: len(schedule_map[label]))
    )
    discount = exp(-rate * maturity)
    dt = maturity / fine_steps
    drift = (rate - dividend_yield - 0.5 * volatility**2) * dt
    diffusion = volatility * sqrt(dt)
    log_barrier = log(barrier)
    rng = np.random.default_rng(seed)

    totals = {
        schedule: {
            "out": 0.0,
            "out_square": 0.0,
            "in": 0.0,
            "in_square": 0.0,
            "max_parity_error": 0.0,
        }
        for schedule in schedules
    }
    vanilla_sum = 0.0
    vanilla_square = 0.0
    bridge_sum = 0.0
    bridge_square = 0.0

    completed = 0
    while completed < paths:
        batch = min(batch_size, paths - completed)
        increments = drift + diffusion * rng.standard_normal(
            (batch, fine_steps)
        )
        log_nodes = np.empty((batch, fine_steps + 1))
        log_nodes[:, 0] = log(spot)
        log_nodes[:, 1:] = log(spot) + np.cumsum(increments, axis=1)
        terminal = np.exp(log_nodes[:, -1])
        vanilla = discount * np.maximum(terminal - strike, 0.0)
        vanilla_sum += float(np.sum(vanilla))
        vanilla_square += float(np.dot(vanilla, vanilla))

        for schedule in schedules:
            indices = np.asarray(schedule_map[schedule], dtype=int)
            alive = np.min(log_nodes[:, indices], axis=1) > log_barrier
            out = vanilla * alive
            knock_in = vanilla - out
            parity_error = np.abs(vanilla - out - knock_in)
            totals[schedule]["out"] += float(np.sum(out))
            totals[schedule]["out_square"] += float(np.dot(out, out))
            totals[schedule]["in"] += float(np.sum(knock_in))
            totals[schedule]["in_square"] += float(
                np.dot(knock_in, knock_in)
            )
            totals[schedule]["max_parity_error"] = max(
                totals[schedule]["max_parity_error"],
                float(np.max(parity_error)),
            )

        left = log_nodes[:, :-1] - log_barrier
        right = log_nodes[:, 1:] - log_barrier
        endpoint_alive = (left > 0.0) & (right > 0.0)
        crossing_probability = np.ones_like(left)
        positive = endpoint_alive
        crossing_probability[positive] = np.exp(
            -2.0
            * left[positive]
            * right[positive]
            / (volatility**2 * dt)
        )
        interval_survival = np.where(
            endpoint_alive,
            1.0 - crossing_probability,
            0.0,
        )
        bridge_survival = np.prod(interval_survival, axis=1)
        bridge_out = vanilla * bridge_survival
        bridge_sum += float(np.sum(bridge_out))
        bridge_square += float(np.dot(bridge_out, bridge_out))
        completed += batch

    vanilla_mean, vanilla_se = _summary(
        vanilla_sum,
        vanilla_square,
        paths,
    )
    schedule_results: dict[str, dict[str, Any]] = {}
    for schedule in schedules:
        out_mean, out_se = _summary(
            totals[schedule]["out"],
            totals[schedule]["out_square"],
            paths,
        )
        in_mean, in_se = _summary(
            totals[schedule]["in"],
            totals[schedule]["in_square"],
            paths,
        )
        schedule_results[schedule] = {
            "observation_indices": schedule_map[schedule],
            "observation_count": len(schedule_map[schedule]),
            "knock_out_value": out_mean,
            "knock_out_standard_error": out_se,
            "knock_in_value": in_mean,
            "knock_in_standard_error": in_se,
            "in_out_value_residual": vanilla_mean - in_mean - out_mean,
            "maximum_pathwise_parity_error": totals[schedule][
                "max_parity_error"
            ],
        }
    bridge_mean, bridge_se = _summary(
        bridge_sum,
        bridge_square,
        paths,
    )
    analytic = analytic_down_and_out_call(
        spot=spot,
        strike=strike,
        barrier=barrier,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity,
    )
    reference_limit = (
        reference_sigma_multiple * bridge_se + reference_absolute_tolerance
    )
    reference_pass = abs(bridge_mean - analytic) <= reference_limit
    parity_pass = all(
        result["maximum_pathwise_parity_error"] <= parity_tolerance
        and abs(result["in_out_value_residual"])
        <= reference_sigma_multiple
        * (
            vanilla_se
            + result["knock_out_standard_error"]
            + result["knock_in_standard_error"]
        )
        + parity_tolerance
        for result in schedule_results.values()
    )
    ordering_pass = True
    for coarse, fine in zip(schedules, schedules[1:]):
        coarse_result = schedule_results[coarse]
        fine_result = schedule_results[fine]
        uncertainty = reference_sigma_multiple * (
            coarse_result["knock_out_standard_error"]
            + fine_result["knock_out_standard_error"]
        )
        if coarse_result["knock_out_value"] + uncertainty < fine_result["knock_out_value"]:
            ordering_pass = False
    status = "PASS" if reference_pass and parity_pass and ordering_pass else "REJECT"
    return {
        "vanilla_value": vanilla_mean,
        "vanilla_standard_error": vanilla_se,
        "discrete": schedule_results,
        "bridge_continuous_value": bridge_mean,
        "bridge_standard_error": bridge_se,
        "analytic_continuous_value": analytic,
        "bridge_reference_error": bridge_mean - analytic,
        "bridge_reference_limit": reference_limit,
        "reference_gate_status": "PASS" if reference_pass else "REJECT",
        "pathwise_parity_gate_status": "PASS" if parity_pass else "REJECT",
        "schedule_ordering_gate_status": "PASS" if ordering_pass else "REJECT",
        "fine_steps": fine_steps,
        "paths": paths,
        "seed": seed,
        "rebate": 0.0,
        "monitoring": "explicit discrete index schedules plus continuous bridge",
        "observation_time_zero_included": False,
        "rebate_timing": "none_zero_rebate",
        "status": status,
    }


def down_and_out_call_mc(
    *,
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    volatility: float,
    maturity: float,
    steps: int,
    paths: int,
    seed: int = 43,
) -> tuple[float, float]:
    """Compatibility wrapper for one discretely monitored schedule."""

    result = barrier_monitoring_experiment(
        spot=spot,
        strike=strike,
        barrier=barrier,
        rate=rate,
        dividend_yield=0.0,
        volatility=volatility,
        maturity=maturity,
        fine_steps=steps,
        paths=paths,
        monitoring_indices={"requested": tuple(range(1, steps + 1))},
        seed=seed,
    )
    record = result["discrete"]["requested"]
    return (
        float(record["knock_out_value"]),
        float(record["knock_out_standard_error"]),
    )
