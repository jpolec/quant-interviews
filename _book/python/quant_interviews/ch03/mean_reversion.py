"""Exact and Euler evidence for a forced mean-reverting ODE."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np


def exact_mean_reversion_step(
    state: float,
    kappa: float,
    theta: float,
    forcing: float,
    time_step: float,
) -> float:
    """Advance x'=kappa(theta-x)+forcing with stable small-step algebra."""

    if (
        not all(
            math.isfinite(value)
            for value in (state, kappa, theta, forcing, time_step)
        )
        or kappa < 0.0
        or time_step < 0.0
    ):
        raise ValueError("finite state and nonnegative kappa/time step required")
    if kappa == 0.0:
        return state + forcing * time_step
    scaled_step = kappa * time_step
    decay = math.exp(-scaled_step)
    one_minus_decay = -math.expm1(-scaled_step)
    forcing_weight = (
        time_step
        * (
            1.0
            - 0.5 * scaled_step
            + scaled_step**2 / 6.0
        )
        if abs(scaled_step) < 1e-8
        else one_minus_decay / kappa
    )
    return (
        state * decay
        + theta * one_minus_decay
        + forcing * forcing_weight
    )


def linear_mean_reversion(
    initial: float,
    kappa: float,
    theta: float,
    times: Sequence[float],
    forcing: float = 0.0,
) -> np.ndarray:
    """Evaluate the exact constant-forcing path on an increasing time grid."""

    grid = np.asarray(times, dtype=float)
    if (
        grid.ndim != 1
        or grid.size == 0
        or not np.all(np.isfinite(grid))
        or grid[0] < 0.0
        or np.any(np.diff(grid) < 0.0)
    ):
        raise ValueError("finite nonnegative increasing times required")
    values = np.empty_like(grid)
    values[0] = exact_mean_reversion_step(
        initial,
        kappa,
        theta,
        forcing,
        float(grid[0]),
    )
    for index in range(1, grid.size):
        values[index] = exact_mean_reversion_step(
            values[index - 1],
            kappa,
            theta,
            forcing,
            float(grid[index] - grid[index - 1]),
        )
    return values


def euler_mean_reversion(
    initial: float,
    kappa: float,
    theta: float,
    forcing: float,
    time_step: float,
    steps: int,
) -> np.ndarray:
    """Advance the same forced ODE with explicit Euler."""

    if (
        not all(
            math.isfinite(value)
            for value in (initial, kappa, theta, forcing, time_step)
        )
        or time_step <= 0.0
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 0
        or kappa < 0.0
    ):
        raise ValueError("positive step, nonnegative kappa and steps required")
    values = np.empty(steps + 1)
    values[0] = initial
    for index in range(steps):
        drift = kappa * (theta - values[index]) + forcing
        values[index + 1] = values[index] + time_step * drift
    return values


def mean_reversion_convergence(
    initial: float,
    kappa: float,
    theta: float,
    forcing: float,
    horizon: float,
    step_counts: Sequence[int],
) -> list[dict[str, float | int | str | None]]:
    """Report exact-versus-Euler error, order and stability regime."""

    counts = list(step_counts)
    if (
        not all(
            math.isfinite(value)
            for value in (initial, kappa, theta, forcing, horizon)
        )
        or kappa < 0.0
        or horizon <= 0.0
        or not counts
        or any(
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps <= 0
            for steps in counts
        )
        or any(
            current <= previous
            for previous, current in zip(counts, counts[1:], strict=False)
        )
    ):
        raise ValueError(
            "finite model inputs and positive increasing integer steps required"
        )
    exact = exact_mean_reversion_step(
        initial,
        kappa,
        theta,
        forcing,
        horizon,
    )
    rows: list[dict[str, float | int | str | None]] = []
    previous_error: float | None = None
    previous_steps: int | None = None
    physical_half_life = (
        float("inf") if kappa == 0.0 else math.log(2.0) / kappa
    )
    for steps in counts:
        time_step = horizon / steps
        terminal = euler_mean_reversion(
            initial,
            kappa,
            theta,
            forcing,
            time_step,
            steps,
        )[-1]
        error = abs(terminal - exact)
        multiplier = 1.0 - kappa * time_step
        multiplier_magnitude = abs(multiplier)
        regime = (
            "STABLE"
            if 0.0 <= multiplier <= 1.0
            else "OSCILLATORY"
            if -1.0 < multiplier < 0.0
            else "UNSTABLE"
        )
        effective_half_life: float | None
        if multiplier_magnitude == 0.0:
            effective_half_life = 0.0
        elif multiplier_magnitude < 1.0:
            effective_half_life = (
                -time_step * math.log(2.0) / math.log(multiplier_magnitude)
            )
        elif multiplier_magnitude == 1.0:
            effective_half_life = float("inf")
        else:
            effective_half_life = None
        rows.append(
            {
                "steps": steps,
                "time_step": time_step,
                "exact_terminal": exact,
                "euler_terminal": terminal,
                "absolute_error": error,
                "observed_order": (
                    None
                    if (
                        previous_error is None
                        or previous_steps is None
                        or previous_error == 0.0
                        or error == 0.0
                    )
                    else math.log(previous_error / error)
                    / math.log(steps / previous_steps)
                ),
                "physical_half_life": physical_half_life,
                "effective_half_life": effective_half_life,
                "stability": regime,
            }
        )
        previous_error = error
        previous_steps = steps
    return rows
