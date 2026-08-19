"""Stable heat-equation time stepping and Rannacher evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class HeatSolveResult:
    """Structured finite-difference output and time-accounting evidence."""

    values: np.ndarray
    cfl: float
    stability_margin: float | None
    final_time: float
    backward_euler_half_steps: int
    crank_nicolson_steps: int


def _heat_inputs(
    initial: ArrayLike,
    diffusivity: float,
    spacing: float,
    time_step: float,
    steps: int,
) -> np.ndarray:
    values = np.asarray(initial, dtype=float)
    if (
        values.ndim != 1
        or values.size < 3
        or not np.all(np.isfinite(values))
        or not math.isfinite(diffusivity)
        or not math.isfinite(spacing)
        or not math.isfinite(time_step)
        or diffusivity < 0.0
        or spacing <= 0.0
        or time_step <= 0.0
        or steps < 0
    ):
        raise ValueError("finite grid and nonnegative diffusion controls required")
    return values.copy()


def explicit_heat(
    initial: ArrayLike,
    diffusivity: float,
    spacing: float,
    time_step: float,
    steps: int,
) -> HeatSolveResult:
    """Advance the heat equation only inside the discrete stability gate."""

    values = _heat_inputs(initial, diffusivity, spacing, time_step, steps)
    cfl = diffusivity * time_step / spacing**2
    if not 0.0 <= cfl <= 0.5:
        raise ValueError("explicit CFL must lie in [0, 0.5]")
    for _ in range(steps):
        previous = values.copy()
        values[1:-1] = (
            previous[1:-1]
            + cfl
            * (previous[2:] - 2.0 * previous[1:-1] + previous[:-2])
        )
    return HeatSolveResult(
        values,
        cfl,
        0.5 - cfl,
        steps * time_step,
        0,
        0,
    )


def _theta_heat_step(
    values: np.ndarray,
    cfl: float,
    theta: float,
) -> np.ndarray:
    """Apply one constant-boundary theta step to the interior grid."""

    interior_count = values.size - 2
    explicit_weight = (1.0 - theta) * cfl
    rhs = (
        (1.0 - 2.0 * explicit_weight) * values[1:-1]
        + explicit_weight * (values[:-2] + values[2:])
    )
    implicit_weight = theta * cfl
    rhs[0] += implicit_weight * values[0]
    rhs[-1] += implicit_weight * values[-1]
    bands = np.zeros((3, interior_count))
    bands[0, 1:] = -implicit_weight
    bands[1] = 1.0 + 2.0 * implicit_weight
    bands[2, :-1] = -implicit_weight
    updated = values.copy()
    updated[1:-1] = solve_banded((1, 1), bands, rhs)
    return updated


def crank_nicolson_heat(
    initial: ArrayLike,
    diffusivity: float,
    spacing: float,
    time_step: float,
    steps: int,
    rannacher_full_steps: int = 0,
) -> HeatSolveResult:
    """Use two BE half-steps for each replaced Rannacher startup step."""

    values = _heat_inputs(initial, diffusivity, spacing, time_step, steps)
    if not 0 <= rannacher_full_steps <= steps:
        raise ValueError("Rannacher startup must fit inside the time grid")
    cfl = diffusivity * time_step / spacing**2
    for _ in range(rannacher_full_steps):
        values = _theta_heat_step(values, 0.5 * cfl, theta=1.0)
        values = _theta_heat_step(values, 0.5 * cfl, theta=1.0)
    for _ in range(steps - rannacher_full_steps):
        values = _theta_heat_step(values, cfl, theta=0.5)
    return HeatSolveResult(
        values,
        cfl,
        None,
        steps * time_step,
        2 * rannacher_full_steps,
        steps - rannacher_full_steps,
    )


def rannacher_comparison(
    initial: ArrayLike,
    diffusivity: float,
    spacing: float,
    time_step: float,
    steps: int,
    *,
    rannacher_full_steps: int = 1,
) -> dict[str, float | int | np.ndarray]:
    """Compare CN and Rannacher curvature on the same final horizon."""

    crank_nicolson = crank_nicolson_heat(
        initial,
        diffusivity,
        spacing,
        time_step,
        steps,
    )
    rannacher = crank_nicolson_heat(
        initial,
        diffusivity,
        spacing,
        time_step,
        steps,
        rannacher_full_steps=rannacher_full_steps,
    )
    cn_gamma = np.diff(crank_nicolson.values, n=2) / spacing**2
    rannacher_gamma = np.diff(rannacher.values, n=2) / spacing**2

    def gamma_metrics(gamma: np.ndarray) -> tuple[float, int]:
        total_variation = float(
            np.sum(np.abs(np.diff(gamma))) * spacing
        )
        material = gamma[np.abs(gamma) > 1e-10]
        sign_flips = int(
            np.sum(np.diff(np.sign(material)) != 0)
            if material.size > 1
            else 0
        )
        return total_variation, sign_flips

    cn_variation, cn_sign_flips = gamma_metrics(cn_gamma)
    rannacher_variation, rannacher_sign_flips = gamma_metrics(
        rannacher_gamma
    )
    return {
        "crank_nicolson": crank_nicolson.values,
        "rannacher": rannacher.values,
        "final_time": rannacher.final_time,
        "rannacher_full_steps": rannacher_full_steps,
        "cn_gamma_total_variation": cn_variation,
        "rannacher_gamma_total_variation": rannacher_variation,
        "cn_gamma_sign_flips": cn_sign_flips,
        "rannacher_gamma_sign_flips": rannacher_sign_flips,
        "cn_minimum_value": float(np.min(crank_nicolson.values)),
        "rannacher_minimum_value": float(np.min(rannacher.values)),
        "gamma_variation_reduction": cn_variation - rannacher_variation,
        "cn_curvature_oscillation": cn_variation,
        "rannacher_curvature_oscillation": rannacher_variation,
    }


def rannacher_startup_study(
    initial: ArrayLike,
    diffusivity: float,
    spacing: float,
    time_step: float,
    steps: int,
) -> list[dict[str, float | int]]:
    """Compare zero, one and two replaced Rannacher full steps."""

    rows: list[dict[str, float | int]] = []
    for startup_steps in range(3):
        evidence = rannacher_comparison(
            initial,
            diffusivity,
            spacing,
            time_step,
            steps,
            rannacher_full_steps=min(startup_steps, steps),
        )
        rows.append(
            {
                "rannacher_full_steps": startup_steps,
                "gamma_total_variation": float(
                    evidence["rannacher_gamma_total_variation"]
                ),
                "gamma_sign_flips": int(
                    evidence["rannacher_gamma_sign_flips"]
                ),
                "minimum_value": float(
                    evidence["rannacher_minimum_value"]
                ),
            }
        )
    return rows
