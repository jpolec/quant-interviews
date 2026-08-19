"""Barrier-contract semantics and finite-difference evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from math import exp, log, sqrt

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import solve_banded
from scipy.special import ndtr


def barrier_terminal_grid(
    spots: ArrayLike,
    strike: float,
    barrier: float,
    *,
    rebate: float = 0.0,
) -> np.ndarray:
    """Apply an inclusive down-and-out terminal contract."""

    grid = np.asarray(spots, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 3
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
        or not all(math.isfinite(value) for value in (strike, barrier, rebate))
        or strike <= 0.0
        or barrier <= 0.0
        or rebate < 0.0
    ):
        raise ValueError("increasing finite grid and valid contract required")
    return np.where(
        grid <= barrier,
        rebate,
        np.maximum(grid - strike, 0.0),
    )


def enforce_absorbing_boundary(
    values: ArrayLike,
    rebate: float = 0.0,
) -> np.ndarray:
    """Set the aligned inclusive barrier node."""

    result = np.asarray(values, dtype=float).copy()
    if (
        result.ndim != 1
        or result.size < 3
        or not np.all(np.isfinite(result))
        or not math.isfinite(rebate)
        or rebate < 0.0
    ):
        raise ValueError("finite value grid and nonnegative rebate required")
    result[0] = rebate
    return result


def analytic_down_and_out_call(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
) -> float:
    """Closed-form continuous down-and-out call for barrier below strike."""

    if (
        not all(
            math.isfinite(value)
            for value in (
                spot,
                strike,
                barrier,
                rate,
                dividend,
                volatility,
                maturity,
            )
        )
        or not 0.0 < barrier < min(spot, strike)
        or volatility <= 0.0
        or maturity <= 0.0
    ):
        raise ValueError("requires 0 < barrier < min(spot, strike)")
    standard_deviation = volatility * sqrt(maturity)
    mu = (rate - dividend) / volatility**2 - 0.5
    mu_sigma = (1.0 + mu) * standard_deviation
    x1 = log(spot / strike) / standard_deviation + mu_sigma
    y1 = log(barrier**2 / (spot * strike)) / standard_deviation + mu_sigma
    vanilla = (
        spot * exp(-dividend * maturity) * ndtr(x1)
        - strike * exp(-rate * maturity) * ndtr(x1 - standard_deviation)
    )
    ratio = barrier / spot
    image = (
        spot
        * exp(-dividend * maturity)
        * ratio ** (2.0 * (mu + 1.0))
        * ndtr(y1)
        - strike
        * exp(-rate * maturity)
        * ratio ** (2.0 * mu)
        * ndtr(y1 - standard_deviation)
    )
    return float(vanilla - image)


def analytic_down_and_out_greeks(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    *,
    spot_bump: float = 0.05,
) -> tuple[float, float, float]:
    """Return independent formula price plus central-difference Greeks."""

    if (
        not math.isfinite(spot_bump)
        or spot_bump <= 0.0
        or spot - spot_bump <= barrier
    ):
        raise ValueError("positive bump must keep both spots above barrier")
    prices = [
        analytic_down_and_out_call(
            shifted_spot,
            strike,
            barrier,
            rate,
            dividend,
            volatility,
            maturity,
        )
        for shifted_spot in (
            spot - spot_bump,
            spot,
            spot + spot_bump,
        )
    ]
    delta = (prices[2] - prices[0]) / (2.0 * spot_bump)
    gamma = (prices[2] - 2.0 * prices[1] + prices[0]) / spot_bump**2
    return prices[1], delta, gamma


@dataclass(frozen=True)
class BarrierPDEResult:
    """Numerical result plus the boundary and grid contract."""

    price: float
    delta: float
    gamma: float
    spots: np.ndarray
    values: np.ndarray
    barrier_aligned: bool
    monitoring: str
    barrier_inclusive: bool
    rebate_timing: str
    space_steps: int
    time_steps: int
    s_max: float


def _barrier_value(
    rebate: float,
    rate: float,
    time_to_expiry: float,
    timing: str,
) -> float:
    if (
        not all(math.isfinite(value) for value in (rebate, rate, time_to_expiry))
        or rebate < 0.0
        or time_to_expiry < 0.0
    ):
        raise ValueError("finite nonnegative rebate and time are required")
    if timing == "immediate":
        return rebate
    if timing == "maturity":
        return rebate * exp(-rate * time_to_expiry)
    raise ValueError("rebate timing must be immediate or maturity")


def _validate_barrier_pde_contract(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    s_max: float,
    space_steps: int,
    time_steps: int,
    rannacher_full_steps: int,
    rebate: float,
    rebate_timing: str,
    monitoring: str,
) -> None:
    scalars = (
        spot,
        strike,
        barrier,
        rate,
        dividend,
        volatility,
        maturity,
        s_max,
        rebate,
    )
    if (
        not all(math.isfinite(value) for value in scalars)
        or not 0.0 < barrier < spot < s_max
        or strike <= 0.0
        or volatility <= 0.0
        or maturity <= 0.0
        or rebate < 0.0
        or monitoring != "continuous"
        or rebate_timing not in {"immediate", "maturity"}
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (space_steps, time_steps, rannacher_full_steps)
        )
        or space_steps < 20
        or time_steps < 2
        or not 0 <= rannacher_full_steps <= time_steps
    ):
        raise ValueError(
            "finite continuous barrier contract and valid integer grid required"
        )


def _theta_step(
    values: np.ndarray,
    spots: np.ndarray,
    time_step: float,
    theta: float,
    rate: float,
    dividend: float,
    volatility: float,
    old_barrier: float,
    new_barrier: float,
    old_far: float,
    new_far: float,
) -> np.ndarray:
    """Advance one Black-Scholes theta step with explicit boundary terms."""

    spacing = spots[1] - spots[0]
    interior = spots[1:-1]
    diffusion = 0.5 * volatility**2 * interior**2 / spacing**2
    convection = 0.5 * (rate - dividend) * interior / spacing
    lower = diffusion - convection
    diagonal = -2.0 * diffusion - rate
    upper = diffusion + convection

    old = values.copy()
    old[0], old[-1] = old_barrier, old_far
    operator_old = (
        lower * old[:-2]
        + diagonal * old[1:-1]
        + upper * old[2:]
    )
    rhs = old[1:-1] + (1.0 - theta) * time_step * operator_old
    rhs[0] += theta * time_step * lower[0] * new_barrier
    rhs[-1] += theta * time_step * upper[-1] * new_far

    bands = np.zeros((3, interior.size))
    bands[0, 1:] = -theta * time_step * upper[:-1]
    bands[1] = 1.0 - theta * time_step * diagonal
    bands[2, :-1] = -theta * time_step * lower[1:]
    updated = old.copy()
    updated[0], updated[-1] = new_barrier, new_far
    updated[1:-1] = solve_banded((1, 1), bands, rhs)
    return updated


def barrier_pde_algorithm_core(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    *,
    s_max: float,
    space_steps: int,
    time_steps: int,
    rannacher_full_steps: int,
    rebate: float,
    rebate_timing: str,
    monitoring: str = "continuous",
) -> BarrierPDEResult:
    """Execute aligned boundaries, Rannacher startup and CN evolution."""

    _validate_barrier_pde_contract(
        spot,
        strike,
        barrier,
        rate,
        dividend,
        volatility,
        maturity,
        s_max,
        space_steps,
        time_steps,
        rannacher_full_steps,
        rebate,
        rebate_timing,
        monitoring,
    )
    spots = np.linspace(barrier, s_max, space_steps + 1)
    values = enforce_absorbing_boundary(
        barrier_terminal_grid(
            spots,
            strike,
            barrier,
            rebate=rebate,
        ),
        rebate,
    )
    time_step = maturity / time_steps
    elapsed = 0.0

    def boundaries(time_to_expiry: float) -> tuple[float, float]:
        lower = _barrier_value(
            rebate,
            rate,
            time_to_expiry,
            rebate_timing,
        )
        upper = (
            s_max * exp(-dividend * time_to_expiry)
            - strike * exp(-rate * time_to_expiry)
        )
        return lower, max(upper, 0.0)

    def advance(step: float, theta: float) -> None:
        nonlocal elapsed, values
        old_lower, old_upper = boundaries(elapsed)
        new_elapsed = elapsed + step
        new_lower, new_upper = boundaries(new_elapsed)
        values = _theta_step(
            values,
            spots,
            step,
            theta,
            rate,
            dividend,
            volatility,
            old_lower,
            new_lower,
            old_upper,
            new_upper,
        )
        values = enforce_absorbing_boundary(values, new_lower)
        elapsed = new_elapsed

    for _ in range(rannacher_full_steps):
        advance(0.5 * time_step, 1.0)
        advance(0.5 * time_step, 1.0)
    for _ in range(time_steps - rannacher_full_steps):
        advance(time_step, 0.5)

    first_derivative = np.gradient(values, spots)
    second_derivative = np.gradient(first_derivative, spots)
    return BarrierPDEResult(
        float(np.interp(spot, spots, values)),
        float(np.interp(spot, spots, first_derivative)),
        float(np.interp(spot, spots, second_derivative)),
        spots,
        values,
        bool(spots[0] == barrier),
        monitoring,
        True,
        rebate_timing,
        space_steps,
        time_steps,
        s_max,
    )


def down_and_out_call_pde(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    *,
    s_max: float,
    space_steps: int,
    time_steps: int,
    rannacher_full_steps: int = 2,
    rebate: float = 0.0,
    rebate_timing: str = "immediate",
    monitoring: str = "continuous",
) -> BarrierPDEResult:
    """Price a continuously monitored inclusive down-and-out call."""

    _validate_barrier_pde_contract(
        spot,
        strike,
        barrier,
        rate,
        dividend,
        volatility,
        maturity,
        s_max,
        space_steps,
        time_steps,
        rannacher_full_steps,
        rebate,
        rebate_timing,
        monitoring,
    )
    return barrier_pde_algorithm_core(
        spot,
        strike,
        barrier,
        rate,
        dividend,
        volatility,
        maturity,
        s_max=s_max,
        space_steps=space_steps,
        time_steps=time_steps,
        rannacher_full_steps=rannacher_full_steps,
        rebate=rebate,
        rebate_timing=rebate_timing,
        monitoring=monitoring,
    )


def barrier_pde_refinement(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    *,
    s_max: float,
    space_steps: Sequence[int],
) -> list[dict[str, float | int | None]]:
    """Compare price and Greek refinements with an independent formula."""

    reference_price, reference_delta, reference_gamma = (
        analytic_down_and_out_greeks(
            spot,
            strike,
            barrier,
            rate,
            dividend,
            volatility,
            maturity,
        )
    )
    steps_values = list(space_steps)
    if (
        not steps_values
        or any(
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 20
            for steps in steps_values
        )
        or any(
            current <= previous
            for previous, current in zip(
                steps_values,
                steps_values[1:],
                strict=False,
            )
        )
    ):
        raise ValueError("space steps must be increasing integers of at least 20")
    rows: list[dict[str, float | int | None]] = []
    previous_steps: int | None = None
    previous_errors: dict[str, float] | None = None
    for steps in steps_values:
        result = down_and_out_call_pde(
            spot,
            strike,
            barrier,
            rate,
            dividend,
            volatility,
            maturity,
            s_max=s_max,
            space_steps=steps,
            time_steps=2 * steps,
        )
        errors = {
            "price": abs(result.price - reference_price),
            "delta": abs(result.delta - reference_delta),
            "gamma": abs(result.gamma - reference_gamma),
        }
        row: dict[str, float | int | None] = {
            "space_steps": steps,
            "time_steps": 2 * steps,
            "spatial_spacing": (s_max - barrier) / steps,
            "time_step": maturity / (2 * steps),
            "price": result.price,
            "delta": result.delta,
            "gamma": result.gamma,
            "absolute_price_error": errors["price"],
            "absolute_delta_error": errors["delta"],
            "absolute_gamma_error": errors["gamma"],
            "absolute_reference_error": errors["price"],
            "price_observed_order": None,
            "delta_observed_order": None,
            "gamma_observed_order": None,
        }
        if previous_steps is not None and previous_errors is not None:
            refinement = np.log(steps / previous_steps)
            for quantity in ("price", "delta", "gamma"):
                old_error = previous_errors[quantity]
                new_error = errors[quantity]
                if old_error > 0.0 and new_error > 0.0:
                    row[f"{quantity}_observed_order"] = float(
                        np.log(old_error / new_error) / refinement
                    )
        rows.append(row)
        previous_steps = steps
        previous_errors = errors
    return rows


def barrier_far_boundary_study(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    *,
    s_max_values: Sequence[float],
    spatial_spacing: float,
    time_steps: int,
) -> list[dict[str, float | int]]:
    """Vary only the far boundary while holding grid scales fixed."""

    if (
        not math.isfinite(spatial_spacing)
        or spatial_spacing <= 0.0
        or isinstance(time_steps, bool)
        or not isinstance(time_steps, int)
        or time_steps < 2
        or len(s_max_values) < 2
        or any(not math.isfinite(s_max) for s_max in s_max_values)
        or any(
            current <= previous
            for previous, current in zip(
                s_max_values,
                s_max_values[1:],
                strict=False,
            )
        )
    ):
        raise ValueError("positive grid controls and two boundaries required")
    rows: list[dict[str, float | int]] = []
    for s_max in s_max_values:
        steps = int(round((s_max - barrier) / spatial_spacing))
        if steps < 20:
            raise ValueError("far boundary must define a valid spatial grid")
        result = down_and_out_call_pde(
            spot,
            strike,
            barrier,
            rate,
            dividend,
            volatility,
            maturity,
            s_max=s_max,
            space_steps=steps,
            time_steps=time_steps,
        )
        rows.append(
            {
                "s_max": s_max,
                "space_steps": steps,
                "time_steps": time_steps,
                "spatial_spacing": (s_max - barrier) / steps,
                "time_step": maturity / time_steps,
                "price": result.price,
                "delta": result.delta,
                "gamma": result.gamma,
            }
        )
    return rows
