"""Quadrature rules and observable convergence evidence for Chapter 3."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

import numpy as np
from numpy.typing import ArrayLike


def _validated_grid(lower: float, upper: float, intervals: int) -> np.ndarray:
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or not lower < upper
        or isinstance(intervals, bool)
        or not isinstance(intervals, int)
        or intervals <= 0
    ):
        raise ValueError("finite ordered interval and positive intervals required")
    return np.linspace(lower, upper, intervals + 1)


def composite_trapezoid(
    function: Callable[[np.ndarray], ArrayLike],
    lower: float,
    upper: float,
    intervals: int,
) -> float:
    """Apply the composite trapezoid rule."""

    grid = _validated_grid(lower, upper, intervals)
    values = np.asarray(function(grid), dtype=float)
    if values.shape != grid.shape or not np.all(np.isfinite(values)):
        raise ValueError("integrand must return one finite value per grid point")
    return float(np.trapezoid(values, grid))


def composite_simpson(
    function: Callable[[np.ndarray], ArrayLike],
    lower: float,
    upper: float,
    intervals: int,
) -> float:
    """Apply composite Simpson quadrature on an even grid."""

    if intervals % 2:
        raise ValueError("Simpson intervals must be even")
    grid = _validated_grid(lower, upper, intervals)
    values = np.asarray(function(grid), dtype=float)
    if values.shape != grid.shape or not np.all(np.isfinite(values)):
        raise ValueError("integrand must return one finite value per grid point")
    spacing = (upper - lower) / intervals
    return float(
        spacing
        / 3.0
        * (
            values[0]
            + values[-1]
            + 4.0 * values[1:-1:2].sum()
            + 2.0 * values[2:-1:2].sum()
        )
    )


def singularity_transform_integral(
    function: Callable[[np.ndarray], ArrayLike],
    lower: float,
    upper: float,
    intervals: int,
    *,
    transformed_endpoint_limit: float,
) -> float:
    """Use x=a+(b-a)u² without evaluating a singular endpoint."""

    _validated_grid(lower, upper, intervals)
    if not math.isfinite(transformed_endpoint_limit):
        raise ValueError("finite transformed endpoint limit required")
    unit_grid = _validated_grid(0.0, 1.0, intervals)
    transformed = lower + (upper - lower) * unit_grid[1:] ** 2
    transformed_values = np.asarray(function(transformed), dtype=float)
    if transformed_values.shape != transformed.shape:
        raise ValueError("integrand must return one value per transformed point")
    values = np.empty_like(unit_grid)
    values[0] = transformed_endpoint_limit
    values[1:] = (
        transformed_values
        * 2.0
        * (upper - lower)
        * unit_grid[1:]
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("transformed integrand must be finite away from endpoint")
    return float(np.trapezoid(values, unit_grid))


def quadrature_convergence_table(
    method: Callable[[Callable[[np.ndarray], ArrayLike], float, float, int], float],
    function: Callable[[np.ndarray], ArrayLike],
    lower: float,
    upper: float,
    exact: float,
    interval_counts: Sequence[int],
) -> list[dict[str, float | int | None]]:
    """Report estimates, errors and mesh-ratio-aware observed orders."""

    rows: list[dict[str, float | int | None]] = []
    previous_error: float | None = None
    previous_intervals: int | None = None
    counts = list(interval_counts)
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or not lower < upper
        or not math.isfinite(exact)
        or not counts
        or any(
            isinstance(intervals, bool)
            or not isinstance(intervals, int)
            or intervals <= 0
            for intervals in counts
        )
        or any(
            current <= previous
            for previous, current in zip(counts, counts[1:], strict=False)
        )
    ):
        raise ValueError("interval counts must be positive and strictly increasing")
    error_floor = 10.0 * np.finfo(float).eps * max(1.0, abs(exact))
    for intervals in counts:
        estimate = float(method(function, lower, upper, intervals))
        if not math.isfinite(estimate):
            raise ValueError("quadrature estimates must be finite")
        error = abs(estimate - exact)
        order = (
            None
            if (
                previous_error is None
                or previous_intervals is None
                or previous_error <= error_floor
                or error <= error_floor
            )
            else math.log(previous_error / error)
            / math.log(intervals / previous_intervals)
        )
        rows.append(
            {
                "intervals": intervals,
                "estimate": estimate,
                "absolute_error": error,
                "observed_order": order,
            }
        )
        previous_error = error
        previous_intervals = intervals
    return rows
