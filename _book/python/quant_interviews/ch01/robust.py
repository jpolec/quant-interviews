"""Robust location estimators for Chapter 1."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike


def median_mad(values: ArrayLike) -> dict[str, float]:
    """Return the median, MAD and normal-consistent robust scale."""

    observations = np.asarray(values, dtype=float)
    if (
        observations.ndim != 1
        or observations.size == 0
        or not np.all(np.isfinite(observations))
    ):
        raise ValueError("a finite one-dimensional sample is required")
    median = float(np.median(observations))
    mad = float(np.median(np.abs(observations - median)))
    return {
        "median": median,
        "mad": mad,
        "sigma_consistent": 1.4826 * mad,
    }


def huber_location(
    values: ArrayLike,
    tuning: float = 1.345,
    iterations: int = 50,
    *,
    scale: float | None = None,
    tolerance: float = 1e-12,
    scale_floor: float = 1e-12,
) -> dict[str, float | int | bool | str]:
    """Return a Huber estimate and explicit convergence diagnostics."""

    observations = np.asarray(values, dtype=float)
    if (
        not np.isfinite(tuning)
        or not np.isfinite(tolerance)
        or not np.isfinite(scale_floor)
        or tuning <= 0.0
        or isinstance(iterations, bool)
        or not isinstance(iterations, Integral)
        or iterations <= 0
        or tolerance <= 0.0
        or scale_floor <= 0.0
        or observations.ndim != 1
        or observations.size == 0
        or not np.all(np.isfinite(observations))
    ):
        raise ValueError("valid sample and positive solver controls are required")

    location = float(np.median(observations))
    if scale is None:
        effective_scale = max(
            float(1.4826 * np.median(np.abs(observations - location))),
            scale_floor,
        )
        scale_method = "normal_consistent_mad"
    else:
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        effective_scale = float(scale)
        scale_method = "caller_supplied"

    final_step = float("inf")
    for iteration in range(1, iterations + 1):
        standardized = (observations - location) / effective_scale
        weights = np.ones_like(standardized)
        outliers = np.abs(standardized) > tuning
        weights[outliers] = tuning / np.abs(standardized[outliers])
        updated_location = float(
            np.sum(weights * observations) / np.sum(weights)
        )
        final_step = abs(updated_location - location)
        location = updated_location
        if final_step <= tolerance:
            return {
                "estimate": location,
                "scale": effective_scale,
                "scale_method": scale_method,
                "iterations_used": iteration,
                "converged": True,
                "final_step": final_step,
                "tolerance": tolerance,
            }
    return {
        "estimate": location,
        "scale": effective_scale,
        "scale_method": scale_method,
        "iterations_used": iterations,
        "converged": False,
        "final_step": final_step,
        "tolerance": tolerance,
    }
