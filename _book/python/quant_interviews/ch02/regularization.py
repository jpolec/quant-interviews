"""Covariance shrinkage and nearest-correlation projection for Chapter 2."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _finite_square(matrix: ArrayLike, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or values.shape[0] == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"{name} must be a finite square matrix")
    return values


def linear_shrinkage(
    sample: ArrayLike,
    target: ArrayLike,
    intensity: float,
) -> np.ndarray:
    """Return a validated convex blend of sample and target covariance."""

    sample_covariance = _finite_square(sample, "sample covariance")
    target_covariance = _finite_square(target, "target covariance")
    if (
        sample_covariance.shape != target_covariance.shape
        or not np.isfinite(intensity)
        or not 0.0 <= intensity <= 1.0
    ):
        raise ValueError("matching matrices and intensity in [0, 1] required")
    return (
        (1.0 - intensity) * sample_covariance
        + intensity * target_covariance
    )


def _gmv_weights(covariance: np.ndarray) -> np.ndarray:
    ones = np.ones(covariance.shape[0])
    raw = np.linalg.solve(covariance, ones)
    denominator = float(ones @ raw)
    if denominator <= 0.0:
        raise ValueError("covariance does not define positive GMV normalization")
    return raw / denominator


def _gmv_evidence(
    covariance: np.ndarray,
) -> tuple[str, np.ndarray, float]:
    """Return GMV weights or an explicit unavailable state."""

    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    spectral_scale = max(
        float(np.max(np.abs(eigenvalues))),
        np.finfo(float).tiny,
    )
    floor = np.finfo(float).eps * covariance.shape[0] * spectral_scale
    if eigenvalues[0] <= floor:
        return (
            "UNAVAILABLE",
            np.full(covariance.shape[0], np.nan),
            float("nan"),
        )
    weights = _gmv_weights(covariance)
    return "AVAILABLE", weights, float(np.abs(weights).sum())


def oas_intensity(sample: np.ndarray, sample_count: int) -> float:
    """Return the Chen--Wiesel--Hero OAS intensity, clipped to [0, 1]."""

    sample = _finite_square(sample, "sample covariance")
    asset_count = sample.shape[0]
    if (
        isinstance(sample_count, (bool, np.bool_))
        or not isinstance(sample_count, (int, np.integer))
        or sample_count <= 0
        or not np.allclose(sample, sample.T, atol=1e-14, rtol=1e-10)
    ):
        raise ValueError("symmetric sample covariance and positive sample count required")
    trace = float(np.trace(sample))
    trace_squared = float(np.sum(sample * sample.T))
    dimension_correction = 2.0 / asset_count
    numerator = (1.0 - dimension_correction) * trace_squared + trace**2
    denominator = (sample_count + 1.0 - dimension_correction) * (
        trace_squared - trace**2 / asset_count
    )
    if denominator <= 0.0:
        return 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def oas_shrinkage_stability(
    returns: ArrayLike,
) -> dict[str, float | np.ndarray]:
    """Estimate OAS intensity and expose its portfolio-stability effect."""

    observations = np.asarray(returns, dtype=float)
    if (
        observations.ndim != 2
        or observations.shape[0] <= 2
        or observations.shape[1] <= 1
        or not np.all(np.isfinite(observations))
    ):
        raise ValueError("finite return panel with multiple rows/assets required")
    centered = observations - observations.mean(axis=0, keepdims=True)
    sample_count, asset_count = centered.shape
    sample = centered.T @ centered / sample_count
    target_level = float(np.trace(sample) / asset_count)
    intensity = oas_intensity(sample, sample_count)
    target = target_level * np.eye(asset_count)
    shrunk = linear_shrinkage(sample, target, intensity)
    sample_status, sample_weights, sample_leverage = _gmv_evidence(sample)
    shrunk_status, shrunk_weights, shrunk_leverage = _gmv_evidence(shrunk)
    sample_condition = (
        float("inf")
        if sample_status == "UNAVAILABLE"
        else float(np.linalg.cond(sample))
    )
    return {
        "sample_covariance": sample,
        "shrunk_covariance": shrunk,
        "target_covariance": target,
        "intensity": float(intensity),
        "sample_covariance_divisor": float(sample_count),
        "sample_condition_number": sample_condition,
        "shrunk_condition_number": float(np.linalg.cond(shrunk)),
        "sample_gmv_status": sample_status,
        "shrunk_gmv_status": shrunk_status,
        "sample_gmv_weights": sample_weights,
        "shrunk_gmv_weights": shrunk_weights,
        "sample_gross_leverage": sample_leverage,
        "shrunk_gross_leverage": shrunk_leverage,
    }


def dykstra_nearest_correlation_core(
    source: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, float, bool]:
    """Alternate exact PSD and unit-diagonal projections with correction."""

    if (
        source.ndim != 2
        or source.shape[0] != source.shape[1]
        or source.shape[0] == 0
        or not np.all(np.isfinite(source))
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
        or isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations <= 0
    ):
        raise ValueError("finite square input and positive controls required")
    current = source.copy()
    correction = np.zeros_like(source)
    relative_change = float("inf")
    for iteration in range(1, max_iterations + 1):
        residual = current - correction
        values, vectors = np.linalg.eigh(0.5 * (residual + residual.T))
        psd = (vectors * np.maximum(values, 0.0)) @ vectors.T
        correction = psd - residual
        next_matrix = psd.copy()
        np.fill_diagonal(next_matrix, 1.0)
        next_matrix = 0.5 * (next_matrix + next_matrix.T)
        relative_change = float(
            np.linalg.norm(next_matrix - current, ord="fro")
            / max(np.linalg.norm(next_matrix, ord="fro"), np.finfo(float).tiny)
        )
        current = next_matrix
        if relative_change <= tolerance:
            return current, iteration, relative_change, True
    return current, max_iterations, relative_change, False


def nearest_correlation(
    matrix: ArrayLike,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> dict[str, float | int | str | bool | np.ndarray]:
    """Project onto PSD and unit-diagonal sets with Dykstra correction."""

    source = _finite_square(matrix, "correlation input")
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0.0
        or isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations <= 0
        or not np.allclose(source, source.T, atol=tolerance, rtol=0.0)
    ):
        raise ValueError("symmetric input, positive tolerance and iterations required")
    source = 0.5 * (source + source.T)
    if not np.allclose(np.diag(source), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("correlation input must have unit diagonal")

    current, iteration, relative_change, converged = (
        dykstra_nearest_correlation_core(
            source,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    )

    eigenvalues = np.linalg.eigvalsh(current)
    minimum_eigenvalue = float(eigenvalues[0])
    scale = max(
        float(np.max(np.abs(eigenvalues))),
        np.finfo(float).tiny,
    )
    diagonal_error = float(
        np.max(np.abs(np.diag(current) - 1.0))
    )
    psd_violation = max(0.0, -minimum_eigenvalue)
    finite_output = bool(np.all(np.isfinite(current)))
    passed = (
        converged
        and finite_output
        and diagonal_error <= tolerance
        and psd_violation <= tolerance * scale
    )
    status = (
        "PASS"
        if passed
        else "CONDITIONAL_PASS"
        if finite_output
        and diagonal_error <= 10.0 * tolerance
        and psd_violation <= 10.0 * tolerance * scale
        else "REJECT"
    )
    return {
        "correlation": current,
        "converged": converged,
        "passed": passed,
        "status": status,
        "iterations": iteration,
        "relative_change": relative_change,
        "minimum_eigenvalue": minimum_eigenvalue,
        "diagonal_error": diagonal_error,
        "psd_violation": psd_violation,
        "finite_output": finite_output,
        "repair_frobenius_norm": float(
            np.linalg.norm(current - source, ord="fro")
        ),
    }
