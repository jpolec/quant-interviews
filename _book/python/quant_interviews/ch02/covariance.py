"""Covariance geometry and spectral diagnostics for Chapter 2."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _symmetric_square(
    matrix: ArrayLike,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    name: str,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or values.shape[0] == 0
        or not np.all(np.isfinite(values))
        or not np.isfinite(absolute_tolerance)
        or not np.isfinite(relative_tolerance)
        or absolute_tolerance < 0.0
        or relative_tolerance < 0.0
    ):
        raise ValueError(f"{name} must be a finite square matrix")
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    symmetry_tolerance = absolute_tolerance + relative_tolerance * scale
    if float(np.max(np.abs(values - values.T))) > symmetry_tolerance:
        raise ValueError(f"{name} must be symmetric within tolerance")
    return 0.5 * (values + values.T)


def covariance_psd_certificate(
    matrix: ArrayLike,
    *,
    absolute_tolerance: float = 1e-14,
    relative_tolerance: float = 1e-10,
) -> dict[str, float | int | bool]:
    """Return an absolute-plus-relative PSD covariance certificate."""

    if (
        not np.isfinite(absolute_tolerance)
        or not np.isfinite(relative_tolerance)
        or absolute_tolerance < 0.0
        or relative_tolerance < 0.0
    ):
        raise ValueError("nonnegative absolute and relative tolerances required")
    covariance = _symmetric_square(
        matrix,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        name="covariance",
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    spectral_scale = float(np.max(np.abs(eigenvalues)))
    numerical_floor = (
        absolute_tolerance + relative_tolerance * spectral_scale
    )
    return {
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "trace": float(np.trace(covariance)),
        "effective_rank": int(np.sum(eigenvalues > numerical_floor)),
        "is_psd": bool(eigenvalues[0] >= -numerical_floor),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "numerical_floor": numerical_floor,
    }


def equicorrelation_eigenvalues(n: int, rho: float) -> np.ndarray:
    """Return the common-mode and relative-mode eigenvalues."""

    if (
        isinstance(n, (bool, np.bool_))
        or not isinstance(n, (int, np.integer))
        or n < 2
        or not np.isfinite(rho)
    ):
        raise ValueError("rho must define a valid equicorrelation matrix")
    n = int(n)
    lower_bound = -1.0 / (n - 1)
    if not lower_bound <= rho <= 1.0:
        raise ValueError("rho must define a valid equicorrelation matrix")
    return np.array(
        [1.0 + (n - 1) * rho] + [1.0 - rho] * (n - 1),
        dtype=float,
    )


def equicorrelation_condition_number(n: int, rho: float) -> float:
    """Return the spectral condition number, including negative rho."""

    eigenvalues = equicorrelation_eigenvalues(n, rho)
    minimum = float(np.min(eigenvalues))
    if minimum <= 0.0:
        return float("inf")
    return float(np.max(eigenvalues) / minimum)


def condition_diagnostics(matrix: ArrayLike) -> dict[str, float | int]:
    """Return singular-value diagnostics without forming an inverse."""

    values = np.asarray(matrix, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("matrix must be finite and two-dimensional")
    singular_values = np.linalg.svd(values, compute_uv=False)
    maximum = float(singular_values[0])
    minimum = float(singular_values[-1])
    threshold = np.finfo(float).eps * max(values.shape) * maximum
    effective_rank = int(np.sum(singular_values > threshold))
    condition_number = (
        float("inf") if minimum <= threshold else maximum / minimum
    )
    return {
        "condition_number": condition_number,
        "minimum_singular": minimum,
        "maximum_singular": maximum,
        "effective_rank": effective_rank,
    }


def governed_pca_components(
    raw_eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    weights: np.ndarray,
    *,
    numerical_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clip only numerical negatives and compute PCA risk contributions."""

    raw_eigenvalues = np.asarray(raw_eigenvalues, dtype=float)
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    weights = np.asarray(weights, dtype=float)
    component_count = raw_eigenvalues.size
    if (
        raw_eigenvalues.shape != (component_count,)
        or eigenvectors.shape != (component_count, component_count)
        or weights.shape != (component_count,)
        or component_count == 0
        or not np.all(np.isfinite(raw_eigenvalues))
        or not np.all(np.isfinite(eigenvectors))
        or not np.all(np.isfinite(weights))
        or not np.isfinite(numerical_floor)
        or numerical_floor < 0.0
    ):
        raise ValueError("finite aligned PCA inputs and numerical floor required")
    if float(np.min(raw_eigenvalues)) < -numerical_floor:
        raise ValueError("materially indefinite covariance")
    used_eigenvalues = np.maximum(raw_eigenvalues, 0.0)
    order = np.argsort(used_eigenvalues)[::-1]
    ordered_raw = raw_eigenvalues[order]
    ordered_used = used_eigenvalues[order]
    ordered_vectors = eigenvectors[:, order]
    exposures = ordered_vectors.T @ weights
    contributions = ordered_used * exposures**2
    return ordered_raw, ordered_used, exposures, contributions


def pca_risk_decomposition(
    covariance: ArrayLike,
    weights: ArrayLike,
    *,
    absolute_tolerance: float = 1e-14,
    relative_tolerance: float = 1e-10,
) -> dict[str, np.ndarray | float]:
    """Decompose variance without treating numerical negatives as risk."""

    if (
        not np.isfinite(absolute_tolerance)
        or not np.isfinite(relative_tolerance)
        or absolute_tolerance < 0.0
        or relative_tolerance < 0.0
    ):
        raise ValueError("nonnegative absolute and relative tolerances required")
    matrix = _symmetric_square(
        covariance,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        name="covariance",
    )
    portfolio = np.asarray(weights, dtype=float)
    if portfolio.shape != (matrix.shape[0],) or not np.all(
        np.isfinite(portfolio)
    ):
        raise ValueError("weights must be finite and aligned to covariance")
    raw_eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    spectral_scale = float(np.max(np.abs(raw_eigenvalues)))
    numerical_floor = (
        absolute_tolerance + relative_tolerance * spectral_scale
    )
    if raw_eigenvalues[0] < -numerical_floor:
        raise ValueError("covariance must be positive semidefinite")
    (
        raw_eigenvalues,
        used_eigenvalues,
        exposures,
        contributions,
    ) = governed_pca_components(
        raw_eigenvalues,
        eigenvectors,
        portfolio,
        numerical_floor=numerical_floor,
    )
    direct_variance = float(portfolio @ matrix @ portfolio)
    reconstructed_variance = float(contributions.sum())
    return {
        "raw_eigenvalues": raw_eigenvalues,
        "used_eigenvalues": used_eigenvalues,
        "factor_exposures": exposures,
        "variance_contributions": contributions,
        "direct_variance": direct_variance,
        "reconstructed_variance": reconstructed_variance,
        "reconciliation_error": abs(
            reconstructed_variance - direct_variance
        ),
        "numerical_repair_norm": float(
            np.linalg.norm(used_eigenvalues - raw_eigenvalues)
        ),
        "total_variance": reconstructed_variance,
    }
