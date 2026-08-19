"""Structured covariance solves and governed simulation for Chapter 2."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def woodbury_factor_solve_core(
    specific_variance: np.ndarray,
    loadings: np.ndarray,
    factor_root: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the low-rank solve after inputs and the factor root are validated."""

    reduced_loadings = loadings @ factor_root
    scaled_loadings = reduced_loadings / specific_variance[:, None]
    specific_solution = rhs / specific_variance
    factor_system = (
        np.eye(factor_root.shape[1])
        + reduced_loadings.T @ scaled_loadings
    )
    adjustment = np.linalg.solve(
        factor_system,
        reduced_loadings.T @ specific_solution,
    )
    return specific_solution - scaled_loadings @ adjustment, factor_system


def woodbury_solve(
    specific_variance: ArrayLike,
    loadings: ArrayLike,
    factor_covariance: ArrayLike,
    rhs: ArrayLike,
    *,
    tolerance: float = 1e-10,
) -> dict[str, float | int | np.ndarray]:
    """Solve (D + B F B.T)x = rhs without forming any inverse."""

    diagonal = np.asarray(specific_variance, dtype=float)
    exposures = np.asarray(loadings, dtype=float)
    factor = np.asarray(factor_covariance, dtype=float)
    target = np.asarray(rhs, dtype=float)
    if (
        diagonal.ndim != 1
        or exposures.ndim != 2
        or factor.ndim != 2
        or factor.shape[0] != factor.shape[1]
        or exposures.shape != (diagonal.size, factor.shape[0])
        or target.shape != (diagonal.size,)
        or not np.all(np.isfinite(diagonal))
        or not np.all(np.isfinite(exposures))
        or not np.all(np.isfinite(factor))
        or not np.all(np.isfinite(target))
        or np.any(diagonal <= 0.0)
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("finite aligned inputs and positive specific risk required")
    factor_scale = max(float(np.max(np.abs(factor))), np.finfo(float).tiny)
    if float(np.max(np.abs(factor - factor.T))) > tolerance * factor_scale:
        raise ValueError("factor covariance must be symmetric")

    factor_values, factor_vectors = np.linalg.eigh(0.5 * (factor + factor.T))
    scale = max(
        float(np.max(np.abs(factor_values))),
        np.finfo(float).tiny,
    )
    if factor_values[0] < -tolerance * scale:
        raise ValueError("factor covariance must be positive semidefinite")
    retained = factor_values > tolerance * scale
    factor_root = (
        factor_vectors[:, retained]
        * np.sqrt(factor_values[retained])
    )
    if not np.any(retained):
        solution = target / diagonal
        factor_system = np.eye(0)
        factor_system_condition = 1.0
    else:
        solution, factor_system = woodbury_factor_solve_core(
            diagonal,
            exposures,
            factor_root,
            target,
        )
        factor_system_condition = float(np.linalg.cond(factor_system))

    covariance = (
        np.diag(diagonal)
        + exposures @ factor @ exposures.T
    )
    residual = covariance @ solution - target
    relative_residual = np.linalg.norm(residual) / max(
        np.linalg.norm(target),
        np.finfo(float).tiny,
    )
    return {
        "solution": solution,
        "relative_residual": float(relative_residual),
        "factor_system_condition_number": factor_system_condition,
        "minimum_specific_variance": float(diagonal.min()),
        "factor_rank": int(retained.sum()),
        "factor_exposure": exposures.T @ solution,
    }


def correlation_eigen_root(
    correlation: np.ndarray,
    *,
    numerical_tolerance: float,
) -> dict[str, float | int | str | np.ndarray]:
    """Build an auditable PSD correlation root after structural validation."""

    correlation = np.asarray(correlation, dtype=float)
    if (
        correlation.ndim != 2
        or correlation.shape[0] != correlation.shape[1]
        or correlation.shape[0] == 0
        or not np.all(np.isfinite(correlation))
        or not np.isfinite(numerical_tolerance)
        or numerical_tolerance < 0.0
    ):
        raise ValueError("finite square correlation and tolerance required")
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    if eigenvalues[0] < -numerical_tolerance:
        raise ValueError("materially indefinite correlation requires governance")
    used = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(used), np.finfo(float).tiny))
    used = used / np.outer(diagonal, diagonal)
    np.fill_diagonal(used, 1.0)
    used = 0.5 * (used + used.T)
    used_values, used_vectors = np.linalg.eigh(used)
    if used_values[0] > numerical_tolerance:
        root, method = np.linalg.cholesky(used), "cholesky"
    else:
        root = used_vectors @ np.diag(np.sqrt(np.maximum(used_values, 0.0)))
        method = "eigen-root"
    return {
        "root": root,
        "correlation": used,
        "minimum_eigenvalue": float(used_values[0]),
        "effective_rank": int(np.sum(used_values > numerical_tolerance)),
        "method": method,
    }


def simulate_correlated(
    correlation: ArrayLike,
    draws: int,
    seed: int = 0,
    *,
    numerical_tolerance: float = 1e-10,
    maximum_repair_norm: float = 1e-8,
) -> dict[str, float | int | str | np.ndarray]:
    """Generate normals only after an auditable correlation-matrix gate."""

    raw = np.asarray(correlation, dtype=float)
    if (
        raw.ndim != 2
        or raw.shape[0] != raw.shape[1]
        or raw.shape[0] == 0
        or isinstance(draws, (bool, np.bool_))
        or not isinstance(draws, (int, np.integer))
        or draws <= 0
        or isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
        or not np.isfinite(numerical_tolerance)
        or numerical_tolerance < 0.0
        or not np.isfinite(maximum_repair_norm)
        or maximum_repair_norm < 0.0
        or not np.all(np.isfinite(raw))
    ):
        raise ValueError("finite square correlation and positive draws required")
    if not np.allclose(raw, raw.T, atol=numerical_tolerance, rtol=0.0):
        raise ValueError("correlation must be symmetric")
    if not np.allclose(
        np.diag(raw),
        1.0,
        atol=numerical_tolerance,
        rtol=0.0,
    ):
        raise ValueError("correlation diagonal must equal one")

    symmetric = 0.5 * (raw + raw.T)
    raw_minimum = float(np.linalg.eigvalsh(symmetric)[0])
    factorization = correlation_eigen_root(
        symmetric,
        numerical_tolerance=numerical_tolerance,
    )
    used = np.asarray(factorization["correlation"])
    repair_norm = float(np.linalg.norm(used - symmetric, ord="fro"))
    status = (
        "PASS"
        if repair_norm <= np.finfo(float).eps * raw.shape[0]
        else "CONDITIONAL_PASS"
        if repair_norm <= maximum_repair_norm
        else "REJECT"
    )
    if status == "REJECT":
        samples = np.empty((0, raw.shape[0]))
    else:
        generator = np.random.default_rng(seed)
        root = np.asarray(factorization["root"])
        samples = generator.standard_normal((draws, raw.shape[0])) @ root.T
    return {
        "samples": samples,
        "raw_minimum_eigenvalue": raw_minimum,
        "used_minimum_eigenvalue": factorization["minimum_eigenvalue"],
        "repair_frobenius_norm": repair_norm,
        "factorization_method": factorization["method"],
        "effective_rank": factorization["effective_rank"],
        "maximum_repair_norm": maximum_repair_norm,
        "status": status,
        "seed": seed,
    }
