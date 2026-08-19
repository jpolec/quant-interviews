"""Adjoint sensitivities and independent directional checks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def _finite_vector(values: ArrayLike, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite aligned vector")
    return vector


def linear_system_adjoint(
    matrix: ArrayLike,
    objective_gradient: ArrayLike,
) -> np.ndarray:
    """Solve A.T lambda = J_x without forming an inverse."""

    coefficients = np.asarray(matrix, dtype=float)
    if (
        coefficients.ndim != 2
        or coefficients.shape[0] != coefficients.shape[1]
        or not np.all(np.isfinite(coefficients))
        or np.linalg.matrix_rank(coefficients) < coefficients.shape[0]
    ):
        raise ValueError("finite nonsingular square matrix required")
    gradient = _finite_vector(
        objective_gradient,
        coefficients.shape[0],
        "objective gradient",
    )
    return np.linalg.solve(coefficients.T, gradient)


def parameter_gradient(
    adjoint: ArrayLike,
    state: ArrayLike,
    matrix_derivatives: Sequence[ArrayLike],
    rhs_derivatives: Sequence[ArrayLike],
    explicit_objective_derivative: ArrayLike | None = None,
) -> np.ndarray:
    """Apply J_theta + lambda.T(db/dtheta - dA/dtheta x)."""

    solution = np.asarray(state, dtype=float)
    dual = _finite_vector(adjoint, solution.size, "adjoint")
    if len(matrix_derivatives) != len(rhs_derivatives):
        raise ValueError("matrix and rhs derivative counts must match")
    explicit = (
        np.zeros(len(matrix_derivatives))
        if explicit_objective_derivative is None
        else _finite_vector(
            explicit_objective_derivative,
            len(matrix_derivatives),
            "explicit objective derivative",
        )
    )
    sensitivities = []
    for matrix_derivative, rhs_derivative, direct in zip(
        matrix_derivatives,
        rhs_derivatives,
        explicit,
        strict=True,
    ):
        derivative_matrix = np.asarray(matrix_derivative, dtype=float)
        derivative_rhs = _finite_vector(
            rhs_derivative,
            solution.size,
            "rhs derivative",
        )
        if (
            derivative_matrix.shape != (solution.size, solution.size)
            or not np.all(np.isfinite(derivative_matrix))
        ):
            raise ValueError("matrix derivatives must be finite and aligned")
        sensitivities.append(
            direct + dual @ (derivative_rhs - derivative_matrix @ solution)
        )
    return np.asarray(sensitivities)


def linear_adjoint_audit(
    matrix: ArrayLike,
    rhs: ArrayLike,
    objective_gradient: ArrayLike,
    matrix_derivatives: Sequence[ArrayLike],
    rhs_derivatives: Sequence[ArrayLike],
    directions: ArrayLike,
    *,
    epsilon: float = 1e-6,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 1e-8,
) -> dict[str, float | np.ndarray]:
    """Check the adjoint gradient against independent directional solves."""

    coefficients = np.asarray(matrix, dtype=float)
    target = np.asarray(rhs, dtype=float)
    objective = np.asarray(objective_gradient, dtype=float)
    state = np.linalg.solve(coefficients, target)
    adjoint = linear_system_adjoint(coefficients, objective)
    gradient = parameter_gradient(
        adjoint,
        state,
        matrix_derivatives,
        rhs_derivatives,
    )
    direction_matrix = np.asarray(directions, dtype=float)
    if (
        direction_matrix.ndim != 2
        or direction_matrix.shape[1] != gradient.size
        or not np.all(np.isfinite(direction_matrix))
        or epsilon <= 0.0
        or absolute_tolerance < 0.0
        or relative_tolerance < 0.0
    ):
        raise ValueError(
            "finite directions, positive epsilon and nonnegative tolerances required"
        )

    finite_difference = []
    for direction in direction_matrix:
        matrix_shift = sum(
            coefficient * np.asarray(derivative)
            for coefficient, derivative in zip(
                direction,
                matrix_derivatives,
                strict=True,
            )
        )
        rhs_shift = sum(
            coefficient * np.asarray(derivative)
            for coefficient, derivative in zip(
                direction,
                rhs_derivatives,
                strict=True,
            )
        )
        upper_state = np.linalg.solve(
            coefficients + epsilon * matrix_shift,
            target + epsilon * rhs_shift,
        )
        lower_state = np.linalg.solve(
            coefficients - epsilon * matrix_shift,
            target - epsilon * rhs_shift,
        )
        finite_difference.append(
            objective @ (upper_state - lower_state) / (2.0 * epsilon)
        )
    finite_difference_values = np.asarray(finite_difference)
    adjoint_directional = direction_matrix @ gradient
    absolute_errors = np.abs(
        adjoint_directional - finite_difference_values
    )
    mixed_scales = (
        absolute_tolerance
        + relative_tolerance
        * np.maximum(
            np.abs(adjoint_directional),
            np.abs(finite_difference_values),
        )
    )
    scaled_errors = absolute_errors / np.maximum(
        mixed_scales,
        np.finfo(float).tiny,
    )
    return {
        "state": state,
        "adjoint": adjoint,
        "gradient": gradient,
        "primal_residual": float(np.linalg.norm(coefficients @ state - target)),
        "adjoint_residual": float(
            np.linalg.norm(coefficients.T @ adjoint - objective)
        ),
        "directional_adjoint": adjoint_directional,
        "directional_finite_difference": finite_difference_values,
        "maximum_absolute_gradient_error": float(
            np.max(absolute_errors)
        ),
        "maximum_scaled_gradient_error": float(np.max(scaled_errors)),
        "smallest_reference_directional_derivative": float(
            np.min(np.abs(finite_difference_values))
        ),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }
