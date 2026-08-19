"""Stable rank-one updates and solve diagnostics for Chapter 2."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rank_one_inverse_core(
    inverse: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Apply Sherman--Morrison only after guarding its denominator."""

    inverse_left = inverse @ left
    right_inverse = right @ inverse
    denominator = float(1.0 + right @ inverse_left)
    denominator_scale = (
        1.0
        + np.linalg.norm(right)
        * np.linalg.norm(inverse, ord=2)
        * np.linalg.norm(left)
    )
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0.0
        or not np.isfinite(denominator)
        or abs(denominator) <= tolerance * denominator_scale
    ):
        raise ValueError("rank-one update denominator is numerically singular")
    updated = inverse - np.outer(inverse_left, right_inverse) / denominator
    return updated, denominator


def sherman_morrison_inverse(
    inverse: ArrayLike,
    u: ArrayLike,
    v: ArrayLike,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Return a guarded rank-one inverse update."""

    current_inverse = np.asarray(inverse, dtype=float)
    left = np.asarray(u, dtype=float)
    right = np.asarray(v, dtype=float)
    if (
        current_inverse.ndim != 2
        or current_inverse.shape[0] != current_inverse.shape[1]
        or left.shape != (current_inverse.shape[0],)
        or right.shape != left.shape
        or not np.all(np.isfinite(current_inverse))
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("finite aligned inverse and update vectors required")
    updated, denominator = rank_one_inverse_core(
        current_inverse,
        left,
        right,
        tolerance=tolerance,
    )
    return updated


def symmetric_rank_one_update_audit(
    matrix: ArrayLike,
    inverse: ArrayLike,
    vector: ArrayLike,
    scale: float,
    *,
    tolerance: float = 1e-12,
) -> dict[str, float | np.ndarray]:
    """Audit a symmetric covariance update or downdate against its residual."""

    current = np.asarray(matrix, dtype=float)
    current_inverse = np.asarray(inverse, dtype=float)
    direction = np.asarray(vector, dtype=float)
    if (
        current.ndim != 2
        or current.shape[0] != current.shape[1]
        or current_inverse.shape != current.shape
        or direction.shape != (current.shape[0],)
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(current_inverse))
        or not np.all(np.isfinite(direction))
        or not np.isfinite(scale)
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("finite aligned matrix, inverse and direction required")
    matrix_scale = max(float(np.max(np.abs(current))), np.finfo(float).tiny)
    inverse_scale = max(
        float(np.max(np.abs(current_inverse))),
        np.finfo(float).tiny,
    )
    if float(np.max(np.abs(current - current.T))) > tolerance * matrix_scale:
        raise ValueError("current matrix must be symmetric")
    if (
        float(np.max(np.abs(current_inverse - current_inverse.T)))
        > tolerance * inverse_scale
    ):
        raise ValueError("current inverse must be symmetric")
    current = 0.5 * (current + current.T)
    current_inverse = 0.5 * (current_inverse + current_inverse.T)
    try:
        np.linalg.cholesky(current)
    except np.linalg.LinAlgError as error:
        raise ValueError("current matrix must be positive definite") from error
    identity = np.eye(current.shape[0])
    pre_update_residual = np.linalg.norm(
        current @ current_inverse - identity,
        ord="fro",
    ) / np.linalg.norm(identity, ord="fro")
    if pre_update_residual > tolerance:
        raise ValueError("current inverse does not match canonical matrix")
    inverse_direction = current_inverse @ direction
    denominator = float(1.0 + scale * direction @ inverse_direction)
    denominator_scale = (
        1.0
        + abs(scale)
        * np.linalg.norm(direction) ** 2
        * np.linalg.norm(current_inverse, ord=2)
    )
    if denominator <= tolerance * denominator_scale:
        raise ValueError("rank-one downdate loses positive definiteness")
    updated_matrix = current + scale * np.outer(direction, direction)
    try:
        np.linalg.cholesky(updated_matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("rank-one update loses positive definiteness") from error
    updated_inverse = (
        current_inverse
        - scale * np.outer(inverse_direction, inverse_direction) / denominator
    )
    residual = np.linalg.norm(
        updated_matrix @ updated_inverse - identity,
        ord="fro",
    ) / np.linalg.norm(identity, ord="fro")
    return {
        "matrix": updated_matrix,
        "inverse": updated_inverse,
        "denominator": denominator,
        "pre_update_residual": float(pre_update_residual),
        "post_update_residual": float(residual),
        "relative_inverse_residual": float(residual),
    }


def normwise_backward_error(
    matrix: np.ndarray,
    solution: np.ndarray,
    rhs: np.ndarray,
) -> float:
    """Measure solve error in the scale of the original linear system."""

    residual = rhs - matrix @ solution
    denominator = (
        np.linalg.norm(matrix, ord=2) * np.linalg.norm(solution)
        + np.linalg.norm(rhs)
    )
    return float(
        np.linalg.norm(residual)
        / max(denominator, np.finfo(float).tiny)
    )


def solve_audit(
    matrix: ArrayLike,
    rhs: ArrayLike,
    *,
    reference_solution: ArrayLike | None = None,
) -> dict[str, float | np.ndarray]:
    """Compare a direct solve with an explicit-inverse baseline."""

    coefficients = np.asarray(matrix, dtype=float)
    target = np.asarray(rhs, dtype=float)
    if (
        coefficients.ndim != 2
        or coefficients.shape[0] != coefficients.shape[1]
        or target.shape != (coefficients.shape[0],)
        or not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("finite square system and aligned right-hand side required")
    direct = np.linalg.solve(coefficients, target)
    inverse_baseline = np.linalg.inv(coefficients) @ target

    def metrics(solution: np.ndarray) -> tuple[float, float]:
        residual = target - coefficients @ solution
        relative_residual = np.linalg.norm(residual) / max(
            np.linalg.norm(target),
            np.finfo(float).tiny,
        )
        backward_error = normwise_backward_error(
            coefficients,
            solution,
            target,
        )
        return float(relative_residual), float(backward_error)

    direct_residual, direct_backward = metrics(direct)
    inverse_residual, inverse_backward = metrics(inverse_baseline)
    result: dict[str, float | np.ndarray] = {
        "solution": direct,
        "inverse_baseline_solution": inverse_baseline,
        "condition_number": float(np.linalg.cond(coefficients)),
        "relative_residual": direct_residual,
        "backward_error": direct_backward,
        "inverse_relative_residual": inverse_residual,
        "inverse_backward_error": inverse_backward,
    }
    if reference_solution is not None:
        reference = np.asarray(reference_solution, dtype=float)
        if reference.shape != direct.shape or not np.all(np.isfinite(reference)):
            raise ValueError("finite aligned reference solution required")
        reference_norm = max(np.linalg.norm(reference), np.finfo(float).tiny)
        result["forward_error"] = float(
            np.linalg.norm(direct - reference) / reference_norm
        )
        result["inverse_forward_error"] = float(
            np.linalg.norm(inverse_baseline - reference) / reference_norm
        )
    return result


def least_squares_svd_audit(
    design: ArrayLike,
    rhs: ArrayLike,
    *,
    relative_tolerance: float | None = None,
) -> dict[str, float | int | bool | np.ndarray]:
    """Expose SVD rank and residuals for an identifiable least-squares solve."""

    matrix = np.asarray(design, dtype=float)
    target = np.asarray(rhs, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] == 0
        or target.shape != (matrix.shape[0],)
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(target))
        or (
            relative_tolerance is not None
            and (
                not np.isfinite(relative_tolerance)
                or relative_tolerance < 0.0
            )
        )
    ):
        raise ValueError("finite design and aligned right-hand side required")
    solution, _, rank, singular_values = np.linalg.lstsq(
        matrix,
        target,
        rcond=relative_tolerance,
    )
    residual = matrix @ solution - target
    maximum = float(singular_values[0])
    minimum = float(singular_values[-1])
    cutoff = (
        np.finfo(float).eps * max(matrix.shape) * maximum
        if relative_tolerance is None
        else relative_tolerance * maximum
    )
    retained = singular_values[singular_values > cutoff]
    row_count, column_count = matrix.shape
    full_row_rank = bool(rank == row_count)
    full_column_rank = bool(rank == column_count)
    is_underdetermined = bool(row_count < column_count)
    is_parameter_identified = full_column_rank
    is_rank_deficient = bool(rank < min(matrix.shape))
    raw_condition = (
        float("inf")
        if minimum == 0.0
        else maximum / minimum
    )
    retained_condition = (
        float("inf")
        if retained.size == 0
        else maximum / float(retained[-1])
    )
    return {
        "solution": solution,
        "rank": int(rank),
        "singular_values": singular_values,
        "is_rank_deficient": is_rank_deficient,
        "full_row_rank": full_row_rank,
        "full_column_rank": full_column_rank,
        "is_underdetermined": is_underdetermined,
        "is_parameter_identified": is_parameter_identified,
        "identification_status": (
            "IDENTIFIED" if is_parameter_identified else "NOT_IDENTIFIED"
        ),
        "raw_condition_number": raw_condition,
        "retained_condition_number": retained_condition,
        "condition_number": (
            float("inf") if not is_parameter_identified else raw_condition
        ),
        "rank_tolerance": cutoff,
        "relative_residual": float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(target), np.finfo(float).tiny)
        ),
    }
