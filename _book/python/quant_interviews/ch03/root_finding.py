"""Safeguarded scalar root finding for Chapter 3."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import NamedTuple


@dataclass(frozen=True)
class RootResult:
    """Auditable terminal state of a bracketed scalar solve."""

    root: float
    residual: float
    bracket_lower: float
    bracket_upper: float
    bracket_width: float
    iterations: int
    newton_steps: int
    bisection_steps: int
    converged: bool
    status: str


class RootStep(NamedTuple):
    """One audited Newton/bisection transition."""

    estimate: float
    residual: float
    left: float
    right: float
    left_value: float
    used_newton: bool
    converged: bool


def _finite_evaluation(
    function: Callable[[float], float],
    argument: float,
    name: str,
) -> float:
    value = float(function(argument))
    if not math.isfinite(value):
        raise ValueError(f"{name} returned a non-finite value")
    return value


def safeguarded_candidate(
    estimate: float,
    value: float,
    slope: float,
    lower: float,
    upper: float,
    derivative_tolerance: float,
) -> tuple[float, bool]:
    """Return an admissible Newton proposal or the bisection fallback."""

    midpoint = 0.5 * (lower + upper)
    if abs(slope) <= derivative_tolerance:
        return midpoint, False
    proposal = estimate - value / slope
    adequate_progress = abs(proposal - estimate) <= 0.5 * (upper - lower)
    if lower < proposal < upper and adequate_progress:
        return proposal, True
    return midpoint, False


def safeguarded_root_step(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    estimate: float,
    left: float,
    right: float,
    left_value: float,
    *,
    residual_tolerance: float,
    bracket_tolerance: float,
    derivative_tolerance: float,
    residual_threshold: Callable[[float], float] | None = None,
    bracket_threshold: Callable[[float], float] | None = None,
) -> RootStep:
    """Advance one step while retaining the sign-changing bracket."""

    value = _finite_evaluation(function, estimate, "function")
    slope = _finite_evaluation(derivative, estimate, "derivative")
    candidate, used_newton = safeguarded_candidate(
        estimate,
        value,
        slope,
        left,
        right,
        derivative_tolerance,
    )
    candidate_value = _finite_evaluation(
        function,
        candidate,
        "function",
    )
    if left_value * candidate_value <= 0.0:
        right = candidate
    else:
        left, left_value = candidate, candidate_value
    residual = abs(candidate_value)
    residual_limit = (
        residual_tolerance
        if residual_threshold is None
        else float(residual_threshold(candidate))
    )
    bracket_limit = (
        bracket_tolerance
        if bracket_threshold is None
        else float(bracket_threshold(candidate))
    )
    if (
        not math.isfinite(residual_limit)
        or residual_limit <= 0.0
        or not math.isfinite(bracket_limit)
        or bracket_limit <= 0.0
    ):
        raise ValueError("stopping thresholds must be finite and positive")
    converged = residual <= residual_limit and right - left <= bracket_limit
    return RootStep(
        candidate,
        residual,
        left,
        right,
        left_value,
        used_newton,
        converged,
    )


def safeguarded_newton_core(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    left: float,
    right: float,
    left_value: float,
    initial: float,
    *,
    residual_tolerance: float,
    bracket_tolerance: float,
    derivative_tolerance: float,
    max_iterations: int,
    residual_threshold: Callable[[float], float] | None = None,
    bracket_threshold: Callable[[float], float] | None = None,
) -> RootResult:
    """Run the sign-preserving Newton/bisection state machine."""

    estimate = min(max(float(initial), left), right)
    newton_steps = 0
    bisection_steps = 0
    residual = abs(_finite_evaluation(function, estimate, "function"))
    for iteration in range(1, max_iterations + 1):
        step = safeguarded_root_step(
            function,
            derivative,
            estimate,
            left,
            right,
            left_value,
            residual_tolerance=residual_tolerance,
            bracket_tolerance=bracket_tolerance,
            derivative_tolerance=derivative_tolerance,
            residual_threshold=residual_threshold,
            bracket_threshold=bracket_threshold,
        )
        estimate, residual = step.estimate, step.residual
        left, right, left_value = step.left, step.right, step.left_value
        newton_steps += int(step.used_newton)
        bisection_steps += int(not step.used_newton)
        width = right - left
        if step.converged:
            return RootResult(
                estimate,
                residual,
                left,
                right,
                width,
                iteration,
                newton_steps,
                bisection_steps,
                True,
                "CONVERGED",
            )

    return RootResult(
        estimate,
        residual,
        left,
        right,
        right - left,
        max_iterations,
        newton_steps,
        bisection_steps,
        False,
        "MAX_ITERATIONS",
    )


def safeguarded_newton(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    lower: float,
    upper: float,
    initial: float,
    *,
    residual_tolerance: float = 1e-12,
    bracket_tolerance: float = 1e-12,
    derivative_tolerance: float = 1e-14,
    max_iterations: int = 100,
    residual_threshold: Callable[[float], float] | None = None,
    bracket_threshold: Callable[[float], float] | None = None,
) -> RootResult:
    """Combine Newton steps with a sign-preserving bisection bracket."""

    scalars = (
        lower,
        upper,
        initial,
        residual_tolerance,
        bracket_tolerance,
        derivative_tolerance,
    )
    if (
        not all(math.isfinite(value) for value in scalars)
        or not lower < upper
        or residual_tolerance <= 0.0
        or bracket_tolerance <= 0.0
        or derivative_tolerance <= 0.0
        or (residual_threshold is not None and not callable(residual_threshold))
        or (bracket_threshold is not None and not callable(bracket_threshold))
        or isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise ValueError("finite ordered bracket and positive controls required")

    left, right = float(lower), float(upper)
    left_value = _finite_evaluation(function, left, "function")
    right_value = _finite_evaluation(function, right, "function")
    left_residual_limit = (
        residual_tolerance
        if residual_threshold is None
        else float(residual_threshold(left))
    )
    right_residual_limit = (
        residual_tolerance
        if residual_threshold is None
        else float(residual_threshold(right))
    )
    if (
        not math.isfinite(left_residual_limit)
        or left_residual_limit <= 0.0
        or not math.isfinite(right_residual_limit)
        or right_residual_limit <= 0.0
    ):
        raise ValueError("stopping thresholds must be finite and positive")
    if abs(left_value) <= left_residual_limit:
        return RootResult(
            left,
            abs(left_value),
            left,
            left,
            0.0,
            0,
            0,
            0,
            True,
            "CONVERGED_ENDPOINT",
        )
    if abs(right_value) <= right_residual_limit:
        return RootResult(
            right,
            abs(right_value),
            right,
            right,
            0.0,
            0,
            0,
            0,
            True,
            "CONVERGED_ENDPOINT",
        )
    if left_value * right_value > 0.0:
        raise ValueError("root is not bracketed by a sign change")

    return safeguarded_newton_core(
        function,
        derivative,
        left,
        right,
        left_value,
        initial,
        residual_tolerance=residual_tolerance,
        bracket_tolerance=bracket_tolerance,
        derivative_tolerance=derivative_tolerance,
        max_iterations=max_iterations,
        residual_threshold=residual_threshold,
        bracket_threshold=bracket_threshold,
    )
