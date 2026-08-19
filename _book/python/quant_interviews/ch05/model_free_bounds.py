from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog


@dataclass(frozen=True)
class RobustBoundCertificate:
    state_coordinates: NDArray[np.float64]
    state_range: tuple[float, float]
    maximum_state_mesh: float
    unresolved_target_breakpoints: NDArray[np.float64]
    scope: str
    continuum_certified: bool
    lower_bound: float
    upper_bound: float
    subhedge_positions: NDArray[np.float64]
    superhedge_positions: NDArray[np.float64]
    lower_dual_state_prices: NDArray[np.float64]
    upper_dual_state_prices: NDArray[np.float64]
    subhedge_residual: NDArray[np.float64]
    superhedge_residual: NDArray[np.float64]
    lower_duality_gap: float
    upper_duality_gap: float
    lower_binding_states: NDArray[np.int64]
    upper_binding_states: NDArray[np.int64]
    solver_status: str


def discrete_model_free_bounds(
    state_coordinates: NDArray[np.float64],
    payoff: NDArray[np.float64],
    hedge_payoffs: NDArray[np.float64],
    hedge_prices: NDArray[np.float64],
    *,
    tolerance: float = 1e-9,
    target_breakpoints: NDArray[np.float64] | None = None,
) -> RobustBoundCertificate:
    """Compute bounds on an explicitly finite state grid.

    The result does not certify domination between grid nodes or outside the
    reported state range.  Target breakpoints absent from the grid are exposed
    as unresolved rather than silently promoted to continuum evidence.
    """

    state_coordinates = np.asarray(state_coordinates, dtype=float)
    payoff = np.asarray(payoff, dtype=float)
    hedge_payoffs = np.asarray(hedge_payoffs, dtype=float)
    hedge_prices = np.asarray(hedge_prices, dtype=float)
    if (
        state_coordinates.ndim != 1
        or state_coordinates.shape != payoff.shape
        or payoff.ndim != 1
        or hedge_payoffs.ndim != 2
        or hedge_payoffs.shape[0] != payoff.size
        or hedge_payoffs.shape[1] != hedge_prices.size
        or payoff.size < 2
        or hedge_prices.size == 0
    ):
        raise ValueError("robust-bound inputs have inconsistent shape")
    if (
        not np.all(np.isfinite(state_coordinates))
        or not np.all(np.isfinite(payoff))
        or not np.all(np.isfinite(hedge_payoffs))
        or not np.all(np.isfinite(hedge_prices))
        or tolerance <= 0.0
        or np.any(np.diff(state_coordinates) <= 0.0)
    ):
        raise ValueError("robust-bound inputs must be finite")
    if target_breakpoints is None:
        breakpoints = np.array([], dtype=float)
    else:
        breakpoints = np.asarray(target_breakpoints, dtype=float)
        if breakpoints.ndim != 1 or not np.all(np.isfinite(breakpoints)):
            raise ValueError("target breakpoints must be a finite vector")
    unresolved = np.asarray(
        [
            value
            for value in breakpoints
            if state_coordinates[0] < value < state_coordinates[-1]
            and not np.any(np.isclose(state_coordinates, value, rtol=0.0, atol=tolerance))
        ],
        dtype=float,
    )

    bounds = [(None, None)] * hedge_prices.size
    upper_primal = linprog(
        hedge_prices,
        A_ub=-hedge_payoffs,
        b_ub=-payoff,
        bounds=bounds,
        method="highs",
    )
    lower_primal = linprog(
        -hedge_prices,
        A_ub=hedge_payoffs,
        b_ub=payoff,
        bounds=bounds,
        method="highs",
    )
    upper_dual = linprog(
        -payoff,
        A_eq=hedge_payoffs.T,
        b_eq=hedge_prices,
        bounds=[(0.0, None)] * payoff.size,
        method="highs",
    )
    lower_dual = linprog(
        payoff,
        A_eq=hedge_payoffs.T,
        b_eq=hedge_prices,
        bounds=[(0.0, None)] * payoff.size,
        method="highs",
    )
    results = (upper_primal, lower_primal, upper_dual, lower_dual)
    if not all(result.success for result in results):
        messages = "; ".join(result.message for result in results)
        raise RuntimeError(f"robust-bound LP failed: {messages}")

    upper = float(upper_primal.fun)
    lower = float(-lower_primal.fun)
    upper_dual_value = float(-upper_dual.fun)
    lower_dual_value = float(lower_dual.fun)
    super_residual = hedge_payoffs @ upper_primal.x - payoff
    sub_residual = payoff - hedge_payoffs @ lower_primal.x
    upper_gap = upper - upper_dual_value
    lower_gap = lower_dual_value - lower
    passed = (
        lower <= upper + tolerance
        and np.min(super_residual) >= -tolerance
        and np.min(sub_residual) >= -tolerance
        and abs(upper_gap) <= tolerance
        and abs(lower_gap) <= tolerance
    )
    return RobustBoundCertificate(
        state_coordinates=state_coordinates,
        state_range=(float(state_coordinates[0]), float(state_coordinates[-1])),
        maximum_state_mesh=float(np.max(np.diff(state_coordinates))),
        unresolved_target_breakpoints=unresolved,
        scope="DISCRETE_FINITE_STATE_GRID",
        continuum_certified=False,
        lower_bound=lower,
        upper_bound=upper,
        subhedge_positions=lower_primal.x,
        superhedge_positions=upper_primal.x,
        lower_dual_state_prices=lower_dual.x,
        upper_dual_state_prices=upper_dual.x,
        subhedge_residual=sub_residual,
        superhedge_residual=super_residual,
        lower_duality_gap=lower_gap,
        upper_duality_gap=upper_gap,
        lower_binding_states=np.flatnonzero(sub_residual <= tolerance),
        upper_binding_states=np.flatnonzero(super_residual <= tolerance),
        solver_status="PASS" if passed else "REJECT",
    )


def discrete_superhedge(
    state_coordinates: NDArray[np.float64],
    payoff: NDArray[np.float64],
    hedge_payoffs: NDArray[np.float64],
    hedge_prices: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Compatibility wrapper returning the upper primal only."""

    certificate = discrete_model_free_bounds(
        state_coordinates,
        payoff,
        hedge_payoffs,
        hedge_prices,
    )
    return certificate.upper_bound, certificate.superhedge_positions
