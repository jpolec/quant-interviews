"""Auditable constrained portfolio optimization for Chapter 2."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import LinearConstraint, linprog, minimize


def _finite_scalar(value: float, *, name: str) -> float:
    """Return one finite real scalar or raise a stable validation error."""

    candidate = np.asarray(value)
    if candidate.ndim != 0 or candidate.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite scalar")
    scalar = float(candidate)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar")
    return scalar


def _positive_integer(value: int, *, name: str) -> int:
    """Return a positive integer while rejecting booleans and floats."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a positive integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _positive_definite_covariance(covariance: ArrayLike) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] == 0
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("finite symmetric square covariance required")
    symmetry_scale = max(
        float(np.linalg.norm(matrix, ord=np.inf)),
        np.finfo(float).tiny,
    )
    if float(np.max(np.abs(matrix - matrix.T))) > (
        1e-14 + 1e-10 * symmetry_scale
    ):
        raise ValueError("finite symmetric square covariance required")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    positive_floor = (
        np.finfo(float).eps * matrix.shape[0]
        * max(float(eigenvalues[-1]), np.finfo(float).tiny)
    )
    if eigenvalues[0] <= positive_floor:
        raise ValueError("positive-definite covariance required")
    return matrix


def equality_kkt_core(
    covariance: np.ndarray,
    constraints: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the validated equality-constrained minimum-variance system."""

    asset_count = covariance.shape[0]
    constraint_count = constraints.shape[1]
    system = np.block(
        [
            [covariance, -constraints],
            [constraints.T, np.zeros((constraint_count, constraint_count))],
        ]
    )
    solution = np.linalg.solve(
        system,
        np.concatenate([np.zeros(asset_count), targets]),
    )
    return solution[:asset_count], solution[asset_count:]


def markowitz_kkt(
    covariance: ArrayLike,
    constraints: ArrayLike,
    targets: ArrayLike,
) -> dict[str, float | np.ndarray]:
    """Solve equality-constrained minimum variance as one KKT system."""

    matrix = _positive_definite_covariance(covariance)
    constraint_matrix = np.asarray(constraints, dtype=float)
    target = np.asarray(targets, dtype=float)
    asset_count = matrix.shape[0]
    if (
        constraint_matrix.ndim != 2
        or constraint_matrix.shape[0] != asset_count
        or target.shape != (constraint_matrix.shape[1],)
        or not np.all(np.isfinite(constraint_matrix))
        or not np.all(np.isfinite(target))
        or np.linalg.matrix_rank(constraint_matrix)
        < constraint_matrix.shape[1]
    ):
        raise ValueError("finite aligned full-rank constraints required")

    weights, multipliers = equality_kkt_core(
        matrix,
        constraint_matrix,
        target,
    )
    stationarity = matrix @ weights - constraint_matrix @ multipliers
    primal = constraint_matrix.T @ weights - target
    return {
        "weights": weights,
        "multipliers": multipliers,
        "objective": float(0.5 * weights @ matrix @ weights),
        "primal_residual": float(np.linalg.norm(primal)),
        "stationarity_residual": float(np.linalg.norm(stationarity)),
    }


def long_only_markowitz_audit(
    covariance: ArrayLike,
    expected_returns: ArrayLike,
    *,
    risk_aversion: float,
    factor_loadings: ArrayLike | None = None,
    factor_targets: ArrayLike | None = None,
    upper_bound: float = 1.0,
    tolerance: float = 1e-8,
) -> dict[str, float | int | bool | np.ndarray]:
    """Solve a bounded portfolio and recover a numerical KKT certificate."""

    matrix = _positive_definite_covariance(covariance)
    expected = np.asarray(expected_returns, dtype=float)
    asset_count = matrix.shape[0]
    risk_aversion = _finite_scalar(risk_aversion, name="risk_aversion")
    upper_bound = _finite_scalar(upper_bound, name="upper_bound")
    tolerance = _finite_scalar(tolerance, name="tolerance")
    if (
        expected.shape != (asset_count,)
        or not np.all(np.isfinite(expected))
        or risk_aversion <= 0.0
        or not 0.0 < upper_bound <= 1.0
        or tolerance <= 0.0
        or asset_count * upper_bound < 1.0
    ):
        raise ValueError("valid returns, risk aversion and bound required")
    if factor_loadings is None:
        equalities = np.ones((asset_count, 1))
        targets = np.array([1.0])
    else:
        factors = np.asarray(factor_loadings, dtype=float)
        factor_target = np.asarray(factor_targets, dtype=float)
        if (
            factors.ndim != 2
            or factors.shape[0] != asset_count
            or factor_target.shape != (factors.shape[1],)
            or not np.all(np.isfinite(factors))
            or not np.all(np.isfinite(factor_target))
        ):
            raise ValueError("aligned factor equalities and targets required")
        equalities = np.column_stack([np.ones(asset_count), factors])
        targets = np.concatenate([[1.0], factor_target])

    def objective(weights: np.ndarray) -> float:
        return float(
            0.5 * risk_aversion * weights @ matrix @ weights
            - expected @ weights
        )

    result = minimize(
        objective,
        np.full(asset_count, 1.0 / asset_count),
        method="SLSQP",
        bounds=[(0.0, upper_bound)] * asset_count,
        constraints={
            "type": "eq",
            "fun": lambda weights: equalities.T @ weights - targets,
        },
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.asarray(result.x)
    gradient = risk_aversion * matrix @ weights - expected
    lower_active = np.flatnonzero(weights <= tolerance)
    upper_active = np.flatnonzero(upper_bound - weights <= tolerance)
    lower_basis = np.eye(asset_count)[:, lower_active]
    upper_basis = np.eye(asset_count)[:, upper_active]
    multiplier_design = np.column_stack(
        [equalities, lower_basis, -upper_basis]
    )
    multipliers, *_ = np.linalg.lstsq(
        multiplier_design,
        gradient,
        rcond=None,
    )
    stationarity = gradient - multiplier_design @ multipliers
    equality_count = equalities.shape[1]
    lower_multipliers = multipliers[
        equality_count : equality_count + lower_active.size
    ]
    upper_multipliers = multipliers[equality_count + lower_active.size :]
    dual_violation = max(
        0.0,
        float(-lower_multipliers.min()) if lower_multipliers.size else 0.0,
        float(-upper_multipliers.min()) if upper_multipliers.size else 0.0,
    )
    complementarity = max(
        float(
            np.max(np.abs(lower_multipliers * weights[lower_active]))
        )
        if lower_active.size
        else 0.0,
        float(
            np.max(
                np.abs(
                    upper_multipliers
                    * (upper_bound - weights[upper_active])
                )
            )
        )
        if upper_active.size
        else 0.0,
    )
    primal_residual = max(
        float(np.linalg.norm(equalities.T @ weights - targets)),
        float(max(0.0, -weights.min())),
        float(max(0.0, weights.max() - upper_bound)),
    )
    return {
        "weights": weights,
        "objective": objective(weights),
        "equality_multipliers": multipliers[:equality_count],
        "lower_bound_multipliers": lower_multipliers,
        "upper_bound_multipliers": upper_multipliers,
        "primal_residual": primal_residual,
        "stationarity_residual": float(np.linalg.norm(stationarity)),
        "complementarity_residual": complementarity,
        "dual_feasibility_violation": dual_violation,
        "active_lower_bounds": int(lower_active.size),
        "active_upper_bounds": int(upper_active.size),
        "status": (
            "PASS"
            if max(
                primal_residual,
                float(np.linalg.norm(stationarity)),
                complementarity,
                dual_violation,
            )
            <= tolerance
            else "CONDITIONAL_PASS"
        ),
    }


def l1_turnover_constraint(
    previous: ArrayLike,
) -> tuple[LinearConstraint, LinearConstraint]:
    """Encode budget and absolute-trade epigraph constraints."""

    previous = np.asarray(previous, dtype=float)
    if (
        previous.ndim != 1
        or previous.size == 0
        or not np.all(np.isfinite(previous))
    ):
        raise ValueError("finite nonempty previous portfolio required")
    asset_count = previous.size
    identity = np.eye(asset_count)
    budget = np.hstack(
        [np.ones((1, asset_count)), np.zeros((1, asset_count))]
    )
    trade_bounds = np.vstack(
        [
            np.hstack([identity, -identity]),
            np.hstack([-identity, -identity]),
        ]
    )
    return (
        LinearConstraint(budget, 1.0, 1.0),
        LinearConstraint(
            trade_bounds,
            np.full(2 * asset_count, -np.inf),
            np.concatenate([previous, -previous]),
        ),
    )


def _validated_equalities(
    constraints: ArrayLike,
    targets: ArrayLike,
    *,
    asset_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate an aligned, independent equality-constraint system."""

    constraint_matrix = np.asarray(constraints, dtype=float)
    target = np.asarray(targets, dtype=float)
    if (
        constraint_matrix.ndim != 2
        or constraint_matrix.shape[0] != asset_count
        or constraint_matrix.shape[1] == 0
        or target.shape != (constraint_matrix.shape[1],)
        or not np.all(np.isfinite(constraint_matrix))
        or not np.all(np.isfinite(target))
        or np.linalg.matrix_rank(constraint_matrix)
        < constraint_matrix.shape[1]
    ):
        raise ValueError("finite aligned full-rank equalities required")
    return constraint_matrix, target


def _equality_quadratic_solution(
    covariance: np.ndarray,
    expected_returns: np.ndarray,
    constraints: np.ndarray,
    targets: np.ndarray,
    *,
    risk_aversion: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the equality-constrained differentiable allocation exactly."""

    asset_count = covariance.shape[0]
    constraint_count = constraints.shape[1]
    system = np.block(
        [
            [
                risk_aversion * covariance,
                constraints,
            ],
            [
                constraints.T,
                np.zeros((constraint_count, constraint_count)),
            ],
        ]
    )
    solution = np.linalg.solve(
        system,
        np.concatenate([expected_returns, targets]),
    )
    return solution[:asset_count], solution[asset_count:]


def _l1_subgradient_certificate(
    base_gradient: np.ndarray,
    trade: np.ndarray,
    constraints: np.ndarray,
    *,
    turnover_penalty: float,
    active_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Recover equality multipliers and a valid L1 subgradient certificate."""

    constraint_count = constraints.shape[1]
    if turnover_penalty == 0.0:
        multipliers, *_ = np.linalg.lstsq(
            constraints,
            -base_gradient,
            rcond=None,
        )
        subgradient = np.zeros(trade.size)
        residual = float(
            np.linalg.norm(
                base_gradient + constraints @ multipliers,
                ord=np.inf,
            )
        )
        return multipliers, subgradient, residual

    active = np.abs(trade) > active_tolerance
    sign = np.sign(trade)
    # Minimize the largest violation of the active-sign equations and the
    # inactive subgradient interval.  This small LP is a deterministic KKT
    # certificate; it is not a second allocation problem.
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    for index in range(trade.size):
        row = np.zeros(constraint_count + 1)
        row[:constraint_count] = constraints[index]
        row[-1] = -1.0
        if active[index]:
            target = -base_gradient[index] - turnover_penalty * sign[index]
            opposite_row = -row
            opposite_row[-1] = -1.0
            rows.extend([row, opposite_row])
            bounds.extend([target, -target])
        else:
            opposite_row = -row
            opposite_row[-1] = -1.0
            rows.extend([row, opposite_row])
            bounds.extend(
                [
                    turnover_penalty - base_gradient[index],
                    turnover_penalty + base_gradient[index],
                ]
            )
    certificate = linprog(
        np.concatenate([np.zeros(constraint_count), [1.0]]),
        A_ub=np.vstack(rows),
        b_ub=np.asarray(bounds),
        bounds=[(None, None)] * constraint_count + [(0.0, None)],
        method="highs",
    )
    if not certificate.success:
        raise RuntimeError(
            f"unable to recover L1 KKT certificate: {certificate.message}"
        )
    multipliers = np.asarray(certificate.x[:constraint_count])
    reduced_gradient = base_gradient + constraints @ multipliers
    subgradient = np.empty(trade.size)
    subgradient[active] = sign[active]
    subgradient[~active] = np.clip(
        -reduced_gradient[~active] / turnover_penalty,
        -1.0,
        1.0,
    )
    stationarity = (
        reduced_gradient + turnover_penalty * subgradient
    )
    return (
        multipliers,
        subgradient,
        float(np.linalg.norm(stationarity, ord=np.inf)),
    )


def constrained_turnover_regularized(
    covariance: ArrayLike,
    expected_returns: ArrayLike,
    previous: ArrayLike,
    constraints: ArrayLike,
    targets: ArrayLike,
    *,
    risk_aversion: float,
    turnover_penalty: float,
    tolerance: float = 1e-8,
    max_iterations: int = 4_000,
) -> dict[str, float | int | str | np.ndarray]:
    """Solve one equality-constrained L1-turnover portfolio with KKT evidence.

    The returned weights are the sole allocation used for constraint
    residuals, turnover, transaction cost and the complete optimality
    certificate.
    """

    matrix = _positive_definite_covariance(covariance)
    expected = np.asarray(expected_returns, dtype=float)
    current = np.asarray(previous, dtype=float)
    asset_count = matrix.shape[0]
    risk_aversion = _finite_scalar(risk_aversion, name="risk_aversion")
    turnover_penalty = _finite_scalar(
        turnover_penalty,
        name="turnover_penalty",
    )
    tolerance = _finite_scalar(tolerance, name="tolerance")
    max_iterations = _positive_integer(
        max_iterations,
        name="max_iterations",
    )
    if (
        expected.shape != (asset_count,)
        or current.shape != (asset_count,)
        or not np.all(np.isfinite(expected))
        or not np.all(np.isfinite(current))
        or risk_aversion <= 0.0
        or turnover_penalty < 0.0
        or tolerance <= 0.0
    ):
        raise ValueError("valid aligned portfolio and finite controls required")
    constraint_matrix, target = _validated_equalities(
        constraints,
        targets,
        asset_count=asset_count,
    )

    if turnover_penalty == 0.0:
        weights, equality_multipliers = _equality_quadratic_solution(
            matrix,
            expected,
            constraint_matrix,
            target,
            risk_aversion=risk_aversion,
        )
        trade_bounds = np.abs(weights - current)
        solver_iterations = 0
        solver_message = "exact equality-constrained linear solve"
    else:
        gram = constraint_matrix.T @ constraint_matrix
        feasible_weights = current + constraint_matrix @ np.linalg.solve(
            gram,
            target - constraint_matrix.T @ current,
        )
        initial_trade = feasible_weights - current
        initial = np.concatenate(
            [feasible_weights, np.abs(initial_trade) + tolerance]
        )
        identity = np.eye(asset_count)
        equality_rows = np.hstack(
            [
                constraint_matrix.T,
                np.zeros((constraint_matrix.shape[1], asset_count)),
            ]
        )
        trade_rows = np.vstack(
            [
                np.hstack([identity, -identity]),
                np.hstack([-identity, -identity]),
            ]
        )

        def objective(variables: np.ndarray) -> float:
            weights = variables[:asset_count]
            epigraph = variables[asset_count:]
            return float(
                0.5 * risk_aversion * weights @ matrix @ weights
                - expected @ weights
                + turnover_penalty * epigraph.sum()
            )

        def gradient(variables: np.ndarray) -> np.ndarray:
            weights = variables[:asset_count]
            return np.concatenate(
                [
                    risk_aversion * matrix @ weights - expected,
                    np.full(asset_count, turnover_penalty),
                ]
            )

        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            bounds=[(None, None)] * asset_count
            + [(0.0, None)] * asset_count,
            constraints=[
                LinearConstraint(equality_rows, target, target),
                LinearConstraint(
                    trade_rows,
                    np.full(2 * asset_count, -np.inf),
                    np.concatenate([current, -current]),
                ),
            ],
            options={
                "ftol": min(1e-12, tolerance * 1e-2),
                "maxiter": max_iterations,
            },
        )
        if not result.success:
            raise RuntimeError(result.message)
        weights = np.asarray(result.x[:asset_count])
        trade_bounds = np.asarray(result.x[asset_count:])
        solver_iterations = int(result.nit)
        solver_message = str(result.message)
        equality_multipliers = np.zeros(constraint_matrix.shape[1])

    trade = weights - current
    active_tolerance = max(10.0 * tolerance, 1e-9)
    base_gradient = risk_aversion * matrix @ weights - expected
    (
        equality_multipliers,
        subgradient,
        stationarity_residual,
    ) = _l1_subgradient_certificate(
        base_gradient,
        trade,
        constraint_matrix,
        turnover_penalty=turnover_penalty,
        active_tolerance=active_tolerance,
    )
    equality_residuals = constraint_matrix.T @ weights - target
    epigraph_violation = float(
        np.max(np.maximum(np.abs(trade) - trade_bounds, 0.0))
    )
    primal_residual = max(
        float(np.linalg.norm(equality_residuals, ord=np.inf)),
        epigraph_violation,
    )
    if turnover_penalty == 0.0:
        dual_feasibility_violation = 0.0
        complementarity_residual = 0.0
    else:
        positive_multiplier = turnover_penalty * np.maximum(subgradient, 0.0)
        negative_multiplier = turnover_penalty * np.maximum(-subgradient, 0.0)
        lower_multiplier = (
            turnover_penalty - positive_multiplier - negative_multiplier
        )
        dual_feasibility_violation = float(
            max(
                0.0,
                -positive_multiplier.min(),
                -negative_multiplier.min(),
                -lower_multiplier.min(),
            )
        )
        complementarity_residual = float(
            max(
                np.max(
                    np.abs(
                        positive_multiplier * (trade_bounds - trade)
                    )
                ),
                np.max(
                    np.abs(
                        negative_multiplier * (trade_bounds + trade)
                    )
                ),
                np.max(np.abs(lower_multiplier * trade_bounds)),
            )
        )
    maximum_kkt_violation = max(
        primal_residual,
        stationarity_residual,
        dual_feasibility_violation,
        complementarity_residual,
    )
    turnover = float(np.abs(trade).sum())
    factor_residuals = equality_residuals[1:]
    factor_neutrality_residual = (
        float(np.linalg.norm(factor_residuals, ord=np.inf))
        if factor_residuals.size
        else 0.0
    )
    no_trade_mask = np.abs(trade) <= active_tolerance
    base_objective = float(
        0.5 * risk_aversion * weights @ matrix @ weights
        - expected @ weights
    )
    turnover_cost = float(turnover_penalty * turnover)
    return {
        "weights": weights,
        "trade": trade,
        "turnover": turnover,
        "turnover_cost": turnover_cost,
        "base_objective": base_objective,
        "objective": base_objective + turnover_cost,
        "constraint_residuals": equality_residuals,
        "budget_residual": float(abs(equality_residuals[0])),
        "factor_residuals": factor_residuals,
        "factor_neutrality_residual": factor_neutrality_residual,
        "equality_multipliers": equality_multipliers,
        "subgradient": subgradient,
        "stationarity_residual": stationarity_residual,
        "subgradient_residual": stationarity_residual,
        "dual_feasibility_violation": dual_feasibility_violation,
        "complementarity_residual": complementarity_residual,
        "epigraph_violation": epigraph_violation,
        "primal_residual": primal_residual,
        "maximum_kkt_violation": maximum_kkt_violation,
        "no_trade_positions": int(np.sum(no_trade_mask)),
        "no_trade_mask": no_trade_mask,
        "solver_iterations": solver_iterations,
        "solver_message": solver_message,
        "status": "PASS" if maximum_kkt_violation <= tolerance else "REJECT",
    }


def turnover_regularized(
    covariance: ArrayLike,
    expected_returns: ArrayLike,
    previous: ArrayLike,
    *,
    risk_aversion: float,
    turnover_penalty: float,
    penalty: str,
) -> dict[str, float | int | str | np.ndarray]:
    """Solve budget-constrained L1 or L2 trade regularization."""

    matrix = _positive_definite_covariance(covariance)
    expected = np.asarray(expected_returns, dtype=float)
    current = np.asarray(previous, dtype=float)
    asset_count = matrix.shape[0]
    risk_aversion = _finite_scalar(risk_aversion, name="risk_aversion")
    turnover_penalty = _finite_scalar(
        turnover_penalty,
        name="turnover_penalty",
    )
    if (
        expected.shape != (asset_count,)
        or current.shape != (asset_count,)
        or not np.all(np.isfinite(expected))
        or not np.all(np.isfinite(current))
        or risk_aversion <= 0.0
        or turnover_penalty < 0.0
        or penalty not in {"l1", "l2"}
        or not np.isclose(current.sum(), 1.0)
    ):
        raise ValueError("valid aligned portfolio and penalty contract required")

    if penalty == "l2":
        def objective(weights: np.ndarray) -> float:
            trade = weights - current
            return float(
                0.5 * risk_aversion * weights @ matrix @ weights
                - expected @ weights
                + 0.5 * turnover_penalty * trade @ trade
            )

        result = minimize(
            objective,
            current,
            method="SLSQP",
            constraints={
                "type": "eq",
                "fun": lambda weights: weights.sum() - 1.0,
            },
            options={"ftol": 1e-12, "maxiter": 2_000},
        )
        weights = np.asarray(result.x)
    else:
        def objective(variables: np.ndarray) -> float:
            weights = variables[:asset_count]
            trade_bounds = variables[asset_count:]
            return float(
                0.5 * risk_aversion * weights @ matrix @ weights
                - expected @ weights
                + turnover_penalty * trade_bounds.sum()
            )

        initial = np.concatenate([current, np.zeros(asset_count)])
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(None, None)] * asset_count + [(0.0, None)] * asset_count,
            constraints=l1_turnover_constraint(current),
            options={"ftol": 1e-12, "maxiter": 4_000},
        )
        weights = np.asarray(result.x[:asset_count])
    if not result.success:
        raise RuntimeError(result.message)
    trade = weights - current
    base_gradient = risk_aversion * matrix @ weights - expected
    if penalty == "l2":
        equality_multiplier = -float(
            np.mean(base_gradient + turnover_penalty * trade)
        )
        reduced_gradient = base_gradient + equality_multiplier
        stationarity = (
            reduced_gradient + turnover_penalty * trade
        )
        no_trade_mask = np.abs(trade) <= 1e-8
        subgradient_residual = float(
            np.linalg.norm(stationarity, ord=np.inf)
        )
    else:
        active_tolerance = 1e-7
        active = np.abs(trade) > active_tolerance
        if np.any(active):
            equality_multiplier = float(
                np.median(
                    -base_gradient[active]
                    - turnover_penalty * np.sign(trade[active])
                )
            )
        elif turnover_penalty > 0.0:
            lower = np.max(-turnover_penalty - base_gradient)
            upper = np.min(turnover_penalty - base_gradient)
            equality_multiplier = float(0.5 * (lower + upper))
        else:
            equality_multiplier = -float(np.mean(base_gradient))
        reduced_gradient = base_gradient + equality_multiplier
        subgradient = np.zeros(asset_count)
        subgradient[active] = np.sign(trade[active])
        if turnover_penalty > 0.0:
            subgradient[~active] = np.clip(
                -reduced_gradient[~active] / turnover_penalty,
                -1.0,
                1.0,
            )
        stationarity = (
            reduced_gradient + turnover_penalty * subgradient
        )
        no_trade_mask = ~active
        subgradient_residual = float(
            np.linalg.norm(stationarity, ord=np.inf)
        )
    budget_residual = float(abs(weights.sum() - 1.0))
    maximum_kkt_violation = max(
        budget_residual,
        subgradient_residual,
    )
    return {
        "weights": weights,
        "trade": trade,
        "turnover": float(np.abs(trade).sum()),
        "no_trade_positions": int(np.sum(no_trade_mask)),
        "no_trade_mask": no_trade_mask,
        "reduced_gradient": reduced_gradient,
        "equality_multiplier": equality_multiplier,
        "subgradient_residual": subgradient_residual,
        "maximum_kkt_violation": maximum_kkt_violation,
        "budget_residual": budget_residual,
        "penalty": penalty,
        "objective": float(result.fun),
        "status": (
            "PASS"
            if maximum_kkt_violation <= 1e-6
            else "CONDITIONAL_PASS"
        ),
    }
