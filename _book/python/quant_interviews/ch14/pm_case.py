from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass(frozen=True)
class CovarianceCertificate:
    volatility: float
    risk_contributions: NDArray[np.float64]
    contribution_residual: float
    minimum_eigenvalue: float
    condition_number: float | None
    status: Literal["PASS", "REJECT_SINGULAR", "REJECT_RECONCILIATION"]


@dataclass(frozen=True)
class CapitalCertificate:
    optimal_capital: float
    utility: float
    derivative_residual: float
    boundary: bool
    status: str


@dataclass(frozen=True)
class AllocationCertificate:
    weights: NDArray[np.float64]
    raw_expected_return: float
    shrunk_margin_adjusted_return: float
    net_expected_return: float
    volatility: float
    gross_traded_weight: float
    one_way_turnover: float
    linear_turnover_cost: float
    nonlinear_impact_cost: float
    borrow_cost: float
    risk_penalty: float
    objective: float
    budget_residual: float
    factor_residual: float
    maximum_bound_violation: float
    kkt_residual: float
    return_shrinkage: float
    solver_method: str
    solver_status: str
    iterations: int
    decision: Literal["PASS", "REJECT"]


@dataclass(frozen=True)
class FourSleeveFixture:
    names: tuple[str, ...]
    expected_returns: NDArray[np.float64]
    covariance: NDArray[np.float64]
    current_weights: NDArray[np.float64]
    factor_loadings: NDArray[np.float64]
    factor_target: float
    borrow_cost_rates: NDArray[np.float64]
    turnover_cost_rates: NDArray[np.float64]
    impact_coefficients: NDArray[np.float64]
    capacity_units: NDArray[np.float64]
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]


def canonical_four_sleeve_fixture() -> FourSleeveFixture:
    names = (
        "equity_stat_arb",
        "macro_trend",
        "volatility_relative_value",
        "defensive_carry",
    )
    volatilities = np.array([0.20, 0.16, 0.14, 0.10])
    correlation = np.array(
        [
            [1.00, 0.35, 0.10, 0.20],
            [0.35, 1.00, -0.05, 0.25],
            [0.10, -0.05, 1.00, 0.15],
            [0.20, 0.25, 0.15, 1.00],
        ]
    )
    covariance = correlation * np.outer(volatilities, volatilities)
    return FourSleeveFixture(
        names=names,
        expected_returns=np.array([0.080, 0.065, 0.055, 0.040]),
        covariance=covariance,
        current_weights=np.array([0.30, 0.25, 0.20, 0.25]),
        factor_loadings=np.array([-0.25, 0.15, 0.35, -0.05]),
        factor_target=0.05,
        borrow_cost_rates=np.array([0.0050, 0.0020, 0.0040, 0.0010]),
        turnover_cost_rates=np.array([0.0020, 0.0015, 0.0025, 0.0010]),
        impact_coefficients=np.array([0.010, 0.008, 0.012, 0.006]),
        capacity_units=np.array([4.0, 6.0, 3.0, 8.0]),
        lower_bounds=np.full(4, 0.05),
        upper_bounds=np.full(4, 0.60),
    )


def _validated_covariance(
    weights: NDArray[np.float64],
    covariance: NDArray[np.float64],
    *,
    symmetry_atol: float = 1e-12,
    psd_atol: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    if w.ndim != 1 or sigma.shape != (w.size, w.size) or w.size == 0:
        raise ValueError("weights and covariance have incompatible shapes")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(sigma)):
        raise ValueError("weights and covariance must be finite")
    if not np.allclose(sigma, sigma.T, atol=symmetry_atol, rtol=0.0):
        raise ValueError("covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(sigma)
    if float(eigenvalues[0]) < -psd_atol:
        raise ValueError("covariance is materially indefinite")
    return w, sigma


def covariance_certificate(
    weights: NDArray[np.float64],
    covariance: NDArray[np.float64],
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> CovarianceCertificate:
    w, sigma = _validated_covariance(weights, covariance, psd_atol=atol)
    eigenvalues = np.linalg.eigvalsh(sigma)
    minimum = float(eigenvalues[0])
    variance = float(w @ sigma @ w)
    if not isfinite(variance) or variance <= 0.0:
        raise ValueError("portfolio variance must be finite and positive")
    volatility = sqrt(variance)
    contributions = w * (sigma @ w) / volatility
    residual = float(np.sum(contributions) - volatility)
    if not np.all(np.isfinite(contributions)) or not isfinite(residual):
        raise FloatingPointError("risk-contribution reconciliation is non-finite")
    tolerance = atol + rtol * volatility
    if minimum <= atol:
        condition = None
        status: Literal[
            "PASS", "REJECT_SINGULAR", "REJECT_RECONCILIATION"
        ] = "REJECT_SINGULAR"
    else:
        condition = float(eigenvalues[-1] / minimum)
        status = "PASS" if abs(residual) <= tolerance else "REJECT_RECONCILIATION"
    return CovarianceCertificate(
        volatility=volatility,
        risk_contributions=contributions,
        contribution_residual=residual,
        minimum_eigenvalue=minimum,
        condition_number=condition,
        status=status,
    )


def portfolio_volatility(
    weights: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> float:
    certificate = covariance_certificate(weights, covariance)
    if certificate.status != "PASS":
        raise ValueError(f"covariance certificate rejected: {certificate.status}")
    return certificate.volatility


def risk_contributions(
    weights: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> NDArray[np.float64]:
    certificate = covariance_certificate(weights, covariance)
    if certificate.status != "PASS":
        raise ValueError(f"covariance certificate rejected: {certificate.status}")
    return certificate.risk_contributions


def capacity_utility(
    capital: float,
    alpha_rate: float,
    impact_coefficient: float,
    risk_penalty: float,
) -> float:
    if not all(
        isfinite(value)
        for value in (capital, alpha_rate, impact_coefficient, risk_penalty)
    ):
        raise ValueError("capacity inputs must be finite")
    if capital < 0.0 or impact_coefficient < 0.0 or risk_penalty < 0.0:
        raise ValueError("capital and penalty coefficients must be non-negative")
    result = (
        alpha_rate * capital
        - impact_coefficient * capital**1.5
        - risk_penalty * capital * capital
    )
    if not isfinite(result):
        raise FloatingPointError("capacity utility is non-finite")
    return result


def optimal_capital(
    alpha_rate: float,
    impact_coefficient: float,
    risk_penalty: float,
    *,
    maximum_capital: float | None = None,
) -> CapitalCertificate:
    values = (alpha_rate, impact_coefficient, risk_penalty)
    if not all(isfinite(value) for value in values):
        raise ValueError("capacity parameters must be finite")
    if alpha_rate < 0.0 or impact_coefficient < 0.0 or risk_penalty < 0.0:
        raise ValueError("capacity parameters must be non-negative")
    if maximum_capital is not None and (
        not isfinite(maximum_capital) or maximum_capital <= 0.0
    ):
        raise ValueError("maximum_capital must be finite and positive")
    if alpha_rate == 0.0:
        unconstrained = 0.0
    elif risk_penalty > 0.0:
        root = (
            -1.5 * impact_coefficient
            + sqrt(2.25 * impact_coefficient**2 + 8.0 * risk_penalty * alpha_rate)
        ) / (4.0 * risk_penalty)
        unconstrained = max(0.0, root) ** 2
    elif impact_coefficient > 0.0:
        unconstrained = (2.0 * alpha_rate / (3.0 * impact_coefficient)) ** 2
    else:
        unconstrained = float("inf")

    boundary = maximum_capital is not None and unconstrained >= maximum_capital
    selected = maximum_capital if boundary else unconstrained
    if not isfinite(selected):
        raise ValueError("utility is unbounded without impact, risk or capital limit")
    derivative = (
        alpha_rate
        - 1.5 * impact_coefficient * sqrt(selected)
        - 2.0 * risk_penalty * selected
    )
    status = "CONDITIONAL_PASS" if boundary else "PASS"
    return CapitalCertificate(
        optimal_capital=selected,
        utility=capacity_utility(
            selected, alpha_rate, impact_coefficient, risk_penalty
        ),
        derivative_residual=derivative,
        boundary=boundary,
        status=status,
    )


def _vector(
    value: NDArray[np.float64] | None,
    size: int,
    *,
    default: float,
    name: str,
    nonnegative: bool = True,
) -> NDArray[np.float64]:
    result = np.full(size, default) if value is None else np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return result


def allocate_sleeves(
    expected_returns: NDArray[np.float64],
    covariance: NDArray[np.float64],
    current_weights: NDArray[np.float64],
    factor_loadings: NDArray[np.float64],
    factor_target: float,
    *,
    risk_aversion: float,
    borrow_cost_rates: NDArray[np.float64] | None = None,
    turnover_cost_rates: NDArray[np.float64] | None = None,
    impact_coefficients: NDArray[np.float64] | None = None,
    capacity_units: NDArray[np.float64] | None = None,
    lower_bounds: NDArray[np.float64] | None = None,
    upper_bounds: NDArray[np.float64] | None = None,
    return_shrinkage: float = 1.0,
    uncertainty_margin: float = 0.0,
    tolerance: float = 1.0e-7,
) -> AllocationCertificate:
    mu = np.asarray(expected_returns, dtype=float)
    w0 = np.asarray(current_weights, dtype=float)
    factor = np.asarray(factor_loadings, dtype=float)
    _, sigma = _validated_covariance(w0, covariance)
    size = w0.size
    if mu.shape != w0.shape or factor.shape != w0.shape:
        raise ValueError("return, weight and factor vectors must align")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(factor)):
        raise ValueError("allocation inputs must be finite")
    scalars = (factor_target, risk_aversion, return_shrinkage, uncertainty_margin, tolerance)
    if not all(isfinite(value) for value in scalars):
        raise ValueError("allocation configuration must be finite")
    if (
        risk_aversion <= 0.0
        or not 0.0 <= return_shrinkage <= 1.0
        or uncertainty_margin < 0.0
        or tolerance <= 0.0
    ):
        raise ValueError("invalid risk, shrinkage, margin or tolerance")

    borrow = _vector(
        borrow_cost_rates, size, default=0.0, name="borrow_cost_rates"
    )
    linear_rate = _vector(
        turnover_cost_rates, size, default=0.0, name="turnover_cost_rates"
    )
    impact = _vector(
        impact_coefficients, size, default=0.0, name="impact_coefficients"
    )
    capacity = _vector(
        capacity_units, size, default=1.0, name="capacity_units"
    )
    if np.any(capacity <= 0.0):
        raise ValueError("capacity_units must be strictly positive")
    lower = _vector(
        lower_bounds, size, default=0.0, name="lower_bounds", nonnegative=False
    )
    upper = _vector(
        upper_bounds, size, default=1.0, name="upper_bounds", nonnegative=False
    )
    if np.any(lower >= upper) or np.any(w0 < lower) or np.any(w0 > upper):
        raise ValueError("bounds are invalid or exclude current_weights")
    if abs(float(np.sum(w0)) - 1.0) > tolerance:
        raise ValueError("current_weights must satisfy the budget")
    shrunk_mu = return_shrinkage * mu - uncertainty_margin

    def components(weights: NDArray[np.float64]) -> tuple[float, ...]:
        trade = weights - w0
        raw_return = float(weights @ mu)
        governed_return = float(weights @ shrunk_mu)
        linear = float(linear_rate @ np.abs(trade))
        nonlinear = float(np.sum(impact * np.abs(trade) ** 1.5 / np.sqrt(capacity)))
        borrow_cost = float(borrow @ np.maximum(-weights, 0.0))
        risk = 0.5 * risk_aversion * float(weights @ sigma @ weights)
        objective_value = governed_return - linear - nonlinear - borrow_cost - risk
        return (
            raw_return,
            governed_return,
            linear,
            nonlinear,
            borrow_cost,
            risk,
            objective_value,
        )

    def objective(weights: NDArray[np.float64]) -> float:
        return -components(weights)[-1]

    constraints = (
        {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        {
            "type": "eq",
            "fun": lambda weights: float(weights @ factor - factor_target),
        },
    )
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=constraints,
        options={"ftol": tolerance * 0.01, "maxiter": 1_000},
    )
    if not result.success:
        raise RuntimeError(f"allocation solver rejected: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    if not np.all(np.isfinite(weights)):
        raise FloatingPointError("allocation solver returned non-finite weights")
    (
        raw_return,
        governed_return,
        linear,
        nonlinear,
        borrow_cost,
        risk,
        utility,
    ) = components(weights)
    trade = weights - w0
    budget_residual = float(np.sum(weights) - 1.0)
    factor_residual = float(weights @ factor - factor_target)
    bound_violation = float(
        max(np.max(lower - weights), np.max(weights - upper), 0.0)
    )

    sign_trade = np.sign(trade)
    gradient = (
        -shrunk_mu
        + risk_aversion * (sigma @ weights)
        + linear_rate * sign_trade
        + 1.5
        * impact
        * np.sqrt(np.abs(trade))
        * sign_trade
        / np.sqrt(capacity)
        - borrow * (weights < 0.0)
    )
    equality = np.column_stack((np.ones(size), factor))
    free = (weights > lower + tolerance) & (weights < upper - tolerance)
    solve_rows = equality[free] if np.any(free) else equality
    solve_gradient = gradient[free] if np.any(free) else gradient
    multipliers = np.linalg.lstsq(solve_rows, -solve_gradient, rcond=None)[0]
    reduced = gradient + equality @ multipliers
    residuals = np.where(
        weights <= lower + tolerance,
        np.minimum(reduced, 0.0),
        np.where(weights >= upper - tolerance, np.maximum(reduced, 0.0), reduced),
    )
    kkt_residual = float(np.max(np.abs(residuals)))
    values = (
        raw_return,
        governed_return,
        linear,
        nonlinear,
        borrow_cost,
        risk,
        utility,
        budget_residual,
        factor_residual,
        bound_violation,
        kkt_residual,
    )
    if not all(isfinite(value) for value in values):
        raise FloatingPointError("allocation post-condition is non-finite")
    decision: Literal["PASS", "REJECT"] = (
        "PASS"
        if abs(budget_residual) <= tolerance
        and abs(factor_residual) <= tolerance
        and bound_violation <= tolerance
        and kkt_residual <= 1.0e-4
        else "REJECT"
    )
    volatility = sqrt(float(weights @ sigma @ weights))
    return AllocationCertificate(
        weights=weights,
        raw_expected_return=raw_return,
        shrunk_margin_adjusted_return=governed_return,
        net_expected_return=governed_return - linear - nonlinear - borrow_cost,
        volatility=volatility,
        gross_traded_weight=float(np.sum(np.abs(trade))),
        one_way_turnover=0.5 * float(np.sum(np.abs(trade))),
        linear_turnover_cost=linear,
        nonlinear_impact_cost=nonlinear,
        borrow_cost=borrow_cost,
        risk_penalty=risk,
        objective=utility,
        budget_residual=budget_residual,
        factor_residual=factor_residual,
        maximum_bound_violation=bound_violation,
        kkt_residual=kkt_residual,
        return_shrinkage=return_shrinkage,
        solver_method="SLSQP",
        solver_status=str(result.message),
        iterations=int(result.nit),
        decision=decision,
    )
