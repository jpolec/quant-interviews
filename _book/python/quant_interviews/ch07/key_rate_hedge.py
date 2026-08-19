from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear

from .bond_risk import BASIS_POINT, BondCashflows
from .curve_bootstrap import (
    DiscountCurve,
    bootstrap_curve_certificate,
)
from .evidence import (
    ConventionSchema,
    EvidenceCertificate,
    InvariantResult,
    TolerancePolicy,
    Violation,
    decision_from_invariants,
    stable_manifest_hash,
)


@dataclass(frozen=True)
class HedgeResult:
    notionals: NDArray[np.float64]
    residual_dv01: NDArray[np.float64]
    residual_norm: float
    condition_number: float
    solver: str
    weighted_residual_norm: float
    objective_value: float
    active_constraints: tuple[str, ...]


@dataclass(frozen=True)
class KeyRateHedgeResult:
    quote_coordinates: tuple[str, ...]
    base_curve: DiscountCurve
    portfolio_key_rate_dv01: NDArray[np.float64]
    hedge_key_rate_dv01: NDArray[np.float64]
    hedge_notionals: NDArray[np.float64]
    rounded_notionals: NDArray[np.float64]
    residual_before_rounding: NDArray[np.float64]
    residual_after_rounding: NDArray[np.float64]
    hedge_matrix_condition_number: float
    maximum_curve_repricing_residual: float
    weighted_objective_before_rounding: float
    weighted_objective_after_rounding: float
    active_constraints: tuple[str, ...]
    tradable_hedges: tuple[bool, ...]


def _weight_square_root(
    bucket_weights: NDArray[np.float64] | None,
    bucket_count: int,
) -> NDArray[np.float64]:
    if bucket_weights is None:
        return np.eye(bucket_count)
    weights = np.asarray(bucket_weights, dtype=float)
    if not np.all(np.isfinite(weights)):
        raise ValueError("bucket_weights must be finite")
    if weights.ndim == 1:
        if weights.shape != (bucket_count,) or np.any(weights <= 0.0):
            raise ValueError("vector bucket_weights must be positive per bucket")
        return np.diag(np.sqrt(weights))
    if weights.shape != (bucket_count, bucket_count):
        raise ValueError("matrix bucket_weights must be square by bucket")
    if not np.allclose(weights, weights.T, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("matrix bucket_weights must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(weights)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("matrix bucket_weights must be positive definite")
    return np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def solve_key_rate_hedge(
    portfolio_dv01: NDArray[np.float64],
    hedge_dv01_matrix: NDArray[np.float64],
    ridge_penalty: float = 0.0,
    *,
    bucket_weights: NDArray[np.float64] | None = None,
    lower_bounds: NDArray[np.float64] | None = None,
    upper_bounds: NDArray[np.float64] | None = None,
    tradable: NDArray[np.bool_] | None = None,
) -> HedgeResult:
    """Solve a weighted, bounded ridge hedge without normal equations."""

    b = np.asarray(portfolio_dv01, dtype=float)
    h = np.asarray(hedge_dv01_matrix, dtype=float)
    if h.ndim != 2 or b.ndim != 1 or h.shape[0] != b.size:
        raise ValueError("hedge matrix rows must match portfolio DV01 buckets")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(b)):
        raise ValueError("hedge inputs must be finite")
    if not np.isfinite(ridge_penalty) or ridge_penalty < 0.0:
        raise ValueError("ridge_penalty must be finite and non-negative")
    hedge_count = h.shape[1]
    sqrt_weight = _weight_square_root(bucket_weights, b.size)
    weighted_h = sqrt_weight @ h
    weighted_target = -(sqrt_weight @ b)
    lower = (
        np.full(hedge_count, -np.inf)
        if lower_bounds is None
        else np.asarray(lower_bounds, dtype=float)
    )
    upper = (
        np.full(hedge_count, np.inf)
        if upper_bounds is None
        else np.asarray(upper_bounds, dtype=float)
    )
    tradable_mask = (
        np.ones(hedge_count, dtype=bool)
        if tradable is None
        else np.asarray(tradable, dtype=bool)
    )
    if (
        lower.shape != (hedge_count,)
        or upper.shape != (hedge_count,)
        or tradable_mask.shape != (hedge_count,)
        or np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower > upper)
    ):
        raise ValueError("invalid hedge bounds or tradability mask")
    equal_bounds = np.isclose(lower, upper, atol=0.0, rtol=0.0)
    fixed = ~tradable_mask | equal_bounds
    fixed_values = np.zeros(hedge_count)
    fixed_values[equal_bounds & np.isfinite(lower)] = lower[
        equal_bounds & np.isfinite(lower)
    ]
    if np.any(
        fixed
        & (
            (fixed_values < lower)
            | (fixed_values > upper)
        )
    ):
        raise ValueError("non-tradable hedge has bounds that exclude zero")
    free = ~fixed
    notionals = fixed_values.copy()
    adjusted_target = weighted_target - weighted_h[:, fixed] @ notionals[fixed]
    free_h = weighted_h[:, free]
    if ridge_penalty > 0.0:
        design = np.vstack(
            [free_h, sqrt(ridge_penalty) * np.eye(int(np.sum(free)))]
        )
        target = np.r_[adjusted_target, np.zeros(int(np.sum(free)))]
    else:
        design = free_h
        target = adjusted_target
    finite_bounds = np.any(np.isfinite(lower[free])) or np.any(
        np.isfinite(upper[free])
    )
    if not np.any(free):
        solver = "fixed_constraints"
    elif finite_bounds:
        solution = lsq_linear(
            design,
            target,
            bounds=(lower[free], upper[free]),
            method="trf",
            lsmr_tol="auto",
        )
        if not solution.success or not np.all(np.isfinite(solution.x)):
            raise ArithmeticError("bounded key-rate hedge solve failed")
        notionals[free] = solution.x
        solver = (
            "weighted_bounded_augmented_lsq"
            if ridge_penalty > 0.0
            else "weighted_bounded_lsq"
        )
    else:
        free_solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        notionals[free] = free_solution
        solver = (
            "augmented_svd_ridge"
            if ridge_penalty > 0.0
            else "svd_least_squares"
        )
    residual = b + h @ notionals
    weighted_residual = sqrt_weight @ residual
    active: list[str] = [
        f"hedge:{index}:nontradable"
        for index in range(hedge_count)
        if not tradable_mask[index]
    ]
    for index in range(hedge_count):
        scale = max(1.0, abs(notionals[index]))
        if np.isfinite(lower[index]) and abs(notionals[index] - lower[index]) <= 1.0e-10 * scale:
            active.append(f"hedge:{index}:lower")
        if np.isfinite(upper[index]) and abs(notionals[index] - upper[index]) <= 1.0e-10 * scale:
            active.append(f"hedge:{index}:upper")
    return HedgeResult(
        notionals=notionals,
        residual_dv01=residual,
        residual_norm=float(np.linalg.norm(residual)),
        condition_number=float(np.linalg.cond(weighted_h)),
        solver=solver,
        weighted_residual_norm=float(np.linalg.norm(weighted_residual)),
        objective_value=float(
            np.dot(weighted_residual, weighted_residual)
            + ridge_penalty * np.dot(notionals, notionals)
        ),
        active_constraints=tuple(active),
    )


def _cashflow_price(
    bond: BondCashflows,
    curve: DiscountCurve,
) -> float:
    if bond.times[-1] > curve.times[-1]:
        raise ValueError("cash flow lies beyond the calibrated curve")
    return float(
        np.dot(
            bond.amounts,
            np.array([curve.df(float(time)) for time in bond.times]),
        )
    )


def _quote_coordinates(
    deposit_quotes: dict[float, float],
    swap_quotes: dict[int, float],
) -> tuple[tuple[str, float | int], ...]:
    deposits = tuple(
        (f"deposit:{maturity:g}Y", maturity)
        for maturity in sorted(deposit_quotes)
    )
    swaps = tuple(
        (f"swap:{maturity:d}Y", maturity)
        for maturity in sorted(swap_quotes)
    )
    return deposits + swaps


def _bumped_curve(
    deposit_quotes: dict[float, float],
    swap_quotes: dict[int, float],
    coordinate: str,
    maturity: float | int,
    bump: float,
) -> DiscountCurve:
    deposits = dict(deposit_quotes)
    swaps = dict(swap_quotes)
    if coordinate.startswith("deposit:"):
        deposits[float(maturity)] += bump
    else:
        swaps[int(maturity)] += bump
    return bootstrap_curve_certificate(deposits, swaps).result.curve


def key_rate_hedge_certificate(
    *,
    portfolio_bonds: tuple[BondCashflows, ...],
    portfolio_notionals: NDArray[np.float64],
    hedge_bonds: tuple[BondCashflows, ...],
    deposit_quotes: dict[float, float],
    swap_quotes: dict[int, float],
    bump_bp: float = 1.0,
    ridge_penalty: float = 0.0,
    notional_rounding: float = 1.0e-4,
    bucket_weights: NDArray[np.float64] | None = None,
    lower_bounds: NDArray[np.float64] | None = None,
    upper_bounds: NDArray[np.float64] | None = None,
    lot_sizes: NDArray[np.float64] | None = None,
    tradable: NDArray[np.bool_] | None = None,
    tolerance: TolerancePolicy = TolerancePolicy(1.0e-8, 1.0e-6),
) -> EvidenceCertificate[KeyRateHedgeResult]:
    notionals = np.asarray(portfolio_notionals, dtype=float)
    if notionals.shape != (len(portfolio_bonds),):
        raise ValueError("portfolio_notionals must match portfolio_bonds")
    if not portfolio_bonds or not hedge_bonds:
        raise ValueError("portfolio and hedge instruments must be non-empty")
    if (
        not np.all(np.isfinite(notionals))
        or not np.isfinite(bump_bp)
        or not np.isfinite(notional_rounding)
        or bump_bp <= 0.0
        or notional_rounding <= 0.0
    ):
        raise ValueError("notionals, bump and notional_rounding must be finite")

    base_certificate = bootstrap_curve_certificate(
        deposit_quotes,
        swap_quotes,
    )
    coordinates = _quote_coordinates(deposit_quotes, swap_quotes)
    bond_universe = portfolio_bonds + hedge_bonds
    key_rate_rows: list[list[float]] = []
    bump = bump_bp * BASIS_POINT
    for coordinate, maturity in coordinates:
        curve_down = _bumped_curve(
            deposit_quotes,
            swap_quotes,
            coordinate,
            maturity,
            -bump,
        )
        curve_up = _bumped_curve(
            deposit_quotes,
            swap_quotes,
            coordinate,
            maturity,
            bump,
        )
        row = [
            (
                _cashflow_price(bond, curve_down)
                - _cashflow_price(bond, curve_up)
            )
            / (2.0 * bump_bp)
            for bond in bond_universe
        ]
        key_rate_rows.append(row)

    key_rate_matrix = np.asarray(key_rate_rows, dtype=float)
    portfolio_columns = key_rate_matrix[:, : len(portfolio_bonds)]
    hedge_columns = key_rate_matrix[:, len(portfolio_bonds) :]
    portfolio_risk = portfolio_columns @ notionals
    hedge = solve_key_rate_hedge(
        portfolio_risk,
        hedge_columns,
        ridge_penalty,
        bucket_weights=bucket_weights,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        tradable=tradable,
    )
    hedge_count = len(hedge_bonds)
    lots = (
        np.full(hedge_count, notional_rounding)
        if lot_sizes is None
        else np.asarray(lot_sizes, dtype=float)
    )
    if (
        lots.shape != (hedge_count,)
        or not np.all(np.isfinite(lots))
        or np.any(lots <= 0.0)
    ):
        raise ValueError("lot_sizes must be finite and positive per hedge")
    lower = (
        np.full(hedge_count, -np.inf)
        if lower_bounds is None
        else np.asarray(lower_bounds, dtype=float)
    )
    upper = (
        np.full(hedge_count, np.inf)
        if upper_bounds is None
        else np.asarray(upper_bounds, dtype=float)
    )
    tradable_mask = (
        np.ones(hedge_count, dtype=bool)
        if tradable is None
        else np.asarray(tradable, dtype=bool)
    )
    rounded = np.round(hedge.notionals / lots) * lots
    rounded = np.minimum(np.maximum(rounded, lower), upper)
    rounded[~tradable_mask] = 0.0
    rounded_residual = portfolio_risk + hedge_columns @ rounded
    sqrt_weight = _weight_square_root(bucket_weights, portfolio_risk.size)
    weighted_rounded_residual = sqrt_weight @ rounded_residual
    rounded_objective = float(
        np.dot(weighted_rounded_residual, weighted_rounded_residual)
        + ridge_penalty * np.dot(rounded, rounded)
    )
    risk_scale = max(
        1.0,
        float(np.linalg.norm(sqrt_weight @ portfolio_risk)),
    )
    residual_limit = tolerance.limit(risk_scale)
    rounded_limit = tolerance.conditional_multiplier * residual_limit
    invariants = (
        InvariantResult(
            "base_curve_repricing",
            base_certificate.result.maximum_repricing_residual,
            0.0,
            tolerance.limit(1.0),
            base_certificate.result.maximum_repricing_residual
            <= tolerance.limit(1.0),
        ),
        InvariantResult(
            "hedge_residual_before_rounding",
            hedge.weighted_residual_norm,
            0.0,
            residual_limit,
            hedge.weighted_residual_norm <= residual_limit,
        ),
        InvariantResult(
            "hedge_residual_after_rounding",
            float(np.linalg.norm(weighted_rounded_residual)),
            0.0,
            rounded_limit,
            float(np.linalg.norm(weighted_rounded_residual)) <= rounded_limit,
        ),
    )
    conditional = (
        float(np.linalg.norm(weighted_rounded_residual)) > residual_limit
        or hedge.condition_number > 1.0e8
    )
    violations = tuple(
        Violation(
            code=f"KRD_{invariant.name.upper()}",
            message=f"key-rate invariant failed: {invariant.name}",
            observed=str(invariant.observed),
            limit=str(invariant.tolerance),
        )
        for invariant in invariants
        if not invariant.passed
    )
    if conditional and all(invariant.passed for invariant in invariants):
        if float(np.linalg.norm(weighted_rounded_residual)) > residual_limit:
            violations += (
                Violation(
                    code="KRD_ROUNDING_EXCEPTION",
                    message="rounded hedge exceeds the primary residual limit",
                    observed=str(float(np.linalg.norm(weighted_rounded_residual))),
                    limit=str(residual_limit),
                ),
            )
        if hedge.condition_number > 1.0e8:
            violations += (
                Violation(
                    code="KRD_CONDITIONING_EXCEPTION",
                    message="weighted hedge matrix exceeds the conditioning policy",
                    observed=str(hedge.condition_number),
                    limit="100000000.0",
                ),
            )
    result = KeyRateHedgeResult(
        quote_coordinates=tuple(name for name, _ in coordinates),
        base_curve=base_certificate.result.curve,
        portfolio_key_rate_dv01=portfolio_risk,
        hedge_key_rate_dv01=hedge_columns,
        hedge_notionals=hedge.notionals,
        rounded_notionals=rounded,
        residual_before_rounding=hedge.residual_dv01,
        residual_after_rounding=rounded_residual,
        hedge_matrix_condition_number=hedge.condition_number,
        maximum_curve_repricing_residual=(
            base_certificate.result.maximum_repricing_residual
        ),
        weighted_objective_before_rounding=hedge.objective_value,
        weighted_objective_after_rounding=rounded_objective,
        active_constraints=hedge.active_constraints,
        tradable_hedges=tuple(bool(value) for value in tradable_mask),
    )
    return EvidenceCertificate(
        status=decision_from_invariants(
            invariants,
            conditional=conditional,
        ),
        result=result,
        invariants=invariants,
        violations=violations,
        conventions=ConventionSchema(
            currency="price units per one basis point",
            rate_compounding="instrument quote conventions",
            day_count="annual educational fixture",
            discounting_curve="bump, rebuild and reprice",
        ),
        tolerances=tolerance,
        input_manifest_hash=stable_manifest_hash(
            {
                "portfolio_bonds": portfolio_bonds,
                "portfolio_notionals": notionals,
                "hedge_bonds": hedge_bonds,
                "deposit_quotes": deposit_quotes,
                "swap_quotes": swap_quotes,
                "bump_bp": bump_bp,
                "ridge_penalty": ridge_penalty,
                "notional_rounding": notional_rounding,
                "bucket_weights": bucket_weights,
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
                "lot_sizes": lots,
                "tradable": tradable_mask,
                "tolerance": tolerance,
            }
        ),
        implementation_version="ch07.key_rate_hedge.v0.10",
    )


if __name__ == "__main__":
    portfolio = BondCashflows(
        times=np.array([1.0, 2.0, 3.0]),
        amounts=np.array([3.0, 3.0, 103.0]),
    )
    hedges = (
        BondCashflows(np.array([1.0]), np.array([100.0])),
        BondCashflows(np.array([2.0]), np.array([100.0])),
        BondCashflows(np.array([3.0]), np.array([100.0])),
    )
    certificate = key_rate_hedge_certificate(
        portfolio_bonds=(portfolio,),
        portfolio_notionals=np.array([1.0]),
        hedge_bonds=hedges,
        deposit_quotes={1.0: 0.02},
        swap_quotes={2: 0.025, 3: 0.03},
    )
    print("status=", certificate.status)
    print("rounded residual=", certificate.result.residual_after_rounding)
