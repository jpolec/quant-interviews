from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .evidence import (
    ConventionSchema,
    EvidenceCertificate,
    InvariantResult,
    TolerancePolicy,
    Violation,
    decision_from_invariants,
    stable_manifest_hash,
)

BASIS_POINT = 1.0e-4


@dataclass(frozen=True)
class CallableBond:
    face: float
    coupon_rate: float
    maturity_years: int
    call_price: float
    first_call_year: int

    def __post_init__(self) -> None:
        if (
            not isfinite(self.face)
            or not isfinite(self.coupon_rate)
            or not isfinite(self.call_price)
            or self.face <= 0.0
            or self.coupon_rate < 0.0
            or self.call_price <= 0.0
            or self.maturity_years <= 0
        ):
            raise ValueError("invalid callable-bond terms")
        if not (1 <= self.first_call_year < self.maturity_years):
            raise ValueError(
                "first_call_year must be before maturity; maturity redemption is not a call date"
            )


@dataclass(frozen=True)
class CalibratedRateTree:
    market_discount_factors: NDArray[np.float64]
    rate_step: float
    level_shifts: NDArray[np.float64]
    reproduced_discount_factors: NDArray[np.float64]
    maximum_curve_residual: float


@dataclass(frozen=True)
class TreePriceResult:
    price: float
    exercise_masks: tuple[NDArray[np.bool_], ...]
    exercise_probability_by_year: NDArray[np.float64]
    exercise_boundary_rate_by_year: tuple[float | None, ...]
    first_nonexercise_rate_by_year: tuple[float | None, ...]
    exercise_rule: str


@dataclass(frozen=True)
class CallableBondOASResult:
    calibrated_tree: CalibratedRateTree
    market_price: float
    straight_price: float
    callable_price: float
    embedded_option_value: float
    oas: float
    oas_repricing_residual: float
    effective_duration: float
    effective_convexity: float
    exercise_probability_by_year: NDArray[np.float64]
    exercise_boundary_rate_by_year: tuple[float | None, ...]
    first_nonexercise_rate_by_year: tuple[float | None, ...]
    exercise_rule: str
    solver_iterations: int
    oas_bracket: tuple[float, float]
    market_price_source: str
    market_price_evidence_hash: str | None


def calibrate_additive_short_rate_tree(
    market_discount_factors: NDArray[np.float64],
    rate_step: float,
) -> CalibratedRateTree:
    dfs = np.asarray(market_discount_factors, dtype=float)
    if (
        dfs.ndim != 1
        or dfs.size == 0
        or not np.all(np.isfinite(dfs))
        or np.any(dfs <= 0.0)
        or not isfinite(rate_step)
        or rate_step < 0.0
    ):
        raise ValueError("invalid tree calibration inputs")
    state_prices = np.array([1.0])
    shifts: list[float] = []
    reproduced: list[float] = []
    for step, target_df in enumerate(dfs):
        state_indices = np.arange(step + 1)
        factors = np.exp(
            -(2.0 * state_indices - step) * rate_step
        )
        unshifted_discounted_mass = float(np.dot(state_prices, factors))
        shift = log(unshifted_discounted_mass / target_df)
        shifts.append(shift)
        discounted_state_prices = (
            state_prices
            * np.exp(
                -(
                    shift
                    + (2.0 * state_indices - step) * rate_step
                )
            )
        )
        next_state_prices = np.zeros(step + 2)
        next_state_prices[:-1] += 0.5 * discounted_state_prices
        next_state_prices[1:] += 0.5 * discounted_state_prices
        state_prices = next_state_prices
        reproduced.append(float(np.sum(state_prices)))
    reproduced_array = np.asarray(reproduced)
    return CalibratedRateTree(
        market_discount_factors=dfs,
        rate_step=rate_step,
        level_shifts=np.asarray(shifts),
        reproduced_discount_factors=reproduced_array,
        maximum_curve_residual=float(
            np.max(np.abs(reproduced_array - dfs))
        ),
    )


def price_on_calibrated_tree(
    bond: CallableBond,
    tree: CalibratedRateTree,
    *,
    callable_: bool,
    oas: float = 0.0,
) -> TreePriceResult:
    if tree.market_discount_factors.size != bond.maturity_years:
        raise ValueError("tree horizon must equal bond maturity")
    if not isfinite(oas):
        raise ValueError("oas must be finite")
    coupon = bond.face * bond.coupon_rate
    values = np.full(bond.maturity_years + 1, bond.face)
    masks: list[NDArray[np.bool_]] = [
        np.zeros(step + 1, dtype=bool)
        for step in range(bond.maturity_years)
    ]
    boundaries: list[float | None] = [None] * bond.maturity_years
    first_nonexercise_rates: list[float | None] = [None] * bond.maturity_years
    for step in range(bond.maturity_years - 1, -1, -1):
        state_indices = np.arange(step + 1)
        rates = (
            tree.level_shifts[step]
            + (2.0 * state_indices - step) * tree.rate_step
        )
        continuation = np.exp(-(rates + oas)) * 0.5 * (
            values[:-1] + coupon + values[1:] + coupon
        )
        exercise = np.zeros(step + 1, dtype=bool)
        if callable_ and step >= bond.first_call_year:
            exercise = continuation > bond.call_price
            continuation = np.minimum(continuation, bond.call_price)
            if np.any(exercise):
                boundaries[step] = float(np.max(rates[exercise]))
                if np.any(~exercise):
                    first_nonexercise_rates[step] = float(
                        np.min(rates[~exercise])
                    )
        masks[step] = exercise
        values = continuation

    alive = np.array([1.0])
    exercise_probabilities = np.zeros(bond.maturity_years)
    for step in range(bond.maturity_years):
        exercise = masks[step]
        exercise_probabilities[step] = float(np.sum(alive[exercise]))
        survivors = alive.copy()
        survivors[exercise] = 0.0
        next_alive = np.zeros(step + 2)
        next_alive[:-1] += 0.5 * survivors
        next_alive[1:] += 0.5 * survivors
        alive = next_alive
    return TreePriceResult(
        price=float(values[0]),
        exercise_masks=tuple(masks),
        exercise_probability_by_year=exercise_probabilities,
        exercise_boundary_rate_by_year=tuple(boundaries),
        first_nonexercise_rate_by_year=tuple(first_nonexercise_rates),
        exercise_rule="issuer exercises iff short rate <= boundary on eligible dates",
    )


def callable_bond_oas_certificate(
    *,
    bond: CallableBond,
    market_discount_factors: NDArray[np.float64],
    rate_step: float,
    market_price: float,
    market_price_source: Literal[
        "independent", "self_roundtrip", "unclassified"
    ] = "unclassified",
    market_price_evidence_hash: str | None = None,
    bump_bp: float = 10.0,
    initial_oas_bracket: tuple[float, float] = (-0.05, 0.05),
    maximum_absolute_oas: float = 5.0,
    tolerance: TolerancePolicy = TolerancePolicy(1.0e-10, 1.0e-9),
) -> EvidenceCertificate[CallableBondOASResult]:
    if not isfinite(market_price) or market_price <= 0.0:
        raise ValueError("market_price must be positive and finite")
    if market_price_source not in {
        "independent",
        "self_roundtrip",
        "unclassified",
    }:
        raise ValueError("invalid market_price_source")
    if market_price_source == "independent":
        if (
            market_price_evidence_hash is None
            or len(market_price_evidence_hash) != 64
            or market_price_evidence_hash.lower() != market_price_evidence_hash
            or any(character not in "0123456789abcdef" for character in market_price_evidence_hash)
        ):
            raise ValueError(
                "independent market price requires a lowercase SHA-256 evidence hash"
            )
    elif market_price_evidence_hash is not None:
        raise ValueError(
            "market_price_evidence_hash is reserved for independent evidence"
        )
    if bump_bp <= 0.0 or maximum_absolute_oas <= 0.0:
        raise ValueError("invalid OAS audit configuration")
    lower, upper = initial_oas_bracket
    if not lower < upper:
        raise ValueError("invalid initial OAS bracket")
    tree = calibrate_additive_short_rate_tree(
        market_discount_factors,
        rate_step,
    )

    def residual(spread: float) -> float:
        return (
            price_on_calibrated_tree(
                bond,
                tree,
                callable_=True,
                oas=spread,
            ).price
            - market_price
        )

    lower_value = residual(lower)
    upper_value = residual(upper)
    while lower_value * upper_value > 0.0 and (
        abs(lower) < maximum_absolute_oas
        or abs(upper) < maximum_absolute_oas
    ):
        lower = max(2.0 * lower, -maximum_absolute_oas)
        upper = min(2.0 * upper, maximum_absolute_oas)
        lower_value = residual(lower)
        upper_value = residual(upper)
    no_bracket = lower_value * upper_value > 0.0
    if no_bracket:
        oas = float("nan")
        iterations = 0
        callable_result = price_on_calibrated_tree(
            bond,
            tree,
            callable_=True,
        )
        oas_residual = callable_result.price - market_price
    else:
        oas, solver = brentq(
            residual,
            lower,
            upper,
            xtol=tolerance.absolute,
            rtol=max(tolerance.relative, 4.0 * np.finfo(float).eps),
            full_output=True,
        )
        iterations = int(solver.iterations)
        callable_result = price_on_calibrated_tree(
            bond,
            tree,
            callable_=True,
            oas=float(oas),
        )
        oas_residual = callable_result.price - market_price
    straight_result = price_on_calibrated_tree(
        bond,
        tree,
        callable_=False,
        oas=0.0 if no_bracket else float(oas),
    )

    times = np.arange(1.0, bond.maturity_years + 1.0)
    bump = bump_bp * BASIS_POINT
    down_dfs = tree.market_discount_factors * np.exp(bump * times)
    up_dfs = tree.market_discount_factors * np.exp(-bump * times)
    down_tree = calibrate_additive_short_rate_tree(down_dfs, rate_step)
    up_tree = calibrate_additive_short_rate_tree(up_dfs, rate_step)
    spread = 0.0 if no_bracket else float(oas)
    down_price = price_on_calibrated_tree(
        bond,
        down_tree,
        callable_=True,
        oas=spread,
    ).price
    up_price = price_on_calibrated_tree(
        bond,
        up_tree,
        callable_=True,
        oas=spread,
    ).price
    base_price = callable_result.price
    effective_duration = (
        (down_price - up_price) / (2.0 * base_price * bump)
    )
    effective_convexity = (
        (down_price + up_price - 2.0 * base_price)
        / (base_price * bump**2)
    )
    curve_limit = tolerance.limit(
        float(np.max(tree.market_discount_factors))
    )
    price_limit = tolerance.limit(market_price)
    invariants = (
        InvariantResult(
            "initial_curve_reproduction",
            tree.maximum_curve_residual,
            0.0,
            curve_limit,
            tree.maximum_curve_residual <= curve_limit,
        ),
        InvariantResult(
            "oas_bracket",
            not no_bracket,
            True,
            None,
            not no_bracket,
        ),
        InvariantResult(
            "oas_market_repricing",
            abs(oas_residual),
            0.0,
            price_limit,
            not no_bracket and abs(oas_residual) <= price_limit,
        ),
        InvariantResult(
            "callable_not_above_straight",
            callable_result.price - straight_result.price,
            "<= 0",
            price_limit,
            callable_result.price <= straight_result.price + price_limit,
        ),
    )
    violations = tuple(
        Violation(
            code=f"OAS_{invariant.name.upper()}",
            message=f"callable-bond invariant failed: {invariant.name}",
            observed=str(invariant.observed),
            limit=str(invariant.expected),
        )
        for invariant in invariants
        if not invariant.passed
    )
    if market_price_source != "independent" and all(
        invariant.passed for invariant in invariants
    ):
        violations += (
            Violation(
                code="OAS_NONINDEPENDENT_MARKET_PRICE",
                message=(
                    "solver evidence is reproducible, but the market-price source "
                    "is not an independent validation benchmark"
                ),
                observed=market_price_source,
                limit="independent",
            ),
        )
    result = CallableBondOASResult(
        calibrated_tree=tree,
        market_price=market_price,
        straight_price=straight_result.price,
        callable_price=callable_result.price,
        embedded_option_value=(
            straight_result.price - callable_result.price
        ),
        oas=float(oas),
        oas_repricing_residual=float(oas_residual),
        effective_duration=float(effective_duration),
        effective_convexity=float(effective_convexity),
        exercise_probability_by_year=(
            callable_result.exercise_probability_by_year
        ),
        exercise_boundary_rate_by_year=(
            callable_result.exercise_boundary_rate_by_year
        ),
        first_nonexercise_rate_by_year=(
            callable_result.first_nonexercise_rate_by_year
        ),
        exercise_rule=callable_result.exercise_rule,
        solver_iterations=iterations,
        oas_bracket=(lower, upper),
        market_price_source=market_price_source,
        market_price_evidence_hash=market_price_evidence_hash,
    )
    return EvidenceCertificate(
        status=decision_from_invariants(
            invariants,
            conditional=market_price_source != "independent",
        ),
        result=result,
        invariants=invariants,
        violations=violations,
        conventions=ConventionSchema(
            currency="bond price per face amount",
            rate_compounding="annual tree discounting",
            day_count="annual coupon fixture",
            discounting_curve="calibrated market discount factors",
            exercise_convention="issuer call after coupon at eligible tree dates",
        ),
        tolerances=tolerance,
        input_manifest_hash=stable_manifest_hash(
            {
                "bond": bond,
                "market_discount_factors": market_discount_factors,
                "rate_step": rate_step,
                "market_price": market_price,
                "market_price_source": market_price_source,
                "market_price_evidence_hash": market_price_evidence_hash,
                "bump_bp": bump_bp,
                "initial_oas_bracket": initial_oas_bracket,
                "maximum_absolute_oas": maximum_absolute_oas,
                "tolerance": tolerance,
            }
        ),
        implementation_version="ch07.callable_bond_tree.v0.10",
    )


def price_on_additive_short_rate_tree(
    bond: CallableBond,
    initial_rate: float,
    rate_step: float,
    callable_: bool,
    parallel_shift: float = 0.0,
) -> float:
    times = np.arange(1.0, bond.maturity_years + 1.0)
    curve = np.exp(-(initial_rate + parallel_shift) * times)
    tree = calibrate_additive_short_rate_tree(curve, rate_step)
    return price_on_calibrated_tree(
        bond,
        tree,
        callable_=callable_,
    ).price


def effective_duration(
    bond: CallableBond,
    initial_rate: float,
    rate_step: float,
    callable_: bool,
    bump_bp: float = 10.0,
) -> float:
    bump = bump_bp * BASIS_POINT
    base = price_on_additive_short_rate_tree(
        bond,
        initial_rate,
        rate_step,
        callable_,
    )
    down = price_on_additive_short_rate_tree(
        bond,
        initial_rate,
        rate_step,
        callable_,
        -bump,
    )
    up = price_on_additive_short_rate_tree(
        bond,
        initial_rate,
        rate_step,
        callable_,
        bump,
    )
    return (down - up) / (2.0 * base * bump)


if __name__ == "__main__":
    example = CallableBond(100.0, 0.06, 5, 100.0, 2)
    market_dfs = np.exp(-0.04 * np.arange(1.0, 6.0))
    base_tree = calibrate_additive_short_rate_tree(market_dfs, 0.01)
    target = price_on_calibrated_tree(
        example,
        base_tree,
        callable_=True,
        oas=0.0125,
    ).price
    certificate = callable_bond_oas_certificate(
        bond=example,
        market_discount_factors=market_dfs,
        rate_step=0.01,
        market_price=target,
        market_price_source="self_roundtrip",
    )
    print("status=", certificate.status)
    print("OAS=", certificate.result.oas)
    print("effective duration=", certificate.result.effective_duration)
    print("effective convexity=", certificate.result.effective_convexity)
