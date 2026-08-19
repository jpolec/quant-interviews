from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, sqrt

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AmericanPutLatticeResult:
    price: float
    risk_neutral_probability: float | None
    exercise_boundary: NDArray[np.float64]
    exercise_mask: tuple[NDArray[np.bool_], ...]
    intrinsic_values: tuple[NDArray[np.float64], ...]
    continuation_values: tuple[NDArray[np.float64], ...]
    status: str


@dataclass(frozen=True)
class AmericanLatticeCertificate:
    paired_steps: tuple[tuple[int, int], ...]
    paired_prices: tuple[tuple[float, float], ...]
    pair_averages: tuple[float, ...]
    pair_spreads: tuple[float, ...]
    finest_price: float
    european_lower_bound: float
    intrinsic_lower_bound: float
    maximum_policy_residual: float
    boundary_stability_error: float
    price_tolerance: float
    boundary_tolerance: float
    status: str


def american_put_lattice(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    steps: int,
    tolerance: float = 1e-12,
    exercise_style: str = "AMERICAN_PUT",
    settlement_convention: str = "PHYSICAL_AT_EXERCISE",
) -> AmericanPutLatticeResult:
    numeric = (
        spot,
        strike,
        rate,
        dividend_yield,
        volatility,
        maturity,
        tolerance,
    )
    if not all(isfinite(value) for value in numeric):
        raise ValueError("binomial inputs must be finite")
    if (
        steps <= 0
        or spot <= 0.0
        or strike < 0.0
        or volatility <= 0.0
        or maturity < 0.0
        or tolerance < 0.0
        or exercise_style != "AMERICAN_PUT"
        or settlement_convention != "PHYSICAL_AT_EXERCISE"
    ):
        raise ValueError("invalid binomial inputs")
    if maturity == 0.0:
        intrinsic = np.array([max(strike - spot, 0.0)])
        return AmericanPutLatticeResult(
            price=float(intrinsic[0]),
            risk_neutral_probability=None,
            exercise_boundary=np.array([spot if intrinsic[0] > 0.0 else np.nan]),
            exercise_mask=(intrinsic > 0.0,),
            intrinsic_values=(intrinsic,),
            continuation_values=(np.array([np.nan]),),
            status="EXPIRY",
        )

    dt = maturity / steps
    up = exp(volatility * sqrt(dt))
    down = 1.0 / up
    growth = exp((rate - dividend_yield) * dt)
    denominator = up - down
    if denominator <= 0.0:
        raise ValueError("degenerate binomial grid")
    probability = (growth - down) / denominator
    if not 0.0 <= probability <= 1.0:
        raise ValueError("invalid risk-neutral probability")
    discount = exp(-rate * dt)

    terminal_spots = spot * up ** np.arange(steps + 1) * down ** (
        steps - np.arange(steps + 1)
    )
    values = np.maximum(strike - terminal_spots, 0.0)
    masks: list[NDArray[np.bool_]] = [np.array([], dtype=bool)] * (steps + 1)
    intrinsic_records: list[NDArray[np.float64]] = [
        np.array([], dtype=float)
    ] * (steps + 1)
    continuation_records: list[NDArray[np.float64]] = [
        np.array([], dtype=float)
    ] * (steps + 1)
    masks[steps] = values > tolerance
    intrinsic_records[steps] = values.copy()
    continuation_records[steps] = np.full(steps + 1, np.nan)
    boundary = np.full(steps + 1, np.nan)
    if np.any(masks[steps]):
        boundary[steps] = float(np.max(terminal_spots[masks[steps]]))

    for step in range(steps - 1, -1, -1):
        indices = np.arange(step + 1)
        state_spots = spot * up**indices * down ** (step - indices)
        continuation = discount * (
            probability * values[1:] + (1.0 - probability) * values[:-1]
        )
        intrinsic = np.maximum(strike - state_spots, 0.0)
        exercise = (intrinsic > continuation + tolerance) & (
            intrinsic > tolerance
        )
        values = np.maximum(intrinsic, continuation)
        masks[step] = exercise
        intrinsic_records[step] = intrinsic
        continuation_records[step] = continuation
        if np.any(exercise):
            boundary[step] = float(np.max(state_spots[exercise]))

    return AmericanPutLatticeResult(
        price=float(values[0]),
        risk_neutral_probability=float(probability),
        exercise_boundary=boundary,
        exercise_mask=tuple(masks),
        intrinsic_values=tuple(intrinsic_records),
        continuation_values=tuple(continuation_records),
        status="RAW_GRID_ONLY",
    )


def american_lattice_convergence(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    base_steps: tuple[int, ...] = (100, 200, 400),
    price_absolute_tolerance: float = 2.5e-3,
    price_relative_tolerance: float = 2.5e-4,
    boundary_tolerance: float = 3.0,
    policy_tolerance: float = 1e-10,
) -> AmericanLatticeCertificate:
    """Certify an American put using paired even/odd lattice refinements."""

    controls = (
        spot,
        strike,
        rate,
        dividend_yield,
        volatility,
        maturity,
        price_absolute_tolerance,
        price_relative_tolerance,
        boundary_tolerance,
        policy_tolerance,
    )
    if (
        len(base_steps) < 3
        or any(not isinstance(step, int) or isinstance(step, bool) for step in base_steps)
        or any(step < 2 for step in base_steps)
        or any(base_steps[index + 1] <= base_steps[index] for index in range(len(base_steps) - 1))
        or not all(isfinite(value) for value in controls)
        or min(
            price_absolute_tolerance,
            price_relative_tolerance,
            boundary_tolerance,
            policy_tolerance,
        )
        < 0.0
        or maturity <= 0.0
    ):
        raise ValueError("invalid American convergence policy")

    results: list[tuple[AmericanPutLatticeResult, AmericanPutLatticeResult]] = []
    for steps in base_steps:
        left = american_put_lattice(
            spot=spot,
            strike=strike,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity=maturity,
            steps=steps,
        )
        right = american_put_lattice(
            spot=spot,
            strike=strike,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity=maturity,
            steps=steps + 1,
        )
        results.append((left, right))

    paired_prices = tuple((left.price, right.price) for left, right in results)
    pair_averages = tuple(0.5 * (left + right) for left, right in paired_prices)
    pair_spreads = tuple(abs(left - right) for left, right in paired_prices)
    finest_price = pair_averages[-1]
    discounted_strike = strike * exp(-rate * maturity)
    discounted_spot = spot * exp(-dividend_yield * maturity)
    european_lower = max(discounted_strike - discounted_spot, 0.0)
    intrinsic_lower = max(strike - spot, 0.0)

    maximum_policy_residual = 0.0
    for result_pair in results:
        for result in result_pair:
            for intrinsic, continuation, mask in zip(
                result.intrinsic_values[:-1],
                result.continuation_values[:-1],
                result.exercise_mask[:-1],
            ):
                selected = np.maximum(intrinsic, continuation)
                reconstructed = np.where(mask, intrinsic, continuation)
                maximum_policy_residual = max(
                    maximum_policy_residual,
                    float(np.max(np.abs(selected - reconstructed))),
                )

    left_boundary = results[-1][0].exercise_boundary
    right_boundary = results[-1][1].exercise_boundary
    common = np.linspace(0.0, 1.0, 101)

    def interpolated_boundary(values: NDArray[np.float64]) -> NDArray[np.float64]:
        time = np.linspace(0.0, 1.0, values.size)
        finite = np.isfinite(values)
        if np.count_nonzero(finite) < 2:
            return np.full(common.shape, np.nan)
        return np.interp(common, time[finite], values[finite])

    first_boundary = interpolated_boundary(left_boundary)
    second_boundary = interpolated_boundary(right_boundary)
    comparable = np.isfinite(first_boundary) & np.isfinite(second_boundary)
    boundary_error = (
        float(np.max(np.abs(first_boundary[comparable] - second_boundary[comparable])))
        if np.any(comparable)
        else float("inf")
    )
    price_limit = price_absolute_tolerance + price_relative_tolerance * abs(finest_price)
    refinement_changes = np.abs(np.diff(np.asarray(pair_averages)))
    passed = (
        pair_spreads[-1] <= price_limit
        and refinement_changes[-1] <= price_limit
        and finest_price + price_limit >= max(european_lower, intrinsic_lower)
        and maximum_policy_residual <= policy_tolerance
        and boundary_error <= boundary_tolerance
    )
    return AmericanLatticeCertificate(
        paired_steps=tuple((steps, steps + 1) for steps in base_steps),
        paired_prices=paired_prices,
        pair_averages=pair_averages,
        pair_spreads=pair_spreads,
        finest_price=finest_price,
        european_lower_bound=european_lower,
        intrinsic_lower_bound=intrinsic_lower,
        maximum_policy_residual=maximum_policy_residual,
        boundary_stability_error=boundary_error,
        price_tolerance=price_limit,
        boundary_tolerance=boundary_tolerance,
        status="PASS" if passed else "REJECT",
    )


def american_put_binomial(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    steps: int,
) -> float:
    """Compatibility wrapper returning only the governed lattice price."""

    return american_put_lattice(
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity,
        steps=steps,
    ).price
