from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BlackScholesPDEResult:
    price: float
    delta: float
    gamma: float
    spot_steps: int
    time_steps: int
    rannacher_full_steps: int
    final_time: float
    status: str


@dataclass(frozen=True)
class BlackScholesPDECertificate:
    grids: tuple[tuple[int, int, float], ...]
    prices: tuple[float, ...]
    price_errors: tuple[float, ...]
    deltas: tuple[float, ...]
    delta_errors: tuple[float, ...]
    gammas: tuple[float, ...]
    gamma_errors: tuple[float, ...]
    observed_price_orders: tuple[float | None, ...]
    far_boundary_price_difference: float
    analytic_price: float
    analytic_delta: float
    analytic_gamma: float
    price_tolerance: float
    delta_tolerance: float
    gamma_tolerance: float
    far_boundary_tolerance: float
    status: str


def _solve_tridiagonal(
    lower: NDArray[np.float64],
    diagonal: NDArray[np.float64],
    upper: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    arrays = tuple(np.asarray(value, dtype=float) for value in (lower, diagonal, upper, rhs))
    original_lower, original_diagonal, original_upper, original_rhs = arrays
    if (
        original_diagonal.ndim != 1
        or original_rhs.shape != original_diagonal.shape
        or original_lower.shape != (max(original_diagonal.size - 1, 0),)
        or original_upper.shape != (max(original_diagonal.size - 1, 0),)
        or original_diagonal.size == 0
        or not all(np.all(np.isfinite(value)) for value in arrays)
    ):
        raise ValueError("invalid tridiagonal system")
    lower = original_lower.copy()
    diagonal = original_diagonal.copy()
    upper = original_upper.copy()
    rhs = original_rhs.copy()
    scale = max(
        float(np.max(np.abs(original_diagonal))),
        float(np.max(np.abs(original_lower), initial=0.0)),
        float(np.max(np.abs(original_upper), initial=0.0)),
        1.0,
    )
    pivot_floor = 64.0 * np.finfo(float).eps * scale
    for index in range(1, diagonal.size):
        if abs(diagonal[index - 1]) <= pivot_floor:
            raise ArithmeticError("near-zero tridiagonal pivot")
        multiplier = lower[index - 1] / diagonal[index - 1]
        diagonal[index] -= multiplier * upper[index - 1]
        rhs[index] -= multiplier * rhs[index - 1]
    solution = np.empty_like(rhs)
    if abs(diagonal[-1]) <= pivot_floor:
        raise ArithmeticError("near-zero final tridiagonal pivot")
    solution[-1] = rhs[-1] / diagonal[-1]
    for index in range(diagonal.size - 2, -1, -1):
        solution[index] = (
            rhs[index] - upper[index] * solution[index + 1]
        ) / diagonal[index]
    residual = original_diagonal * solution - original_rhs
    residual[1:] += original_lower * solution[:-1]
    residual[:-1] += original_upper * solution[1:]
    matrix_scale = (
        np.max(np.abs(original_diagonal))
        + np.max(np.abs(original_lower), initial=0.0)
        + np.max(np.abs(original_upper), initial=0.0)
    )
    denominator = matrix_scale * np.max(np.abs(solution)) + np.max(np.abs(original_rhs))
    backward_error = float(np.max(np.abs(residual)) / max(denominator, 1e-300))
    if not np.all(np.isfinite(solution)) or backward_error > 1e-11:
        raise ArithmeticError("tridiagonal residual exceeds tolerance")
    return solution


def black_scholes_pde_call(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    maximum_spot: float,
    spot_steps: int,
    time_steps: int,
    rannacher_full_steps: int = 2,
) -> BlackScholesPDEResult:
    """Price a European call with Rannacher-started Crank-Nicolson."""

    numeric = (
        spot,
        strike,
        rate,
        dividend_yield,
        volatility,
        maturity,
        maximum_spot,
    )
    if not all(isfinite(value) for value in numeric):
        raise ValueError("PDE inputs must be finite")
    if (
        spot <= 0.0
        or strike <= 0.0
        or volatility <= 0.0
        or maturity <= 0.0
        or maximum_spot <= max(spot, strike)
        or spot_steps < 20
        or time_steps < 10
        or not 0 <= rannacher_full_steps <= time_steps
    ):
        raise ValueError("invalid PDE controls")

    grid = np.linspace(0.0, maximum_spot, spot_steps + 1)
    spacing = grid[1] - grid[0]
    values = np.maximum(grid - strike, 0.0)
    interior_index = np.arange(1, spot_steps, dtype=float)
    diffusion = 0.5 * volatility**2 * interior_index**2
    convection = 0.5 * (rate - dividend_yield) * interior_index
    lower_operator = diffusion - convection
    diagonal_operator = -2.0 * diffusion - rate
    upper_operator = diffusion + convection

    elapsed = 0.0
    full_dt = maturity / time_steps

    def advance(
        current: NDArray[np.float64],
        *,
        step_size: float,
        theta: float,
        new_elapsed: float,
    ) -> NDArray[np.float64]:
        interior = current[1:-1]
        rhs = (
            (1.0 + (1.0 - theta) * step_size * diagonal_operator)
            * interior
        )
        rhs[1:] += (
            (1.0 - theta)
            * step_size
            * lower_operator[1:]
            * current[1:-2]
        )
        rhs[:-1] += (
            (1.0 - theta)
            * step_size
            * upper_operator[:-1]
            * current[2:-1]
        )

        upper_boundary = (
            maximum_spot * exp(-dividend_yield * new_elapsed)
            - strike * exp(-rate * new_elapsed)
        )
        rhs[-1] += (
            (1.0 - theta)
            * step_size
            * upper_operator[-1]
            * current[-1]
        )
        rhs[-1] += (
            theta
            * step_size
            * upper_operator[-1]
            * upper_boundary
        )
        lower = -theta * step_size * lower_operator[1:]
        diagonal = 1.0 - theta * step_size * diagonal_operator
        upper = -theta * step_size * upper_operator[:-1]
        next_values = current.copy()
        next_values[0] = 0.0
        next_values[-1] = upper_boundary
        next_values[1:-1] = _solve_tridiagonal(
            lower,
            diagonal,
            upper,
            rhs,
        )
        return next_values

    for _ in range(rannacher_full_steps):
        half = 0.5 * full_dt
        elapsed += half
        values = advance(
            values,
            step_size=half,
            theta=1.0,
            new_elapsed=elapsed,
        )
        elapsed += half
        values = advance(
            values,
            step_size=half,
            theta=1.0,
            new_elapsed=elapsed,
        )
    for _ in range(time_steps - rannacher_full_steps):
        elapsed += full_dt
        values = advance(
            values,
            step_size=full_dt,
            theta=0.5,
            new_elapsed=elapsed,
        )

    price = float(np.interp(spot, grid, values))
    first = np.gradient(values, spacing, edge_order=2)
    second = np.gradient(first, spacing, edge_order=2)
    delta = float(np.interp(spot, grid, first))
    gamma = float(np.interp(spot, grid, second))
    return BlackScholesPDEResult(
        price=price,
        delta=delta,
        gamma=gamma,
        spot_steps=spot_steps,
        time_steps=time_steps,
        rannacher_full_steps=rannacher_full_steps,
        final_time=elapsed,
        status="RAW_GRID_ONLY",
    )


def black_scholes_pde_convergence_certificate(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    maximum_spot: float,
    grids: tuple[tuple[int, int], ...] = ((80, 160), (160, 320), (320, 640)),
    rannacher_full_steps: int = 2,
    price_tolerance: float = 5e-3,
    delta_tolerance: float = 3e-3,
    gamma_tolerance: float = 5e-4,
    far_boundary_tolerance: float = 3e-3,
) -> BlackScholesPDECertificate:
    """Run nested grids, an analytic Greek cross-check and a boundary refinement."""

    controls = (
        spot,
        strike,
        rate,
        dividend_yield,
        volatility,
        maturity,
        maximum_spot,
        price_tolerance,
        delta_tolerance,
        gamma_tolerance,
        far_boundary_tolerance,
    )
    if (
        len(grids) < 3
        or any(
            not isinstance(space, int)
            or isinstance(space, bool)
            or not isinstance(time, int)
            or isinstance(time, bool)
            or space < 20
            or time < 10
            for space, time in grids
        )
        or any(
            grids[index + 1][0] <= grids[index][0]
            or grids[index + 1][1] <= grids[index][1]
            for index in range(len(grids) - 1)
        )
        or not all(isfinite(value) for value in controls)
        or min(price_tolerance, delta_tolerance, gamma_tolerance, far_boundary_tolerance) <= 0.0
        or maximum_spot < 2.5 * max(spot, strike)
    ):
        raise ValueError("invalid PDE convergence policy")

    from .black_scholes import black_scholes_call, normal_cdf, normal_pdf

    analytic_price = black_scholes_call(
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity,
    )
    root_t = sqrt(maturity)
    d1 = (
        log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility**2) * maturity
    ) / (volatility * root_t)
    analytic_delta = exp(-dividend_yield * maturity) * normal_cdf(d1)
    analytic_gamma = (
        exp(-dividend_yield * maturity)
        * normal_pdf(d1)
        / (spot * volatility * root_t)
    )

    results = tuple(
        black_scholes_pde_call(
            spot=spot,
            strike=strike,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            maturity=maturity,
            maximum_spot=maximum_spot,
            spot_steps=space,
            time_steps=time,
            rannacher_full_steps=rannacher_full_steps,
        )
        for space, time in grids
    )
    prices = tuple(result.price for result in results)
    deltas = tuple(result.delta for result in results)
    gammas = tuple(result.gamma for result in results)
    price_errors = tuple(value - analytic_price for value in prices)
    delta_errors = tuple(value - analytic_delta for value in deltas)
    gamma_errors = tuple(value - analytic_gamma for value in gammas)
    orders: list[float | None] = [None]
    for index in range(1, len(grids)):
        previous_error = abs(price_errors[index - 1])
        error = abs(price_errors[index])
        mesh_ratio = grids[index][0] / grids[index - 1][0]
        orders.append(
            log(previous_error / error) / log(mesh_ratio)
            if previous_error > 0.0 and error > 0.0
            else None
        )

    finest_space, finest_time = grids[-1]
    wider = black_scholes_pde_call(
        spot=spot,
        strike=strike,
        rate=rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        maturity=maturity,
        maximum_spot=1.5 * maximum_spot,
        spot_steps=int(round(1.5 * finest_space)),
        time_steps=finest_time,
        rannacher_full_steps=rannacher_full_steps,
    )
    far_boundary_difference = abs(wider.price - prices[-1])
    monotone_price_improvement = abs(price_errors[-1]) < abs(price_errors[-2])
    passed = (
        monotone_price_improvement
        and abs(price_errors[-1]) <= price_tolerance
        and abs(delta_errors[-1]) <= delta_tolerance
        and abs(gamma_errors[-1]) <= gamma_tolerance
        and far_boundary_difference <= far_boundary_tolerance
        and all(np.isclose(result.final_time, maturity, rtol=0.0, atol=1e-12) for result in results)
    )
    return BlackScholesPDECertificate(
        grids=tuple((space, time, maximum_spot) for space, time in grids),
        prices=prices,
        price_errors=price_errors,
        deltas=deltas,
        delta_errors=delta_errors,
        gammas=gammas,
        gamma_errors=gamma_errors,
        observed_price_orders=tuple(orders),
        far_boundary_price_difference=far_boundary_difference,
        analytic_price=analytic_price,
        analytic_delta=analytic_delta,
        analytic_gamma=analytic_gamma,
        price_tolerance=price_tolerance,
        delta_tolerance=delta_tolerance,
        gamma_tolerance=gamma_tolerance,
        far_boundary_tolerance=far_boundary_tolerance,
        status="PASS" if passed else "REJECT",
    )
