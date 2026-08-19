from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, sqrt

import numpy as np


@dataclass(frozen=True)
class FeynmanKacResult:
    estimate: float
    standard_error: float
    analytic_value: float
    analytic_grid_value: float
    sampling_bias: float
    discretization_bias: float
    confidence_interval: tuple[float, float]
    sampling_status: str
    discretization_status: str
    status: str


def _discounted_integral(rate: float, maturity: float) -> float:
    """Return integral_0^T exp(-rate*s) ds stably for either sign."""

    if abs(rate * maturity) < 1e-8:
        x = rate * maturity
        return maturity * (1.0 - x / 2.0 + x * x / 6.0 - x**3 / 24.0)
    return -np.expm1(-rate * maturity) / rate


def discounted_gbm_call_mc(
    *,
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    paths: int = 200_000,
    seed: int = 29,
) -> tuple[float, float]:
    values = np.asarray([spot, strike, rate, volatility, maturity], dtype=float)
    if (
        not np.all(np.isfinite(values))
        or spot <= 0.0
        or strike < 0.0
        or volatility < 0.0
        or maturity <= 0.0
        or not isinstance(paths, int)
        or isinstance(paths, bool)
        or paths < 2
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise ValueError("invalid pricing inputs")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(paths)
    terminal = spot * np.exp(
        (rate - 0.5 * volatility**2) * maturity
        + volatility * np.sqrt(maturity) * z
    )
    discounted = np.exp(-rate * maturity) * np.maximum(terminal - strike, 0.0)
    return float(np.mean(discounted)), float(np.std(discounted, ddof=1) / sqrt(paths))


def black_scholes_call_reference(
    *,
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    values = np.asarray([spot, strike, rate, volatility, maturity], dtype=float)
    if (
        not np.all(np.isfinite(values))
        or spot <= 0.0
        or strike < 0.0
        or volatility < 0.0
        or maturity <= 0.0
    ):
        raise ValueError("invalid Black-Scholes inputs")
    if strike == 0.0:
        return spot
    if volatility == 0.0:
        return max(spot - strike * exp(-rate * maturity), 0.0)
    denominator = volatility * sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / denominator
    d2 = d1 - denominator
    normal_cdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    return float(spot * normal_cdf(d1) - strike * exp(-rate * maturity) * normal_cdf(d2))


def ou_terminal_and_running_payoff(
    *,
    state: float,
    kappa: float,
    theta: float,
    sigma: float,
    rate: float,
    maturity: float,
    running_weight: float,
    steps: int = 128,
    paths: int = 80_000,
    seed: int = 29,
    confidence_z: float = 1.96,
    maximum_absolute_discretization_bias: float = 1e-4,
) -> FeynmanKacResult:
    """Verify time-of-payment discounting for an OU terminal-plus-running claim."""

    values = np.asarray(
        [
            state,
            kappa,
            theta,
            sigma,
            rate,
            maturity,
            running_weight,
            confidence_z,
            maximum_absolute_discretization_bias,
        ],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(values))
        or kappa <= 0.0
        or sigma < 0.0
        or maturity <= 0.0
        or confidence_z <= 0.0
        or maximum_absolute_discretization_bias < 0.0
        or not isinstance(steps, int)
        or isinstance(steps, bool)
        or steps < 2
        or not isinstance(paths, int)
        or isinstance(paths, bool)
        or paths < 2
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise ValueError("invalid OU Feynman-Kac inputs")

    dt = maturity / steps
    decay = np.exp(-kappa * dt)
    innovation_variance = sigma**2 * (-np.expm1(-2.0 * kappa * dt)) / (2.0 * kappa)
    rng = np.random.default_rng(seed)
    x = np.full(paths, state, dtype=float)
    running_present_value = np.zeros(paths, dtype=float)
    expected_grid_running = 0.0
    for step in range(1, steps + 1):
        previous = x
        mean = theta + (previous - theta) * decay
        x = mean + sqrt(innovation_variance) * rng.standard_normal(paths)
        t_left = (step - 1) * dt
        t_right = step * dt
        left_discount = exp(-rate * t_left)
        right_discount = exp(-rate * t_right)
        running_present_value += 0.5 * (
            left_discount * previous + right_discount * x
        ) * dt
        expected_left = theta + (state - theta) * exp(-kappa * t_left)
        expected_right = theta + (state - theta) * exp(-kappa * t_right)
        expected_grid_running += 0.5 * (
            left_discount * expected_left + right_discount * expected_right
        ) * dt

    terminal_discount = exp(-rate * maturity)
    discounted = terminal_discount * x + running_weight * running_present_value
    estimate = float(np.mean(discounted))
    standard_error = float(np.std(discounted, ddof=1) / sqrt(paths))

    terminal_mean = theta + (state - theta) * exp(-kappa * maturity)
    exact_running = theta * _discounted_integral(rate, maturity) + (
        state - theta
    ) * _discounted_integral(rate + kappa, maturity)
    analytic = float(terminal_discount * terminal_mean + running_weight * exact_running)
    analytic_grid = float(
        terminal_discount * terminal_mean + running_weight * expected_grid_running
    )
    interval = (
        estimate - confidence_z * standard_error,
        estimate + confidence_z * standard_error,
    )
    sampling_status = "PASS" if interval[0] <= analytic_grid <= interval[1] else "REJECT"
    discretization_bias = analytic_grid - analytic
    discretization_status = (
        "PASS"
        if abs(discretization_bias) <= maximum_absolute_discretization_bias
        else "REJECT"
    )
    return FeynmanKacResult(
        estimate=estimate,
        standard_error=standard_error,
        analytic_value=analytic,
        analytic_grid_value=analytic_grid,
        sampling_bias=estimate - analytic_grid,
        discretization_bias=discretization_bias,
        confidence_interval=interval,
        sampling_status=sampling_status,
        discretization_status=discretization_status,
        status=(
            "PASS"
            if sampling_status == "PASS" and discretization_status == "PASS"
            else "REJECT"
        ),
    )
