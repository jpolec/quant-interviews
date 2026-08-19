from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any
from math import isfinite


def discounted_expectation(
    payoffs: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    discount_factor: float,
) -> float:
    if payoffs.shape != probabilities.shape:
        raise ValueError("shape mismatch")
    if np.any(probabilities < 0.0) or not np.isclose(np.sum(probabilities), 1.0):
        raise ValueError("invalid probabilities")
    if (
        not np.all(np.isfinite(payoffs))
        or not np.all(np.isfinite(probabilities))
        or not np.isfinite(discount_factor)
        or discount_factor <= 0.0
    ):
        raise ValueError("expectation inputs must be finite")
    return float(discount_factor * np.dot(payoffs, probabilities))


def physical_probability_invariance_certificate(
    *,
    payoffs: NDArray[np.float64],
    state_prices: NDArray[np.float64],
    physical_probabilities_a: NDArray[np.float64],
    physical_probabilities_b: NDArray[np.float64],
    cash_price: float,
    cash_payoff: float,
    traded_payoffs: NDArray[np.float64],
    traded_prices: NDArray[np.float64],
    pricing_tolerance: float = 1e-10,
    require_physical_contrast: bool = True,
) -> dict[str, Any]:
    """Separate no-arbitrage price from physical expected payoff."""

    arrays = tuple(
        np.asarray(array, dtype=float)
        for array in (
            payoffs,
            state_prices,
            physical_probabilities_a,
            physical_probabilities_b,
        )
    )
    payoff, prices, physical_a, physical_b = arrays
    traded_payoffs = np.asarray(traded_payoffs, dtype=float)
    traded_prices = np.asarray(traded_prices, dtype=float)
    if not all(array.shape == payoff.shape for array in arrays):
        raise ValueError("state vectors must align")
    if payoff.ndim != 1 or payoff.size < 2:
        raise ValueError("at least two states are required")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("state vectors must be finite")
    if np.any(prices < 0.0) or np.sum(prices) <= 0.0:
        raise ValueError("invalid state prices")
    if (
        traded_payoffs.ndim != 2
        or traded_payoffs.shape[0] != payoff.size
        or traded_payoffs.shape[1] != traded_prices.size
        or traded_prices.size == 0
        or not np.all(np.isfinite(traded_payoffs))
        or not np.all(np.isfinite(traded_prices))
        or not all(isfinite(value) for value in (cash_price, cash_payoff, pricing_tolerance))
        or cash_price <= 0.0
        or cash_payoff <= 0.0
        or pricing_tolerance < 0.0
        or not isinstance(require_physical_contrast, bool)
    ):
        raise ValueError("invalid traded-market certificate")
    if (
        np.any(physical_a < 0.0)
        or np.any(physical_b < 0.0)
        or not np.isclose(np.sum(physical_a), 1.0)
        or not np.isclose(np.sum(physical_b), 1.0)
    ):
        raise ValueError("invalid physical probabilities")
    no_arbitrage_price = float(prices @ payoff)
    expectation_a = float(physical_a @ payoff)
    expectation_b = float(physical_b @ payoff)
    cash_residual = float(np.sum(prices) * cash_payoff - cash_price)
    traded_residuals = prices @ traded_payoffs - traded_prices
    maximum_pricing_residual = max(
        abs(cash_residual),
        float(np.max(np.abs(traded_residuals))),
    )
    contrast = not np.isclose(
        expectation_a,
        expectation_b,
        rtol=0.0,
        atol=pricing_tolerance,
    )
    passed = maximum_pricing_residual <= pricing_tolerance and (
        contrast or not require_physical_contrast
    )
    return {
        "no_arbitrage_price_a": no_arbitrage_price,
        "no_arbitrage_price_b": no_arbitrage_price,
        "physical_expected_payoff_a": expectation_a,
        "physical_expected_payoff_b": expectation_b,
        "physical_expectations_differ": contrast,
        "cash_numeraire_residual": cash_residual,
        "traded_pricing_residuals": traded_residuals,
        "maximum_pricing_residual": maximum_pricing_residual,
        "pricing_tolerance": pricing_tolerance,
        "require_physical_contrast": require_physical_contrast,
        "status": "PASS" if passed else "REJECT",
    }
