from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np


@dataclass(frozen=True)
class RevenueSimulation:
    estimated_revenue: float
    standard_error: float
    analytic_revenue: float
    residual: float
    tie_rule: str
    status: str


VALID_TIE_RULES = {"uniform_among_highest_bids", "lowest_bidder_id"}


def _validate_tie_rule(tie_rule: str) -> None:
    if tie_rule not in VALID_TIE_RULES:
        raise ValueError("tie_rule must be versioned and supported")


def _validate_parameters(value: float, bidders: int, risk_aversion: float) -> None:
    if (
        not isfinite(value)
        or not isfinite(risk_aversion)
        or not 0.0 <= value <= 1.0
        or isinstance(bidders, bool)
        or not isinstance(bidders, int)
        or bidders < 2
        or not 0.0 <= risk_aversion < 1.0
    ):
        raise ValueError("require value in [0,1], bidders >= 2 and 0 <= rho < 1")


def equilibrium_bid(
    value: float,
    bidders: int,
    risk_aversion: float = 0.0,
) -> float:
    _validate_parameters(value, bidders, risk_aversion)
    return (bidders - 1.0) / (bidders - risk_aversion) * value


def deviation_utility(
    value: float,
    bid: float,
    bidders: int,
    risk_aversion: float = 0.0,
    tie_rule: str = "uniform_among_highest_bids",
) -> float:
    """Return normalized CRRA utility; a positive constant is omitted."""
    _validate_parameters(value, bidders, risk_aversion)
    _validate_tie_rule(tie_rule)
    if not isfinite(bid) or bid < 0.0 or bid > value:
        raise ValueError("bid must be finite and lie in [0,value]")
    exponent = 1.0 - risk_aversion
    inferred_type = min(
        1.0,
        bid * (bidders - risk_aversion) / (bidders - 1.0),
    )
    win_probability = inferred_type ** (bidders - 1)
    return (value - bid) ** exponent * win_probability


def best_response_grid(
    value: float,
    bidders: int,
    risk_aversion: float = 0.0,
    grid_size: int = 5001,
    tie_rule: str = "uniform_among_highest_bids",
) -> tuple[float, float]:
    _validate_parameters(value, bidders, risk_aversion)
    _validate_tie_rule(tie_rule)
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    bids = np.linspace(0.0, value, grid_size)
    utilities = np.array(
        [
            deviation_utility(value, float(bid), bidders, risk_aversion, tie_rule)
            for bid in bids
        ]
    )
    index = int(np.argmax(utilities))
    return float(bids[index]), float(utilities[index])


def simulate_revenue_certificate(
    bidders: int,
    trials: int = 200_000,
    seed: int = 13,
    risk_aversion: float = 0.0,
    tie_rule: str = "uniform_among_highest_bids",
) -> RevenueSimulation:
    _validate_parameters(0.5, bidders, risk_aversion)
    _validate_tie_rule(tie_rule)
    if trials <= 1:
        raise ValueError("trials must exceed one")
    rng = np.random.default_rng(seed)
    values = rng.random((trials, bidders))
    coefficient = (bidders - 1.0) / (bidders - risk_aversion)
    revenues = coefficient * np.max(values, axis=1)
    estimate = float(np.mean(revenues))
    standard_error = float(np.std(revenues, ddof=1) / sqrt(trials))
    analytic = coefficient * bidders / (bidders + 1.0)
    residual = estimate - analytic
    return RevenueSimulation(
        estimated_revenue=estimate,
        standard_error=standard_error,
        analytic_revenue=analytic,
        residual=residual,
        tie_rule=tie_rule,
        status="PASS" if abs(residual) <= 4.0 * standard_error else "REJECT",
    )


def simulate_revenue(
    bidders: int,
    trials: int = 200_000,
    seed: int = 13,
    risk_aversion: float = 0.0,
    tie_rule: str = "uniform_among_highest_bids",
) -> float:
    return simulate_revenue_certificate(
        bidders,
        trials,
        seed,
        risk_aversion,
        tie_rule,
    ).estimated_revenue
