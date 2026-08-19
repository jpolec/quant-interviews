from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrderChoiceInputs:
    half_spread_bps: float
    market_fee_bps: float
    passive_fee_bps: float
    passive_rebate_bps: float
    market_impact_bps: float
    fill_probability: float
    adverse_selection_bps_if_filled: float
    nonfill_opportunity_cost_bps: float
    replacement_cost_bps: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.half_spread_bps,
                self.market_fee_bps,
                self.passive_fee_bps,
                self.passive_rebate_bps,
                self.market_impact_bps,
                self.fill_probability,
                self.adverse_selection_bps_if_filled,
                self.nonfill_opportunity_cost_bps,
                self.replacement_cost_bps,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("all order-choice inputs must be finite")
        if not 0.0 <= self.fill_probability <= 1.0:
            raise ValueError("fill_probability must lie in [0,1]")
        non_negative = (
            self.half_spread_bps,
            self.market_fee_bps,
            self.passive_fee_bps,
            self.passive_rebate_bps,
            self.market_impact_bps,
            self.adverse_selection_bps_if_filled,
            self.nonfill_opportunity_cost_bps,
            self.replacement_cost_bps,
        )
        if any(value < 0.0 for value in non_negative):
            raise ValueError("order-choice costs and rebates cannot be negative")


def expected_market_cost_bps(x: OrderChoiceInputs) -> float:
    return x.half_spread_bps + x.market_fee_bps + x.market_impact_bps


def expected_passive_cost_bps(x: OrderChoiceInputs) -> float:
    filled_cost = (
        -x.half_spread_bps
        + x.passive_fee_bps
        - x.passive_rebate_bps
        + x.adverse_selection_bps_if_filled
    )
    nonfilled_cost = x.nonfill_opportunity_cost_bps + x.replacement_cost_bps
    return x.fill_probability * filled_cost + (1.0 - x.fill_probability) * nonfilled_cost


@dataclass(frozen=True)
class OrderChoiceDecision:
    decision: str
    market_cost_bps: float
    passive_cost_bps: float
    break_even_fill_probability: float | None
    threshold_direction: str
    status: str


def break_even_fill_probability(
    x: OrderChoiceInputs,
    *,
    threshold_tolerance: float = 1e-12,
) -> tuple[float | None, str]:
    if not np.isfinite(threshold_tolerance) or threshold_tolerance < 0.0:
        raise ValueError("threshold_tolerance must be finite and non-negative")
    market = expected_market_cost_bps(x)
    filled = (
        -x.half_spread_bps
        + x.passive_fee_bps
        - x.passive_rebate_bps
        + x.adverse_selection_bps_if_filled
    )
    nonfilled = x.nonfill_opportunity_cost_bps + x.replacement_cost_bps
    denominator = nonfilled - filled
    if denominator <= 0.0:
        return None, "NO_MONOTONE_HIGHER_FILL_ADVANTAGE"
    threshold = (nonfilled - market) / denominator
    if threshold < -threshold_tolerance:
        return threshold, "LIMIT_FOR_ALL_FEASIBLE_PROBABILITIES"
    if abs(threshold) <= threshold_tolerance:
        return threshold, "LIMIT_IF_P_GT_0_TIE_AT_0"
    if threshold > 1.0 + threshold_tolerance:
        return threshold, "MARKET_FOR_ALL_FEASIBLE_PROBABILITIES"
    if abs(threshold - 1.0) <= threshold_tolerance:
        return threshold, "MARKET_IF_P_LT_1_TIE_AT_1"
    return threshold, "CHOOSE_LIMIT_IF_FILL_PROBABILITY_EXCEEDS_THRESHOLD"


def choose_order_type(
    x: OrderChoiceInputs,
    *,
    tie_tolerance_bps: float = 1e-9,
    threshold_tolerance: float = 1e-12,
) -> OrderChoiceDecision:
    if not np.isfinite(tie_tolerance_bps) or tie_tolerance_bps < 0.0:
        raise ValueError("tie_tolerance_bps must be finite and non-negative")
    market = expected_market_cost_bps(x)
    passive = expected_passive_cost_bps(x)
    difference = passive - market
    if abs(difference) <= tie_tolerance_bps:
        decision = "REVIEW"
        status = "CONDITIONAL_PASS"
    else:
        decision = "LIMIT" if difference < 0.0 else "MARKET"
        status = "PASS"
    threshold, direction = break_even_fill_probability(
        x, threshold_tolerance=threshold_tolerance
    )
    return OrderChoiceDecision(
        decision,
        market,
        passive,
        threshold,
        direction,
        status,
    )
