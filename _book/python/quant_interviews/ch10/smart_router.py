from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class VenueQuote:
    venue: str
    displayed_size: int
    fill_probability: float
    half_spread_bps: float
    fee_bps: float
    rebate_bps: float
    adverse_selection_bps: float
    latency_penalty_bps: float
    replacement_cost_bps: float
    information_leakage_bps: float
    order_type: str = "PASSIVE"
    eligible: bool = True

    def __post_init__(self) -> None:
        if not self.venue:
            raise ValueError("venue must be non-empty")
        if (
            isinstance(self.displayed_size, bool)
            or not isinstance(self.displayed_size, int)
            or self.displayed_size < 0
        ):
            raise ValueError("displayed_size must be a non-negative integer")
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be boolean")

    def expected_cost_bps(self) -> float:
        numeric = (
            self.fill_probability,
            self.half_spread_bps,
            self.fee_bps,
            self.rebate_bps,
            self.adverse_selection_bps,
            self.latency_penalty_bps,
            self.replacement_cost_bps,
            self.information_leakage_bps,
        )
        if (
            self.displayed_size < 0
            or not all(isfinite(value) for value in numeric)
            or not 0.0 <= self.fill_probability <= 1.0
        ):
            raise ValueError("invalid displayed size or fill probability")
        order_type = self.order_type.upper()
        if order_type not in {"PASSIVE", "AGGRESSIVE"}:
            raise ValueError("order_type must be PASSIVE or AGGRESSIVE")
        signed_spread = -self.half_spread_bps if order_type == "PASSIVE" else self.half_spread_bps
        filled = (
            signed_spread
            + self.fee_bps
            - self.rebate_bps
            + self.adverse_selection_bps
            + self.latency_penalty_bps
            + self.information_leakage_bps
        )
        nonfill = (
            self.replacement_cost_bps
            + self.latency_penalty_bps
            + self.information_leakage_bps
        )
        return self.fill_probability * filled + (1.0 - self.fill_probability) * nonfill


@dataclass(frozen=True)
class RouteAllocation:
    venue: str
    submitted_quantity: int
    expected_filled_quantity: float
    expected_cost_bps: float
    expected_cost_contribution_bps_quantity: float


@dataclass(frozen=True)
class RoutePlan:
    allocations: tuple[RouteAllocation, ...]
    target_quantity: int
    total_submitted_quantity: int
    expected_filled_quantity: float
    maximum_overfill_exposure: int
    expected_completion_fraction: float
    completion_shortfall: float
    non_completion_penalty_bps: float
    total_objective_bps: float
    release_mode: str
    decision_reason: str
    status: str


def route_order(
    quantity: int,
    venues: list[VenueQuote],
    *,
    non_completion_penalty_bps: float = 10.0,
    minimum_expected_completion_fraction: float = 0.75,
    release_mode: str = "SIMULTANEOUS_HARD_CAP",
) -> RoutePlan:
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("quantity must be a positive integer")
    if (
        not isfinite(non_completion_penalty_bps)
        or non_completion_penalty_bps < 0.0
        or not isfinite(minimum_expected_completion_fraction)
        or not 0.0 <= minimum_expected_completion_fraction <= 1.0
    ):
        raise ValueError("routing policy controls are invalid")
    if release_mode != "SIMULTANEOUS_HARD_CAP":
        raise ValueError("only SIMULTANEOUS_HARD_CAP is implemented")
    names = [venue.venue for venue in venues]
    if len(names) != len(set(names)):
        raise ValueError("venue identifiers must be unique")
    eligible = [venue for venue in venues if venue.eligible and venue.displayed_size > 0]
    if not eligible:
        raise ValueError("no eligible venue capacity")
    ranked = sorted(eligible, key=lambda v: (v.expected_cost_bps(), -v.fill_probability))
    live_remaining = quantity
    result: list[RouteAllocation] = []
    for venue in ranked:
        if live_remaining <= 0:
            break
        if venue.fill_probability <= 0.0:
            continue
        venue_cost = venue.expected_cost_bps()
        if venue_cost >= non_completion_penalty_bps:
            continue
        submit = min(venue.displayed_size, live_remaining)
        if submit <= 0:
            continue
        expected_fill = submit * venue.fill_probability
        result.append(
            RouteAllocation(
                venue=venue.venue,
                submitted_quantity=submit,
                expected_filled_quantity=expected_fill,
                expected_cost_bps=venue_cost,
                expected_cost_contribution_bps_quantity=expected_fill * venue_cost,
            )
        )
        live_remaining -= submit
    expected_filled = sum(allocation.expected_filled_quantity for allocation in result)
    submitted = sum(allocation.submitted_quantity for allocation in result)
    if submitted > quantity:
        raise AssertionError("simultaneous live quantity exceeded the hard parent cap")
    completion_shortfall = max(0.0, quantity - expected_filled)
    completion_fraction = expected_filled / quantity
    expected_cost_quantity = sum(
        allocation.expected_cost_contribution_bps_quantity for allocation in result
    )
    objective = (
        expected_cost_quantity
        + completion_shortfall * non_completion_penalty_bps
    ) / quantity
    if completion_fraction < minimum_expected_completion_fraction:
        status = "REJECT"
        reason = "EXPECTED_COMPLETION_BELOW_POLICY"
    elif completion_shortfall > 1e-9:
        status = "CONDITIONAL_PASS"
        reason = "HARD_CAP_PRESERVED_WITH_NONCOMPLETION_RISK"
    else:
        status = "PASS"
        reason = "HARD_CAP_AND_COMPLETION_POLICY_PASS"
    return RoutePlan(
        tuple(result),
        quantity,
        submitted,
        expected_filled,
        0,
        completion_fraction,
        completion_shortfall,
        non_completion_penalty_bps,
        objective,
        release_mode,
        reason,
        status,
    )
