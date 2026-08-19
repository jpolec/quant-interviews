from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal


@dataclass(frozen=True)
class DiscreteDistribution:
    values: tuple[float, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.probabilities) or not self.values:
            raise ValueError("values and probabilities must have equal non-zero length")
        if (
            not all(isfinite(value) for value in self.values)
            or not all(isfinite(probability) for probability in self.probabilities)
            or any(probability < 0.0 for probability in self.probabilities)
            or abs(sum(self.probabilities) - 1.0) > 1e-12
        ):
            raise ValueError("distribution must be finite and probabilities must sum to one")


@dataclass(frozen=True)
class SearchDecision:
    value: float
    continuation_value: float
    stop_value: float
    action: Literal["stop", "continue"]
    bellman_residual: float
    policy_consistency_residual: float
    verified_states: int
    status: str


def finite_horizon_decision(
    dist: DiscreteDistribution,
    cost: float,
    draws_left: int,
    best: float,
) -> SearchDecision:
    if (
        not isfinite(cost)
        or not isfinite(best)
        or cost < 0.0
        or draws_left < 0
    ):
        raise ValueError("invalid cost, horizon or current best")

    # Build the complete finite state table from frozen previous-horizon layers.
    states = tuple(sorted({best, *dist.values}))
    layers: list[dict[float, float]] = [
        {state: state for state in states}
    ]
    for _remaining in range(1, draws_left + 1):
        previous = layers[-1]
        current: dict[float, float] = {}
        for state in states:
            continuation = -cost + sum(
                probability * previous[max(state, outcome)]
                for outcome, probability in zip(dist.values, dist.probabilities)
            )
            current[state] = max(state, continuation)
        layers.append(current)

    maximum_residual = 0.0
    maximum_policy_residual = 0.0
    for remaining in range(1, draws_left + 1):
        previous = layers[remaining - 1]
        current = layers[remaining]
        for state in states:
            independent_continuation = -cost + sum(
                probability * previous[max(state, outcome)]
                for outcome, probability in zip(dist.values, dist.probabilities)
            )
            independent_rhs = max(state, independent_continuation)
            maximum_residual = max(
                maximum_residual,
                abs(current[state] - independent_rhs),
            )
            if current[state] == state:
                violation = max(0.0, independent_continuation - state)
            else:
                violation = max(0.0, state - independent_continuation)
            maximum_policy_residual = max(maximum_policy_residual, violation)

    stop_value = best
    continuation_value = best if draws_left == 0 else -cost + sum(
        probability * layers[draws_left - 1][max(best, outcome)]
        for outcome, probability in zip(dist.values, dist.probabilities)
    )
    optimal_value = layers[draws_left][best]
    action: Literal["stop", "continue"] = (
        "continue"
        if draws_left > 0 and continuation_value > stop_value
        else "stop"
    )
    requested_residual = abs(optimal_value - max(stop_value, continuation_value))
    residual = max(maximum_residual, requested_residual)
    return SearchDecision(
        value=optimal_value,
        continuation_value=continuation_value,
        stop_value=stop_value,
        action=action,
        bellman_residual=residual,
        policy_consistency_residual=maximum_policy_residual,
        verified_states=len(states) * (draws_left + 1),
        status=(
            "PASS"
            if residual <= 1e-12 and maximum_policy_residual <= 1e-12
            else "REJECT"
        ),
    )


def finite_horizon_value(
    dist: DiscreteDistribution,
    cost: float,
    draws_left: int,
    best: float,
) -> float:
    return finite_horizon_decision(dist, cost, draws_left, best).value


def reservation_value_uniform(cost: float) -> float:
    if not isfinite(cost) or cost < 0.0:
        raise ValueError("cost must be finite and non-negative")
    if cost >= 0.5:
        return 0.0
    if cost == 0.0:
        return 1.0
    return 1.0 - sqrt(2.0 * cost)


def policy(
    dist: DiscreteDistribution,
    cost: float,
    draws_left: int,
    best: float,
) -> str:
    return finite_horizon_decision(dist, cost, draws_left, best).action
