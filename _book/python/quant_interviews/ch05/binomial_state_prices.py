from __future__ import annotations

from math import isfinite
from typing import Any


def state_prices(
    *,
    spot: float,
    up: float,
    down: float,
    gross_rate: float,
) -> tuple[float, float]:
    if not all(isfinite(value) for value in (spot, up, down, gross_rate)):
        raise ValueError("state-price inputs must be finite")
    if spot <= 0.0 or gross_rate <= 0.0 or up <= down:
        raise ValueError("invalid one-period market")
    if not down < gross_rate * spot < up:
        raise ValueError("one-period market violates no-arbitrage ordering")
    q = (gross_rate * spot - down) / (up - down)
    return q / gross_rate, (1.0 - q) / gross_rate


def replicate_two_state(
    *,
    up: float,
    down: float,
    payoff_up: float,
    payoff_down: float,
    gross_rate: float,
) -> tuple[float, float]:
    if not all(
        isfinite(value)
        for value in (up, down, payoff_up, payoff_down, gross_rate)
    ):
        raise ValueError("replication inputs must be finite")
    if up <= down or gross_rate <= 0.0:
        raise ValueError("invalid replication market")
    delta = (payoff_up - payoff_down) / (up - down)
    bond = (up * payoff_down - down * payoff_up) / (gross_rate * (up - down))
    return delta, bond


def two_state_replication_certificate(
    *,
    spot: float,
    up: float,
    down: float,
    payoff_up: float,
    payoff_down: float,
    gross_rate: float,
) -> dict[str, Any]:
    """Reconcile replication, state-price valuation and both terminal states."""

    state_up, state_down = state_prices(
        spot=spot,
        up=up,
        down=down,
        gross_rate=gross_rate,
    )
    delta, bond = replicate_two_state(
        up=up,
        down=down,
        payoff_up=payoff_up,
        payoff_down=payoff_down,
        gross_rate=gross_rate,
    )
    replication_price = delta * spot + bond
    state_price_value = state_up * payoff_up + state_down * payoff_down
    residual_up = delta * up + bond * gross_rate - payoff_up
    residual_down = delta * down + bond * gross_rate - payoff_down
    q_up = gross_rate * state_up
    q_down = gross_rate * state_down
    return {
        "risk_neutral_probabilities": (q_up, q_down),
        "state_prices": (state_up, state_down),
        "delta": delta,
        "bond": bond,
        "replication_price": replication_price,
        "state_price_value": state_price_value,
        "up_state_residual": residual_up,
        "down_state_residual": residual_down,
        "pricing_residual": replication_price - state_price_value,
        "status": (
            "PASS"
            if max(
                abs(residual_up),
                abs(residual_down),
                abs(replication_price - state_price_value),
            )
            <= 1e-12
            else "REJECT"
        ),
    }
