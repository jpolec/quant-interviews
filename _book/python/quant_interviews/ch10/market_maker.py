from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite, log1p


@dataclass(frozen=True)
class Quotes:
    reservation_price: float
    bid: float | None
    ask: float | None
    half_spread: float
    inventory_limit_breached: bool
    bid_suppression_reason: str | None
    ask_suppression_reason: str | None
    status: str


def avellaneda_stoikov_quotes(
    mid: float,
    inventory: float,
    risk_aversion: float,
    volatility: float,
    time_remaining: float,
    order_arrival_decay: float,
    *,
    tick_size: float,
    max_absolute_inventory: float,
    maker_fee: float = 0.0,
    order_size: float = 1.0,
    working_bid_quantity: float = 0.0,
    working_ask_quantity: float = 0.0,
) -> Quotes:
    values = (
        mid,
        inventory,
        risk_aversion,
        volatility,
        time_remaining,
        order_arrival_decay,
        tick_size,
        max_absolute_inventory,
        maker_fee,
        order_size,
        working_bid_quantity,
        working_ask_quantity,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("market-making inputs must be finite")
    if mid <= 0.0 or risk_aversion < 0.0 or volatility < 0.0:
        raise ValueError("invalid market or risk parameters")
    if (
        time_remaining < 0.0
        or order_arrival_decay <= 0.0
        or tick_size <= 0.0
        or max_absolute_inventory <= 0.0
        or order_size <= 0.0
        or working_bid_quantity < 0.0
        or working_ask_quantity < 0.0
    ):
        raise ValueError("invalid horizon or arrival decay")
    if abs(inventory) > max_absolute_inventory:
        return Quotes(
            mid,
            None,
            None,
            float("nan"),
            True,
            "CURRENT_INVENTORY_OUTSIDE_LIMIT",
            "CURRENT_INVENTORY_OUTSIDE_LIMIT",
            "REJECT",
        )
    reservation = mid - inventory * risk_aversion * volatility**2 * time_remaining
    liquidity_term = (
        1.0 / order_arrival_decay
        if risk_aversion == 0.0
        else log1p(risk_aversion / order_arrival_decay) / risk_aversion
    )
    half = (
        liquidity_term
        + 0.5 * risk_aversion * volatility**2 * time_remaining
        + maker_fee
    )
    raw_bid = reservation - half
    raw_ask = reservation + half
    bid = floor(raw_bid / tick_size) * tick_size
    ask = ceil(raw_ask / tick_size) * tick_size
    if bid <= 0.0 or ask <= bid:
        return Quotes(
            reservation,
            bid,
            ask,
            half,
            False,
            "INVALID_QUOTE_GEOMETRY",
            "INVALID_QUOTE_GEOMETRY",
            "REJECT",
        )
    bid_reason = (
        "PROJECTED_LONG_INVENTORY_LIMIT"
        if inventory + working_bid_quantity + order_size > max_absolute_inventory
        else None
    )
    ask_reason = (
        "PROJECTED_SHORT_INVENTORY_LIMIT"
        if inventory - working_ask_quantity - order_size < -max_absolute_inventory
        else None
    )
    governed_bid = None if bid_reason else bid
    governed_ask = None if ask_reason else ask
    if governed_bid is None and governed_ask is None:
        status = "REJECT"
    elif governed_bid is None or governed_ask is None:
        status = "CONDITIONAL_PASS"
    else:
        status = "PASS"
    return Quotes(
        reservation,
        governed_bid,
        governed_ask,
        half,
        False,
        bid_reason,
        ask_reason,
        status,
    )
