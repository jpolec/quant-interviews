from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class RiskLimits:
    max_order_quantity: int
    max_notional: float
    max_participation: float
    price_collar_bps: float
    max_market_volume_age_ms: int
    limit_snapshot_version: str
    max_position_after_order: float
    max_credit_notional: float
    kill_switch_active: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_order_quantity, bool)
            or not isinstance(self.max_order_quantity, int)
            or self.max_order_quantity <= 0
            or not isfinite(self.max_notional)
            or self.max_notional <= 0.0
            or not isfinite(self.max_participation)
            or not 0.0 < self.max_participation <= 1.0
            or not isfinite(self.price_collar_bps)
            or self.price_collar_bps < 0.0
            or isinstance(self.max_market_volume_age_ms, bool)
            or not isinstance(self.max_market_volume_age_ms, int)
            or self.max_market_volume_age_ms < 0
            or not self.limit_snapshot_version
            or not isfinite(self.max_position_after_order)
            or self.max_position_after_order < 0.0
            or not isfinite(self.max_credit_notional)
            or self.max_credit_notional <= 0.0
            or not isinstance(self.kill_switch_active, bool)
        ):
            raise ValueError("invalid risk-limit snapshot")


@dataclass(frozen=True)
class OrderIntent:
    quantity: int
    price: float
    reference_price: float
    forecast_market_volume: float
    market_volume_age_ms: int
    limit_snapshot_version: str
    projected_position: float
    available_credit_notional: float
    self_trade_risk: bool


def pretrade_check(order: OrderIntent, limits: RiskLimits) -> list[str]:
    breaches: list[str] = []
    if (
        order.quantity <= 0
        or not isfinite(order.price)
        or order.price <= 0.0
        or not isfinite(order.reference_price)
        or order.reference_price <= 0.0
    ):
        breaches.append("INVALID_ORDER_FIELDS")
        return breaches
    if limits.kill_switch_active:
        breaches.append("KILL_SWITCH_ACTIVE")
    if order.limit_snapshot_version != limits.limit_snapshot_version:
        breaches.append("STALE_LIMIT_SNAPSHOT")
    if order.self_trade_risk:
        breaches.append("SELF_TRADE_RISK")
    if order.quantity > limits.max_order_quantity:
        breaches.append("MAX_ORDER_QUANTITY")
    notional = order.quantity * order.price
    if notional > limits.max_notional:
        breaches.append("MAX_NOTIONAL")
    if (
        not isfinite(order.forecast_market_volume)
        or order.forecast_market_volume <= 0.0
        or order.market_volume_age_ms < 0
        or order.market_volume_age_ms > limits.max_market_volume_age_ms
    ):
        breaches.append("INVALID_OR_STALE_MARKET_VOLUME")
    else:
        participation = order.quantity / order.forecast_market_volume
        if participation > limits.max_participation:
            breaches.append("MAX_PARTICIPATION")
    deviation_bps = abs(order.price / order.reference_price - 1.0) * 10_000.0
    if deviation_bps > limits.price_collar_bps:
        breaches.append("PRICE_COLLAR")
    if not isfinite(order.projected_position):
        breaches.append("UNKNOWN_POSITION_STATE")
    elif abs(order.projected_position) > limits.max_position_after_order:
        breaches.append("POSITION_LIMIT")
    if not isfinite(order.available_credit_notional) or order.available_credit_notional < 0.0:
        breaches.append("UNKNOWN_CREDIT_STATE")
    elif (
        notional > order.available_credit_notional
        or notional > limits.max_credit_notional
    ):
        breaches.append("CREDIT_LIMIT")
    return breaches
