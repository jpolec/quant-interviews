from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"


class OrderState(str, Enum):
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    STOPPED = "STOPPED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


TERMINAL_STATES = {
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.REJECTED,
    OrderState.EXPIRED,
}


@dataclass(frozen=True)
class Order:
    order_id: str
    side: Side
    order_type: OrderType
    time_in_force: TimeInForce
    quantity_lots: int
    venue: str
    submitted_at: datetime
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    oco_group_id: str | None = None
    state: OrderState = OrderState.NEW
    filled_lots: int = 0
    closed_unfilled_lots: int = 0
    cumulative_value: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    acknowledged_at: datetime | None = None
    closed_at: datetime | None = None
    close_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.order_id or not self.venue:
            raise ValueError("order_id and venue are required")
        if self.quantity_lots <= 0:
            raise ValueError("quantity_lots must be positive")
        _require_aware_utc(self.submitted_at, "submitted_at")
        if self.acknowledged_at is not None:
            _require_aware_utc(self.acknowledged_at, "acknowledged_at")
            if self.acknowledged_at < self.submitted_at:
                raise ValueError("acknowledgement cannot precede submission")
        if self.closed_at is not None:
            _require_aware_utc(self.closed_at, "closed_at")
            if self.closed_at < (self.acknowledged_at or self.submitted_at):
                raise ValueError("closure cannot precede the live order")
        if min(self.filled_lots, self.closed_unfilled_lots) < 0:
            raise ValueError("filled and closed quantities must be non-negative")
        if self.filled_lots + self.closed_unfilled_lots > self.quantity_lots:
            raise ValueError("order quantity is over-allocated")
        if (
            not self.cumulative_value.is_finite()
            or not self.fees.is_finite()
            or self.cumulative_value < 0
            or self.fees < 0
        ):
            raise ValueError("value and fees must be non-negative")
        if self.order_type is OrderType.LIMIT:
            if (
                self.limit_price is None
                or not self.limit_price.is_finite()
                or self.limit_price <= 0
            ):
                raise ValueError("a limit order requires a positive limit price")
        if self.order_type is OrderType.STOP:
            if (
                self.stop_price is None
                or not self.stop_price.is_finite()
                or self.stop_price <= 0
            ):
                raise ValueError("a stop order requires a positive stop price")
        if self.filled_lots == 0 and self.cumulative_value != 0:
            raise ValueError("zero-filled order cannot have execution value")
        if self.state is OrderState.FILLED and self.filled_lots != self.quantity_lots:
            raise ValueError("FILLED state requires full quantity")
        if self.state is OrderState.NEW and (
            self.acknowledged_at is not None
            or self.filled_lots
            or self.closed_unfilled_lots
            or self.closed_at is not None
        ):
            raise ValueError("NEW state cannot carry acknowledgement, fill or closure")
        if self.state in {
            OrderState.ACKNOWLEDGED,
            OrderState.STOPPED,
            OrderState.PARTIALLY_FILLED,
        }:
            if self.acknowledged_at is None or self.remaining_lots <= 0:
                raise ValueError("live states require acknowledgement and open quantity")
        if self.state is OrderState.PARTIALLY_FILLED and self.filled_lots <= 0:
            raise ValueError("PARTIALLY_FILLED requires a positive fill")
        if self.state in TERMINAL_STATES:
            if self.filled_lots + self.closed_unfilled_lots != self.quantity_lots:
                raise ValueError("terminal order must conserve the full quantity")
            if self.closed_at is None or not self.close_reason:
                raise ValueError("terminal state requires closure time and reason")

    @property
    def remaining_lots(self) -> int:
        return self.quantity_lots - self.filled_lots - self.closed_unfilled_lots

    @property
    def average_price(self) -> Decimal | None:
        if self.filled_lots == 0:
            return None
        return self.cumulative_value / Decimal(self.filled_lots)

    @property
    def quantity_conserved(self) -> bool:
        return (
            self.filled_lots + self.closed_unfilled_lots + self.remaining_lots
            == self.quantity_lots
        )


def acknowledge(order: Order, acknowledged_at: datetime) -> Order:
    if order.state is not OrderState.NEW:
        raise ValueError("only NEW orders can be acknowledged")
    _require_aware_utc(acknowledged_at, "acknowledged_at")
    if acknowledged_at < order.submitted_at:
        raise ValueError("acknowledgement cannot precede submission")
    initial_state = (
        OrderState.STOPPED
        if order.order_type is OrderType.STOP
        else OrderState.ACKNOWLEDGED
    )
    return replace(
        order,
        state=initial_state,
        acknowledged_at=acknowledged_at,
    )


def activate_stop(
    order: Order,
    trigger_price: Decimal,
) -> Order:
    if order.state is not OrderState.STOPPED or order.stop_price is None:
        raise ValueError("order is not an inactive stop")
    if not trigger_price.is_finite() or trigger_price <= 0:
        raise ValueError("trigger price must be finite and positive")
    triggered = (
        trigger_price >= order.stop_price
        if order.side is Side.BUY
        else trigger_price <= order.stop_price
    )
    if not triggered:
        return order
    return replace(order, state=OrderState.ACKNOWLEDGED)


def apply_fill(
    order: Order,
    quantity_lots: int,
    price: Decimal,
    filled_at: datetime,
    fee: Decimal = Decimal("0"),
) -> Order:
    if order.state not in {
        OrderState.ACKNOWLEDGED,
        OrderState.PARTIALLY_FILLED,
    }:
        raise ValueError("order is not fillable")
    _require_aware_utc(filled_at, "filled_at")
    if filled_at < (order.acknowledged_at or order.submitted_at):
        raise ValueError("fill cannot precede acknowledgement")
    if quantity_lots <= 0 or quantity_lots > order.remaining_lots:
        raise ValueError("invalid fill quantity")
    if (
        not price.is_finite()
        or not fee.is_finite()
        or price <= 0
        or fee < 0
    ):
        raise ValueError("price must be positive and fee non-negative")
    if order.order_type is OrderType.LIMIT and order.limit_price is not None:
        if order.side is Side.BUY and price > order.limit_price:
            raise ValueError("buy fill exceeds limit")
        if order.side is Side.SELL and price < order.limit_price:
            raise ValueError("sell fill is below limit")
    new_filled = order.filled_lots + quantity_lots
    state = (
        OrderState.FILLED
        if new_filled == order.quantity_lots
        else OrderState.PARTIALLY_FILLED
    )
    return replace(
        order,
        state=state,
        filled_lots=new_filled,
        cumulative_value=(
            order.cumulative_value + Decimal(quantity_lots) * price
        ),
        fees=order.fees + fee,
        closed_at=order.closed_at if state is not OrderState.FILLED else filled_at,
        close_reason=order.close_reason if state is not OrderState.FILLED else "filled",
    )


def _close(
    order: Order,
    state: OrderState,
    closed_at: datetime,
    reason: str,
) -> Order:
    if order.state in TERMINAL_STATES:
        raise ValueError("terminal order cannot transition again")
    _require_aware_utc(closed_at, "closed_at")
    if closed_at < (order.acknowledged_at or order.submitted_at):
        raise ValueError("closure cannot precede submission/acknowledgement")
    if not reason:
        raise ValueError("close reason is required")
    return replace(
        order,
        state=state,
        closed_unfilled_lots=order.closed_unfilled_lots + order.remaining_lots,
        closed_at=closed_at,
        close_reason=reason,
    )


def cancel(order: Order, closed_at: datetime, reason: str) -> Order:
    return _close(order, OrderState.CANCELLED, closed_at, reason)


def reject(order: Order, closed_at: datetime, reason: str) -> Order:
    return _close(order, OrderState.REJECTED, closed_at, reason)


def expire(order: Order, closed_at: datetime, reason: str = "time_in_force") -> Order:
    return _close(order, OrderState.EXPIRED, closed_at, reason)


def cancel_oco_sibling(
    filled_order: Order,
    sibling: Order,
    event_time: datetime,
) -> Order:
    if not filled_order.oco_group_id or (
        sibling.oco_group_id != filled_order.oco_group_id
    ):
        raise ValueError("orders are not in one OCO group")
    if filled_order.state is not OrderState.FILLED:
        raise ValueError("OCO cancellation requires a terminally filled sibling")
    return cancel(sibling, event_time, f"oco:{filled_order.order_id}")


def _require_aware_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field} must be normalized to UTC")
