from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class EventType(str, Enum):
    SUBMIT = "SUBMIT"
    ACK = "ACK"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    CANCEL_REQUEST = "CANCEL_REQUEST"
    CANCEL_ACK = "CANCEL_ACK"
    REJECT = "REJECT"


@dataclass(frozen=True)
class OrderEvent:
    event_id: str
    sequence: int
    event_type: EventType
    quantity: int = 0
    price: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
        ):
            raise ValueError("sequence must be a positive integer")
        if not isinstance(self.event_type, EventType):
            raise ValueError("event_type must be an EventType")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise ValueError("event quantity must be an integer")
        if self.event_type == EventType.SUBMIT:
            if self.quantity <= 0 or self.price is not None:
                raise ValueError("SUBMIT requires positive quantity and no price")
        elif self.event_type in {EventType.PARTIAL_FILL, EventType.FILL}:
            if (
                self.quantity <= 0
                or self.price is None
                or not isfinite(self.price)
                or self.price <= 0.0
            ):
                raise ValueError("fill events require positive finite quantity and price")
        elif self.quantity != 0 or self.price is not None:
            raise ValueError("non-quantity events cannot carry quantity or price")


@dataclass(frozen=True)
class OrderState:
    submitted_quantity: int
    filled_quantity: int
    average_fill_price: float | None
    status: str
    cancel_requested: bool
    terminal_reason: str | None
    last_sequence: int
    applied_event_ids: tuple[str, ...]


_TERMINAL = {
    "FILLED",
    "CANCELLED",
    "PARTIALLY_FILLED_CANCELLED",
    "REJECTED",
    "PARTIALLY_FILLED_REJECTED",
}


def reconstruct_order(events: list[OrderEvent]) -> OrderState:
    if not events:
        raise ValueError("at least one event is required")
    submitted = 0
    filled = 0
    notional = 0.0
    status = "UNKNOWN"
    cancel_pending = False
    last = 0
    applied: dict[str, OrderEvent] = {}
    for event in events:
        if event.event_id in applied:
            if applied[event.event_id] != event:
                raise ValueError("event_id reused with different payload")
            continue
        if event.sequence <= last:
            raise ValueError("event sequences must be strictly increasing")
        if event.sequence != last + 1:
            raise ValueError("event sequence gap")
        last = event.sequence
        if status in _TERMINAL:
            raise ValueError("non-correction event after terminal state")
        if event.event_type == EventType.SUBMIT:
            if status != "UNKNOWN" or submitted != 0 or event.quantity <= 0:
                raise ValueError("invalid duplicate or non-positive submit")
            submitted = event.quantity
            status = "PENDING_ACK"
        elif event.event_type == EventType.ACK:
            if status not in {"PENDING_ACK", "PARTIALLY_FILLED"}:
                raise ValueError("ACK is invalid in current state")
            if status == "PENDING_ACK":
                status = "WORKING"
        elif event.event_type in {EventType.PARTIAL_FILL, EventType.FILL}:
            if status not in {
                "PENDING_ACK",
                "WORKING",
                "PARTIALLY_FILLED",
                "CANCEL_PENDING",
                "PARTIALLY_FILLED_CANCEL_PENDING",
            }:
                raise ValueError("fill is invalid in current state")
            filled += event.quantity
            assert event.price is not None
            notional += event.quantity * event.price
            if filled > submitted:
                raise ValueError("overfill")
            if event.event_type == EventType.FILL and filled != submitted:
                raise ValueError("FILL must complete the submitted quantity")
            if event.event_type == EventType.PARTIAL_FILL and filled == submitted:
                raise ValueError("PARTIAL_FILL cannot complete the order")
            if filled == submitted:
                status = "FILLED"
                cancel_pending = False
            else:
                status = (
                    "PARTIALLY_FILLED_CANCEL_PENDING"
                    if cancel_pending
                    else "PARTIALLY_FILLED"
                )
        elif event.event_type == EventType.CANCEL_REQUEST:
            if status not in {"PENDING_ACK", "WORKING", "PARTIALLY_FILLED"}:
                raise ValueError("cancel request is invalid in current state")
            cancel_pending = True
            status = (
                "PARTIALLY_FILLED_CANCEL_PENDING"
                if filled
                else "CANCEL_PENDING"
            )
        elif event.event_type == EventType.CANCEL_ACK:
            if not cancel_pending:
                raise ValueError("CANCEL_ACK without CANCEL_REQUEST")
            status = "PARTIALLY_FILLED_CANCELLED" if filled else "CANCELLED"
            cancel_pending = False
        elif event.event_type == EventType.REJECT:
            if status not in {"PENDING_ACK", "WORKING", "PARTIALLY_FILLED"}:
                raise ValueError("REJECT is invalid in current state")
            status = "PARTIALLY_FILLED_REJECTED" if filled else "REJECTED"
        applied[event.event_id] = event
    average = notional / filled if filled else None
    terminal_reason = status if status in _TERMINAL else None
    return OrderState(
        submitted,
        filled,
        average,
        status,
        cancel_pending,
        terminal_reason,
        last,
        tuple(applied),
    )
