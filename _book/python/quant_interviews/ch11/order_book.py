from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import heapq
from typing import Literal

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class Order:
    order_id: int
    side: Side
    price_ticks: int
    quantity: int
    sequence: int


@dataclass(frozen=True)
class Fill:
    maker_order_id: int
    taker_order_id: int
    price_ticks: int
    quantity: int


@dataclass(frozen=True)
class ExecutionResult:
    fills: tuple[Fill, ...]
    requested_quantity: int
    remaining_quantity: int
    status: Literal["FILLED", "PARTIAL", "UNFILLED", "RESTING"]

    @property
    def filled_quantity(self) -> int:
        return sum(fill.quantity for fill in self.fills)

    def __post_init__(self) -> None:
        if self.filled_quantity + self.remaining_quantity != self.requested_quantity:
            raise ValueError("execution quantity does not reconcile")


class PriceTimeOrderBook:
    """Readable reference with O(log P) best-level maintenance and O(1) cancel."""

    def __init__(self) -> None:
        self._bids: dict[int, OrderedDict[int, Order]] = {}
        self._asks: dict[int, OrderedDict[int, Order]] = {}
        self._bid_heap: list[int] = []
        self._ask_heap: list[int] = []
        self._orders: dict[int, Order] = {}
        self._seen_ids: set[int] = set()
        self._fill_history: list[Fill] = []
        self._cancel_history: list[tuple[int, int, bool]] = []
        # Authoritative external event sequences start at one and are
        # contiguous.  A gap is a replay failure, not merely a later event.
        self._last_sequence = 0

    @staticmethod
    def _validate_side(side: str) -> Side:
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        return side  # type: ignore[return-value]

    def _next_sequence(self, sequence: int | None) -> int:
        value = self._last_sequence + 1 if sequence is None else sequence
        if value != self._last_sequence + 1:
            raise ValueError("external sequence must equal the next contiguous value")
        self._last_sequence = value
        return value

    @staticmethod
    def _clean_best(levels: dict[int, OrderedDict[int, Order]], heap: list[int], sign: int) -> None:
        while heap and sign * heap[0] not in levels:
            heapq.heappop(heap)

    @property
    def best_bid(self) -> int | None:
        self._clean_best(self._bids, self._bid_heap, -1)
        return None if not self._bid_heap else -self._bid_heap[0]

    @property
    def best_ask(self) -> int | None:
        self._clean_best(self._asks, self._ask_heap, 1)
        return None if not self._ask_heap else self._ask_heap[0]

    def _rest(self, order: Order) -> None:
        levels = self._bids if order.side == "buy" else self._asks
        heap = self._bid_heap if order.side == "buy" else self._ask_heap
        heap_key = -order.price_ticks if order.side == "buy" else order.price_ticks
        if order.price_ticks not in levels:
            levels[order.price_ticks] = OrderedDict()
            heapq.heappush(heap, heap_key)
        levels[order.price_ticks][order.order_id] = order
        self._orders[order.order_id] = order

    def _match(
        self,
        taker_order_id: int,
        side: Side,
        quantity: int,
        limit_price: int | None,
    ) -> ExecutionResult:
        opposite = self._asks if side == "buy" else self._bids
        fills: list[Fill] = []
        remaining = quantity
        while remaining > 0:
            best = self.best_ask if side == "buy" else self.best_bid
            if best is None:
                break
            if limit_price is not None:
                marketable = best <= limit_price if side == "buy" else best >= limit_price
                if not marketable:
                    break
            queue = opposite[best]
            maker_id = next(iter(queue))
            maker = queue[maker_id]
            traded = min(remaining, maker.quantity)
            fills.append(Fill(maker.order_id, taker_order_id, best, traded))
            remaining -= traded
            if traded == maker.quantity:
                del queue[maker_id]
                self._orders.pop(maker_id)
            else:
                residual = Order(
                    maker.order_id,
                    maker.side,
                    maker.price_ticks,
                    maker.quantity - traded,
                    maker.sequence,
                )
                queue[maker_id] = residual
                self._orders[maker_id] = residual
            if not queue:
                del opposite[best]
        if remaining == 0:
            status: Literal["FILLED", "PARTIAL", "UNFILLED", "RESTING"] = "FILLED"
        elif fills:
            status = "PARTIAL"
        else:
            status = "UNFILLED"
        self._fill_history.extend(fills)
        return ExecutionResult(tuple(fills), quantity, remaining, status)

    def add_limit(
        self,
        order_id: int,
        side: Side,
        price_ticks: int,
        quantity: int,
        *,
        sequence: int | None = None,
    ) -> ExecutionResult:
        validated_side = self._validate_side(side)
        if order_id in self._seen_ids:
            raise ValueError("duplicate order_id")
        if price_ticks <= 0 or quantity <= 0:
            raise ValueError("price and quantity must be positive")
        event_sequence = self._next_sequence(sequence)
        self._seen_ids.add(order_id)
        result = self._match(order_id, validated_side, quantity, price_ticks)
        if result.remaining_quantity:
            self._rest(
                Order(
                    order_id,
                    validated_side,
                    price_ticks,
                    result.remaining_quantity,
                    event_sequence,
                )
            )
            status = "RESTING" if not result.fills else "PARTIAL"
            result = ExecutionResult(
                result.fills,
                result.requested_quantity,
                result.remaining_quantity,
                status,
            )
        self._assert_not_crossed()
        return result

    def cancel(self, order_id: int, *, sequence: int | None = None) -> bool:
        event_sequence = self._next_sequence(sequence)
        order = self._orders.pop(order_id, None)
        if order is None:
            self._cancel_history.append((order_id, event_sequence, False))
            return False
        levels = self._bids if order.side == "buy" else self._asks
        queue = levels[order.price_ticks]
        del queue[order_id]
        if not queue:
            del levels[order.price_ticks]
        self._cancel_history.append((order_id, event_sequence, True))
        return True

    def execute_market(
        self,
        taker_order_id: int,
        side: Side,
        quantity: int,
        *,
        sequence: int | None = None,
    ) -> ExecutionResult:
        validated_side = self._validate_side(side)
        if taker_order_id in self._seen_ids:
            raise ValueError("duplicate order_id")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self._next_sequence(sequence)
        self._seen_ids.add(taker_order_id)
        result = self._match(taker_order_id, validated_side, quantity, None)
        self._assert_not_crossed()
        return result

    def _assert_not_crossed(self) -> None:
        if self.best_bid is not None and self.best_ask is not None:
            if self.best_bid >= self.best_ask:
                raise AssertionError("resting book is crossed")

    def state_hash(self) -> str:
        orders = "".join(
            f"{order.order_id},{order.side},{order.price_ticks},"
            f"{order.quantity},{order.sequence};"
            for order in sorted(self._orders.values(), key=lambda item: item.sequence)
        )
        seen = ",".join(str(order_id) for order_id in sorted(self._seen_ids))
        fills = "".join(
            f"{fill.maker_order_id},{fill.taker_order_id},{fill.price_ticks},"
            f"{fill.quantity};"
            for fill in self._fill_history
        )
        cancellations = "".join(
            f"{order_id},{sequence},{int(found)};"
            for order_id, sequence, found in self._cancel_history
        )
        payload = (
            f"v1|last={self._last_sequence}|seen={seen}|orders={orders}|"
            f"fills={fills}|cancels={cancellations}"
        ).encode("ascii")
        value = 14695981039346656037
        for byte in payload:
            value ^= byte
            value = (value * 1099511628211) & ((1 << 64) - 1)
        return f"{value:016x}"
