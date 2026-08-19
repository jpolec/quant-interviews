from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BookMetrics:
    best_bid: float
    best_ask: float
    spread: float
    midpoint: float
    top_imbalance: float
    microprice: float


@dataclass(frozen=True)
class BookSnapshot:
    bid_prices: FloatArray
    bid_sizes: FloatArray
    ask_prices: FloatArray
    ask_sizes: FloatArray
    tick_size: float = 0.01
    lot_size: float = 1.0

    def __post_init__(self) -> None:
        arrays = tuple(
            np.asarray(value, dtype=float)
            for value in (
                self.bid_prices,
                self.bid_sizes,
                self.ask_prices,
                self.ask_sizes,
            )
        )
        for field, value in zip(
            ("bid_prices", "bid_sizes", "ask_prices", "ask_sizes"),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, field, value)
        if any(np.asarray(a).ndim != 1 for a in arrays):
            raise ValueError("book arrays must be one-dimensional")
        if len(self.bid_prices) == 0 or len(self.ask_prices) == 0:
            raise ValueError("both sides of the book are required")
        if len(self.bid_prices) != len(self.bid_sizes):
            raise ValueError("bid price and size lengths differ")
        if len(self.ask_prices) != len(self.ask_sizes):
            raise ValueError("ask price and size lengths differ")
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("book arrays must be finite")
        if (
            not np.isfinite(self.tick_size)
            or self.tick_size <= 0.0
            or not np.isfinite(self.lot_size)
            or self.lot_size <= 0.0
        ):
            raise ValueError("tick_size and lot_size must be finite and positive")
        if np.any(self.bid_prices <= 0.0) or np.any(self.ask_prices <= 0.0):
            raise ValueError("all book prices must be strictly positive")
        if np.any(self.bid_sizes <= 0.0) or np.any(self.ask_sizes <= 0.0):
            raise ValueError("displayed sizes must be strictly positive")
        price_ratios = np.r_[self.bid_prices, self.ask_prices] / self.tick_size
        size_ratios = np.r_[self.bid_sizes, self.ask_sizes] / self.lot_size
        if not np.allclose(price_ratios, np.rint(price_ratios), rtol=0.0, atol=1e-8):
            raise ValueError("book prices must be aligned to the tick grid")
        if not np.allclose(size_ratios, np.rint(size_ratios), rtol=0.0, atol=1e-8):
            raise ValueError("displayed sizes must be aligned to the lot grid")
        if np.any(np.diff(self.bid_prices) >= 0.0):
            raise ValueError("bid prices must be strictly descending")
        if np.any(np.diff(self.ask_prices) <= 0.0):
            raise ValueError("ask prices must be strictly ascending")
        if self.bid_prices[0] >= self.ask_prices[0]:
            raise ValueError("snapshot is crossed or locked")

    def metrics(self) -> BookMetrics:
        bid = float(self.bid_prices[0])
        ask = float(self.ask_prices[0])
        bid_size = float(self.bid_sizes[0])
        ask_size = float(self.ask_sizes[0])
        total = bid_size + ask_size
        imbalance = (bid_size - ask_size) / total
        microprice = (ask * bid_size + bid * ask_size) / total
        return BookMetrics(
            best_bid=bid,
            best_ask=ask,
            spread=ask - bid,
            midpoint=0.5 * (bid + ask),
            top_imbalance=imbalance,
            microprice=microprice,
        )

    def market_order_vwap(self, side: str, quantity: float) -> float:
        if quantity <= 0.0 or not np.isfinite(quantity):
            raise ValueError("quantity must be positive and finite")
        if not np.isclose(
            quantity / self.lot_size,
            round(quantity / self.lot_size),
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError("quantity must be aligned to the lot grid")
        side_norm = side.lower()
        if side_norm == "buy":
            prices, sizes = self.ask_prices, self.ask_sizes
        elif side_norm == "sell":
            prices, sizes = self.bid_prices, self.bid_sizes
        else:
            raise ValueError("side must be 'buy' or 'sell'")
        remaining = float(quantity)
        notional = 0.0
        for price, size in zip(prices, sizes, strict=True):
            take = min(remaining, float(size))
            notional += take * float(price)
            remaining -= take
            if remaining <= 1e-12:
                return notional / quantity
        raise ValueError("insufficient displayed depth for requested quantity")
