from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class _Node:
    key: tuple[float, int]
    priority: int
    size: int = 1
    left: _Node | None = None
    right: _Node | None = None


def _size(node: _Node | None) -> int:
    return 0 if node is None else node.size


def _refresh(node: _Node) -> None:
    node.size = 1 + _size(node.left) + _size(node.right)


def _rotate_right(node: _Node) -> _Node:
    replacement = node.left
    assert replacement is not None
    node.left = replacement.right
    replacement.right = node
    _refresh(node)
    _refresh(replacement)
    return replacement


def _rotate_left(node: _Node) -> _Node:
    replacement = node.right
    assert replacement is not None
    node.right = replacement.left
    replacement.left = node
    _refresh(node)
    _refresh(replacement)
    return replacement


def _priority(sequence: int) -> int:
    # Deterministic SplitMix64 permutation: replay does not depend on process RNG.
    value = (sequence + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return value ^ (value >> 31)


def _insert(node: _Node | None, key: tuple[float, int]) -> _Node:
    if node is None:
        return _Node(key=key, priority=_priority(key[1]))
    if key < node.key:
        node.left = _insert(node.left, key)
        if node.left.priority < node.priority:
            node = _rotate_right(node)
    elif key > node.key:
        node.right = _insert(node.right, key)
        if node.right.priority < node.priority:
            node = _rotate_left(node)
    else:
        raise RuntimeError("duplicate rolling key")
    _refresh(node)
    return node


def _erase(node: _Node | None, key: tuple[float, int]) -> _Node | None:
    if node is None:
        raise RuntimeError("rolling structure lost synchronization")
    if key < node.key:
        node.left = _erase(node.left, key)
    elif key > node.key:
        node.right = _erase(node.right, key)
    elif node.left is None:
        return node.right
    elif node.right is None:
        return node.left
    elif node.left.priority < node.right.priority:
        node = _rotate_right(node)
        node.right = _erase(node.right, key)
    else:
        node = _rotate_left(node)
        node.left = _erase(node.left, key)
    _refresh(node)
    return node


def _select(node: _Node | None, index: int) -> float:
    if node is None or not 0 <= index < _size(node):
        raise IndexError(index)
    left_size = _size(node.left)
    if index < left_size:
        return _select(node.left, index)
    if index == left_size:
        return node.key[0]
    return _select(node.right, index - left_size - 1)


class ExactRollingQuantile:
    """Lower order-statistic quantile with O(log window) insert/delete/select."""

    def __init__(self, window: int, quantile: float, *, min_periods: int | None = None) -> None:
        if window < 1:
            raise ValueError("window must be positive")
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        required = window if min_periods is None else min_periods
        if not 1 <= required <= window:
            raise ValueError("min_periods must be in [1, window]")
        self.window = window
        self.quantile = quantile
        self.min_periods = required
        self._queue: deque[tuple[float, int]] = deque()
        self._root: _Node | None = None
        self._sequence = 0

    def push(self, value: float) -> float | None:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("value must be finite")
        key = (numeric, self._sequence)
        self._sequence += 1
        self._queue.append(key)
        self._root = _insert(self._root, key)
        if len(self._queue) > self.window:
            self._root = _erase(self._root, self._queue.popleft())
        if len(self._queue) < self.min_periods:
            return None
        return self.value()

    def value(self) -> float:
        count = len(self._queue)
        if count < self.min_periods:
            raise ValueError("minimum observations not reached")
        # Complete convention: p=0 -> first, p=1 -> last.
        rank_zero_based = math.floor(self.quantile * (count - 1))
        return _select(self._root, rank_zero_based)


@dataclass(frozen=True)
class HistogramDiagnostics:
    observations: int
    below_range: int
    above_range: int
    bin_width: float


class HistogramQuantile:
    """Rolling fixed-range histogram with O(1) insert/delete and explicit range gate."""

    def __init__(
        self,
        lower: float,
        upper: float,
        bins: int,
        quantile: float,
        *,
        window: int,
        min_periods: int | None = None,
    ) -> None:
        if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
            raise ValueError("finite lower must be smaller than finite upper")
        if bins < 2:
            raise ValueError("bins must be at least two")
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        if window < 1:
            raise ValueError("window must be positive")
        required = window if min_periods is None else min_periods
        if not 1 <= required <= window:
            raise ValueError("min_periods must be in [1, window]")
        self.lower = float(lower)
        self.upper = float(upper)
        self.bins = bins
        self.quantile = quantile
        self.window = window
        self.min_periods = required
        self.edges = np.linspace(self.lower, self.upper, self.bins + 1)
        self.counts = np.zeros(self.bins, dtype=np.int64)
        self._queue: deque[int] = deque()
        self._below_range = 0
        self._above_range = 0

    def _bin_index(self, value: float) -> int:
        if not math.isfinite(value):
            raise ValueError("values must be finite")
        if value < self.lower:
            raise ValueError("value below governed histogram range")
        if value > self.upper:
            raise ValueError("value above governed histogram range")
        if value == self.upper:
            return self.bins - 1
        return int(np.searchsorted(self.edges, value, side="right") - 1)

    def _append_index(self, index: int) -> float | None:
        self._queue.append(index)
        self.counts[index] += 1
        if len(self._queue) > self.window:
            expired = self._queue.popleft()
            self.counts[expired] -= 1
        if len(self._queue) < self.min_periods:
            return None
        return self.value()

    def push(self, value: float) -> float | None:
        numeric = float(value)
        try:
            index = self._bin_index(numeric)
        except ValueError:
            if math.isfinite(numeric) and numeric < self.lower:
                self._below_range += 1
            elif math.isfinite(numeric) and numeric > self.upper:
                self._above_range += 1
            raise
        return self._append_index(index)

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("values must be one-dimensional")
        if np.any(~np.isfinite(x)):
            raise ValueError("values must be finite")
        below = int(np.count_nonzero(x < self.lower))
        above = int(np.count_nonzero(x > self.upper))
        if below or above:
            # Diagnostics record the rejected batch, but the governed rolling
            # state remains atomic: no observation from the batch is applied.
            self._below_range += below
            self._above_range += above
            raise ValueError("bulk update contains a value outside the governed range")
        indexes = tuple(self._bin_index(float(value)) for value in x)
        for index in indexes:
            self._append_index(index)

    def value(self) -> float:
        total = len(self._queue)
        if total < self.min_periods:
            raise ValueError("minimum observations not reached")
        target = math.floor(self.quantile * (total - 1)) + 1
        cumulative = np.cumsum(self.counts)
        index = int(np.searchsorted(cumulative, target, side="left"))
        return 0.5 * (self.edges[index] + self.edges[index + 1])

    def diagnostics(self) -> HistogramDiagnostics:
        return HistogramDiagnostics(
            observations=len(self._queue),
            below_range=self._below_range,
            above_range=self._above_range,
            bin_width=(self.upper - self.lower) / self.bins,
        )
