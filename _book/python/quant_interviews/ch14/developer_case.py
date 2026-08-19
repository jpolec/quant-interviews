from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from heapq import nlargest
from math import isfinite, sqrt


@dataclass
class OnlineStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.count, int) or self.count < 0:
            raise ValueError("count must be a non-negative integer")
        if not isfinite(self.mean) or not isfinite(self.m2) or self.m2 < 0.0:
            raise ValueError("online state must have finite mean and non-negative m2")
        if self.count == 0 and (self.mean != 0.0 or self.m2 != 0.0):
            raise ValueError("an empty state must have zero mean and m2")

    def update(self, value: float) -> None:
        if not isfinite(value):
            raise ValueError("online observations must be finite")
        new_count = self.count + 1
        delta = value - self.mean
        new_mean = self.mean + delta / new_count
        new_m2 = self.m2 + delta * (value - new_mean)
        if not isfinite(new_mean) or not isfinite(new_m2) or new_m2 < 0.0:
            raise FloatingPointError("online update would create a corrupt state")
        # Commit only after all post-conditions pass.
        self.count = new_count
        self.mean = new_mean
        self.m2 = new_m2

    @property
    def variance(self) -> float:
        if self.count < 2:
            raise ValueError("sample variance requires at least two observations")
        result = self.m2 / (self.count - 1)
        if not isfinite(result) or result < 0.0:
            raise FloatingPointError("online variance is invalid")
        return result

    @property
    def standard_deviation(self) -> float:
        return sqrt(self.variance)


@dataclass
class _Node:
    key: tuple[float, str]
    priority: int
    left: _Node | None = None
    right: _Node | None = None


def _priority(identifier: str) -> int:
    return int.from_bytes(sha256(identifier.encode("utf-8")).digest()[:8], "big")


def _rotate_left(root: _Node) -> _Node:
    pivot = root.right
    assert pivot is not None
    root.right = pivot.left
    pivot.left = root
    return pivot


def _rotate_right(root: _Node) -> _Node:
    pivot = root.left
    assert pivot is not None
    root.left = pivot.right
    pivot.right = root
    return pivot


def _insert(root: _Node | None, node: _Node) -> _Node:
    if root is None:
        return node
    if node.key < root.key:
        root.left = _insert(root.left, node)
        if root.left is not None and root.left.priority < root.priority:
            root = _rotate_right(root)
    else:
        root.right = _insert(root.right, node)
        if root.right is not None and root.right.priority < root.priority:
            root = _rotate_left(root)
    return root


def _delete(root: _Node | None, key: tuple[float, str]) -> _Node | None:
    if root is None:
        return None
    if key < root.key:
        root.left = _delete(root.left, key)
    elif key > root.key:
        root.right = _delete(root.right, key)
    elif root.left is None:
        return root.right
    elif root.right is None:
        return root.left
    elif root.left.priority < root.right.priority:
        root = _rotate_right(root)
        root.right = _delete(root.right, key)
    else:
        root = _rotate_left(root)
        root.left = _delete(root.left, key)
    return root


@dataclass
class UpdateableTopK:
    """Expected O(log n) under hash-like priorities; not a worst-case tree."""

    _root: _Node | None = None
    _scores: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self._scores is None:
            self._scores = {}

    def update(self, identifier: str, score: float) -> None:
        if not identifier or not isfinite(score):
            raise ValueError("identifier must be non-empty and score finite")
        assert self._scores is not None
        previous = self._scores.get(identifier)
        if previous is not None:
            self._root = _delete(self._root, (previous, identifier))
        self._scores[identifier] = score
        self._root = _insert(
            self._root,
            _Node((score, identifier), _priority(identifier)),
        )

    def delete(self, identifier: str) -> bool:
        assert self._scores is not None
        previous = self._scores.pop(identifier, None)
        if previous is None:
            return False
        self._root = _delete(self._root, (previous, identifier))
        return True

    def top_k(self, k: int) -> list[tuple[float, str]]:
        if k <= 0:
            raise ValueError("k must be positive")
        result: list[tuple[float, str]] = []

        def reverse_in_order(node: _Node | None) -> None:
            if node is None or len(result) >= k:
                return
            reverse_in_order(node.right)
            if len(result) < k:
                result.append(node.key)
            reverse_in_order(node.left)

        reverse_in_order(self._root)
        return result

    @property
    def tree_height(self) -> int:
        def height(node: _Node | None) -> int:
            return 0 if node is None else 1 + max(height(node.left), height(node.right))

        return height(self._root)

    @property
    def active_count(self) -> int:
        assert self._scores is not None
        return len(self._scores)


def deterministic_top_k(
    updates: list[tuple[float, str]],
    k: int,
) -> list[tuple[float, str]]:
    book = UpdateableTopK()
    for score, identifier in updates:
        book.update(identifier, score)
    return book.top_k(k)


def batch_updateable_top_k(
    updates: list[tuple[float, str]],
    k: int,
) -> list[tuple[float, str]]:
    """Last-update-wins reference: O(n + a log k), a active IDs."""

    if k <= 0:
        raise ValueError("k must be positive")
    active: dict[str, float] = {}
    for score, identifier in updates:
        if not identifier or not isfinite(score):
            raise ValueError("identifier must be non-empty and score finite")
        active[identifier] = score
    return nlargest(k, ((score, identifier) for identifier, score in active.items()))


def relative_error(actual: float, reference: float, floor: float = 1e-12) -> float:
    if not all(isfinite(value) for value in (actual, reference, floor)) or floor <= 0.0:
        raise ValueError("relative-error inputs must be finite and floor positive")
    return abs(actual - reference) / max(abs(reference), floor)
