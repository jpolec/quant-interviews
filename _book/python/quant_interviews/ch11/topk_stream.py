from __future__ import annotations

import heapq
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Generic, Iterable, TypeVar

T = TypeVar("T")


@dataclass(order=True, frozen=True)
class ScoredItem(Generic[T]):
    score: float
    tie_priority: int
    sequence: int = field(compare=False)
    value: T = field(compare=False)
    event_id: str = field(compare=False)


@dataclass(frozen=True)
class StreamingTopKCheckpoint(Generic[T]):
    schema_version: int
    k: int
    next_sequence: int
    retained: tuple[ScoredItem[T], ...]
    seen_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported top-k checkpoint schema")
        if self.k < 0 or self.next_sequence < 0:
            raise ValueError("invalid top-k checkpoint counters")
        if len(self.retained) > self.k:
            raise ValueError("checkpoint retains more than k items")
        if tuple(sorted(self.seen_event_ids)) != self.seen_event_ids:
            raise ValueError("checkpoint event IDs must be sorted")
        if len(set(self.seen_event_ids)) != len(self.seen_event_ids):
            raise ValueError("checkpoint contains duplicate event IDs")
        if self.next_sequence != len(self.seen_event_ids):
            raise ValueError("checkpoint sequence does not reconcile with accepted events")
        if any(item.event_id not in self.seen_event_ids for item in self.retained):
            raise ValueError("retained item is absent from the idempotency history")


def top_k_stream(items: Iterable[tuple[float, T]], k: int) -> list[tuple[float, T]]:
    """Return the k highest-scoring items using O(k) memory.

    Sequence numbers make ties deterministic without requiring T to be orderable.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return []

    heap: list[ScoredItem[T]] = []
    for sequence, (score, value) in enumerate(items):
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError("scores must be finite")
        candidate = ScoredItem(
            numeric_score,
            -sequence,
            sequence,
            value,
            f"event-{sequence}",
        )
        if len(heap) < k:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)

    return [(item.score, item.value) for item in sorted(heap, reverse=True)]


class StreamingTopK(Generic[T]):
    def __init__(self, k: int) -> None:
        if k < 0:
            raise ValueError("k must be non-negative")
        self._k = k
        self._heap: list[ScoredItem[T]] = []
        # Event identity is a replay property, not a property of the retained
        # heap.  Keep the full history so an evicted (or k=0) event cannot be
        # accepted a second time.
        self._seen_event_ids: set[str] = set()
        self._sequence = 0

    def push(self, score: float, value: T, *, event_id: str | None = None) -> None:
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError("scores must be finite")
        stable_event_id = event_id if event_id is not None else f"event-{self._sequence}"
        if not isinstance(stable_event_id, str) or not stable_event_id:
            raise ValueError("event_id must be a non-empty string")
        if stable_event_id in self._seen_event_ids:
            raise ValueError("duplicate event_id")
        candidate = ScoredItem(
            numeric_score,
            -self._sequence,
            self._sequence,
            value,
            stable_event_id,
        )
        self._sequence += 1
        self._seen_event_ids.add(stable_event_id)
        if self._k == 0:
            return
        if len(self._heap) < self._k:
            heapq.heappush(self._heap, candidate)
        elif candidate > self._heap[0]:
            heapq.heapreplace(self._heap, candidate)

    def snapshot(self) -> tuple[tuple[float, T], ...]:
        return tuple((item.score, item.value) for item in sorted(self._heap, reverse=True))

    def replay_snapshot(self) -> tuple[ScoredItem[T], ...]:
        return tuple(sorted(self._heap, reverse=True))

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def seen_event_count(self) -> int:
        return len(self._seen_event_ids)

    def checkpoint(self) -> StreamingTopKCheckpoint[T]:
        """Return an immutable replay checkpoint including durable dedupe state."""

        return StreamingTopKCheckpoint(
            schema_version=1,
            k=self._k,
            next_sequence=self._sequence,
            retained=deepcopy(tuple(sorted(self._heap, reverse=True))),
            seen_event_ids=tuple(sorted(self._seen_event_ids)),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: StreamingTopKCheckpoint[T],
    ) -> StreamingTopK[T]:
        restored = cls(checkpoint.k)
        restored._sequence = checkpoint.next_sequence
        restored._seen_event_ids = set(checkpoint.seen_event_ids)
        restored._heap = deepcopy(list(checkpoint.retained))
        heapq.heapify(restored._heap)
        return restored
