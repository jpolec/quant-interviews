from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from threading import RLock
from typing import Callable, Generic, Iterable, Mapping, TypeVar

S = TypeVar("S")


def _payload_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class DomainEvent:
    event_id: str
    aggregate_id: str
    sequence: int
    event_type: str
    payload_json: bytes
    schema_version: str = ""
    reducer_version: str = ""
    producer_id: str = ""
    event_time: datetime | None = None
    known_at: datetime | None = None
    recorded_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.aggregate_id or not self.event_type:
            raise ValueError("event identity is incomplete")
        if self.sequence < 1:
            raise ValueError("sequence must be positive")
        if not self.schema_version or not self.reducer_version or not self.producer_id:
            raise ValueError("schema, reducer and producer versions are required")
        if self.event_time is None or self.known_at is None or self.recorded_at is None:
            raise ValueError("event, knowledge and recording clocks are required")
        for field, value in (
            ("event_time", self.event_time),
            ("known_at", self.known_at),
            ("recorded_at", self.recorded_at),
        ):
            _require_aware_utc(value, field)
        if self.known_at > self.recorded_at:
            raise ValueError("recording cannot precede knowledge")
        decoded = json.loads(self.payload_json)
        if not isinstance(decoded, dict) or _payload_bytes(decoded) != self.payload_json:
            raise ValueError("payload_json must be a canonical JSON object")

    @classmethod
    def create(
        cls,
        *,
        event_id: str,
        aggregate_id: str,
        sequence: int,
        event_type: str,
        payload: Mapping[str, object],
        schema_version: str,
        reducer_version: str,
        producer_id: str,
        event_time: datetime,
        known_at: datetime,
        recorded_at: datetime,
    ) -> DomainEvent:
        if not event_id or not aggregate_id or not event_type:
            raise ValueError("event identity is incomplete")
        if sequence < 1:
            raise ValueError("sequence must be positive")
        return cls(
            event_id,
            aggregate_id,
            sequence,
            event_type,
            _payload_bytes(payload),
            schema_version,
            reducer_version,
            producer_id,
            event_time,
            known_at,
            recorded_at,
        )

    @property
    def payload(self) -> dict[str, object]:
        decoded = json.loads(self.payload_json)
        if not isinstance(decoded, dict):
            raise TypeError("event payload must decode to an object")
        return decoded

    @property
    def content_hash(self) -> str:
        body = {
            "schema_version": self.schema_version,
            "aggregate_id": self.aggregate_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "payload_hash": sha256(self.payload_json).hexdigest(),
            "reducer_version": self.reducer_version,
            "producer_id": self.producer_id,
            "event_time": self.event_time.astimezone(timezone.utc).isoformat(),
            "known_at": self.known_at.astimezone(timezone.utc).isoformat(),
            "recorded_at": self.recorded_at.astimezone(timezone.utc).isoformat(),
        }
        return sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class AppendResult:
    status: str
    event_id: str
    aggregate_id: str
    sequence: int
    content_hash: str


@dataclass(frozen=True)
class Snapshot(Generic[S]):
    aggregate_id: str
    sequence: int
    state: S
    prefix_hash: str
    reducer_version: str

    def __post_init__(self) -> None:
        if not self.aggregate_id or self.sequence < 0 or not self.reducer_version:
            raise ValueError("invalid snapshot identity")
        if len(self.prefix_hash) != 64 or set(self.prefix_hash) - set("0123456789abcdef"):
            raise ValueError("prefix_hash must be SHA-256")


class EventStore:
    def __init__(self) -> None:
        self._events: list[DomainEvent] = []
        self._by_id: dict[str, DomainEvent] = {}
        self._lock = RLock()

    def append(self, event: DomainEvent) -> AppendResult:
        with self._lock:
            prior = self._by_id.get(event.event_id)
            if prior is not None:
                if prior.content_hash != event.content_hash:
                    raise ValueError(
                        "event_id conflict: payload or aggregate changed"
                    )
                return AppendResult(
                    "DUPLICATE_SAME_CONTENT",
                    event.event_id,
                    event.aggregate_id,
                    event.sequence,
                    event.content_hash,
                )
            aggregate_events = [
                existing
                for existing in self._events
                if existing.aggregate_id == event.aggregate_id
            ]
            expected = (
                aggregate_events[-1].sequence + 1
                if aggregate_events
                else 1
            )
            if event.sequence != expected:
                raise ValueError(
                    f"sequence gap for {event.aggregate_id}: "
                    f"expected {expected}, got {event.sequence}"
                )
            if any(
                existing.aggregate_id == event.aggregate_id
                and existing.sequence == event.sequence
                for existing in self._events
            ):
                raise ValueError("duplicate aggregate sequence")
            self._events.append(event)
            self._by_id[event.event_id] = event
            return AppendResult(
                "APPENDED",
                event.event_id,
                event.aggregate_id,
                event.sequence,
                event.content_hash,
            )

    def stream(self, aggregate_id: str) -> tuple[DomainEvent, ...]:
        with self._lock:
            return tuple(
                event
                for event in self._events
                if event.aggregate_id == aggregate_id
            )


def fold_events(
    initial: S,
    events: Iterable[DomainEvent],
    reducers: Mapping[str, Callable[[S, Mapping[str, object]], S]],
) -> S:
    stream = tuple(events)
    aggregate_ids = {event.aggregate_id for event in stream}
    if len(aggregate_ids) > 1:
        raise ValueError("fold_events accepts one aggregate only")
    state = initial
    last_sequence = 0
    for event in sorted(stream, key=lambda item: item.sequence):
        if event.sequence != last_sequence + 1:
            raise ValueError("non-contiguous aggregate event stream")
        reducer = reducers.get(event.event_type)
        if reducer is None:
            raise ValueError(f"unknown event type: {event.event_type}")
        state = reducer(state, event.payload)
        last_sequence = event.sequence
    return state


def build_snapshot(
    aggregate_id: str,
    initial: S,
    events: Iterable[DomainEvent],
    reducers: Mapping[str, Callable[[S, Mapping[str, object]], S]],
    *,
    reducer_version: str,
) -> Snapshot[S]:
    if not reducer_version:
        raise ValueError("reducer_version is required")
    stream = tuple(events)
    if any(event.aggregate_id != aggregate_id for event in stream):
        raise ValueError("snapshot stream contains a different aggregate")
    state = fold_events(initial, stream, reducers)
    prefix_hash = sha256()
    for event in sorted(stream, key=lambda item: item.sequence):
        prefix_hash.update(event.content_hash.encode())
    return Snapshot(
        aggregate_id=aggregate_id,
        sequence=max((event.sequence for event in stream), default=0),
        state=state,
        prefix_hash=prefix_hash.hexdigest(),
        reducer_version=reducer_version,
    )


def verify_snapshot(
    snapshot: Snapshot[S],
    initial: S,
    events: Iterable[DomainEvent],
    reducers: Mapping[str, Callable[[S, Mapping[str, object]], S]],
) -> bool:
    stream = tuple(events)
    if any(event.aggregate_id != snapshot.aggregate_id for event in stream):
        return False
    ordered = tuple(sorted(stream, key=lambda event: event.sequence))
    try:
        fold_events(initial, ordered, reducers)
    except (TypeError, ValueError):
        return False
    prefix = tuple(
        event for event in ordered if event.sequence <= snapshot.sequence
    )
    if len(prefix) != snapshot.sequence:
        return False
    rebuilt = build_snapshot(
        snapshot.aggregate_id,
        initial,
        prefix,
        reducers,
        reducer_version=snapshot.reducer_version,
    )
    return rebuilt == snapshot


def _require_aware_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field} must be normalized to UTC")
