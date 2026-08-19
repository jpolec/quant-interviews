from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from enum import IntEnum
from heapq import heappop, heappush
from types import MappingProxyType
from typing import Any, Callable, Mapping


class EventPriority(IntEnum):
    MARKET_DATA = 10
    FEATURE = 20
    SIGNAL = 30
    ORDER = 40
    FILL = 50
    ACCOUNTING = 60


CausalKey = tuple[int, int, int, int, str]


def _freeze_payload(payload: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    frozen: list[tuple[str, str]] = []
    for key, value in sorted(payload.items()):
        if not isinstance(key, str):
            raise TypeError("payload keys must be strings")
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        frozen.append((key, encoded))
    return tuple(frozen)


@dataclass(order=True, frozen=True)
class Event:
    available_at_ns: int
    priority: EventPriority
    event_time_ns: int
    sequence: int
    event_id: str
    kind: str = field(compare=False)
    payload_items: tuple[tuple[str, str], ...] = field(
        compare=False, default_factory=tuple
    )

    def __post_init__(self) -> None:
        if min(self.available_at_ns, self.event_time_ns, self.sequence) < 0:
            raise ValueError("event clocks and sequence must be non-negative")
        if not self.event_id or not self.kind:
            raise ValueError("event_id and kind are required")
        if not isinstance(self.priority, EventPriority):
            raise TypeError("priority must be EventPriority")
        if tuple(sorted(self.payload_items)) != self.payload_items:
            raise ValueError("payload_items must be canonically sorted")
        keys = [item[0] for item in self.payload_items]
        if len(keys) != len(set(keys)):
            raise ValueError("payload keys must be unique")
        for key, encoded in self.payload_items:
            if not isinstance(key, str) or not key:
                raise ValueError("payload keys must be non-empty strings")
            decoded = json.loads(encoded)
            if json.dumps(
                decoded,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ) != encoded:
                raise ValueError("payload values must use canonical JSON")

    @classmethod
    def from_payload(
        cls,
        *,
        event_id: str,
        event_time_ns: int,
        available_at_ns: int,
        priority: EventPriority,
        sequence: int,
        kind: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Event:
        return cls(
            available_at_ns,
            priority,
            event_time_ns,
            sequence,
            event_id,
            kind,
            _freeze_payload(payload or {}),
        )

    @property
    def payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {key: json.loads(value) for key, value in self.payload_items}
        )

    @property
    def causal_key(self) -> CausalKey:
        return (
            self.available_at_ns,
            int(self.priority),
            self.event_time_ns,
            self.sequence,
            self.event_id,
        )


@dataclass(frozen=True)
class ReplayResult:
    processed: tuple[Event, ...]
    trace_hash: str
    status: str
    reason: str = ""
    input_snapshot_hash: str = ""
    terminal_state_hash: str = ""


@dataclass(frozen=True)
class HandlerEvidence:
    dependency_causal_keys: tuple[CausalKey, ...] = ()
    output_hashes: tuple[str, ...] = ()


def event_seed(master_seed: int, event_id: str) -> int:
    payload = f"CH12-EVENT-SEED\0{master_seed}\0{event_id}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big")


class EventLoop:
    def __init__(self, *, master_seed: int = 0) -> None:
        self._queue: list[Event] = []
        self._event_ids: set[str] = set()
        self._ordering_keys: set[tuple[int, int, int, int, str]] = set()
        self._current_key: tuple[int, int, int, int, str] | None = None
        self._closed = False
        self.master_seed = master_seed

    def publish(self, event: Event) -> None:
        if self._closed:
            raise RuntimeError("event loop is one-shot and already closed")
        if event.event_id in self._event_ids:
            raise ValueError(f"duplicate event_id: {event.event_id}")
        if event.causal_key in self._ordering_keys:
            raise ValueError(f"duplicate causal key: {event.causal_key}")
        if self._current_key is not None and event.causal_key < self._current_key:
            raise ValueError("retroactive publication violates causal replay")
        self._event_ids.add(event.event_id)
        self._ordering_keys.add(event.causal_key)
        heappush(self._queue, event)

    def run(
        self,
        handlers: Mapping[
            str,
            Callable[[Event, EventLoop, int], HandlerEvidence | None],
        ],
        *,
        handler_versions: Mapping[str, str],
        input_snapshot_hash: str,
        config_hash: str,
    ) -> ReplayResult:
        if self._closed:
            raise RuntimeError("event loop is one-shot and already closed")
        for digest_name, digest in (
            ("input_snapshot_hash", input_snapshot_hash),
            ("config_hash", config_hash),
        ):
            if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
                raise ValueError(f"{digest_name} must be a SHA-256 digest")
        if set(handlers) != set(handler_versions) or any(
            not value for value in handler_versions.values()
        ):
            raise ValueError("every handler needs one frozen implementation version")
        processed: list[Event] = []
        digest = sha256()
        digest.update(
            json.dumps(
                {
                    "schema_version": "event-replay-v0.10",
                    "master_seed": self.master_seed,
                    "input_snapshot_hash": input_snapshot_hash,
                    "config_hash": config_hash,
                    "handler_versions": dict(sorted(handler_versions.items())),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        while self._queue:
            event = heappop(self._queue)
            if self._current_key is not None and event.causal_key < self._current_key:
                raise RuntimeError("event ordering violated")
            self._current_key = event.causal_key
            processed.append(event)
            digest.update(
                json.dumps(
                    {
                        "key": event.causal_key,
                        "kind": event.kind,
                        "payload": event.payload_items,
                    },
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            )
            handler = handlers.get(event.kind)
            if handler is None:
                self._closed = True
                self._queue.clear()
                self._current_key = None
                return ReplayResult(
                    tuple(processed),
                    digest.hexdigest(),
                    "REJECT",
                    f"unknown event kind: {event.kind}",
                    input_snapshot_hash,
                    "",
                )
            evidence = handler(
                event,
                self,
                event_seed(self.master_seed, event.event_id),
            ) or HandlerEvidence()
            for dependency_key in evidence.dependency_causal_keys:
                assert_no_lookahead(event.causal_key, dependency_key)
            for output_hash in evidence.output_hashes:
                if len(output_hash) != 64 or set(output_hash) - set("0123456789abcdef"):
                    raise ValueError("handler output hash must be SHA-256")
            digest.update(
                json.dumps(
                    {
                        "handler_version": handler_versions[event.kind],
                        "dependencies": evidence.dependency_causal_keys,
                        "outputs": evidence.output_hashes,
                    },
                    separators=(",", ":"),
                ).encode()
            )
        self._current_key = None
        self._closed = True
        terminal = sha256(
            json.dumps(
                [event.event_id for event in processed],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return ReplayResult(
            tuple(processed),
            digest.hexdigest(),
            "PASS",
            "all events handled and dependency availability verified",
            input_snapshot_hash,
            terminal,
        )


def assert_no_lookahead(
    decision_causal_key: CausalKey,
    source_causal_key: CausalKey,
) -> None:
    if source_causal_key >= decision_causal_key:
        raise ValueError(
            "look-ahead: source causal key does not precede decision key"
        )
