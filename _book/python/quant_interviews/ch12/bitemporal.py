from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from typing import Iterable


class RowStatus(str, Enum):
    ACTIVE = "ACTIVE"
    TOMBSTONE = "TOMBSTONE"


class SelectionStatus(str, Enum):
    SELECTED = "SELECTED"
    MISSING = "MISSING"
    WITHDRAWN = "WITHDRAWN"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class BitemporalRow:
    key: str
    value: float | None
    valid_from: datetime
    valid_to: datetime | None
    known_from: datetime
    known_to: datetime | None
    version_id: str
    source_id: str
    published_at: datetime
    received_at: datetime
    eligible_at: datetime
    status: RowStatus = RowStatus.ACTIVE
    source_priority: int = 0
    authoritative_row_id: str = ""

    def __post_init__(self) -> None:
        if not self.key or not self.version_id or not self.source_id:
            raise ValueError("key, version_id and source_id are required")
        for field, value in (
            ("valid_from", self.valid_from),
            ("known_from", self.known_from),
            ("published_at", self.published_at),
            ("received_at", self.received_at),
            ("eligible_at", self.eligible_at),
        ):
            _require_aware_utc(value, field)
        for field, value in (("valid_to", self.valid_to), ("known_to", self.known_to)):
            if value is not None:
                _require_aware_utc(value, field)
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("invalid valid-time interval")
        if self.known_to is not None and self.known_to <= self.known_from:
            raise ValueError("invalid knowledge-time interval")
        if self.received_at < self.published_at:
            raise ValueError("received_at cannot precede published_at")
        if self.eligible_at < self.received_at:
            raise ValueError("eligible_at cannot precede receipt")
        if self.known_from < self.published_at:
            raise ValueError("known_from cannot precede publication")
        if self.source_priority < 0:
            raise ValueError("source_priority must be non-negative")
        if self.status is RowStatus.ACTIVE:
            if self.value is None or not isfinite(self.value):
                raise ValueError("an active row requires a finite value")
        elif self.value is not None:
            raise ValueError("a tombstone must not carry a value")

    @property
    def row_id(self) -> str:
        return self.authoritative_row_id or sha256(
            json.dumps(
                {
                    "schema_version": "bitemporal-row-v0.10",
                    "key": self.key,
                    "value": self.value,
                    "valid_from": self.valid_from.isoformat(),
                    "valid_to": None if self.valid_to is None else self.valid_to.isoformat(),
                    "known_from": self.known_from.isoformat(),
                    "known_to": None if self.known_to is None else self.known_to.isoformat(),
                    "published_at": self.published_at.isoformat(),
                    "received_at": self.received_at.isoformat(),
                    "eligible_at": self.eligible_at.isoformat(),
                    "version_id": self.version_id,
                    "source_id": self.source_id,
                    "source_priority": self.source_priority,
                    "status": self.status.value,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def eligible(self, valid_at: datetime, known_at: datetime) -> bool:
        valid = self.valid_from <= valid_at and (
            self.valid_to is None or valid_at < self.valid_to
        )
        known = self.known_from <= known_at and (
            self.known_to is None or known_at < self.known_to
        )
        return valid and known and self.eligible_at <= known_at


@dataclass(frozen=True)
class BitemporalSelection:
    status: SelectionStatus
    row: BitemporalRow | None
    candidates: tuple[BitemporalRow, ...]
    reason: str
    precedence_policy_version: str = "QJ-BITEMPORAL-PRECEDENCE-1"
    selected_row_hash: str | None = None


@dataclass(frozen=True)
class TemporalRequest:
    row_id: str
    key: str
    valid_at: datetime
    known_at: datetime
    decision_at: datetime | None = None
    snapshot_id: str = "UNSPECIFIED"

    def __post_init__(self) -> None:
        if not self.row_id or not self.key:
            raise ValueError("temporal request identity is required")
        _require_aware_utc(self.valid_at, "valid_at")
        _require_aware_utc(self.known_at, "known_at")
        decision = self.known_at if self.decision_at is None else self.decision_at
        _require_aware_utc(decision, "decision_at")
        if self.known_at > decision:
            raise ValueError("knowledge cutoff cannot follow decision time")


def as_of_value(
    rows: Iterable[BitemporalRow],
    key: str,
    valid_at: datetime,
    known_at: datetime,
) -> BitemporalSelection:
    _require_aware_utc(valid_at, "valid_at")
    _require_aware_utc(known_at, "known_at")
    candidates = tuple(
        row for row in rows if row.key == key and row.eligible(valid_at, known_at)
    )
    if not candidates:
        return BitemporalSelection(
            SelectionStatus.MISSING, None, (), "no eligible historical version"
        )
    top_precedence = max(
        (
            row.source_priority,
            row.valid_from,
            row.known_from,
            row.eligible_at,
        )
        for row in candidates
    )
    top = tuple(
        row
        for row in candidates
        if (
            row.source_priority,
            row.valid_from,
            row.known_from,
            row.eligible_at,
        ) == top_precedence
    )
    semantic_rows = {(row.status, row.value) for row in top}
    if len(semantic_rows) != 1:
        return BitemporalSelection(
            SelectionStatus.AMBIGUOUS,
            None,
            top,
            "equal-precedence versions disagree",
        )
    winner = max(
        top,
        key=lambda row: (row.version_id, row.source_id, row.row_id),
    )
    if winner.status is RowStatus.TOMBSTONE:
        return BitemporalSelection(
            SelectionStatus.WITHDRAWN,
            winner,
            top,
            "the selected historical version is a tombstone",
            selected_row_hash=winner.row_id,
        )
    return BitemporalSelection(
        SelectionStatus.SELECTED,
        winner,
        top,
        "one active version has maximum temporal precedence",
        selected_row_hash=winner.row_id,
    )


def temporal_join(
    requests: Iterable[TemporalRequest],
    rows: Iterable[BitemporalRow],
) -> dict[tuple[str, str], BitemporalSelection]:
    materialized_rows = tuple(rows)
    output: dict[tuple[str, str], BitemporalSelection] = {}
    for request in requests:
        output_key = request.row_id, request.key
        if output_key in output:
            raise ValueError(f"duplicate temporal request: {output_key}")
        output[output_key] = as_of_value(
            materialized_rows,
            request.key,
            request.valid_at,
            request.known_at,
        )
    return output


def _require_aware_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field} must be normalized to UTC")
