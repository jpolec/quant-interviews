from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import re
from typing import Iterable


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


@dataclass(frozen=True)
class TemporalValue:
    """One bitemporal feature revision with source-derived knowledge time."""

    key: str
    valid_from: datetime
    valid_to: datetime | None
    published_at: datetime
    ingested_at: datetime
    value: float
    revision_id: int = 1
    superseded_at: datetime | None = None
    source_priority: int = 0
    payload_hash: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key must be nonempty")
        for field, value in (
            ("valid_from", self.valid_from),
            ("published_at", self.published_at),
            ("ingested_at", self.ingested_at),
        ):
            _require_aware(value, field)
        for field, value in (
            ("valid_to", self.valid_to),
            ("superseded_at", self.superseded_at),
        ):
            if value is not None:
                _require_aware(value, field)
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid interval must be nonempty and half-open")
        if self.published_at > self.ingested_at:
            raise ValueError("publication cannot follow ingestion")
        if self.superseded_at is not None and self.superseded_at < self.known_at:
            raise ValueError("supersession cannot precede effective knowledge")
        if not isinstance(self.revision_id, int) or isinstance(self.revision_id, bool):
            raise ValueError("revision_id must be an integer")
        if not isinstance(self.source_priority, int) or isinstance(
            self.source_priority, bool
        ):
            raise ValueError("source_priority must be an integer")
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")
        if not _SHA256_RE.fullmatch(self.payload_hash):
            raise ValueError("payload_hash must be 64 lowercase hexadecimal digits")

    @property
    def known_at(self) -> datetime:
        """Effective knowledge time; callers cannot forge an earlier timestamp."""

        return max(self.published_at, self.ingested_at)


def _selection_key(row: TemporalValue) -> tuple[datetime, datetime, int, int, str]:
    return (
        row.valid_from,
        row.known_at,
        row.revision_id,
        row.source_priority,
        row.payload_hash,
    )


def select_as_known(
    rows: Iterable[TemporalValue],
    *,
    key: str,
    valid_at: datetime,
    decision_time: datetime,
) -> TemporalValue:
    """Select one record under half-open valid and knowledge-time intervals."""

    _require_aware(valid_at, "valid_at")
    _require_aware(decision_time, "decision_time")
    if decision_time < valid_at:
        raise ValueError("decision_time cannot precede the requested valid_at")
    materialized = tuple(rows)
    eligible = [
        row
        for row in materialized
        if (
            row.key == key
            and row.valid_from <= valid_at
            and (row.valid_to is None or valid_at < row.valid_to)
            and row.known_at <= decision_time
            and (
                row.superseded_at is None
                or decision_time < row.superseded_at
            )
        )
    ]
    if not eligible:
        raise LookupError("no point-in-time value is available")
    selected = max(eligible, key=_selection_key)
    tied = [row for row in eligible if _selection_key(row) == _selection_key(selected)]
    if len(tied) != 1:
        raise ValueError("ambiguous point-in-time tie")
    return selected


def temporal_snapshot_hash(rows: Iterable[TemporalValue]) -> str:
    """Hash a canonically ordered, fully timestamped bitemporal payload."""

    payload = [
        {
            "ingested_at": row.ingested_at.isoformat(),
            "key": row.key,
            "payload_hash": row.payload_hash,
            "published_at": row.published_at.isoformat(),
            "revision_id": row.revision_id,
            "source_priority": row.source_priority,
            "superseded_at": (
                None if row.superseded_at is None else row.superseded_at.isoformat()
            ),
            "valid_from": row.valid_from.isoformat(),
            "valid_to": None if row.valid_to is None else row.valid_to.isoformat(),
            "value": format(row.value, ".17g"),
        }
        for row in sorted(
            tuple(rows),
            key=lambda item: (
                item.key,
                item.valid_from,
                item.known_at,
                item.revision_id,
                item.source_priority,
                item.payload_hash,
            ),
        )
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_predictable(
    feature_known_at: datetime,
    order_decision_time: datetime,
) -> None:
    _require_aware(feature_known_at, "feature_known_at")
    _require_aware(order_decision_time, "order_decision_time")
    if feature_known_at > order_decision_time:
        raise ValueError("feature is anticipative at the decision timestamp")
