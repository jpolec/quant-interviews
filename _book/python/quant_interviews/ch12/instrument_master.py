from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from typing import Iterable


class ResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    MISSING = "MISSING"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class IdentityRecord:
    instrument_id: str
    id_type: str
    id_value: str
    valid_from: date
    valid_to: date | None
    known_from: datetime
    known_to: datetime | None
    version_id: str
    source_id: str
    source_priority: int = 0
    override_id: str | None = None
    override_approved_by: str | None = None
    override_expires_at: datetime | None = None
    override_reason: str | None = None
    override_not_before: datetime | None = None
    authoritative_row_id: str | None = None

    def __post_init__(self) -> None:
        text_fields = (
            self.instrument_id,
            self.id_type,
            self.id_value,
            self.version_id,
            self.source_id,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("identity text fields must be non-empty")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid interval must be non-empty and half-open")
        if self.known_to is not None and self.known_to <= self.known_from:
            raise ValueError("knowledge interval must be non-empty and half-open")
        _require_aware_utc(self.known_from, "known_from")
        if self.known_to is not None:
            _require_aware_utc(self.known_to, "known_to")
        if self.source_priority < 0:
            raise ValueError("source_priority must be non-negative")
        if self.authoritative_row_id is not None and not self.authoritative_row_id:
            raise ValueError("authoritative_row_id must be non-empty")
        override_fields = (
            self.override_id,
            self.override_approved_by,
            self.override_expires_at,
            self.override_reason,
            self.override_not_before,
        )
        if any(value is not None for value in override_fields):
            if any(value is None for value in override_fields):
                raise ValueError("an override requires full scope and validity")
            assert self.override_expires_at is not None
            assert self.override_not_before is not None
            _require_aware_utc(self.override_not_before, "override_not_before")
            _require_aware_utc(self.override_expires_at, "override_expires_at")
            if self.override_expires_at <= self.override_not_before:
                raise ValueError("override interval must be non-empty")

    def eligible(self, valid_at: date, known_at: datetime) -> bool:
        valid = self.valid_from <= valid_at and (
            self.valid_to is None or valid_at < self.valid_to
        )
        known = self.known_from <= known_at and (
            self.known_to is None or known_at < self.known_to
        )
        _require_aware_utc(known_at, "known_at")
        override_live = self.override_id is None or (
            self.override_not_before is not None
            and self.override_expires_at is not None
            and self.override_not_before <= known_at < self.override_expires_at
        )
        return valid and known and override_live

    @property
    def precedence(self) -> tuple[int, date, datetime]:
        return self.source_priority, self.valid_from, self.known_from

    @property
    def lineage_hash(self) -> str:
        body = {
            "schema_version": "identity-record-v0.10",
            "instrument_id": self.instrument_id,
            "id_type": self.id_type,
            "id_value": self.id_value,
            "valid_from": self.valid_from.isoformat(),
            "valid_to": None if self.valid_to is None else self.valid_to.isoformat(),
            "known_from": self.known_from.astimezone(timezone.utc).isoformat(),
            "known_to": None
            if self.known_to is None
            else self.known_to.astimezone(timezone.utc).isoformat(),
            "version_id": self.version_id,
            "source_id": self.source_id,
            "source_priority": self.source_priority,
            "override_id": self.override_id,
            "override_approved_by": self.override_approved_by,
            "override_not_before": None
            if self.override_not_before is None
            else self.override_not_before.astimezone(timezone.utc).isoformat(),
            "override_expires_at": None
            if self.override_expires_at is None
            else self.override_expires_at.astimezone(timezone.utc).isoformat(),
            "override_reason": self.override_reason,
            "authoritative_row_id": self.authoritative_row_id,
        }
        return sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class IdentityResolution:
    status: ResolutionStatus
    instrument_id: str | None
    mapping_version: str | None
    source_id: str | None
    candidates: tuple[IdentityRecord, ...]
    reason: str


@dataclass(frozen=True)
class IdentityViolation:
    id_type: str
    id_value: str
    left_version: str
    right_version: str
    code: str


def resolve_identifier(
    records: Iterable[IdentityRecord],
    id_type: str,
    id_value: str,
    valid_at: date,
    known_at: datetime,
) -> IdentityResolution:
    _require_aware_utc(known_at, "known_at")
    candidates = tuple(
        record
        for record in records
        if record.id_type == id_type
        and record.id_value == id_value
        and record.eligible(valid_at, known_at)
    )
    if not candidates:
        return IdentityResolution(
            ResolutionStatus.MISSING,
            None,
            None,
            None,
            (),
            "no eligible mapping at the requested valid/knowledge time",
        )
    top_precedence = max(record.precedence for record in candidates)
    top = tuple(record for record in candidates if record.precedence == top_precedence)
    semantic_instruments = {record.instrument_id for record in top}
    if len(semantic_instruments) != 1:
        return IdentityResolution(
            ResolutionStatus.AMBIGUOUS,
            None,
            None,
            None,
            top,
            "equal-precedence mappings disagree",
        )
    winner = max(
        top,
        key=lambda record: (
            record.version_id,
            record.source_id,
            record.authoritative_row_id or record.lineage_hash,
        ),
    )
    return IdentityResolution(
        ResolutionStatus.RESOLVED,
        winner.instrument_id,
        winner.version_id,
        winner.source_id,
        top,
        "one semantic mapping has maximum precedence",
    )


def _overlaps(
    left_start: date | datetime,
    left_end: date | datetime | None,
    right_start: date | datetime,
    right_end: date | datetime | None,
) -> bool:
    return (
        right_end is None or left_start < right_end
    ) and (
        left_end is None or right_start < left_end
    )


def _require_aware_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field} must be normalized to UTC")


def validate_no_overlap(
    records: Iterable[IdentityRecord],
) -> tuple[IdentityViolation, ...]:
    grouped: dict[tuple[str, str], list[IdentityRecord]] = {}
    for record in records:
        grouped.setdefault((record.id_type, record.id_value), []).append(record)
    violations: list[IdentityViolation] = []
    for (id_type, id_value), group in grouped.items():
        ordered = sorted(
            group,
            key=lambda record: (
                record.valid_from,
                record.known_from,
                record.version_id,
            ),
        )
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if not _overlaps(
                    left.valid_from,
                    left.valid_to,
                    right.valid_from,
                    right.valid_to,
                ):
                    continue
                if not _overlaps(
                    left.known_from,
                    left.known_to,
                    right.known_from,
                    right.known_to,
                ):
                    continue
                code = (
                    "ambiguous_mapping"
                    if left.instrument_id != right.instrument_id
                    else "overlapping_versions"
                )
                violations.append(
                    IdentityViolation(
                        id_type,
                        id_value,
                        left.version_id,
                        right.version_id,
                        code,
                    )
                )
    return tuple(violations)
