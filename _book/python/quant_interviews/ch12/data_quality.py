from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from math import isfinite
from typing import Iterable


class ObservationStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    DELISTED = "DELISTED"
    TOMBSTONE = "TOMBSTONE"


class QualityStatus(str, Enum):
    PASS = "PASS"
    CONDITIONAL_PASS = "CONDITIONAL_PASS"
    REJECT = "REJECT"


@dataclass(frozen=True)
class Observation:
    observation_id: str
    instrument_id: str
    field: str
    effective_at: datetime
    known_at: datetime
    version_id: int
    source_id: str
    value: float | None
    status: ObservationStatus = ObservationStatus.ACTIVE

    def __post_init__(self) -> None:
        if (
            not self.observation_id
            or not self.instrument_id
            or not self.field
            or not self.source_id
        ):
            raise ValueError("observation identity is incomplete")
        if self.version_id < 1:
            raise ValueError("version_id must be positive")
        _require_aware_utc(self.effective_at, "effective_at")
        _require_aware_utc(self.known_at, "known_at")


@dataclass(frozen=True)
class UniverseMember:
    instrument_id: str
    market_value: float

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id is required")
        if not isfinite(self.market_value) or self.market_value < 0.0:
            raise ValueError("market_value must be finite and non-negative")


@dataclass(frozen=True)
class QualityIssue:
    observation_id: str
    instrument_id: str
    code: str
    detail: str


@dataclass(frozen=True)
class CoverageCertificate:
    status: QualityStatus
    count_coverage: float
    market_value_coverage: float
    expected_count: int
    observed_count: int
    missing_instruments: tuple[str, ...]
    issue_count: int
    unknown_observed_instruments: tuple[str, ...] = ()


def validate_observations(
    observations: Iterable[Observation],
    *,
    as_of: datetime,
    stale_after: timedelta,
) -> tuple[QualityIssue, ...]:
    _require_aware_utc(as_of, "as_of")
    if stale_after < timedelta(0):
        raise ValueError("stale_after must be non-negative")
    rows = list(observations)
    issues: list[QualityIssue] = []
    full_seen: set[tuple[str, str, datetime, datetime, int, str]] = set()
    versions_by_natural_key: dict[
        tuple[str, str, datetime], list[Observation]
    ] = {}
    for obs in rows:
        full_key = (
            obs.instrument_id,
            obs.field,
            obs.effective_at,
            obs.known_at,
            obs.version_id,
            obs.source_id,
        )
        if full_key in full_seen:
            issues.append(
                QualityIssue(
                    obs.observation_id,
                    obs.instrument_id,
                    "duplicate_version",
                    repr(full_key),
                )
            )
        full_seen.add(full_key)
        versions_by_natural_key.setdefault(
            (obs.instrument_id, obs.field, obs.effective_at), []
        ).append(obs)
    for natural_key, versions in versions_by_natural_key.items():
        eligible = [
            row
            for row in versions
            if row.known_at <= as_of and row.effective_at <= as_of
        ]
        if not eligible:
            for row in versions:
                if row.known_at > as_of:
                    issues.append(
                        QualityIssue(
                            row.observation_id,
                            row.instrument_id,
                            "future_knowledge",
                            row.known_at.isoformat(),
                        )
                    )
                if row.effective_at > as_of:
                    issues.append(
                        QualityIssue(
                            row.observation_id,
                            row.instrument_id,
                            "future_observation",
                            row.effective_at.isoformat(),
                        )
                    )
            continue
        best_key = max(
            (row.known_at, row.version_id) for row in eligible
        )
        tied = [
            row
            for row in eligible
            if (row.known_at, row.version_id) == best_key
        ]
        meanings = {(row.status, row.value) for row in tied}
        if len(meanings) > 1:
            for row in tied:
                issues.append(
                    QualityIssue(
                        row.observation_id,
                        row.instrument_id,
                        "ambiguous_revision",
                        repr(natural_key),
                    )
                )
            continue
        selected = min(tied, key=lambda row: (row.source_id, row.observation_id))
        if (
            selected.status is ObservationStatus.ACTIVE
            and as_of - selected.effective_at > stale_after
        ):
            issues.append(
                QualityIssue(
                    selected.observation_id,
                    selected.instrument_id,
                    "stale",
                    selected.effective_at.isoformat(),
                )
            )
        if selected.status is ObservationStatus.ACTIVE:
            if selected.value is None:
                issues.append(
                    QualityIssue(
                        selected.observation_id,
                        selected.instrument_id,
                        "missing",
                        selected.field,
                    )
                )
            elif not isfinite(selected.value):
                issues.append(
                    QualityIssue(
                        selected.observation_id,
                        selected.instrument_id,
                        "non_finite",
                        repr(selected.value),
                    )
                )
        elif selected.value is not None:
            issues.append(
                QualityIssue(
                    selected.observation_id,
                    selected.instrument_id,
                    "inactive_has_value",
                    selected.status.value,
                )
            )
    return tuple(issues)


def coverage(
    expected: Iterable[UniverseMember],
    observed_instruments: Iterable[str],
    *,
    issue_count: int = 0,
    minimum_count_coverage: float = 0.99,
    minimum_value_coverage: float = 0.995,
) -> CoverageCertificate:
    members = list(expected)
    if not members:
        raise ValueError("expected universe must not be empty")
    if len({member.instrument_id for member in members}) != len(members):
        raise ValueError("expected universe contains duplicate instruments")
    if not 0.0 <= minimum_count_coverage <= 1.0:
        raise ValueError("invalid minimum_count_coverage")
    if not 0.0 <= minimum_value_coverage <= 1.0:
        raise ValueError("invalid minimum_value_coverage")
    if isinstance(issue_count, bool) or issue_count < 0:
        raise ValueError("issue_count must be a non-negative integer")
    observed = set(observed_instruments)
    expected_ids = {member.instrument_id for member in members}
    unknown = observed - expected_ids
    covered = expected_ids & observed
    count_ratio = len(covered) / len(expected_ids)
    total_value = sum(member.market_value for member in members)
    if total_value <= 0.0:
        raise ValueError("expected universe market value must be positive")
    covered_value = sum(
        member.market_value
        for member in members
        if member.instrument_id in covered
    )
    value_ratio = covered_value / total_value
    if (
        count_ratio >= minimum_count_coverage
        and value_ratio >= minimum_value_coverage
        and issue_count == 0
    ):
        status = QualityStatus.PASS
    elif (
        count_ratio >= minimum_count_coverage
        and value_ratio >= minimum_value_coverage
    ):
        status = QualityStatus.CONDITIONAL_PASS
    else:
        status = QualityStatus.REJECT
    return CoverageCertificate(
        status=status,
        count_coverage=count_ratio,
        market_value_coverage=value_ratio,
        expected_count=len(expected_ids),
        observed_count=len(covered),
        missing_instruments=tuple(sorted(expected_ids - observed)),
        issue_count=issue_count,
        unknown_observed_instruments=tuple(sorted(unknown)),
    )


def _require_aware_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field} must be normalized to UTC")
