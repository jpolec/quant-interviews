from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict
from hashlib import sha256
import json
import math
import re
from typing import Iterable

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class VersionedValue:
    natural_key: str
    valid_time: datetime
    knowledge_time: datetime
    value: float
    version: int
    source_id: str
    row_hash: str

    @property
    def key(self) -> str:
        return self.natural_key

    @property
    def effective_at(self) -> datetime:
        return self.valid_time

    @property
    def known_at(self) -> datetime:
        return self.knowledge_time

    def __post_init__(self) -> None:
        if not self.natural_key:
            raise ValueError("natural_key is required")
        if self.valid_time.tzinfo is None or self.knowledge_time.tzinfo is None:
            raise ValueError("temporal fields must be timezone-aware")
        if self.version < 1:
            raise ValueError("version must be positive")
        if not self.source_id:
            raise ValueError("source_id is required")
        if SHA256_PATTERN.fullmatch(self.row_hash) is None:
            raise ValueError("row_hash must be a lowercase SHA-256 digest")
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")
        expected = self.content_hash_for(
            natural_key=self.natural_key,
            valid_time=self.valid_time,
            knowledge_time=self.knowledge_time,
            value=self.value,
            version=self.version,
            source_id=self.source_id,
        )
        if self.row_hash != expected:
            raise ValueError("row_hash does not bind the canonical temporal row payload")

    @staticmethod
    def content_hash_for(
        *,
        natural_key: str,
        valid_time: datetime,
        knowledge_time: datetime,
        value: float,
        version: int,
        source_id: str,
    ) -> str:
        if valid_time.tzinfo is None or knowledge_time.tzinfo is None:
            raise ValueError("temporal fields must be timezone-aware")
        payload = {
            "knowledge_time": knowledge_time.astimezone(timezone.utc).isoformat(timespec="microseconds"),
            "natural_key": natural_key,
            "schema_version": 1,
            "source_id": source_id,
            "valid_time": valid_time.astimezone(timezone.utc).isoformat(timespec="microseconds"),
            "value_hex": float(value).hex(),
            "version": version,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def point_in_time_lookup(
    rows: Iterable[VersionedValue],
    key: str,
    valid_cutoff: datetime,
    knowledge_cutoff: datetime,
) -> VersionedValue | None:
    if not key:
        raise ValueError("key is required")
    if valid_cutoff.tzinfo is None or knowledge_cutoff.tzinfo is None:
        raise ValueError("cutoffs must be timezone-aware")
    row_list = list(rows)
    grouped: dict[tuple[object, ...], list[VersionedValue]] = defaultdict(list)
    for row in row_list:
        identity = (
            row.natural_key,
            row.valid_time,
            row.knowledge_time,
            row.version,
            row.source_id,
        )
        grouped[identity].append(row)
    for versions in grouped.values():
        if len(versions) <= 1:
            continue
        payloads = {(row.value.hex(), row.row_hash) for row in versions}
        if len(payloads) == 1:
            raise ValueError("exact duplicate temporal row")
        raise ValueError("conflicting payload for complete temporal version key")
    eligible = [
        row
        for row in row_list
        if row.natural_key == key
        and row.valid_time <= valid_cutoff
        and row.knowledge_time <= knowledge_cutoff
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            row.valid_time,
            row.knowledge_time,
            row.version,
            row.source_id,
        ),
    )
