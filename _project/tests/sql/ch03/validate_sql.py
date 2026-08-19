#!/usr/bin/env python3
"""Execute the CH03 bitemporal calibration-snapshot SQL contract."""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "_book" / "python"))

from quant_interviews.ch03.lineage import (  # noqa: E402
    canonical_date,
    canonical_decimal,
    canonical_quote_payload,
    canonical_snapshot_hash,
    canonical_snapshot_row,
    canonical_timestamp,
    quote_payload_hash,
)
from quant_interviews.ch03.case_study import (  # noqa: E402
    CASE_SNAPSHOT_GOLDEN_SHA256,
    _case_snapshot_rows,
    governed_numerical_case,
)
from quant_interviews.ch03.implied_vol import call_price  # noqa: E402


SQL_PATH = REPO_ROOT / "_book" / "sql" / "ch03" / "calibration_snapshot.sql"
DIALECT_DECLARATION = "-- Dialect: PostgreSQL"
EARLY_FIXTURE_HASH = (
    "24e71abf069893d106c4fb30f7d9bdb88413ebe4aabb7a9db931a38890d9bb40"
)


class _StringAggregate:
    def __init__(self) -> None:
        self._values: list[str] = []
        self._separator = ""

    def step(self, value: object | None, separator: object) -> None:
        if value is not None:
            self._values.append(str(value))
            self._separator = str(separator)

    def finalize(self) -> str | None:
        if not self._values:
            return None
        return self._separator.join(self._values)


def _digest(value: object, algorithm: str) -> bytes:
    if algorithm.lower() != "sha256":
        raise ValueError("fixture supports only SHA-256")
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return hashlib.sha256(data).digest()


def _encode(value: bytes, encoding: str) -> str:
    if encoding.lower() != "hex":
        raise ValueError("fixture supports only hexadecimal encoding")
    return bytes(value).hex()


def _translate(value: str, source: str, replacement: str) -> str:
    mapping: dict[int, str | None] = {}
    for index, character in enumerate(source):
        mapping[ord(character)] = (
            replacement[index] if index < len(replacement) else None
        )
    return value.translate(mapping)


def _timezone(zone: str, value: object | None) -> object | None:
    if zone != "UTC":
        raise ValueError("fixture supports only UTC")
    return value


def _to_char(value: object | None, pattern: str) -> str | None:
    if value is None:
        return None
    if pattern == "YYYY-MM-DD":
        return canonical_date(value)
    if pattern == "FM999999999999999990.00000000":
        return canonical_decimal(value)
    if pattern == 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"':
        return canonical_timestamp(value)
    raise ValueError(f"unsupported PostgreSQL TO_CHAR pattern: {pattern}")


def _c_collation(left: str, right: str) -> int:
    left_bytes = left.encode("utf-8")
    right_bytes = right.encode("utf-8")
    return (left_bytes > right_bytes) - (left_bytes < right_bytes)


def audited_connection() -> sqlite3.Connection:
    """Create the PostgreSQL-compatible SQLite execution fixture."""

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.create_function("DIGEST", 2, _digest, deterministic=True)
    connection.create_function("ENCODE", 2, _encode, deterministic=True)
    connection.create_function("TRANSLATE", 3, _translate, deterministic=True)
    connection.create_function("TIMEZONE", 2, _timezone, deterministic=True)
    connection.create_function("TO_CHAR", 2, _to_char, deterministic=True)
    connection.create_aggregate("STRING_AGG", 2, _StringAggregate)
    connection.create_collation("C", _c_collation)
    connection.execute(
        """
        CREATE TABLE calibration_quotes (
            instrument_id TEXT NOT NULL,
            expiry TEXT NOT NULL,
            strike NUMERIC NOT NULL,
            option_type TEXT NOT NULL,
            quote_convention TEXT NOT NULL,
            bid NUMERIC,
            ask NUMERIC,
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            known_at TEXT NOT NULL,
            superseded_at TEXT,
            revision_id INTEGER NOT NULL,
            quote_payload_hash TEXT
        )
        """
    )
    return connection


def read_sql() -> str:
    text = SQL_PATH.read_text(encoding="utf-8")
    assert text.splitlines()[0] == DIALECT_DECLARATION
    assert ":market_as_of" in text
    assert ":knowledge_as_of" in text
    assert ":calibration_time" not in text
    assert "DIGEST(" in text and "'sha256'" in text
    return text


def insert_quote(
    connection: sqlite3.Connection,
    *,
    instrument_id: str,
    expiry: str,
    strike: str,
    option_type: str = "CALL",
    quote_convention: str = "PRICE_USD",
    bid: str,
    ask: str,
    valid_from: str = "2026-06-01T00:00:00Z",
    valid_to: str | None = "2026-07-01T00:00:00Z",
    known_at: str = "2026-06-01T08:00:00Z",
    superseded_at: str | None = None,
    revision_id: int = 1,
    stored_payload_hash: str | None = "AUTO",
) -> None:
    if stored_payload_hash == "AUTO":
        stored_payload_hash = quote_payload_hash(bid, ask)
    expiry = canonical_date(expiry)
    strike = canonical_decimal(strike)
    bid = canonical_decimal(bid)
    ask = canonical_decimal(ask)
    valid_from = canonical_timestamp(valid_from)
    valid_to = None if valid_to is None else canonical_timestamp(valid_to)
    known_at = canonical_timestamp(known_at)
    superseded_at = (
        None if superseded_at is None else canonical_timestamp(superseded_at)
    )
    connection.execute(
        "INSERT INTO calibration_quotes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            instrument_id,
            expiry,
            strike,
            option_type,
            quote_convention,
            bid,
            ask,
            valid_from,
            valid_to,
            known_at,
            superseded_at,
            revision_id,
            stored_payload_hash,
        ),
    )


def positive_fixture(connection: sqlite3.Connection, *, reverse: bool = False) -> None:
    rows = [
        {
            "instrument_id": "SPX-C-100",
            "expiry": "2026-09-18",
            "strike": "100.0000",
            "bid": "8.100000",
            "ask": "8.300000",
            "known_at": "2026-06-01T08:00:00Z",
            "superseded_at": "2026-06-10T08:00:00Z",
            "revision_id": 1,
        },
        {
            "instrument_id": "SPX-C-100",
            "expiry": "2026-09-18",
            "strike": "100.0000",
            "bid": "8.150000",
            "ask": "8.350000",
            "known_at": "2026-06-10T08:00:00Z",
            "revision_id": 2,
        },
        {
            "instrument_id": "SPX-C-110",
            "expiry": "2026-09-18",
            "strike": "110.0000",
            "bid": "4.200000",
            "ask": "4.400000",
            "known_at": "2026-06-02T08:00:00Z",
            "revision_id": 1,
        },
        {
            "instrument_id": "SPX-C-120-FUTURE-MARKET",
            "expiry": "2026-09-18",
            "strike": "120.0000",
            "bid": "1.100000",
            "ask": "1.300000",
            "valid_from": "2026-07-01T00:00:00Z",
            "valid_to": "2026-08-01T00:00:00Z",
            "known_at": "2026-06-01T08:00:00Z",
            "revision_id": 1,
        },
        {
            "instrument_id": "SPX-C-090-EXPIRED",
            "expiry": "2026-09-18",
            "strike": "90.0000",
            "bid": "14.100000",
            "ask": "14.300000",
            "valid_from": "2026-05-01T00:00:00Z",
            "valid_to": "2026-06-15T00:00:00Z",
            "known_at": "2026-05-01T08:00:00Z",
            "revision_id": 1,
        },
    ]
    for row in reversed(rows) if reverse else rows:
        insert_quote(connection, **row)


def execute_snapshot(
    connection: sqlite3.Connection,
    *,
    market_as_of: str,
    knowledge_as_of: str,
    snapshot_id: str,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        read_sql(),
        {
            "market_as_of": canonical_timestamp(market_as_of),
            "knowledge_as_of": canonical_timestamp(knowledge_as_of),
            "snapshot_id": snapshot_id,
        },
    ).fetchall()
    assert rows
    return [dict(row) for row in rows]


def selected_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["instrument_id"] is not None]


def test_point_in_time_snapshot_separates_market_and_knowledge_axes() -> None:
    connection = audited_connection()
    try:
        positive_fixture(connection)
        early = execute_snapshot(
            connection,
            market_as_of="2026-06-15T00:00:00Z",
            knowledge_as_of="2026-06-05T00:00:00Z",
            snapshot_id="SNAPSHOT-EARLY",
        )
        early_selected = selected_rows(early)
        assert early[0]["snapshot_status"] == "PASS"
        assert early[0]["snapshot_reason"] == "COMPLETE_POINT_IN_TIME_SNAPSHOT"
        assert early[0]["row_count"] == 2
        assert {row["instrument_id"] for row in early_selected} == {
            "SPX-C-100",
            "SPX-C-110",
        }
        revisions = {
            row["instrument_id"]: row["revision_id"] for row in early_selected
        }
        assert revisions["SPX-C-100"] == 1
        assert all(
            row["known_at"] <= early[0]["knowledge_as_of"]
            for row in early_selected
        )
        assert all(
            row["valid_from"] <= early[0]["market_as_of"]
            and (
                row["valid_to"] is None
                or row["valid_to"] > early[0]["market_as_of"]
            )
            for row in early_selected
        )

        revised = execute_snapshot(
            connection,
            market_as_of="2026-06-15T00:00:00Z",
            knowledge_as_of="2026-06-12T00:00:00Z",
            snapshot_id="SNAPSHOT-REVISED",
        )
        revised_selected = selected_rows(revised)
        revised_by_id = {row["instrument_id"]: row for row in revised_selected}
        assert revised_by_id["SPX-C-100"]["revision_id"] == 2
        assert revised[0]["snapshot_hash"] != early[0]["snapshot_hash"]
    finally:
        connection.close()


def test_snapshot_hash_is_canonical_ordered_sha256() -> None:
    hashes: list[str] = []
    for reverse in (False, True):
        connection = audited_connection()
        try:
            positive_fixture(connection, reverse=reverse)
            rows = execute_snapshot(
                connection,
                market_as_of="2026-06-15T00:00:00Z",
                knowledge_as_of="2026-06-05T00:00:00Z",
                snapshot_id="SNAPSHOT-EARLY",
            )
            selected = selected_rows(rows)
            assert rows[0]["hash_algorithm"] == "sha256"
            assert rows[0]["serializer_version"] == "length-prefixed-utf8-v1"
            assert rows[0]["schema_version"] == "calibration_snapshot_v2"
            assert all(row["payload_hash_valid"] == 1 for row in selected)
            assert all(
                row["quote_payload_hash"] == row["computed_quote_payload_hash"]
                for row in selected
            )
            assert all(
                row["canonical_quote_payload"]
                == canonical_quote_payload(row["bid"], row["ask"])
                for row in selected
            )
            assert all(
                row["canonical_row"] == canonical_snapshot_row(row)
                for row in selected
            )
            assert rows[0]["snapshot_hash"] == canonical_snapshot_hash(selected)
            assert rows[0]["snapshot_hash"] == EARLY_FIXTURE_HASH
            hashes.append(rows[0]["snapshot_hash"])
        finally:
            connection.close()
    assert hashes[0] == hashes[1]

    lexical_connection = audited_connection()
    try:
        for strike in ("20", "100"):
            insert_quote(
                lexical_connection,
                instrument_id="LEXICAL-ORDER",
                expiry="2026-09-18",
                strike=strike,
                bid="1.00000000",
                ask="1.20000000",
            )
        lexical_rows = execute_snapshot(
            lexical_connection,
            market_as_of="2026-06-15T00:00:00Z",
            knowledge_as_of="2026-06-05T00:00:00Z",
            snapshot_id="LEXICAL-ORDER",
        )
        lexical_selected = selected_rows(lexical_rows)
        assert [canonical_decimal(row["strike"]) for row in lexical_selected] == [
            "20.00000000",
            "100.00000000",
        ]
        assert lexical_rows[0]["snapshot_hash"] == canonical_snapshot_hash(
            lexical_selected
        )
    finally:
        lexical_connection.close()


def test_snapshot_rejects_empty_duplicate_and_invalid_hash_paths() -> None:
    empty_connection = audited_connection()
    try:
        empty = execute_snapshot(
            empty_connection,
            market_as_of="2026-06-15T00:00:00Z",
            knowledge_as_of="2026-06-05T00:00:00Z",
            snapshot_id="EMPTY",
        )
        assert len(empty) == 1
        assert empty[0]["row_count"] == 0
        assert empty[0]["snapshot_status"] == "REJECT"
        assert empty[0]["snapshot_reason"] == "EMPTY_SNAPSHOT"
        assert empty[0]["snapshot_hash"] == hashlib.sha256(b"").hexdigest()
    finally:
        empty_connection.close()

    duplicate_connection = audited_connection()
    try:
        fixture = {
            "instrument_id": "DUPLICATE",
            "expiry": "2026-09-18",
            "strike": "100.0000",
            "bid": "1.000000",
            "ask": "1.200000",
        }
        insert_quote(duplicate_connection, **fixture)
        insert_quote(duplicate_connection, **fixture)
        duplicate = execute_snapshot(
            duplicate_connection,
            market_as_of="2026-06-15T00:00:00Z",
            knowledge_as_of="2026-06-05T00:00:00Z",
            snapshot_id="DUPLICATE",
        )
        assert duplicate[0]["duplicate_version_group_count"] == 1
        assert duplicate[0]["snapshot_status"] == "REJECT"
        assert duplicate[0]["snapshot_reason"] == "DUPLICATE_ELIGIBLE_VERSION"
    finally:
        duplicate_connection.close()

    overlap_connection = audited_connection()
    try:
        insert_quote(
            overlap_connection,
            instrument_id="OVERLAPPING-REVISIONS",
            expiry="2026-09-18",
            strike="100.0000",
            bid="1.000000",
            ask="1.200000",
            known_at="2026-06-01T08:00:00Z",
            superseded_at=None,
            revision_id=2,
        )
        insert_quote(
            overlap_connection,
            instrument_id="OVERLAPPING-REVISIONS",
            expiry="2026-09-18",
            strike="100.0000",
            bid="1.100000",
            ask="1.300000",
            known_at="2026-06-02T08:00:00Z",
            superseded_at=None,
            revision_id=10,
        )
        overlap = execute_snapshot(
            overlap_connection,
            market_as_of="2026-06-15T00:00:00Z",
            knowledge_as_of="2026-06-05T00:00:00Z",
            snapshot_id="OVERLAPPING-REVISIONS",
        )
        assert overlap[0]["duplicate_version_group_count"] == 1
        assert overlap[0]["snapshot_status"] == "REJECT"
        assert overlap[0]["snapshot_reason"] == "DUPLICATE_ELIGIBLE_VERSION"
        overlap_selected = selected_rows(overlap)
        assert len(overlap_selected) == 1
        assert overlap_selected[0]["revision_id"] == 10
        assert overlap[0]["snapshot_hash"] == canonical_snapshot_hash(
            overlap_selected
        )
    finally:
        overlap_connection.close()

    bad_hashes = (None, "not-a-sha256", "0" * 64)
    for index, bad_hash in enumerate(bad_hashes):
        connection = audited_connection()
        try:
            insert_quote(
                connection,
                instrument_id=f"BAD-HASH-{index}",
                expiry="2026-09-18",
                strike="100.0000",
                bid="1.000000",
                ask="1.200000",
                stored_payload_hash=bad_hash,
            )
            rejected = execute_snapshot(
                connection,
                market_as_of="2026-06-15T00:00:00Z",
                knowledge_as_of="2026-06-05T00:00:00Z",
                snapshot_id=f"BAD-HASH-{index}",
            )
            assert rejected[0]["invalid_payload_hash_count"] == 1
            assert rejected[0]["snapshot_status"] == "REJECT"
            assert rejected[0]["snapshot_reason"] == (
                "MISSING_INVALID_OR_MISMATCHED_PAYLOAD_HASH"
            )
        finally:
            connection.close()


def test_governed_case_sql_python_hash_parity() -> None:
    """Execute the exact 24-row governed case through the SQL serializer."""

    spot = 100.0
    rate = 0.03
    dividend = 0.01
    strikes = np.tile(np.array([80.0, 90.0, 100.0, 110.0]), 6)
    maturities = np.repeat(np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0]), 4)
    coordinates = np.column_stack(((strikes / spot) - 1.0, maturities - 1.0))
    parameters = np.array([0.22, -0.04, 0.015])
    volatilities = (
        parameters[0]
        + parameters[1] * coordinates[:, 0]
        + parameters[2] * coordinates[:, 1]
    )
    prices = np.array(
        [
            call_price(
                spot,
                float(strike),
                rate,
                dividend,
                float(volatility),
                float(maturity),
            )
            for strike, volatility, maturity in zip(
                strikes,
                volatilities,
                maturities,
                strict=True,
            )
        ]
    )
    prices[0] = -1.0
    prices[-1] = spot * np.exp(-dividend * maturities[-1]) + 1.0
    case_rows = _case_snapshot_rows(strikes, maturities, prices)

    connection = audited_connection()
    try:
        for row in case_rows:
            insert_quote(
                connection,
                instrument_id=str(row["instrument_id"]),
                expiry=str(row["expiry"]),
                strike=str(row["strike"]),
                option_type=str(row["option_type"]),
                quote_convention=str(row["quote_convention"]),
                bid=str(row["bid"]),
                ask=str(row["ask"]),
                valid_from=str(row["valid_from"]),
                valid_to=str(row["valid_to"]),
                known_at=str(row["known_at"]),
                superseded_at=None,
                revision_id=int(row["revision_id"]),
                stored_payload_hash=str(row["quote_payload_hash"]),
            )
        sql_rows = execute_snapshot(
            connection,
            market_as_of="2026-06-15T00:00:00Z",
            knowledge_as_of="2026-06-15T00:00:00Z",
            snapshot_id="CH03-CASE-V0.11",
        )
        sql_hash = str(sql_rows[0]["snapshot_hash"])
        assert sql_rows[0]["snapshot_status"] == "PASS"
        assert sql_rows[0]["row_count"] == 24
        assert sql_hash == CASE_SNAPSHOT_GOLDEN_SHA256
        assert sql_hash == canonical_snapshot_hash(selected_rows(sql_rows))

        case = governed_numerical_case(snapshot_hash_from_query=sql_hash)
        hash_gate = next(
            gate for gate in case.release_gates if gate.name == "SNAPSHOT_HASH_MATCH"
        )
        assert hash_gate.passed
        assert case.snapshot_hash_matches_artifact
        assert case.decision == "CONDITIONAL PASS"
    finally:
        connection.close()


def main() -> int:
    test_point_in_time_snapshot_separates_market_and_knowledge_axes()
    test_snapshot_hash_is_canonical_ordered_sha256()
    test_snapshot_rejects_empty_duplicate_and_invalid_hash_paths()
    test_governed_case_sql_python_hash_parity()
    print(
        "PASS dialect=PostgreSQL fixture_engine=SQLite "
        f"fixture_hash={EARLY_FIXTURE_HASH} "
        "pit_future_revision=EXCLUDED empty=REJECT duplicate=REJECT "
        "payload_hash_paths=REJECT governed_case_rows=24 hash_parity=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
