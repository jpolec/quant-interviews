from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
SURFACE_SQL = (ROOT / "_book/sql/ch06/option_surface_asof.sql").read_text(
    encoding="utf-8"
)
RECONCILIATION_SQL = (
    ROOT / "_book/sql/ch06/hedging_reconciliation.sql"
).read_text(encoding="utf-8")


def _surface_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE implied_vol_versions (
            underlying_id TEXT,
            expiry TEXT,
            strike REAL,
            option_type TEXT,
            quote_convention TEXT,
            bid_vol REAL,
            ask_vol REAL,
            valid_from TEXT,
            valid_to TEXT,
            known_at TEXT,
            published_at TEXT,
            ingested_at TEXT,
            revision_id INTEGER,
            superseded_at TEXT,
            source_priority INTEGER,
            payload_hash TEXT
        )
        """
    )
    return connection


def _surface_row(
    *,
    strike: float,
    revision: int,
    known: str,
    published: str,
    ingested: str,
    payload_hash: str,
) -> tuple[object, ...]:
    return (
        "SPX",
        "2026-12-18",
        strike,
        "CALL",
        "BLACK",
        0.20 + revision * 0.001,
        0.21 + revision * 0.001,
        "2026-07-01T00:00:00Z",
        "2026-08-01T00:00:00Z",
        known,
        published,
        ingested,
        revision,
        None,
        10,
        payload_hash,
    )


def test_option_surface_snapshot_uses_latest_full_availability_time() -> None:
    connection = _surface_connection()
    rows = [
        _surface_row(
            strike=5000.0,
            revision=1,
            known="2026-07-10T09:00:00Z",
            published="2026-07-10T09:00:01Z",
            ingested="2026-07-10T09:00:02Z",
            payload_hash="hash-a",
        ),
        _surface_row(
            strike=5000.0,
            revision=2,
            known="2026-07-10T09:30:00Z",
            published="2026-07-10T09:30:01Z",
            ingested="2026-07-10T10:30:00Z",
            payload_hash="hash-future-ingest",
        ),
    ]
    connection.executemany(
        "INSERT INTO implied_vol_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    selected = connection.execute(
        SURFACE_SQL,
        {
            "valuation_time": "2026-07-15T00:00:00Z",
            "as_of_knowledge_time": "2026-07-10T10:00:00Z",
        },
    ).fetchall()
    assert len(selected) == 1
    assert selected[0]["revision_id"] == 1
    assert selected[0]["available_at"] == "2026-07-10T09:00:02Z"
    assert selected[0]["snapshot_hash_material"] == "hash-a"
    assert selected[0]["schema_version"] == "ch06-option-surface-v0.10"


def test_option_surface_manifest_hash_material_has_explicit_economic_order() -> None:
    connection = _surface_connection()
    connection.executemany(
        "INSERT INTO implied_vol_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            _surface_row(
                strike=5100.0,
                revision=1,
                known="2026-07-10T09:00:00Z",
                published="2026-07-10T09:00:00Z",
                ingested="2026-07-10T09:00:00Z",
                payload_hash="hash-z",
            ),
            _surface_row(
                strike=4900.0,
                revision=1,
                known="2026-07-10T09:00:00Z",
                published="2026-07-10T09:00:00Z",
                ingested="2026-07-10T09:00:00Z",
                payload_hash="hash-a",
            ),
        ],
    )
    selected = connection.execute(
        SURFACE_SQL,
        {
            "valuation_time": "2026-07-15T00:00:00Z",
            "as_of_knowledge_time": "2026-07-10T10:00:00Z",
        },
    ).fetchall()
    assert len(selected) == 2
    assert {row["snapshot_hash_material"] for row in selected} == {
        "hash-a|hash-z"
    }


def _hedge_connection(events_sql: str) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE hedge_events (
            strategy_id TEXT,
            trade_date TEXT,
            event_id TEXT,
            sequence_number INTEGER,
            option_pnl REAL,
            delta_hedge_pnl REAL,
            financing_pnl REAL,
            dividend_pnl REAL,
            execution_cost REAL,
            accounting_pnl REAL,
            payload_hash TEXT
        );
        """
        + events_sql
    )
    return connection


def _status(connection: sqlite3.Connection, tolerance: float | None) -> str:
    return connection.execute(
        RECONCILIATION_SQL,
        {"reconciliation_tolerance": tolerance},
    ).fetchone()["status"]


def test_hedge_reconciliation_accepts_complete_event_and_aggregate_ledger() -> None:
    connection = _hedge_connection(
        """
        INSERT INTO hedge_events VALUES
          ('S1','2026-07-30','e1',1,1.0,-0.6,0.01,-0.02,0.01,0.38,'h1'),
          ('S1','2026-07-30','e2',2,-0.4,0.3,0.01,-0.01,0.02,-0.12,'h2');
        """
    )
    row = connection.execute(
        RECONCILIATION_SQL, {"reconciliation_tolerance": 1e-12}
    ).fetchone()
    assert row["status"] == "PASS"
    assert row["maximum_absolute_event_residual"] == pytest.approx(0.0)
    assert row["residual"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("events_sql", "expected"),
    [
        (
            """INSERT INTO hedge_events VALUES
            ('S','2026-07-30','e1',1,0,0,0,0,0,0,'h1'),
            ('S','2026-07-30','e2',1,0,0,0,0,0,0,'h2');""",
            "REJECT_DUPLICATE_SEQUENCE",
        ),
        (
            """INSERT INTO hedge_events VALUES
            ('S','2026-07-30','e1',1,0,0,0,0,0,0,'h1'),
            ('S','2026-07-30','e1',2,0,0,0,0,0,0,'h2');""",
            "REJECT_DUPLICATE_EVENT",
        ),
        (
            """INSERT INTO hedge_events VALUES
            ('S','2026-07-30','e1',2,0,0,0,0,0,0,'h1');""",
            "REJECT_SEQUENCE_GAP",
        ),
        (
            """INSERT INTO hedge_events VALUES
            ('S','2026-07-30','e1',1,0,NULL,0,0,0,0,'h1');""",
            "REJECT_NULL",
        ),
    ],
)
def test_hedge_reconciliation_rejects_key_and_null_failures(
    events_sql: str, expected: str
) -> None:
    assert _status(_hedge_connection(events_sql), 1e-12) == expected


def test_hedge_reconciliation_cannot_cancel_opposite_event_errors() -> None:
    connection = _hedge_connection(
        """
        INSERT INTO hedge_events VALUES
          ('S','2026-07-30','e1',1,0,0,0,0,0,1,'h1'),
          ('S','2026-07-30','e2',2,0,0,0,0,0,-1,'h2');
        """
    )
    row = connection.execute(
        RECONCILIATION_SQL, {"reconciliation_tolerance": 1e-12}
    ).fetchone()
    assert row["residual"] == pytest.approx(0.0)
    assert row["maximum_absolute_event_residual"] == pytest.approx(1.0)
    assert row["status"] == "REJECT_EVENT_RESIDUAL"


@pytest.mark.parametrize("tolerance", [-1.0, None])
def test_hedge_reconciliation_rejects_invalid_tolerance(
    tolerance: float | None,
) -> None:
    connection = _hedge_connection(
        """INSERT INTO hedge_events VALUES
        ('S','2026-07-30','e1',1,0,0,0,0,0,0,'h1');"""
    )
    assert _status(connection, tolerance) == "REJECT_INVALID_TOLERANCE"
