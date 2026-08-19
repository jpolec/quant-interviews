#!/usr/bin/env python3
"""Execute deterministic adversarial checks for both CH07 SQLite contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SQL_ROOT = ROOT / "_book" / "sql" / "ch07"
ASOF_SQL = (SQL_ROOT / "fixed_income_asof_market_data.sql").read_text(
    encoding="utf-8"
)
PNL_SQL = (SQL_ROOT / "fixed_income_pnl_reconciliation.sql").read_text(
    encoding="utf-8"
)
CUTOFFS = {
    "valuation_time": "2026-07-30 08:30",
    "knowledge_cutoff": "2026-07-30 08:30",
}
PNL_LIMITS = {
    "absolute_residual_limit": 0.01,
    "relative_residual_limit": 0.001,
    "pnl_floor": 1.0,
}


def asof_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE market_quote_history (
            instrument_id TEXT, curve_id TEXT, tenor TEXT, quote_type TEXT,
            quote_value REAL, valid_from TEXT, valid_to TEXT,
            known_from TEXT, known_to TEXT, revision_id INTEGER,
            source_priority INTEGER, is_cancelled INTEGER, row_hash TEXT
        );
        CREATE TABLE fixing_history (
            index_id TEXT, fixing_date TEXT, fixing_value REAL,
            valid_from TEXT, valid_to TEXT, known_from TEXT, known_to TEXT,
            revision_id INTEGER, source_priority INTEGER, is_cancelled INTEGER,
            row_hash TEXT
        );
        CREATE TABLE expected_market_data (
            record_type TEXT, economic_key TEXT
        );
        """
    )
    return connection


def insert_quote(
    connection: sqlite3.Connection,
    *,
    value: float,
    known_from: str = "2026-07-30 08:00",
    known_to: str | None = None,
    revision_id: int = 1,
    cancelled: int = 0,
    row_hash: str = "a" * 64,
) -> None:
    connection.execute(
        "INSERT INTO market_quote_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "IRS5Y", "OIS", "5Y", "PAR", value,
            "2026-07-30 08:00", None, known_from, known_to,
            revision_id, 10, cancelled, row_hash,
        ),
    )


def asof_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(ASOF_SQL, CUTOFFS).fetchall()


def validate_asof() -> None:
    baseline = asof_connection()
    baseline.executemany(
        "INSERT INTO expected_market_data VALUES (?,?)",
        [
            ("QUOTE", "IRS5Y|OIS|5Y|PAR"),
            ("FIXING", "SOFR|2026-07-29"),
        ],
    )
    insert_quote(
        baseline,
        value=0.0400,
        known_to="2026-07-30 09:00",
        row_hash="a" * 64,
    )
    insert_quote(
        baseline,
        value=0.0410,
        known_from="2026-07-30 09:00",
        revision_id=2,
        row_hash="b" * 64,
    )
    baseline.execute(
        "INSERT INTO fixing_history VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "SOFR", "2026-07-29", 0.0435,
            "2026-07-29 00:00", None,
            "2026-07-30 07:30", None, 1, 10, 0, "c" * 64,
        ),
    )
    rows = asof_rows(baseline)
    assert len(rows) == 2
    assert {row["selection_status"] for row in rows} == {"PASS"}
    quote = next(row for row in rows if row["record_type"] == "QUOTE")
    assert quote["numeric_value"] == 0.0400

    tie_results: list[tuple[float, str]] = []
    for rows_to_insert in (
        ((0.01, "a" * 64), (0.02, "b" * 64)),
        ((0.02, "b" * 64), (0.01, "a" * 64)),
    ):
        tied = asof_connection()
        tied.execute(
            "INSERT INTO expected_market_data VALUES (?,?)",
            ("QUOTE", "IRS5Y|OIS|5Y|PAR"),
        )
        for value, row_hash in rows_to_insert:
            insert_quote(tied, value=value, row_hash=row_hash)
        row = asof_rows(tied)[0]
        assert row["selection_status"] == "REJECT_TOP_RANK_TIE"
        assert row["top_rank_tie_count"] == 2
        tie_results.append(
            (row["numeric_value"], row["snapshot_manifest_payload"])
        )
    assert tie_results[0] == tie_results[1]

    cancelled = asof_connection()
    cancelled.execute(
        "INSERT INTO expected_market_data VALUES (?,?)",
        ("QUOTE", "IRS5Y|OIS|5Y|PAR"),
    )
    insert_quote(
        cancelled,
        value=0.0400,
        known_from="2026-07-30 07:00",
        revision_id=1,
        row_hash="a" * 64,
    )
    insert_quote(
        cancelled,
        value=0.0410,
        known_from="2026-07-30 08:00",
        revision_id=2,
        cancelled=1,
        row_hash="b" * 64,
    )
    cancelled_row = asof_rows(cancelled)[0]
    assert cancelled_row["selection_status"] == "REJECT_CANCELLED_WINNER"
    assert cancelled_row["is_cancelled"] == 1

    missing = asof_connection()
    missing.execute(
        "INSERT INTO expected_market_data VALUES (?,?)",
        ("FIXING", "SOFR|2026-07-29"),
    )
    assert asof_rows(missing)[0]["selection_status"] == "REJECT_MISSING_RECORD"

    invalid_hash = asof_connection()
    invalid_hash.execute(
        "INSERT INTO expected_market_data VALUES (?,?)",
        ("QUOTE", "IRS5Y|OIS|5Y|PAR"),
    )
    insert_quote(invalid_hash, value=0.04, row_hash="not-a-sha256")
    assert (
        asof_rows(invalid_hash)[0]["selection_status"]
        == "REJECT_INVALID_ROW_HASH"
    )

    unexpected = asof_connection()
    insert_quote(unexpected, value=0.04)
    assert asof_rows(unexpected)[0]["selection_status"] == "REJECT_UNEXPECTED_KEY"


def pnl_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE position_pnl (
            valuation_date TEXT, book_id TEXT, instrument_id TEXT,
            state_id TEXT, position_run_id TEXT, risk_run_id TEXT,
            market_run_id TEXT, actual_pnl REAL, carry_pnl REAL,
            roll_down_pnl REAL, basis_pnl REAL, cs01 REAL,
            credit_spread_move_bp REAL, credit_convexity REAL,
            volatility_pnl REAL, cash_and_fixings_pnl REAL,
            exercise_pnl REAL, model_change_pnl REAL,
            data_correction_pnl REAL, expected_curve_bucket_count INTEGER,
            expected_gamma_pair_count INTEGER, stale_input_count INTEGER
        );
        CREATE TABLE curve_risk_bucket (
            valuation_date TEXT, book_id TEXT, instrument_id TEXT,
            state_id TEXT, risk_run_id TEXT, curve_id TEXT,
            key_rate_tenor TEXT, dv01 REAL
        );
        CREATE TABLE curve_market_move (
            valuation_date TEXT, state_id TEXT, market_run_id TEXT,
            curve_id TEXT, key_rate_tenor TEXT, curve_move_bp REAL
        );
        CREATE TABLE curve_cross_gamma (
            valuation_date TEXT, book_id TEXT, instrument_id TEXT,
            state_id TEXT, risk_run_id TEXT,
            curve_id_1 TEXT, key_rate_tenor_1 TEXT,
            curve_id_2 TEXT, key_rate_tenor_2 TEXT, cross_gamma REAL
        );
        """
    )
    position = (
        "2026-07-30", "BOOK1", "BOND1", "state-1",
        "pos-1", "risk-1", "market-1", 132.60, 100.0, 20.0,
        5.0, 2.0, 3.0, 0.1, 1.0, 2.0, 0.0, 0.0, 0.0, 2, 2, 0,
    )
    connection.execute(
        "INSERT INTO position_pnl VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        position,
    )
    connection.executemany(
        "INSERT INTO curve_risk_bucket VALUES (?,?,?,?,?,?,?,?)",
        [
            ("2026-07-30", "BOOK1", "BOND1", "state-1", "risk-1", "OIS", "1Y", 10.0),
            ("2026-07-30", "BOOK1", "BOND1", "state-1", "risk-1", "OIS", "2Y", 20.0),
        ],
    )
    connection.executemany(
        "INSERT INTO curve_market_move VALUES (?,?,?,?,?,?)",
        [
            ("2026-07-30", "state-1", "market-1", "OIS", "1Y", 1.0),
            ("2026-07-30", "state-1", "market-1", "OIS", "2Y", -1.0),
        ],
    )
    connection.executemany(
        "INSERT INTO curve_cross_gamma VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("2026-07-30", "BOOK1", "BOND1", "state-1", "risk-1", "OIS", "1Y", "OIS", "1Y", 0.2),
            ("2026-07-30", "BOOK1", "BOND1", "state-1", "risk-1", "OIS", "2Y", "OIS", "2Y", 0.1),
        ],
    )
    return connection


def pnl_rows(
    connection: sqlite3.Connection,
    limits: dict[str, float] | None = None,
) -> list[sqlite3.Row]:
    return connection.execute(PNL_SQL, limits or PNL_LIMITS).fetchall()


def validate_pnl() -> None:
    baseline = pnl_connection()
    row = pnl_rows(baseline)[0]
    assert row["reconciliation_status"] == "PASS"
    assert abs(row["explained_pnl"] - 132.60) <= 1.0e-12
    assert abs(row["residual_pnl"]) <= 1.0e-12

    duplicate_gamma = pnl_connection()
    duplicate_gamma.execute(
        "INSERT INTO curve_cross_gamma VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("2026-07-30", "BOOK1", "BOND1", "state-1", "risk-1", "OIS", "1Y", "OIS", "1Y", 0.2),
    )
    row = pnl_rows(duplicate_gamma)[0]
    assert row["reconciliation_status"] == "REJECT"
    assert row["duplicate_evidence_count"] > 0

    null_scalar = pnl_connection()
    null_scalar.execute("UPDATE position_pnl SET carry_pnl = NULL")
    row = pnl_rows(null_scalar)[0]
    assert row["reconciliation_status"] == "REJECT"
    assert row["missing_scalar_count"] > 0

    nonfinite_scalar = pnl_connection()
    nonfinite_scalar.execute("UPDATE position_pnl SET carry_pnl = ?", (float("inf"),))
    assert pnl_rows(nonfinite_scalar)[0]["reconciliation_status"] == "REJECT"

    residual_breach = pnl_connection()
    residual_breach.execute("UPDATE position_pnl SET actual_pnl = 150.0")
    assert pnl_rows(residual_breach)[0]["reconciliation_status"] == "REJECT"

    run_mismatch = pnl_connection()
    run_mismatch.execute("UPDATE curve_risk_bucket SET risk_run_id = 'risk-2'")
    run_mismatch.execute("UPDATE curve_cross_gamma SET risk_run_id = 'risk-2'")
    row = pnl_rows(run_mismatch)[0]
    assert row["reconciliation_status"] == "REJECT"
    assert row["misaligned_evidence_count"] > 0

    duplicate_move = pnl_connection()
    duplicate_move.execute(
        "INSERT INTO curve_market_move VALUES (?,?,?,?,?,?)",
        ("2026-07-30", "state-1", "market-1", "OIS", "1Y", 1.0),
    )
    assert pnl_rows(duplicate_move)[0]["reconciliation_status"] == "REJECT"

    missing_gamma = pnl_connection()
    missing_gamma.execute(
        "UPDATE position_pnl SET expected_gamma_pair_count = 3"
    )
    assert pnl_rows(missing_gamma)[0]["reconciliation_status"] == "REJECT"

    duplicate_position = pnl_connection()
    duplicate_position.execute(
        "INSERT INTO position_pnl SELECT * FROM position_pnl LIMIT 1"
    )
    rows = pnl_rows(duplicate_position)
    assert rows and all(row["reconciliation_status"] == "REJECT" for row in rows)

    invalid_policy = pnl_connection()
    invalid_limits = dict(PNL_LIMITS)
    invalid_limits["absolute_residual_limit"] = -1.0
    assert pnl_rows(invalid_policy, invalid_limits)[0]["reconciliation_status"] == "REJECT"


def main() -> int:
    assert ASOF_SQL.startswith("-- Dialect: SQLite 3.35+")
    assert PNL_SQL.startswith("-- Dialect: SQLite 3.35+")
    validate_asof()
    validate_pnl()
    print(
        "PASS dialect=SQLite3 asof_future=EXCLUDED tie=REJECT "
        "cancelled=REJECT missing=REJECT row_hash=REJECT "
        "pnl_null=REJECT gamma_duplicate=REJECT run_mismatch=REJECT "
        "residual_breach=REJECT"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
