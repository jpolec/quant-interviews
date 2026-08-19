from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SQL_DIR = REPO_ROOT / "_book" / "sql" / "ch09"


def _query(name: str) -> str:
    source = (SQL_DIR / name).read_text(encoding="utf-8")
    assert source.splitlines()[0] == "-- Dialect: SQLite 3.x"
    return source


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE positions_bitemporal(
          portfolio_id TEXT, instrument_id TEXT, canonical_instrument_id TEXT,
          currency TEXT, quantity REAL, market_value REAL, valid_time TEXT,
          knowledge_time TEXT, revision_id INTEGER
        );
        CREATE TABLE scenario_run_positions(
          run_id TEXT, portfolio_id TEXT, instrument_id TEXT, market_value REAL
        );
        CREATE TABLE scenario_results(
          run_id TEXT, portfolio_id TEXT, instrument_id TEXT,
          repricing_status TEXT
        );
        CREATE TABLE pnl_components(
          portfolio_id TEXT, business_date TEXT, base_currency TEXT,
          extract_basis TEXT, valuation_timestamp TEXT, market_pnl REAL,
          trading_pnl REAL, carry_pnl REAL, cash_flow_pnl REAL,
          financing_pnl REAL, fx_translation_pnl REAL, fees_pnl REAL
        );
        CREATE TABLE nav_pnl(
          portfolio_id TEXT, business_date TEXT, base_currency TEXT,
          extract_basis TEXT, valuation_timestamp TEXT, nav_version INTEGER,
          nav REAL, total_pnl REAL
        );
        """
    )
    return connection


def test_point_in_time_selection_rejects_full_ordering_tie() -> None:
    connection = _connection()
    connection.executescript(
        """
        INSERT INTO positions_bitemporal VALUES
          ('P','A','CAN-A','USD',1,100,'2026-07-24T10:00:00',
           '2026-07-24T10:01:00',2),
          ('P','A','CAN-A','USD',2,200,'2026-07-24T10:00:00',
           '2026-07-24T10:01:00',2);
        """
    )
    row = connection.execute(
        _query("point_in_time_portfolio_state.sql"),
        {
            "decision_time": "2026-07-24T10:30:00",
            "knowledge_cutoff": "2026-07-24T10:30:00",
        },
    ).fetchone()
    assert row is not None
    assert row["ordering_tie_count"] == 2
    assert row["identity_currency_status"] == "REJECT"


@pytest.mark.parametrize(
    ("run_id", "market_value", "result_count", "expected_status"),
    (
        ("DUP", 100.0, 2, "REJECT"),
        ("ZERO", 0.0, 1, "INCOMPLETE"),
    ),
)
def test_scenario_coverage_rejects_duplicates_and_zero_denominator(
    run_id: str,
    market_value: float,
    result_count: int,
    expected_status: str,
) -> None:
    connection = _connection()
    connection.execute(
        "INSERT INTO scenario_run_positions VALUES (?,?,?,?)",
        (run_id, "P", "A", market_value),
    )
    connection.executemany(
        "INSERT INTO scenario_results VALUES (?,?,?,?)",
        [(run_id, "P", "A", "SUCCESS")] * result_count,
    )
    row = connection.execute(
        _query("scenario_coverage.sql"),
        {"minimum_coverage": 0.99},
    ).fetchone()
    assert row is not None
    assert row["publication_status"] == expected_status
    if result_count > 1:
        assert row["duplicate_results"] == 1
        assert row["universe_instruments"] == 1


@pytest.mark.parametrize(
    ("portfolio_id", "nav", "total_pnl", "component", "expected_status"),
    (
        ("NULL_NAV", None, 0.0, 0.0, "REJECT"),
        ("ZERO_NAV", 0.0, 0.0, 0.0, "REJECT"),
        ("NEG_NAV", -1.0, 0.0, 0.0, "REJECT"),
        ("NULL_PNL", 100.0, None, 0.0, "REJECT"),
        ("ZERO_GROSS_OK", 100.0, 0.0, 0.0, "PASS"),
        ("ZERO_GROSS_BAD", 100.0, 1.0, 0.0, "REJECT"),
        ("NULL_COMPONENT", 100.0, 0.0, None, "REJECT"),
    ),
)
def test_attribution_is_fail_closed_for_undefined_denominators(
    portfolio_id: str,
    nav: float | None,
    total_pnl: float | None,
    component: float | None,
    expected_status: str,
) -> None:
    connection = _connection()
    grain = (portfolio_id, "2026-07-24", "USD", "OFFICIAL", "2026-07-24T17:00:00")
    connection.execute(
        "INSERT INTO nav_pnl VALUES (?,?,?,?,?,?,?,?)",
        (*grain, 1, nav, total_pnl),
    )
    connection.execute(
        "INSERT INTO pnl_components VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (*grain, component, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    row = connection.execute(
        _query("attribution_reconciliation.sql"),
        {
            "absolute_tolerance": 0.0,
            "nav_tolerance": 0.0,
            "gross_tolerance": 0.0,
        },
    ).fetchone()
    assert row is not None
    assert row["publication_status"] == expected_status
    if portfolio_id == "ZERO_GROSS_OK":
        assert row["residual_to_gross_attribution"] == 0.0
