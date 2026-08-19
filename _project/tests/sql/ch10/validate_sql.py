from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
SQL = REPO_ROOT / "_book" / "sql" / "ch10"
DEFAULT_OUTPUT = REPO_ROOT / "_project" / "reports" / "migration" / "ch10"


def _row(connection: sqlite3.Connection, filename: str, parameters: dict[str, Any]) -> dict[str, Any]:
    query = (SQL / filename).read_text(encoding="utf-8")
    rows = connection.execute(query, parameters).fetchall()
    assert len(rows) == 1, (filename, len(rows))
    return dict(rows[0])


def quote_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE quote_history(
            symbol TEXT, venue TEXT, bid_price REAL, bid_size REAL,
            ask_price REAL, ask_size REAL, valid_time INTEGER,
            knowledge_time INTEGER, revision_id INTEGER,
            authoritative_sequence INTEGER, is_cancelled INTEGER,
            is_eligible INTEGER
        );
        """
    )
    connection.executemany(
        "INSERT INTO quote_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("XYZ", "A", 100.00, 110, 100.05, 130, 990, 995, 1, 10, 0, 1),
            ("XYZ", "A", 100.00, 0, 100.05, 0, 991, 999, 2, 11, 1, 1),
            ("XYZ", "B", 99.95, 200, 100.08, 150, 998, 999, 1, 20, 0, 1),
            ("XYZ", "C", 100.20, 500, 100.30, 500, 997, 998, 1, 30, 0, 0),
            ("XYZ", "D", 99.99, 100, 100.02, 100, 900, 900, 1, 40, 0, 1),
        ],
    )
    return connection


def run_nbbo() -> dict[str, Any]:
    parameters = {
        "symbol": "XYZ",
        "decision_time": 1000,
        "knowledge_cutoff": 1000,
        "max_quote_age": 20,
    }
    connection = quote_connection()
    passed = _row(connection, "point_in_time_nbbo.sql", parameters)
    assert passed["nbbo_bid"] == 99.95
    assert passed["nbbo_ask"] == 100.08
    assert passed["nbbo_bid_size"] == 200
    assert passed["eligible_venue_count"] == 1
    assert passed["status"] == "PASS"
    repeated = _row(connection, "point_in_time_nbbo.sql", parameters)
    assert repeated == passed
    connection.execute(
        "INSERT INTO quote_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("XYZ", "B", 99.97, 210, 100.06, 160, 1005, 1005, 2, 21, 0, 1),
    )
    historical_after_revision = _row(connection, "point_in_time_nbbo.sql", parameters)
    assert historical_after_revision == passed
    current = _row(
        connection,
        "point_in_time_nbbo.sql",
        {**parameters, "decision_time": 1010, "knowledge_cutoff": 1010},
    )
    assert current["nbbo_bid"] == 99.97
    assert current["nbbo_ask"] == 100.06
    assert current["status"] == "PASS"

    missing = _row(
        connection,
        "point_in_time_nbbo.sql",
        {**parameters, "symbol": "MISSING"},
    )
    assert missing["status"] == "REJECT_MISSING_SIDE"

    invalid_parameter = _row(
        connection,
        "point_in_time_nbbo.sql",
        {**parameters, "max_quote_age": -1},
    )
    assert invalid_parameter["status"] == "REJECT_INVALID_PARAMETER"

    ambiguous_connection = quote_connection()
    ambiguous_connection.executemany(
        "INSERT INTO quote_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("XYZ", "B", 99.96, 200, 100.08, 150, 998, 999, 1, 20, 0, 1),
        ],
    )
    ambiguous = _row(ambiguous_connection, "point_in_time_nbbo.sql", parameters)
    assert ambiguous["status"] == "REJECT_AMBIGUOUS_RECENCY"

    invalid_connection = quote_connection()
    invalid_connection.execute(
        "INSERT INTO quote_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("XYZ", "E", 101.0, -1, 100.0, 10, 999, 999, 1, 50, 0, 1),
    )
    invalid = _row(invalid_connection, "point_in_time_nbbo.sql", parameters)
    assert invalid["status"] == "REJECT_INVALID_QUOTE"
    return {
        "pass": passed,
        "repeated": repeated,
        "historical_after_revision": historical_after_revision,
        "current_after_revision": current,
        "missing": missing,
        "invalid_parameter": invalid_parameter,
        "ambiguous": ambiguous,
        "invalid_quote": invalid,
    }


def benchmark_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE parent_orders(
            order_id TEXT, symbol TEXT, side TEXT, target_quantity REAL,
            decision_price REAL, arrival_price REAL,
            decision_time INTEGER, arrival_time INTEGER
        );
        CREATE TABLE child_fills(
            fill_id TEXT, order_id TEXT, fill_quantity REAL, fill_price REAL,
            fill_time INTEGER, knowledge_time INTEGER, revision_id INTEGER,
            is_cancelled INTEGER
        );
        CREATE TABLE market_marks(
            symbol TEXT, market_price REAL, valid_time INTEGER,
            knowledge_time INTEGER, revision_id INTEGER
        );
        """
    )
    connection.execute(
        "INSERT INTO parent_orders VALUES (?,?,?,?,?,?,?,?)",
        ("O1", "XYZ", "BUY", 1000, 100.0, 100.1, 900, 910),
    )
    connection.executemany(
        "INSERT INTO child_fills VALUES (?,?,?,?,?,?,?,?)",
        [
            ("F1", "O1", 400, 100.15, 920, 920, 1, 0),
            ("F2", "O1", 400, 100.20, 930, 930, 1, 0),
            ("F2", "O1", 400, 99.00, 930, 1100, 2, 0),
            ("F3", "O1", 50, 100.25, 1010, 940, 1, 0),
        ],
    )
    connection.executemany(
        "INSERT INTO market_marks VALUES (?,?,?,?,?)",
        [
            ("XYZ", 100.40, 990, 995, 1),
            ("XYZ", 100.70, 1000, 1010, 2),
        ],
    )
    return connection


def benchmark_row(connection: sqlite3.Connection, order_id: str = "O1") -> dict[str, Any]:
    return _row(
        connection,
        "execution_benchmark_reconciliation.sql",
        {"order_id": order_id, "knowledge_cutoff": 1000, "end_time": 1000},
    )


def run_shortfall() -> dict[str, Any]:
    connection = benchmark_connection()
    passed = benchmark_row(connection)
    assert passed["executed_quantity"] == 800
    assert passed["average_fill_price"] == 100.175
    assert abs(passed["delay_cost"] - 100.0) < 1e-9
    assert abs(passed["trading_cost"] - 60.0) < 1e-8
    assert abs(passed["opportunity_cost"] - 60.0) < 1e-8
    assert abs(passed["implementation_shortfall"] - 220.0) < 1e-8
    assert passed["status"] == "PASS"

    zero = benchmark_connection()
    zero.execute(
        "INSERT INTO parent_orders VALUES (?,?,?,?,?,?,?,?)",
        ("O2", "ABC", "SELL", 100, 50.0, 49.5, 900, 910),
    )
    zero.execute(
        "INSERT INTO market_marks VALUES (?,?,?,?,?)",
        ("ABC", 49.0, 990, 995, 1),
    )
    zero_fill = benchmark_row(zero, "O2")
    assert zero_fill["executed_quantity"] == 0
    assert zero_fill["average_fill_price"] is None
    assert zero_fill["trading_cost"] == 0.0
    assert zero_fill["status"] == "PASS"

    missing = benchmark_row(connection, "MISSING")
    assert missing["status"] == "REJECT_MISSING_ORDER"

    invalid = benchmark_connection()
    invalid.execute(
        "INSERT INTO child_fills VALUES (?,?,?,?,?,?,?,?)",
        ("BAD", "O1", -1, 0, 940, 940, 1, 0),
    )
    invalid_fill = benchmark_row(invalid)
    assert invalid_fill["status"] == "REJECT_INVALID_FILL"
    assert invalid_fill["executed_quantity"] is None
    assert invalid_fill["implementation_shortfall"] is None

    ambiguous = benchmark_connection()
    ambiguous.execute(
        "INSERT INTO child_fills VALUES (?,?,?,?,?,?,?,?)",
        ("F1", "O1", 401, 100.16, 920, 920, 1, 0),
    )
    ambiguous_fill = benchmark_row(ambiguous)
    assert ambiguous_fill["status"] == "REJECT_AMBIGUOUS_FILL_REVISION"
    assert ambiguous_fill["executed_quantity"] is None

    overfill = benchmark_connection()
    overfill.execute(
        "INSERT INTO child_fills VALUES (?,?,?,?,?,?,?,?)",
        ("F4", "O1", 300, 100.2, 940, 940, 1, 0),
    )
    overfill_result = benchmark_row(overfill)
    assert overfill_result["status"] == "REJECT_OVERFILL"
    return {
        "pass": passed,
        "zero_fill": zero_fill,
        "missing": missing,
        "invalid_fill": invalid_fill,
        "ambiguous_fill": ambiguous_fill,
        "overfill": overfill_result,
    }


def event_connection(rows: list[tuple[object, ...]]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE order_events(
            event_id TEXT, order_id TEXT, sequence_number INTEGER,
            event_type TEXT, event_time INTEGER, knowledge_time INTEGER,
            revision_id INTEGER, is_cancelled INTEGER,
            quantity REAL, price REAL
        );
        """
    )
    connection.executemany("INSERT INTO order_events VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    return connection


def run_event_case(
    rows: list[tuple[object, ...]],
    order_id: str,
    expected_initial_sequence: int = 1,
) -> dict[str, Any]:
    connection = event_connection(rows)
    return _row(
        connection,
        "order_event_reconstruction.sql",
        {
            "order_id": order_id,
            "knowledge_cutoff": 1000,
            "expected_initial_sequence": expected_initial_sequence,
        },
    )


def run_events() -> dict[str, Any]:
    valid_rows = [
        ("E1", "O1", 1, "SUBMIT", 900, 900, 1, 0, 1000, None),
        ("E2", "O1", 2, "ACK", 901, 901, 1, 0, None, None),
        ("E3", "O1", 3, "PARTIAL_FILL", 910, 910, 1, 0, 300, 100.10),
        ("E4", "O1", 4, "PARTIAL_FILL", 920, 920, 1, 0, 200, 100.20),
        ("E5", "O1", 5, "CANCEL_REQUEST", 925, 925, 1, 0, None, None),
        ("E6", "O1", 6, "CANCEL_ACK", 930, 930, 1, 0, None, None),
        ("E3", "O1", 3, "PARTIAL_FILL", 910, 915, 2, 0, 300, 100.10),
    ]
    valid = run_event_case(valid_rows, "O1")
    assert valid["filled_quantity"] == 500
    assert valid["average_fill_price"] == 100.14
    assert valid["reconstructed_state"] == "PARTIALLY_FILLED_CANCELLED"
    assert valid["status"] == "PASS"

    missing_initial = run_event_case(
        [
            ("E7", "O2", 7, "SUBMIT", 900, 900, 1, 0, 100, None),
            ("E8", "O2", 8, "ACK", 901, 901, 1, 0, None, None),
        ],
        "O2",
    )
    assert missing_initial["status"] == "REJECT_SEQUENCE_GAP"

    duplicate_sequence = run_event_case(
        [
            ("A1", "O3", 1, "SUBMIT", 900, 900, 1, 0, 100, None),
            ("A2", "O3", 2, "ACK", 901, 901, 1, 0, None, None),
            ("A3", "O3", 2, "PARTIAL_FILL", 902, 902, 1, 0, 50, 10.0),
        ],
        "O3",
    )
    assert duplicate_sequence["status"] == "REJECT_DUPLICATE_SEQUENCE"
    assert duplicate_sequence["filled_quantity"] is None

    invalid_transition = run_event_case(
        [
            ("B1", "O4", 1, "SUBMIT", 900, 900, 1, 0, 100, None),
            ("B2", "O4", 2, "CANCEL_ACK", 901, 901, 1, 0, None, None),
        ],
        "O4",
    )
    assert invalid_transition["status"] == "REJECT_INVALID_TRANSITION"

    invalid_payload = run_event_case(
        [("C1", "O5", 1, "SUBMIT", 900, 900, 1, 0, 0, None)],
        "O5",
    )
    assert invalid_payload["status"] == "REJECT_INVALID_PAYLOAD"

    ack_only = run_event_case(
        [("D1", "O6", 1, "ACK", 900, 900, 1, 0, None, None)], "O6"
    )
    assert ack_only["status"] == "REJECT_SUBMIT_CARDINALITY"

    duplicate_submit = run_event_case(
        [
            ("E1", "O7", 1, "SUBMIT", 900, 900, 1, 0, 100, None),
            ("E2", "O7", 2, "SUBMIT", 901, 901, 1, 0, 100, None),
        ],
        "O7",
    )
    assert duplicate_submit["status"] == "REJECT_SUBMIT_CARDINALITY"

    negative_fill = run_event_case(
        [
            ("F1", "O8", 1, "SUBMIT", 900, 900, 1, 0, 100, None),
            ("F2", "O8", 2, "ACK", 901, 901, 1, 0, None, None),
            ("F3", "O8", 3, "FILL", 902, 902, 1, 0, -10, 100.0),
        ],
        "O8",
    )
    assert negative_fill["status"] == "REJECT_INVALID_PAYLOAD"
    assert negative_fill["filled_quantity"] is None

    missing = run_event_case([], "MISSING")
    assert missing["status"] == "REJECT_MISSING_ORDER"
    return {
        "pass": valid,
        "missing_initial": missing_initial,
        "duplicate_sequence": duplicate_sequence,
        "invalid_transition": invalid_transition,
        "invalid_payload": invalid_payload,
        "ack_only": ack_only,
        "duplicate_submit": duplicate_submit,
        "negative_fill": negative_fill,
        "missing": missing,
    }


def main() -> None:
    result = {
        "point_in_time_nbbo": run_nbbo(),
        "execution_shortfall": run_shortfall(),
        "event_reconstruction": run_events(),
    }
    output_dir = Path(os.environ.get("QI_CH10_SQL_OUTPUT", DEFAULT_OUTPUT))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sql_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
