from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
SQL = REPO_ROOT / "_book" / "sql" / "ch11"


def query(
    script: str,
    setup: str,
    params: dict[str, Any] | None = None,
) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(setup)
        return connection.execute(
            (SQL / script).read_text(encoding="utf-8"), params or {}
        ).fetchall()
    finally:
        connection.close()


def execute_ingestion(
    connection: sqlite3.Connection,
    params: dict[str, Any],
) -> list[tuple[Any, ...]]:
    connection.create_function(
        "sha256",
        1,
        lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest(),
        deterministic=True,
    )
    final: list[tuple[Any, ...]] = []
    statements = [
        statement.strip()
        for statement in (SQL / "idempotent_event_ingestion.sql")
        .read_text(encoding="utf-8")
        .split(";")
        if statement.strip()
    ]
    for statement in statements:
        cursor = connection.execute(statement, params)
        if cursor.description is not None:
            final = cursor.fetchall()
    return final


results: dict[str, object] = {}

rows = query(
    "dependency_cycle_control.sql",
    """
    CREATE TABLE dependencies(parent TEXT, child TEXT);
    INSERT INTO dependencies VALUES ('a','b'),('b','c'),('c','a');
    """,
    {"max_depth": 10},
)
assert rows and {row[-1] for row in rows} == {"REJECT_CYCLE"}
assert any('"a","b","c","a"' in row[2] for row in rows)
results["dependency_cycle_control.sql"] = rows

rows = query(
    "grouped_calculations.sql",
    """
    CREATE TABLE prices(asset TEXT, timestamp TEXT, price REAL);
    INSERT INTO prices VALUES
      ('A','2026-01-01',100.0),('A','2026-01-02',101.0),
      ('A','2026-01-03',103.0),('A','2026-01-04',102.0);
    """,
)
assert len(rows) == 4 and abs(rows[1][4] - 0.01) < 1e-12
assert all(row[-1] == "PASS" for row in rows)
assert rows[-1][5] is not None
results["grouped_calculations.sql"] = rows

rows = query(
    "point_in_time_asof.sql",
    """
    CREATE TABLE decisions(
      decision_row_id TEXT, decision_id TEXT, instrument_id TEXT,
      effective_at TEXT, knowledge_at TEXT
    );
    CREATE TABLE market_data(
      instrument_id TEXT, effective_at TEXT, known_at TEXT, version INTEGER,
      value REAL, source_id TEXT, row_hash TEXT
    );
    INSERT INTO decisions VALUES
      ('dr1','d1','X','2026-01-01T12:00:00Z','2026-01-01T12:00:00Z'),
      ('dr2','d2','Y','2026-01-01T12:00:00Z','2026-01-01T12:00:00Z');
    INSERT INTO market_data VALUES
      ('X','2026-01-01T10:00:00Z','2026-01-01T10:01:00Z',1,100.0,'feed','hash-1'),
      ('X','2026-01-01T10:00:00Z','2026-01-02T09:00:00Z',2,101.0,'feed','hash-2');
    """,
)
assert len(rows) == 2 and rows[0][5] == 100.0 and rows[0][-1] == "PASS"
assert rows[1][-1] == "REJECT_MISSING_ROW"
results["point_in_time_asof.sql"] = rows

rows = query(
    "point_in_time_feature_join.sql",
    """
    CREATE TABLE decisions(
      decision_row_id TEXT, decision_id TEXT, instrument_id TEXT,
      event_time TEXT, decision_time TEXT
    );
    CREATE TABLE feature_versions(
      instrument_id TEXT, feature_name TEXT, feature_value REAL,
      effective_from TEXT, knowledge_from TEXT, version_id INTEGER,
      source_id TEXT, row_hash TEXT
    );
    INSERT INTO decisions VALUES
      ('dr1','d1','X','2026-01-03T00:00:00Z','2026-01-03T09:00:00Z');
    INSERT INTO feature_versions VALUES
      ('X','momentum',1.5,'2026-01-02T00:00:00Z','2026-01-02T08:00:00Z',1,'calc','hash-1'),
      ('X','momentum',9.9,'2026-01-02T00:00:00Z','2026-01-04T08:00:00Z',2,'calc','hash-2');
    """,
)
assert rows == [(
    "dr1", "d1", "X", "2026-01-03T00:00:00Z",
    "2026-01-03T09:00:00Z", "momentum", 1.5,
    "2026-01-02T00:00:00Z", "2026-01-02T08:00:00Z", 1,
    "calc", "hash-1", "PASS",
)]
results["point_in_time_feature_join.sql"] = rows

rows = query(
    "window_returns_ranks.sql",
    """
    CREATE TABLE prices(
      timestamp TEXT, asset TEXT, sequence_number INTEGER, version_id INTEGER,
      row_id TEXT, row_hash TEXT, price REAL
    );
    INSERT INTO prices VALUES
      ('2026-01-01','A',1,1,'a1','ha1',100.0),
      ('2026-01-02','A',2,1,'a2','ha2',102.0),
      ('2026-01-01','B',1,1,'b1','hb1',100.0),
      ('2026-01-02','B',2,1,'b2','hb2',102.0);
    """,
)
ranked = [row for row in rows if row[0] == "2026-01-02"]
assert [row[8] for row in ranked] == [0.0, 0.0]
assert all(row[-1] == "PASS" for row in rows)
results["window_returns_ranks.sql"] = rows

rows = query(
    "windowed_signal_features.sql",
    """
    CREATE TABLE daily_observations(
      instrument_id TEXT, observation_time TEXT, sequence_number INTEGER,
      version_id INTEGER, knowledge_time TEXT, source_id TEXT, row_hash TEXT,
      return_1d REAL, close_price REAL
    );
    INSERT INTO daily_observations VALUES
      ('A','2026-01-01',1,1,'2026-01-01T12:00:00Z','calc','h1',0.01,100.0),
      ('A','2026-01-02',2,1,'2026-01-02T12:00:00Z','calc','h2',0.02,102.0),
      ('A','2026-01-03',3,1,'2026-01-03T12:00:00Z','calc','h3',-0.01,101.0);
    """,
    {"knowledge_cutoff": "2026-01-04T00:00:00Z"},
)
assert len(rows) == 3 and rows[0][8] is None and rows[1][10] == 100.0
assert all(row[-1] == "PASS" for row in rows)
results["windowed_signal_features.sql"] = rows

connection = sqlite3.connect(":memory:")
try:
    base = {
        "consumer_name": "risk",
        "message_id": "m1",
        "account_id": "a1",
        "amount_minor": 500,
        "currency": "USD",
        "scale": 2,
        "schema_version": 1,
    }
    first = execute_ingestion(connection, base)
    second = execute_ingestion(connection, base)
    changed = execute_ingestion(connection, {**base, "amount_minor": 999})
    assert first[0][0] == "NEWLY_PROCESSED" and len(first[0][1]) == 64
    assert second[0][0] == "DUPLICATE_SAME_PAYLOAD"
    assert changed[0][0] == "REJECT_ID_REUSED_WITH_DIFFERENT_PAYLOAD"
    effect = connection.execute(
        "SELECT amount_minor, currency, scale FROM business_effects"
    ).fetchall()
    assert effect == [(500, "USD", 2)]
    results["idempotent_event_ingestion.sql"] = effect
finally:
    connection.close()

rows = query(
    "schema_evolution_audit.sql",
    """
    CREATE TABLE event_schema_registry(
      event_type TEXT, schema_major INTEGER, schema_minor INTEGER,
      compatibility_state TEXT, required_fields_json TEXT,
      semantic_schema_hash TEXT
    );
    CREATE TABLE event_log(
      event_id TEXT, event_type TEXT, schema_major INTEGER,
      schema_minor INTEGER, payload_json TEXT, ingested_at TEXT
    );
    INSERT INTO event_schema_registry VALUES
      ('trade',1,3,'CURRENT','["instrument_id","quantity"]','schema-trade-1-3'),
      ('trade',1,4,'INCOMPATIBLE','["instrument_id","quantity","venue"]','schema-trade-1-4');
    INSERT INTO event_log VALUES
      ('e1','trade',1,3,'{"instrument_id":"X","quantity":10}','2026-01-01'),
      ('e2','trade',1,3,'{"instrument_id":"X"}','2026-01-02'),
      ('e3','trade',1,4,'{"instrument_id":"X","quantity":2,"venue":"XLON"}','2026-01-03'),
      ('e4','risk',3,0,'{"portfolio":"P"}','2026-01-04');
    """,
)
assert {row[0]: row[-1] for row in rows} == {
    "e2": "REJECT_MISSING_REQUIRED_FIELD",
    "e3": "REJECT_INCOMPATIBLE_SCHEMA",
    "e4": "REJECT_UNKNOWN_SCHEMA",
}
results["schema_evolution_audit.sql"] = rows

connection = sqlite3.connect(":memory:")
try:
    connection.executescript(
        """
        CREATE TABLE prices(
          symbol TEXT, valid_time TEXT, knowledge_time TEXT, version INTEGER,
          source_id TEXT, row_hash TEXT, price REAL
        );
        INSERT INTO prices VALUES
          ('A','2026-01-01','2026-01-01T09:00:00',1,'feed','h1',100.0),
          ('A','2026-01-01','2026-01-01T10:00:00',2,'feed','h2',101.0);
        """
    )
    parts = [
        statement.strip()
        for statement in (SQL / "query_plan_indexes.sql")
        .read_text(encoding="utf-8")
        .split(";")
        if statement.strip()
    ]
    connection.execute(parts[0])
    plan = connection.execute(
        parts[1],
        {"valid_cutoff": "2026-01-02", "knowledge_cutoff": "2026-01-01T10:30:00"},
    ).fetchall()
    plan_text = "\n".join(str(row[-1]) for row in plan)
    assert "idx_prices_symbol_valid_knowledge_version" in plan_text
    assert "LAST TERM" not in plan_text
    results["query_plan_indexes.sql"] = plan
finally:
    connection.close()

rows = query(
    "bitemporal_asof.sql",
    """
    CREATE TABLE instrument_values(
      instrument_id TEXT, valid_from TEXT, valid_to TEXT,
      knowledge_from TEXT, knowledge_to TEXT, value REAL,
      version_id INTEGER, source_id TEXT, row_hash TEXT,
      source_priority INTEGER
    );
    INSERT INTO instrument_values VALUES
      ('X','2026-01-01',NULL,'2026-01-01T08:00:00',NULL,100.0,1,'b','h1',2),
      ('X','2026-01-01',NULL,'2026-01-01T09:00:00',NULL,101.0,2,'a','h2',1),
      ('X','2026-01-01',NULL,'2026-01-02T09:00:00',NULL,999.0,3,'a','h3',1);
    """,
    {"decision_time": "2026-01-01T12:00:00", "knowledge_cutoff": "2026-01-01T10:00:00"},
)
assert rows == [(
    "X", 101.0, "2026-01-01", None, "2026-01-01T09:00:00", None,
    2, "a", "h2", "PASS",
)]
results["bitemporal_asof.sql"] = rows

rows = query(
    "temporal_position_reconstruction.sql",
    """
    CREATE TABLE position_events(
      event_id TEXT, account_id TEXT, instrument_id TEXT, event_time TEXT,
      knowledge_time TEXT, version_id INTEGER, sequence_number INTEGER,
      quantity_delta REAL, source_id TEXT, row_hash TEXT
    );
    INSERT INTO position_events VALUES
      ('e1','a','X','2026-01-01T09:00:00','2026-01-01T09:01:00',1,1,10,'oms','h1'),
      ('e2','a','X','2026-01-01T10:00:00','2026-01-01T10:01:00',1,2,-3,'oms','h2'),
      ('e2','a','X','2026-01-01T10:00:00','2026-01-02T10:00:00',2,2,-4,'oms','h3');
    """,
    {"as_of_time": "2026-01-01T12:00:00", "knowledge_cutoff": "2026-01-01T12:00:00"},
)
assert rows == [("a", "X", 7.0, 2, "PASS")]
results["temporal_position_reconstruction.sql"] = rows

rows = query(
    "window_reconciliation.sql",
    """
    CREATE TABLE cash_events(
      event_id TEXT, account_id TEXT, event_time TEXT, knowledge_time TEXT,
      version_id INTEGER, cash_delta_minor INTEGER, currency TEXT,
      source_id TEXT, row_hash TEXT
    );
    CREATE TABLE account_snapshots(
      account_id TEXT, snapshot_time TEXT, knowledge_time TEXT,
      version_id INTEGER, expected_cash_minor INTEGER, currency TEXT,
      source_id TEXT, row_hash TEXT
    );
    INSERT INTO cash_events VALUES
      ('e1','a','2026-01-01T09:00:00','2026-01-01T09:01:00',1,1000,'USD','ledger','eh1'),
      ('e2','a','2026-01-01T10:00:00','2026-01-01T10:01:00',1,-200,'USD','ledger','eh2');
    INSERT INTO account_snapshots VALUES
      ('a','2026-01-01T10:00:00','2026-01-01T10:02:00',1,750,'USD','custodian','sh1');
    """,
    {
        "knowledge_cutoff": "2026-01-01T12:00:00",
        "tolerance_minor": 1,
        "tolerance_version": "cash-usd-v1",
    },
)
assert rows[0][4:8] == (800, 50, 1, "cash-usd-v1")
assert rows[0][-1] == "REJECT_BREAK"
results["window_reconciliation.sql"] = rows

for name, value in results.items():
    print(f"PASS {name}: {value}")
print(f"TOTAL_PASS={len(results)}")
