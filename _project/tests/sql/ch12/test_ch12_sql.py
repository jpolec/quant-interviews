from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[4]
SQL = ROOT / "_book/sql/ch12"
H64 = "a" * 64


def query(
    connection: sqlite3.Connection,
    name: str,
    parameters: Mapping[str, object] | None = None,
) -> list[tuple[object, ...]]:
    return connection.execute(
        (SQL / name).read_text(encoding="utf-8"), parameters or {}
    ).fetchall()


def script(
    connection: sqlite3.Connection,
    name: str,
    parameters: Mapping[str, object],
) -> list[list[tuple[object, ...]]]:
    outputs: list[list[tuple[object, ...]]] = []
    source = (SQL / name).read_text(encoding="utf-8")
    for statement in (part.strip() for part in source.split(";") if part.strip()):
        cursor = connection.execute(statement, parameters)
        if cursor.description is not None:
            outputs.append(cursor.fetchall())
    return outputs


def test_identity_missing_and_physical_duplicate_are_typed() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """CREATE TABLE instrument_identity(
        mapping_id TEXT, instrument_id TEXT, id_type TEXT, id_value TEXT,
        valid_from TEXT, valid_to TEXT, known_from TEXT, known_to TEXT,
        version_id INTEGER, source_id TEXT, source_priority INTEGER,
        override_id TEXT)"""
    )
    parameters = {
        "id_type": "ticker",
        "id_value": "ABC",
        "valid_at": "2024-01-01",
        "known_at": "2024-01-01",
    }
    assert query(connection, "instrument_identity_asof.sql", parameters)[0][-1] == "MISSING"
    connection.executemany(
        "INSERT INTO instrument_identity VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("m1", "I1", "ticker", "ABC", "2020-01-01", None,
             "2020-01-01", None, 1, "s", 10, None),
            ("m2", "I1", "ticker", "ABC", "2020-01-01", None,
             "2020-01-01", None, 2, "s", 10, None),
        ],
    )
    row = query(connection, "instrument_identity_asof.sql", parameters)[0]
    assert row[0] == "I1" and row[2] == 2 and row[-1] == "RESOLVED"


def test_bitemporal_semantic_tie_is_not_hidden_by_version_id() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE decisions(decision_id TEXT, instrument_id TEXT,
          feature_key TEXT, market_time TEXT, knowledge_time TEXT);
        CREATE TABLE features(feature_id TEXT, instrument_id TEXT,
          feature_key TEXT, value REAL, row_status TEXT, valid_from TEXT,
          valid_to TEXT, known_from TEXT, known_to TEXT, version_id INTEGER,
          source_id TEXT, source_priority INTEGER);
        INSERT INTO decisions VALUES ('d','I','x','2024-01-01','2024-01-01');
        INSERT INTO features VALUES
          ('f1','I','x',1.0,'ACTIVE','2020-01-01',NULL,'2020-01-01',NULL,1,'a',10),
          ('f2','I','x',2.0,'ACTIVE','2020-01-01',NULL,'2020-01-01',NULL,2,'b',10);
        """
    )
    assert query(connection, "bitemporal_feature_join.sql")[0][-1] == "AMBIGUOUS"


def test_quality_emits_all_selected_issues_without_future_double_count() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE observations(observation_id TEXT, instrument_id TEXT,
          field_name TEXT, effective_at TEXT, known_at TEXT, version_id INTEGER,
          source_id TEXT, row_status TEXT, value REAL);
        INSERT INTO observations VALUES
          ('old','A','price','2019-01-01','2019-01-01',1,'s','ACTIVE',NULL),
          ('future','B','price','2019-01-01','2025-01-01',1,'s','ACTIVE',NULL);
        """
    )
    rows = query(
        connection,
        "data_quality_exceptions.sql",
        {"decision_time": "2024-01-01", "stale_cutoff": "2020-01-01"},
    )
    issues = {(row[1], row[-1]) for row in rows}
    assert ("A", "missing") in issues and ("A", "stale") in issues
    assert {issue for instrument, issue in issues if instrument == "B"} == {
        "future_knowledge"
    }


def test_event_ingestion_rejects_bad_hash_gap_and_changed_payload() -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.execute(
        """CREATE TABLE event_log(event_id TEXT PRIMARY KEY,
        aggregate_id TEXT, sequence_no INTEGER, event_type TEXT, payload TEXT,
        payload_hash TEXT, recorded_at TEXT, UNIQUE(aggregate_id, sequence_no))"""
    )
    base = {
        "event_id": "e1", "aggregate_id": "a", "sequence_no": 1,
        "event_type": "created", "payload": "{}", "payload_hash": H64,
    }
    invalid = script(connection, "idempotent_event_ingestion.sql", {
        **base, "event_id": "bad", "payload_hash": "not-a-hash"
    })[-1][0][-1]
    gap = script(connection, "idempotent_event_ingestion.sql", {
        **base, "event_id": "gap", "sequence_no": 2
    })[-1][0][-1]
    assert invalid == "INVALID_HASH" and gap == "SEQUENCE_GAP"
    assert script(connection, "idempotent_event_ingestion.sql", base)[-1][0][-1] == "APPENDED"
    changed = script(connection, "idempotent_event_ingestion.sql", {
        **base, "payload": '{"x":1}'
    })[-1][0][-1]
    assert changed == "CONFLICT"


def test_invalid_tolerances_and_empty_ledger_fail_closed() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE cash_balances(account_id TEXT, currency TEXT, amount REAL,
          as_of_date TEXT);
        CREATE TABLE custodian_cash_balances(account_id TEXT, currency TEXT,
          amount REAL, as_of_date TEXT);
        CREATE TABLE reconciliation_results(run_id TEXT, break_type TEXT,
          entity_key TEXT, internal_value REAL, external_value REAL,
          tolerance REAL);
        INSERT INTO reconciliation_results VALUES
          ('r','cash','USD',1.0,1.0,NULL), ('r','cash','EUR',1.0,1.0,-1.0);
        """
    )
    ledger = query(
        connection,
        "portfolio_ledger_reconciliation.sql",
        {"as_of_date": "2024-01-01", "tolerance": None},
    )
    assert ledger[0][4] == "INVALID_TOLERANCE"
    breaks = query(connection, "reconciliation_breaks.sql", {"run_id": "r"})
    assert {row[5] for row in breaks} == {"INVALID_TOLERANCE"}


def test_run_registry_binds_all_metadata_not_only_output() -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.execute(
        """CREATE TABLE research_runs(run_id TEXT PRIMARY KEY,
        semantic_run_hash TEXT, code_hash TEXT, data_hash TEXT, config_hash TEXT,
        seed INTEGER, output_hash TEXT, created_at TEXT, status TEXT)"""
    )
    base = {
        "run_id": "r", "semantic_run_hash": "a" * 64,
        "code_hash": "b" * 64, "data_hash": "c" * 64,
        "config_hash": "d" * 64, "seed": 7, "output_hash": "e" * 64,
        "status": "SUCCEEDED",
    }
    assert script(connection, "snapshot_run_registry.sql", base)[-1][0][-1] == "REGISTERED"
    conflict = script(connection, "snapshot_run_registry.sql", {
        **base, "code_hash": "f" * 64
    })[-1][0][-1]
    assert conflict == "CONFLICT"


def test_negative_agent_fixture_is_asserted_by_audit_query() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """CREATE TABLE agent_tool_calls(request_id TEXT,idempotency_key TEXT,
        actor_id TEXT,actor_role TEXT,tenant_id TEXT,portfolio_id TEXT,
        action_name TEXT,tool_name TEXT,tool_version TEXT,policy_version TEXT,
        input_schema_version TEXT,output_schema_version TEXT,
        expected_state_hash TEXT,input_hash TEXT,output_hash TEXT,approval_id TEXT,
        approver_id TEXT,precondition_status TEXT,postcondition_status TEXT,
        evidence_hash TEXT,started_at TEXT,completed_at TEXT,status TEXT)"""
    )
    connection.execute(
        "INSERT INTO agent_tool_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "r", "i", "actor", "operator", "t", "p", "publish", "tool",
            "v1", "p1", "in1", "out1", H64, H64, H64, "approval", "owner",
            "PASS", "FAIL", H64, "2024-01-01", "2024-01-02", "SUCCEEDED",
        ),
    )
    rows = query(
        connection,
        "agent_tool_audit.sql",
        {"from_time": "2024-01-01", "to_time": "2025-01-01"},
    )
    assert rows[0][-1] == "INVALID_SUCCESS_EVIDENCE"
