from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[4]
SQL_ROOT = REPO_ROOT / "_book" / "sql" / "ch12"


def read(name: str) -> str:
    return (SQL_ROOT / name).read_text(encoding="utf-8").strip()


def rows(cursor: sqlite3.Cursor) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in cursor.fetchall()]


def execute_script_with_parameters(
    cursor: sqlite3.Cursor,
    name: str,
    parameters: Mapping[str, object],
) -> list[list[tuple[Any, ...]]]:
    outputs: list[list[tuple[Any, ...]]] = []
    for statement in (
        part.strip() for part in read(name).split(";") if part.strip()
    ):
        result = cursor.execute(statement, parameters)
        if result.description is not None:
            outputs.append(rows(result))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/ch12-sql-validation.txt"),
    )
    arguments = parser.parse_args()
    connection = sqlite3.connect(":memory:", isolation_level=None)
    cursor = connection.cursor()
    results: list[str] = []

    cursor.executescript(
        """
        CREATE TABLE instrument_identity(
            mapping_id TEXT,
            instrument_id TEXT,
            id_type TEXT,
            id_value TEXT,
            valid_from TEXT,
            valid_to TEXT,
            known_from TEXT,
            known_to TEXT,
            version_id INTEGER,
            source_id TEXT,
            source_priority INTEGER,
            override_id TEXT
        );
        INSERT INTO instrument_identity VALUES
          ('m1','I_OLD','ticker','ABC','2018-01-01','2020-01-01',
           '2018-01-01',NULL,1,'legacy',5,NULL),
          ('m2','I_NEW','ticker','ABC','2020-01-01',NULL,
           '2019-12-20',NULL,2,'master',10,NULL);
        """
    )
    selected = rows(
        cursor.execute(
            read("instrument_identity_asof.sql"),
            {
                "id_type": "ticker",
                "id_value": "ABC",
                "valid_at": "2021-01-02",
                "known_at": "2021-01-02",
            },
        )
    )
    assert selected[0][0:5] == ("I_NEW", "m2", 2, "master", 10)
    assert selected[0][-1] == "RESOLVED"
    cursor.execute(
        """
        INSERT INTO instrument_identity VALUES
        ('m3','I_CONFLICT','ticker','ABC','2020-01-01',NULL,
         '2019-12-20',NULL,2,'other',10,NULL)
        """
    )
    ambiguous = rows(
        cursor.execute(
            read("instrument_identity_asof.sql"),
            {
                "id_type": "ticker",
                "id_value": "ABC",
                "valid_at": "2021-01-02",
                "known_at": "2021-01-02",
            },
        )
    )
    assert ambiguous[0][-1] == "AMBIGUOUS"
    results.append(
        "instrument_identity_asof.sql: PASS — typed resolved/ambiguous state"
    )

    cursor.executescript(
        """
        CREATE TABLE decisions(
            decision_id TEXT,
            instrument_id TEXT,
            feature_key TEXT,
            market_time TEXT,
            knowledge_time TEXT
        );
        CREATE TABLE features(
            feature_id TEXT,
            instrument_id TEXT,
            feature_key TEXT,
            value REAL,
            row_status TEXT,
            valid_from TEXT,
            valid_to TEXT,
            known_from TEXT,
            known_to TEXT,
            version_id INTEGER,
            source_id TEXT,
            source_priority INTEGER
        );
        INSERT INTO decisions VALUES
          ('d-before','I_NEW','eps','2020-06-30','2020-07-01'),
          ('d-after','I_NEW','eps','2020-06-30','2020-08-15'),
          ('d-other','I_NEW','revenue','2020-06-30','2020-08-15');
        INSERT INTO features VALUES
          ('f1','I_NEW','eps',10.0,'ACTIVE','2020-01-01',NULL,
           '2020-06-01','2020-08-01',1,'vendor',10),
          ('f2','I_NEW','eps',12.0,'ACTIVE','2020-01-01',NULL,
           '2020-08-01',NULL,2,'vendor',10),
          ('f3','I_NEW','revenue',NULL,'TOMBSTONE','2020-01-01',NULL,
           '2020-08-01',NULL,1,'vendor',10);
        """
    )
    feature_rows = rows(cursor.execute(read("bitemporal_feature_join.sql")))
    by_decision = {row[0]: row for row in feature_rows}
    assert by_decision["d-before"][3] == 10.0
    assert by_decision["d-after"][3] == 12.0
    assert by_decision["d-other"][-1] == "WITHDRAWN"
    results.append(
        "bitemporal_feature_join.sql: PASS — feature key, tombstone and lineage"
    )

    cursor.executescript(
        """
        CREATE TABLE corporate_actions(
            action_id TEXT,
            version_id INTEGER,
            source_id TEXT,
            instrument_id TEXT,
            action_type TEXT,
            announced_at TEXT,
            ex_date TEXT,
            record_date TEXT,
            payable_date TEXT,
            known_from TEXT,
            known_to TEXT,
            action_status TEXT,
            ratio REAL,
            cash_amount REAL,
            currency TEXT,
            child_instrument_id TEXT,
            successor_instrument_id TEXT
        );
        INSERT INTO corporate_actions VALUES
          ('a1',1,'exchange','I_NEW','SPLIT','2020-08-01',
           '2020-09-01','2020-09-01','2020-09-02',
           '2020-08-01','2020-08-05','ANNOUNCED',2.0,NULL,'USD',NULL,NULL),
          ('a1',2,'exchange','I_NEW','SPLIT','2020-08-01',
           '2020-09-01','2020-09-01','2020-09-02',
           '2020-08-05',NULL,'CONFIRMED',2.0,NULL,'USD',NULL,NULL),
          ('a2',1,'exchange','I_NEW','CASH_DIVIDEND','2020-09-15',
           '2020-10-01','2020-09-20','2020-10-10',
           '2020-09-15',NULL,'CONFIRMED',1.0,0.5,'USD',NULL,NULL);
        """
    )
    action_rows = rows(
        cursor.execute(
            read("corporate_actions_asof.sql"),
            {
                "market_date": "2020-09-10",
                "knowledge_time": "2020-08-10",
            },
        )
    )
    assert len(action_rows) == 1 and action_rows[0][1] == 2
    assert action_rows[0][11] == "CONFIRMED"
    results.append(
        "corporate_actions_asof.sql: PASS — selected latest known action revision"
    )

    cursor.executescript(
        """
        CREATE TABLE observations(
            observation_id TEXT,
            instrument_id TEXT,
            field_name TEXT,
            effective_at TEXT,
            known_at TEXT,
            version_id INTEGER,
            source_id TEXT,
            row_status TEXT,
            value REAL
        );
        INSERT INTO observations VALUES
          ('o1','future','price','2020-01-10','2020-02-10',1,'v','ACTIVE',1.0),
          ('o2','missing','price','2020-01-10','2020-01-10',1,'v','ACTIVE',NULL),
          ('o3','stale','price','2019-01-01','2019-01-01',1,'v','ACTIVE',2.0),
          ('o4','inactive','price','2020-01-10','2020-01-10',1,'v','TOMBSTONE',3.0);
        """
    )
    quality_rows = rows(
        cursor.execute(
            read("data_quality_exceptions.sql"),
            {
                "decision_time": "2020-02-01",
                "stale_cutoff": "2020-01-01",
            },
        )
    )
    issues = {row[1]: row[-1] for row in quality_rows}
    assert issues == {
        "future": "future_knowledge",
        "missing": "missing",
        "stale": "stale",
        "inactive": "inactive_has_value",
    }
    results.append(
        "data_quality_exceptions.sql: PASS — future/missing/stale/status checks"
    )

    cursor.execute(
        """
        CREATE TABLE event_log(
            event_id TEXT PRIMARY KEY,
            aggregate_id TEXT,
            sequence_no INTEGER,
            event_type TEXT,
            payload TEXT,
            payload_hash TEXT,
            recorded_at TEXT,
            UNIQUE(aggregate_id, sequence_no)
        )
        """
    )
    event_parameters = {
        "event_id": "e1",
        "aggregate_id": "agg",
        "sequence_no": 1,
        "event_type": "created",
        "payload": "{}",
        "payload_hash": "a" * 64,
    }
    first = execute_script_with_parameters(
        cursor, "idempotent_event_ingestion.sql", event_parameters
    )
    second = execute_script_with_parameters(
        cursor, "idempotent_event_ingestion.sql", event_parameters
    )
    conflict = execute_script_with_parameters(
        cursor,
        "idempotent_event_ingestion.sql",
        {**event_parameters, "payload": '{"x":1}', "payload_hash": "b" * 64},
    )
    assert first[-1][0][-1] == "APPENDED"
    assert second[-1][0][-1] == "DUPLICATE_SAME_CONTENT"
    assert conflict[-1][0][-1] == "CONFLICT"
    results.append(
        "idempotent_event_ingestion.sql: PASS — append/replay/conflict states"
    )

    cursor.executescript(
        """
        CREATE TABLE pipeline_partitions(
            job_name TEXT,
            partition_key TEXT,
            run_id TEXT,
            semantic_run_hash TEXT,
            status TEXT,
            input_hash TEXT,
            output_hash TEXT,
            attempts INTEGER,
            maximum_attempts INTEGER,
            last_error_code TEXT
        );
        INSERT INTO pipeline_partitions VALUES
          ('daily','2020-01-01','r1','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
           'SUCCEEDED','bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
           'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',1,3,NULL),
          ('daily','2020-01-02','r2','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
           'FAILED','bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
           NULL,3,3,'TIMEOUT'),
          ('daily','2020-01-03','r3','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
           'SUCCEEDED','bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
           NULL,1,3,NULL);
        """
    )
    partitions = rows(
        cursor.execute(
            read("partition_backfill_control.sql"),
            {
                "job_name": "daily",
                "start_key": "2020-01-01",
                "end_key": "2020-01-31",
            },
        )
    )
    assert [row[1] for row in partitions] == [
        "2020-01-01", "2020-01-02", "2020-01-03"
    ]
    assert [row[-1] for row in partitions] == [
        "COMPLETE",
        "EXHAUSTED",
        "INVALID_METADATA",
    ]
    results.append(
        "partition_backfill_control.sql: PASS — bounded retry and run lineage"
    )

    cursor.executescript(
        """
        CREATE TABLE decision_dates(
            decision_id TEXT,
            decision_date TEXT,
            knowledge_time TEXT
        );
        CREATE TABLE universe_membership(
            membership_id TEXT,
            instrument_id TEXT,
            row_status TEXT,
            valid_from TEXT,
            valid_to TEXT,
            known_from TEXT,
            known_to TEXT,
            version_id INTEGER,
            source_id TEXT
        );
        INSERT INTO decision_dates VALUES
          ('d1','2020-06-30','2020-07-01'),
          ('d2','2020-06-30','2020-08-15');
        INSERT INTO universe_membership VALUES
          ('u1','I_NEW','ACTIVE','2020-01-01',NULL,
           '2020-01-01','2020-08-01',1,'master'),
          ('u2','I_NEW','TOMBSTONE','2020-01-01',NULL,
           '2020-08-01',NULL,2,'master');
        """
    )
    universe = rows(cursor.execute(read("point_in_time_universe.sql")))
    assert [(row[0], row[-1]) for row in universe] == [
        ("d1", "SELECTED"),
        ("d2", "WITHDRAWN"),
    ]
    results.append(
        "point_in_time_universe.sql: PASS — tombstone excluded after knowledge cutoff"
    )

    cursor.executescript(
        """
        CREATE TABLE cash_balances(
            account_id TEXT,
            currency TEXT,
            amount REAL,
            as_of_date TEXT
        );
        CREATE TABLE custodian_cash_balances(
            account_id TEXT,
            currency TEXT,
            amount REAL,
            as_of_date TEXT
        );
        INSERT INTO cash_balances VALUES
          ('A','USD',100.0,'2020-01-31'),
          ('B','EUR',50.0,'2020-01-31');
        INSERT INTO custodian_cash_balances VALUES
          ('A','USD',99.0,'2020-01-31'),
          ('C','GBP',12.0,'2020-01-31');
        """
    )
    ledger_breaks = rows(
        cursor.execute(
            read("portfolio_ledger_reconciliation.sql"),
            {"as_of_date": "2020-01-31", "tolerance": 0.01},
        )
    )
    assert {row[4] for row in ledger_breaks} == {
        "MATERIAL",
        "MISSING_EXTERNAL",
        "MISSING_INTERNAL",
    }
    results.append(
        "portfolio_ledger_reconciliation.sql: PASS — missing sides preserved"
    )

    cursor.executescript(
        """
        CREATE TABLE pretrade_approvals(
            request_id TEXT,
            order_id TEXT,
            actor_id TEXT,
            actor_role TEXT,
            tenant_id TEXT,
            portfolio_id TEXT,
            rule_version TEXT,
            model_version TEXT,
            before_snapshot_id TEXT,
            after_snapshot_id TEXT,
            input_hash TEXT,
            decision TEXT,
            reason TEXT,
            approval_id TEXT,
            approver_id TEXT,
            created_at TEXT
        );
        INSERT INTO pretrade_approvals VALUES
          ('r2','o1','a2','risk','t1','p1','rv1','m1','b','a',
           'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
           'PASS','reviewed','ap1','owner','2020-01-01T10:01:00'),
          ('r1','o1','system','system','t1','p1','rv1','m1','b','a',
           'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
           'REQUIRES_APPROVAL','soft breach',NULL,NULL,'2020-01-01T10:00:00');
        """
    )
    approvals = rows(
        cursor.execute(
            read("pretrade_approval_audit.sql"), {"order_id": "o1"}
        )
    )
    assert [row[0] for row in approvals] == ["r1", "r2"]
    assert approvals[0][10] == "a" * 64
    assert approvals[0][-1] == "PENDING_APPROVAL"
    assert approvals[1][-1] == "EVIDENCE_COMPLETE"
    results.append(
        "pretrade_approval_audit.sql: PASS — ordered full decision evidence"
    )

    cursor.executescript(
        """
        CREATE TABLE reconciliation_results(
            run_id TEXT,
            break_type TEXT,
            entity_key TEXT,
            internal_value REAL,
            external_value REAL,
            tolerance REAL
        );
        INSERT INTO reconciliation_results VALUES
          ('run','position','I1',10.0,8.0,0.1),
          ('run','cash','USD',5.0,5.01,0.1),
          ('run','fee','BROKER',NULL,1.0,0.0);
        """
    )
    reconciliation = rows(
        cursor.execute(read("reconciliation_breaks.sql"), {"run_id": "run"})
    )
    assert {row[1]: row[5] for row in reconciliation} == {
        "I1": "MATERIAL",
        "BROKER": "MISSING_INTERNAL",
    }
    results.append(
        "reconciliation_breaks.sql: PASS — material and missing break states"
    )

    cursor.execute(
        """
        CREATE TABLE research_runs(
            run_id TEXT PRIMARY KEY,
            semantic_run_hash TEXT,
            code_hash TEXT,
            data_hash TEXT,
            config_hash TEXT,
            seed INTEGER,
            output_hash TEXT,
            created_at TEXT,
            status TEXT
        )
        """
    )
    run_parameters = {
        "run_id": "run-1",
        "semantic_run_hash": "a" * 64,
        "code_hash": "b" * 64,
        "data_hash": "c" * 64,
        "config_hash": "d" * 64,
        "seed": 7,
        "output_hash": "e" * 64,
        "status": "SUCCEEDED",
    }
    registered = execute_script_with_parameters(
        cursor, "snapshot_run_registry.sql", run_parameters
    )
    duplicate = execute_script_with_parameters(
        cursor, "snapshot_run_registry.sql", run_parameters
    )
    registry_conflict = execute_script_with_parameters(
        cursor,
        "snapshot_run_registry.sql",
        {**run_parameters, "output_hash": "f" * 64},
    )
    assert registered[-1][0][-1] == "REGISTERED"
    assert duplicate[-1][0][-1] == "DUPLICATE_SAME_RUN"
    assert registry_conflict[-1][0][-1] == "CONFLICT"
    results.append(
        "snapshot_run_registry.sql: PASS — immutable run/output semantics"
    )

    cursor.executescript(
        """
        CREATE TABLE agent_tool_calls(
            request_id TEXT,
            idempotency_key TEXT,
            actor_id TEXT,
            actor_role TEXT,
            tenant_id TEXT,
            portfolio_id TEXT,
            action_name TEXT,
            tool_name TEXT,
            tool_version TEXT,
            policy_version TEXT,
            input_schema_version TEXT,
            output_schema_version TEXT,
            expected_state_hash TEXT,
            input_hash TEXT,
            output_hash TEXT,
            approval_id TEXT,
            approver_id TEXT,
            precondition_status TEXT,
            postcondition_status TEXT,
            evidence_hash TEXT,
            started_at TEXT,
            completed_at TEXT,
            status TEXT
        );
        INSERT INTO agent_tool_calls VALUES
          ('a1','i1','actor','operator','t1','p1','publish','run','1','pv1',
           'is1','os1',
           'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
           'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
           'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
           'ap1','owner','PASS','PASS',
           'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd',
           '2020-01-01T10:00:00','2020-01-01T10:01:00','SUCCEEDED'),
          ('a2','i2','actor','operator','t1','p1','publish','run','1','pv1',
           'is1','os1',
           'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
           'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
           'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
           NULL,NULL,'PASS','FAIL',
           'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd',
           '2020-02-01T10:00:00','2020-02-01T10:01:00','REJECTED');
        """
    )
    audit = rows(
        cursor.execute(
            read("agent_tool_audit.sql"),
            {
                "from_time": "2020-01-01",
                "to_time": "2020-01-31T23:59:59",
            },
        )
    )
    assert len(audit) == 1 and audit[0][0] == "a1"
    assert audit[0][15:19] == ("ap1", "owner", "PASS", "PASS")
    assert audit[0][-1] == "EVIDENCE_COMPLETE"
    results.append(
        "agent_tool_audit.sql: PASS — authorization and post-condition lineage"
    )

    connection.close()
    report = [
        "Chapter 12 SQL validation",
        "=========================",
        "",
        *(f"- {line}" for line in results),
        "",
        (
            f"Summary: {len(results)}/{len(results)} SQL controls executed "
            "successfully on deterministic SQLite 3 fixtures."
        ),
        "",
    ]
    output = "\n".join(report)
    print(output)
    report_path = arguments.output
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
