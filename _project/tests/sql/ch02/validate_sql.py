#!/usr/bin/env python3
"""Execute the CH02 covariance-lineage and optimizer-audit SQL fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
SQL_ROOT = REPO_ROOT / "_book" / "sql" / "ch02"
DIALECT_DECLARATION = "-- Dialect: PostgreSQL"


def read_sql(name: str) -> str:
    """Read a canonical query and enforce its declared reference dialect."""

    text = (SQL_ROOT / name).read_text(encoding="utf-8")
    assert text.splitlines()[0] == DIALECT_DECLARATION
    return text


def audited_connection() -> sqlite3.Connection:
    """Create an isolated database with governed positive reference fixtures."""

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE covariance_run (
            record_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            snapshot_hash TEXT,
            estimator_configuration_hash TEXT,
            estimator TEXT,
            estimator_version TEXT,
            window_start TEXT,
            window_end TEXT,
            knowledge_cutoff TEXT,
            maximum_observation_knowledge_time TEXT,
            created_at TEXT,
            published INTEGER,
            stored_decision_state TEXT
        );
        CREATE TABLE optimizer_run (
            record_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            covariance_hash TEXT,
            constraints_hash TEXT,
            targets_hash TEXT,
            solver_version TEXT,
            configuration_hash TEXT,
            decision_time TEXT,
            covariance_maximum_knowledge_time TEXT,
            constraints_knowledge_time TEXT,
            created_at TEXT,
            expected_asset_count INTEGER NOT NULL,
            expected_constraint_count INTEGER NOT NULL,
            primal_tolerance REAL NOT NULL,
            stationarity_tolerance REAL NOT NULL,
            dual_tolerance REAL NOT NULL,
            complementarity_tolerance REAL NOT NULL
        );
        CREATE TABLE optimizer_output (
            run_id TEXT NOT NULL,
            asset TEXT NOT NULL,
            weight REAL NOT NULL,
            expected_return REAL NOT NULL,
            PRIMARY KEY (run_id, asset)
        );
        CREATE TABLE optimizer_constraint_coefficient (
            run_id TEXT NOT NULL,
            constraint_name TEXT NOT NULL,
            constraint_kind TEXT NOT NULL,
            asset TEXT NOT NULL,
            coefficient REAL NOT NULL,
            target_value REAL NOT NULL,
            tolerance REAL NOT NULL,
            PRIMARY KEY (run_id, constraint_name, asset)
        );
        CREATE TABLE optimizer_kkt_evidence (
            evidence_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            covariance_hash TEXT,
            constraints_hash TEXT,
            configuration_hash TEXT,
            solver_version TEXT,
            multipliers_json TEXT,
            active_constraints_json TEXT,
            primal_residual REAL,
            stationarity_residual REAL,
            dual_feasibility_violation REAL,
            complementarity_residual REAL,
            decision_state TEXT
        );
        """
    )
    insert_lineage_run(connection, "R1", stored_decision_state="PASS")
    insert_optimizer_fixture(connection, "O1")
    return connection


def lineage_row(connection: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    rows = connection.execute(read_sql("covariance_run_lineage.sql")).fetchall()
    matches = [row for row in rows if row["run_id"] == run_id]
    assert len(matches) == 1
    return dict(matches[0])


def optimizer_row(connection: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    rows = connection.execute(read_sql("optimizer_constraint_audit.sql")).fetchall()
    matches = [row for row in rows if row["run_id"] == run_id]
    assert len(matches) == 1
    return dict(matches[0])


def insert_lineage_run(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    record_suffix: str = "a",
    snapshot_hash: str | None = "sha256:snapshot",
    maximum_knowledge_time: str = "2026-01-01T18:00:00Z",
    stored_decision_state: str = "REJECT",
) -> None:
    connection.execute(
        "INSERT INTO covariance_run VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            f"record-{run_id}-{record_suffix}",
            run_id,
            snapshot_hash,
            "sha256:estimator-configuration",
            "oas",
            "cwh-2010-v1",
            "2025-01-01T00:00:00Z",
            "2025-12-31T23:59:59Z",
            "2026-01-02T00:00:00Z",
            maximum_knowledge_time,
            "2026-01-02T00:05:00Z",
            1,
            stored_decision_state,
        ),
    )


def insert_optimizer_fixture(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    covariance_hash: str | None = "sha256:covariance",
    covariance_knowledge_time: str = "2026-07-28T22:00:00Z",
    weights: tuple[float, float] = (0.6, 0.4),
    stationarity_residual: float = 8.2e-10,
    stored_decision_state: str = "PASS",
    evidence_covariance_hash: str = "sha256:covariance",
    record_suffix: str = "a",
    include_evidence: bool = True,
) -> None:
    connection.execute(
        "INSERT INTO optimizer_run VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            f"run-record-{run_id}-{record_suffix}",
            run_id,
            covariance_hash,
            "sha256:constraints",
            "sha256:targets",
            "constrained-turnover-kkt-v1",
            "sha256:solver-configuration",
            "2026-07-29T00:00:00Z",
            covariance_knowledge_time,
            "2026-07-28T23:00:00Z",
            "2026-07-29T00:01:00Z",
            2,
            2,
            1.0e-8,
            1.0e-8,
            1.0e-8,
            1.0e-8,
        ),
    )
    if record_suffix != "a":
        return
    connection.executemany(
        "INSERT INTO optimizer_output VALUES (?,?,?,?)",
        (
            (run_id, "A", weights[0], 0.08),
            (run_id, "B", weights[1], 0.03),
        ),
    )
    connection.executemany(
        "INSERT INTO optimizer_constraint_coefficient VALUES (?,?,?,?,?,?,?)",
        (
            (run_id, "budget", "BUDGET", "A", 1.0, 1.0, 1.0e-12),
            (run_id, "budget", "BUDGET", "B", 1.0, 1.0, 1.0e-12),
            (
                run_id,
                "value_factor",
                "FACTOR_NEUTRALITY",
                "A",
                1.0,
                0.0,
                1.0e-12,
            ),
            (
                run_id,
                "value_factor",
                "FACTOR_NEUTRALITY",
                "B",
                -1.5,
                0.0,
                1.0e-12,
            ),
        ),
    )
    if include_evidence:
        connection.execute(
            "INSERT INTO optimizer_kkt_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"evidence-{run_id}",
                run_id,
                evidence_covariance_hash,
                "sha256:constraints",
                "sha256:solver-configuration",
                "constrained-turnover-kkt-v1",
                '{"budget":0.014,"value_factor":-0.003}',
                '{"lower":[],"upper":[]}',
                2.1e-15,
                stationarity_residual,
                0.0,
                3.4e-11,
                stored_decision_state,
            ),
        )


def test_covariance_lineage_positive_and_negative_fixtures() -> None:
    connection = audited_connection()
    try:
        passed = lineage_row(connection, "R1")
        assert passed["computed_decision_state"] == "PASS"
        assert passed["stored_decision_state"] == "PASS"
        assert passed["violation_reason"] == "COMPLETE_POINT_IN_TIME_LINEAGE"

        insert_lineage_run(connection, "R_MISSING_HASH", snapshot_hash=None)
        missing_hash = lineage_row(connection, "R_MISSING_HASH")
        assert missing_hash["computed_decision_state"] == "REJECT"
        assert missing_hash["violation_reason"] == "MISSING_LINEAGE_HASH"

        insert_lineage_run(
            connection,
            "R_FUTURE",
            maximum_knowledge_time="2026-01-03T00:00:00Z",
        )
        future = lineage_row(connection, "R_FUTURE")
        assert future["computed_decision_state"] == "REJECT"
        assert future["violation_reason"] == "FUTURE_KNOWLEDGE"

        insert_lineage_run(connection, "R_DUPLICATE", record_suffix="a")
        insert_lineage_run(connection, "R_DUPLICATE", record_suffix="b")
        duplicate = lineage_row(connection, "R_DUPLICATE")
        assert duplicate["record_count"] == 2
        assert duplicate["computed_decision_state"] == "REJECT"
        assert duplicate["violation_reason"] == "DUPLICATE_RUN_ID"
    finally:
        connection.close()


def test_optimizer_joins_constraints_kkt_and_lineage() -> None:
    connection = audited_connection()
    try:
        passed = optimizer_row(connection, "O1")
        assert passed["output_asset_count"] == passed["expected_asset_count"] == 2
        assert (
            passed["evaluated_constraint_count"]
            == passed["expected_constraint_count"]
            == 2
        )
        assert abs(passed["budget"] - 1.0) <= 1.0e-15
        assert passed["factor_constraint_breach_count"] == 0
        assert passed["computed_decision_state"] == "PASS"
        assert passed["stored_decision_state"] == "PASS"
        assert passed["violation_reason"] == "ALL_SIGNED_GATES_PASS"

        assert passed["maximum_constraint_residual"] <= 1.0e-15
    finally:
        connection.close()


def test_optimizer_negative_fixtures() -> None:
    connection = audited_connection()
    try:
        insert_optimizer_fixture(
            connection,
            "O_MISSING_HASH",
            covariance_hash=None,
            stored_decision_state="REJECT",
        )
        assert optimizer_row(connection, "O_MISSING_HASH")["violation_reason"] == (
            "MISSING_LINEAGE_HASH_OR_VERSION"
        )

        insert_optimizer_fixture(
            connection,
            "O_FUTURE",
            covariance_knowledge_time="2026-07-30T00:00:00Z",
            stored_decision_state="REJECT",
        )
        assert optimizer_row(connection, "O_FUTURE")["violation_reason"] == (
            "FUTURE_KNOWLEDGE"
        )

        insert_optimizer_fixture(
            connection,
            "O_LINEAGE_MISMATCH",
            evidence_covariance_hash="sha256:different-covariance",
            stored_decision_state="REJECT",
        )
        assert optimizer_row(connection, "O_LINEAGE_MISMATCH")[
            "violation_reason"
        ] == "LINEAGE_MISMATCH"

        insert_optimizer_fixture(
            connection,
            "O_DUPLICATE",
            stored_decision_state="REJECT",
        )
        insert_optimizer_fixture(
            connection,
            "O_DUPLICATE",
            stored_decision_state="REJECT",
            record_suffix="b",
        )
        duplicate = optimizer_row(connection, "O_DUPLICATE")
        assert duplicate["computed_decision_state"] == "REJECT"
        assert duplicate["violation_reason"] == "DUPLICATE_RUN_ID"

        insert_optimizer_fixture(
            connection,
            "O_FACTOR",
            weights=(0.7, 0.3),
            stored_decision_state="REJECT",
        )
        factor = optimizer_row(connection, "O_FACTOR")
        assert factor["factor_constraint_breach_count"] == 1
        assert factor["computed_decision_state"] == "REJECT"
        assert factor["violation_reason"] == "FACTOR_NEUTRALITY_BREACH"

        insert_optimizer_fixture(
            connection,
            "O_KKT",
            stationarity_residual=1.0e-4,
            stored_decision_state="REJECT",
        )
        kkt = optimizer_row(connection, "O_KKT")
        assert kkt["computed_decision_state"] == "REJECT"
        assert kkt["violation_reason"] == "KKT_RESIDUAL_LIMIT_EXCEEDED"

        insert_optimizer_fixture(
            connection,
            "O_STORED_MISMATCH",
            stored_decision_state="REJECT",
        )
        mismatch = optimizer_row(connection, "O_STORED_MISMATCH")
        assert mismatch["computed_decision_state"] == "PASS"
        assert mismatch["stored_decision_state"] == "REJECT"
        assert mismatch["violation_reason"] == "STORED_DECISION_MISMATCH"
    finally:
        connection.close()


def main() -> int:
    test_covariance_lineage_positive_and_negative_fixtures()
    test_optimizer_joins_constraints_kkt_and_lineage()
    test_optimizer_negative_fixtures()
    print(
        "PASS dialect=PostgreSQL fixture_engine=SQLite files=2 "
        "lineage_cases=4 optimizer_cases=8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
