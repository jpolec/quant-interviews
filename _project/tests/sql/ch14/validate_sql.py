#!/usr/bin/env python3
"""Execute CH14 SQL controls and their fail-closed counterexamples in SQLite."""

from __future__ import annotations

from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[4]
SQL = ROOT / "_book/sql/ch14"
H0 = "0" * 64
H1 = "1" * 64


def read(name: str) -> str:
    text = (SQL / name).read_text(encoding="utf-8")
    if not text.startswith("-- Dialect: SQLite 3.45+"):
        raise AssertionError(f"missing dialect declaration: {name}")
    return text


def evidence_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """CREATE TABLE interview_evidence (
        candidate_id TEXT, competency TEXT, score REAL, confidence REAL,
        observed_at TEXT, evidence_id INTEGER, source_id TEXT,
        schema_version TEXT)"""
    )
    return connection


def validate_interview_evidence() -> None:
    connection = evidence_connection()
    connection.executemany(
        "INSERT INTO interview_evidence VALUES (?,?,?,?,?,?,?,?)",
        [
            ("c1", "math", 3.0, 0.8, "2026-01-01T10:00:00Z", 1, "panel", "1.0"),
            ("c1", "math", 4.0, 0.9, "2026-01-02T10:00:00Z", 2, "panel", "1.0"),
            ("c1", "code", 5.0, 1.0, "2026-02-01T10:00:00Z", 3, "panel", "1.0"),
        ],
    )
    rows = connection.execute(
        read("interview_evidence.sql"),
        {"decision_time": "2026-01-15T00:00:00Z"},
    ).fetchall()
    assert rows == [("c1", "math", 4.0, 0.9, "2026-01-02T10:00:00Z", 2, "panel", "1.0")]

    malformed = evidence_connection()
    malformed.execute(
        "INSERT INTO interview_evidence VALUES (?,?,?,?,?,?,?,?)",
        ("c1", "math", 99.0, -5.0, "1999-not-a-time", 1, "", "999"),
    )
    assert malformed.execute(
        read("interview_evidence.sql"),
        {"decision_time": "2026-01-15T00:00:00Z"},
    ).fetchall() == []


def calibration_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE interview_scores (
            interviewer_id TEXT, interview_id TEXT, interviewer_score REAL
        );
        CREATE TABLE benchmark_case_anchor (
            interview_id TEXT, anchor_score REAL, anchor_version TEXT
        );
        """
    )
    return connection


def validate_calibration() -> None:
    connection = calibration_connection()
    connection.executescript(
        """
        INSERT INTO interview_scores VALUES
          ('i1','case1',4.0),('i2','case1',3.0),
          ('i1','case2',3.0),('i2','case2',4.0);
        INSERT INTO benchmark_case_anchor VALUES
          ('case1',3.5,'anchor-v1'),('case2',3.5,'anchor-v1');
        """
    )
    rows = connection.execute(read("score_calibration.sql")).fetchall()
    assert len(rows) == 2
    assert {row[0] for row in rows} == {"i1", "i2"}
    assert {row[-1] for row in rows} == {"PASS"}

    invalid = calibration_connection()
    invalid.executescript(
        """
        INSERT INTO interview_scores VALUES ('bad','case1',99),('bad','case1',99);
        INSERT INTO benchmark_case_anchor VALUES
          ('case1',3.0,'v1'),('case1',4.0,'v2');
        """
    )
    assert invalid.execute(read("score_calibration.sql")).fetchone()[-1].startswith("INVALID_")


def debrief_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE competency_evidence (
            interview_id TEXT, competency TEXT, quality REAL, score REAL,
            observed_at TEXT, evidence_id INTEGER
        );
        CREATE TABLE interview_decision (
            interview_id TEXT, role_id TEXT, policy_version TEXT,
            decision_time TEXT, final_decision TEXT, final_score REAL,
            completeness_threshold REAL
        );
        CREATE TABLE required_competency (
            role_id TEXT, policy_version TEXT, competency TEXT, weight REAL,
            minimum_quality REAL, hard_floor REAL
        );
        """
    )
    return connection


def validate_debrief_decision() -> None:
    connection = debrief_connection()
    connection.executescript(
        """
        INSERT INTO interview_decision VALUES
          ('ok','quant','p1','2026-01-03T00:00:00Z','ACCEPT',4.2,0.8),
          ('missing','quant','p1','2026-01-03T00:00:00Z','ACCEPT',4.2,0.8);
        INSERT INTO required_competency VALUES
          ('quant','p1','correctness',0.7,0.8,2.0),
          ('quant','p1','judgment',0.3,0.8,2.0);
        INSERT INTO competency_evidence VALUES
          ('ok','correctness',1.0,4.0,'2026-01-01T00:00:00Z',1),
          ('ok','judgment',1.0,4.0,'2026-01-02T00:00:00Z',2),
          ('missing','correctness',1.0,4.0,'2026-01-01T00:00:00Z',3);
        """
    )
    rows = {row[0]: row for row in connection.execute(read("debrief_consistency.sql"))}
    assert rows["ok"][-1] == "consistent"
    assert rows["missing"][-2:] == ("INSUFFICIENT_EVIDENCE", "decision_conflict")


def takehome_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE takehome_submission (
            submission_id TEXT, candidate_id TEXT, schema_version TEXT,
            code_commit TEXT, data_hash TEXT, config_hash TEXT,
            environment_hash TEXT, result_hash TEXT
        );
        CREATE TABLE takehome_reproduction (
            reproduction_id INTEGER, submission_id TEXT,
            reproduced_code_commit TEXT, reproduced_data_hash TEXT,
            reproduced_config_hash TEXT, reproduced_environment_hash TEXT,
            reproduced_result_hash TEXT, tolerance_status TEXT,
            metric_residual REAL, metric_tolerance REAL,
            failure_reason TEXT, executed_at TEXT
        );
        """
    )
    return connection


def validate_takehome_latest_reproduction() -> None:
    connection = takehome_connection()
    connection.execute(
        "INSERT INTO takehome_submission VALUES (?,?,?,?,?,?,?,?)",
        ("s1", "c1", "1.0", "a" * 40, H0, H0, H0, H1),
    )
    connection.executemany(
        "INSERT INTO takehome_reproduction VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (1, "s1", "a" * 40, H0, H0, H0, H0, "FAIL", 1.0, 0.01, "mismatch", "2026-01-01T00:00:00Z"),
            (2, "s1", "a" * 40, H0, H0, H0, H1, "PASS", 0.0, 1e-12, None, "2026-01-02T00:00:00Z"),
        ],
    )
    row = connection.execute(read("takehome_lineage.sql")).fetchone()
    assert row is not None and row[-1] == "bitwise_reproduced"

    invalid = takehome_connection()
    invalid.execute(
        "INSERT INTO takehome_submission VALUES (?,?,?,?,?,?,?,?)",
        ("s2", "c2", "999", "bad", "", "", "", ""),
    )
    invalid.execute(
        "INSERT INTO takehome_reproduction VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (1, "s2", "a" * 40, H0, H0, H0, H1, "PASS", None, None, None, "2026-01-02T00:00:00Z"),
    )
    assert invalid.execute(read("takehome_lineage.sql")).fetchone()[-1] == "invalid_manifest"

    mismatched = takehome_connection()
    mismatched.execute(
        "INSERT INTO takehome_submission VALUES (?,?,?,?,?,?,?,?)",
        ("s3", "c3", "1.0", "a" * 40, H0, H0, H0, H1),
    )
    mismatched.execute(
        "INSERT INTO takehome_reproduction VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (1, "s3", "a" * 40, H1, H0, H0, H1, "PASS", 0.0, 1e-12, None, "2026-01-02T00:00:00Z"),
    )
    assert mismatched.execute(read("takehome_lineage.sql")).fetchone()[-1] == "input_commitment_mismatch"


def main() -> int:
    validate_interview_evidence()
    validate_calibration()
    validate_debrief_decision()
    validate_takehome_latest_reproduction()
    print("PASS dialect=SQLite3 files=4 executed_controls=4 adversarial_controls=4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
