#!/usr/bin/env python3
"""Execute the CH01 SQLite point-in-time and selection-audit contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SQL_ROOT = REPO_ROOT / "_book" / "sql" / "ch01"
DIALECT_DECLARATION = "-- Dialect: SQLite 3"


def read_sql(name: str) -> str:
    text = (SQL_ROOT / name).read_text(encoding="utf-8")
    assert text.splitlines()[0] == DIALECT_DECLARATION
    return text


def test_point_in_time_outcome_rejects_future_revisions() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(read_sql("point_in_time_outcome.sql"))
        rows = {
            row["forecast_id"]: dict(row)
            for row in connection.execute(
                "SELECT * FROM selected_outcome ORDER BY forecast_id"
            )
        }
        assert set(rows) == {"P1", "P2"}

        p1 = rows["P1"]
        assert p1["decision_time"] == "2026-01-02T10:00:00"
        assert p1["evaluation_knowledge_cutoff"] == "2026-01-04T00:00:00"
        assert p1["label"] == 1.0
        assert p1["revision"] == 1
        assert p1["outcome_row_hash"] == "o-hash-p1-r1"
        assert p1["selection_status"] == "SELECTED_AS_OF_CUTOFF"

        future_revision = connection.execute(
            """
            SELECT known_at, outcome_row_hash
            FROM outcome_revision
            WHERE forecast_id = 'P1' AND revision = 2
            """
        ).fetchone()
        assert future_revision is not None
        assert future_revision["known_at"] > p1["evaluation_knowledge_cutoff"]
        assert future_revision["outcome_row_hash"] not in {
            row["outcome_row_hash"] for row in rows.values()
        }

        # Same timestamp and revision are resolved by declared lineage keys.
        p2 = rows["P2"]
        assert p2["outcome_source_id"] == "vendor-b"
        assert p2["outcome_row_hash"] == "o-hash-p2-b"
        assert all(
            row["known_at"] <= row["evaluation_knowledge_cutoff"]
            for row in rows.values()
        )
        assert all(
            row["decision_time"] <= row["evaluation_knowledge_cutoff"]
            for row in rows.values()
        )
        assert all(
            row["model_version"]
            and row["forecast_source_id"]
            and row["forecast_row_hash"]
            and row["outcome_source_id"]
            and row["outcome_row_hash"]
            for row in rows.values()
        )
    finally:
        connection.close()


def test_research_selection_audit_has_positive_and_negative_fixtures() -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(read_sql("research_selection_audit.sql"))
        rows = {
            row["family_id"]: dict(row)
            for row in connection.execute(
                "SELECT * FROM selection_audit ORDER BY family_id"
            )
        }

        passed = rows["F_PASS"]
        assert passed["audit_status"] == "PASS"
        assert passed["audit_reason"] == "COMPLETE_SINGLE_SELECTION"
        assert passed["visible_trial_count"] == passed["declared_trial_count"] == 3
        assert passed["selected_count"] == 1
        assert passed["missing_score_count"] == 0
        assert passed["selected_trial_id"] == "T2"
        assert passed["selected_code_version"] == "git-a1"
        assert passed["selected_trial_row_hash"] == "hash-t2"

        expected_rejections = {
            "F_NONE_SELECTED": "NO_SELECTED_TRIAL",
            "F_MULTI_SELECTED": "MULTIPLE_SELECTED_TRIALS",
            "F_MISSING_SCORE": "MISSING_VISIBLE_SCORE",
            "F_INCOMPLETE_AS_OF": "INCOMPLETE_AS_OF_CUTOFF",
        }
        for family_id, reason in expected_rejections.items():
            assert rows[family_id]["audit_status"] == "REJECT"
            assert rows[family_id]["audit_reason"] == reason

        incomplete = rows["F_INCOMPLETE_AS_OF"]
        assert incomplete["visible_trial_count"] == 1
        assert incomplete["declared_trial_count"] == 2
        assert incomplete["future_knowledge_count"] == 1
    finally:
        connection.close()


def main() -> int:
    test_point_in_time_outcome_rejects_future_revisions()
    test_research_selection_audit_has_positive_and_negative_fixtures()
    print(
        "PASS dialect=SQLite3 files=2 "
        "pit_future_revision=REJECT selection_fixtures=5"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
