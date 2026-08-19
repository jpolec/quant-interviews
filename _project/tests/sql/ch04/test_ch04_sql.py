from __future__ import annotations

from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[4]


def execute_script(relative: str) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript((ROOT / relative).read_text(encoding="utf-8"))
    return connection


def test_asof_feature_snapshot() -> None:
    connection = execute_script("_book/sql/ch04/asof_feature_selection.sql")
    rows = connection.execute(
        """
        SELECT feature_id, value, revision_id
        FROM selected_feature
        ORDER BY feature_id
        """
    ).fetchall()
    assert rows == [("eps", 4.2, 1), ("volatility", 0.2, 1)]
    manifest = connection.execute(
        """
        SELECT snapshot_id, row_count, minimum_known_at, maximum_known_at
        FROM feature_snapshot_manifest
        """
    ).fetchone()
    assert manifest == (
        "ch04-snapshot-001",
        2,
        "2026-01-20T08:00:00Z",
        "2026-01-20T08:02:00Z",
    )


def test_asof_rejects_inconsistent_knowledge_and_expired_validity() -> None:
    connection = execute_script("_book/sql/ch04/asof_feature_selection.sql")
    inconsistent = connection.execute(
        """
        SELECT COUNT(*) FROM feature_value
        WHERE known_at <> MAX(published_at, ingested_at)
        """
    ).fetchone()[0]
    assert inconsistent == 1
    invalid_selected = connection.execute(
        """
        SELECT COUNT(*) FROM selected_feature
        WHERE feature_id IN ('bad-known', 'expired')
        """
    ).fetchone()[0]
    assert invalid_selected == 0


def test_correlation_structural_audit() -> None:
    connection = execute_script("_book/sql/ch04/correlation_matrix_audit.sql")
    rows = dict(
        connection.execute(
            """
            SELECT matrix_id, structural_status
            FROM correlation_matrix_audit
            """
        ).fetchall()
    )
    assert rows == {
        "BAD3": "REJECT_UNSUPPORTED_DIMENSION",
        "BAD4": "REJECT_UNSUPPORTED_DIMENSION",
        "INVALID": "REJECT_STRUCTURAL",
        "SINGULAR": "PASS_SINGULAR_PSD_2X2",
        "SPD": "PASS_SPD_2X2",
    }


if __name__ == "__main__":
    test_asof_feature_snapshot()
    test_correlation_structural_audit()
    print("PASS ch04 deterministic SQL fixtures")
