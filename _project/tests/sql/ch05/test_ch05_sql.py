from __future__ import annotations

import sqlite3
from pathlib import Path

from quant_interviews.ch05.integrated_case import (
    canonical_snapshot_hash,
    chapter_release_case,
)


ROOT = Path(__file__).resolve().parents[4]


def execute_fixture(relative: str) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    sql = (ROOT / relative).read_text(encoding="utf-8")
    connection.executescript(sql)
    return connection


def main() -> int:
    test_ch05_sql_fixtures()
    print("PASS ch05 deterministic SQL fixtures")
    return 0


def test_ch05_sql_fixtures() -> None:
    parity = execute_fixture("_book/sql/ch05/option_quote_parity.sql")
    row = parity.execute(
        """
        SELECT
            call_quote_id,
            put_quote_id,
            call_revision_id,
            put_revision_id,
            ROUND(market_event_skew_seconds),
            snapshot_status
        FROM parity_snapshot
        """
    ).fetchone()
    assert row == ("call-r2", "put-r1", 2, 1, 1.0, "PASS")
    assert parity.execute("SELECT COUNT(*) FROM parity_snapshot").fetchone() == (
        1,
    )
    selected = parity.execute(
        "SELECT call_market_event_time, put_market_event_time, call_known_at, put_known_at FROM parity_snapshot"
    ).fetchone()
    assert selected == (
        "2026-07-25T10:00:00.000000Z",
        "2026-07-25T10:00:01.000000Z",
        "2026-07-25T10:00:03.000000Z",
        "2026-07-25T10:00:02.000000Z",
    )
    rows = tuple(
        {
            "quote_id": quote_id,
            "revision_id": revision_id,
            "bid": bid,
            "ask": ask,
            "market_event_time": market_event_time,
            "known_at": known_at,
            "payload_hash": payload_hash,
        }
        for (
            quote_id,
            revision_id,
            bid,
            ask,
            market_event_time,
            known_at,
            payload_hash,
        ) in parity.execute(
            """
            SELECT quote_id, revision_id, bid, ask, market_event_time,
                   known_at, payload_hash
            FROM option_quote
            WHERE quote_id IN ('call-r2', 'put-r1')
            ORDER BY quote_id
            """
        )
    )
    sql_hash = canonical_snapshot_hash(rows)
    case_hash = chapter_release_case()["snapshot"]["snapshot_sha256"]
    assert sql_hash == case_hash
    assert sql_hash == "abf1e9c1d0c9cc42a7557adb52afdfe4c15c347641d4e2bd140b5eb501132063"
    status_sql = """
        SELECT CASE
          WHEN ABS((JULIANDAY(?) - JULIANDAY(?)) * 86400.0) > 2
          THEN 'REJECT_MARKET_EVENT_SKEW'
          ELSE 'PASS'
        END
    """
    assert parity.execute(
        status_sql,
        (
            "2026-07-25T09:00:00.000000Z",
            "2026-07-25T09:10:00.000000Z",
        ),
    ).fetchone() == ("REJECT_MARKET_EVENT_SKEW",)
    assert parity.execute(
        status_sql,
        (
            "2026-07-25T10:00:00.000000Z",
            "2026-07-25T10:00:01.000000Z",
        ),
    ).fetchone() == ("PASS",)
    with __import__("pytest").raises(sqlite3.IntegrityError):
        parity.execute(
            """
            INSERT INTO option_quote VALUES (
              'crossed', 1, 1, 'XYZ', '2026-12-18', 100.0, 'C',
              10.0, 9.0, '2026-07-25T10:00:00.000000Z',
              '2026-07-25T10:00:00.000000Z', NULL,
              '2026-07-25T10:00:00.000000Z', NULL,
              'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
            )
            """
        )
    with __import__("pytest").raises(sqlite3.IntegrityError):
        parity.execute(
            """
            INSERT INTO option_quote VALUES (
              'negative', 1, 1, 'XYZ', '2026-12-18', 100.0, 'P',
              -1.0, 1.0, '2026-07-25T10:00:00.000000Z',
              '2026-07-25T10:00:00.000000Z', NULL,
              '2026-07-25T10:00:00.000000Z', NULL,
              'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
            )
            """
        )

    lineage = execute_fixture("_book/sql/ch05/surface_lineage.sql")
    assert lineage.execute(
        """
        SELECT missing_lineage, unsigned_repair, invalid_status
        FROM surface_audit
        """
    ).fetchone() == (0, 0, 0)


if __name__ == "__main__":
    raise SystemExit(main())
