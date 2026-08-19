#!/usr/bin/env python3
"""Validate the declared CH08 PostgreSQL point-in-time contracts."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[4]
SQL = ROOT / "_book/sql/ch08"


def require(path: Path, *patterns: str) -> None:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("-- Dialect: PostgreSQL 16+"):
        raise AssertionError(f"{path.name}: missing PostgreSQL dialect declaration")
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is None:
            raise AssertionError(f"{path.name}: missing governed pattern {pattern!r}")
    if "SELECT\n    *" in text or "SELECT *" in text:
        raise AssertionError(f"{path.name}: wildcard output loses schema control")
    if not text.rstrip().endswith(";"):
        raise AssertionError(f"{path.name}: SQL statement must end with semicolon")


def forbid(path: Path, *patterns: str) -> None:
    text = path.read_text(encoding="utf-8")
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) is not None:
            raise AssertionError(f"{path.name}: forbidden fail-open pattern {pattern!r}")


def main() -> int:
    point_in_time = SQL / "point_in_time_feature_join.sql"
    realized = SQL / "realized_volatility.sql"
    drift = SQL / "live_drift_evidence.sql"
    require(
        point_in_time,
        r"f\.feature_name\s*=\s*p\.feature_name",
        r"f\.effective_time\s*<=\s*p\.feature_cutoff_time",
        r"f\.knowledge_time\s*<=\s*p\.decision_time",
        r"f\.effective_time DESC NULLS LAST,\s+f\.knowledge_time DESC NULLS LAST",
        r"version_id",
        r"source_id",
        r"row_hash",
        r"selection_status",
        r"TOMBSTONED",
        r"INVALID_NULL",
        r"AMBIGUOUS_REVISION",
        r"WHERE row_rank = 1",
    )
    forbid(
        point_in_time,
        r"f\.is_tombstone\s*=\s*FALSE",
        r"WHERE\s+row_rank\s*=\s*1\s+OR",
    )
    require(
        realized,
        r"trading_bar_schedule",
        r"PARTITION BY bar\.instrument_id,\s*bar\.trade_date,\s*bar\.bar_end_time",
        r"bar\.knowledge_time\s*<=\s*:decision_time",
        r"bar\.version_id DESC,\s*bar\.source_id DESC",
        r"overnight_version_id",
        r"maximum_bar_knowledge_time",
        r"MISSING_OVERNIGHT",
        r"REJECT_INCOMPLETE",
        r"contributing_bar_hashes",
    )
    forbid(
        realized,
        r"is_tombstone\s*=\s*FALSE",
        r"JOIN ranked_overnight",
    )
    require(
        drift,
        r"outcome\.label_known_time\s*<=\s*:monitoring_cutoff",
        r"score\.knowledge_time\s*<=\s*:monitoring_cutoff",
        r"pnl_run_id\s*=\s*:pnl_run_id",
        r"label_coverage",
        r"pnl_coverage",
        r"invalid_probability_count",
        r"invalid_label_count",
        r"ambiguous_revision_count",
        r"expected_prediction_keys",
        r"REJECT_DATA_QUALITY",
    )
    forbid(
        drift,
        r"WHERE\s+score\.knowledge_time[^)]*AND\s+score\.probability\s+BETWEEN",
    )
    print(
        "PASS dialect=PostgreSQL16 files=3 fail_closed_structural_contracts=3 "
        "execution=NOT_RUN_NO_POSTGRESQL_RUNTIME"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
