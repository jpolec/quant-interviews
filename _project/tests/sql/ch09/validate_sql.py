from __future__ import annotations

import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SQL_DIR = REPO_ROOT / "_book" / "sql" / "ch09"

connection = sqlite3.connect(":memory:")
connection.row_factory = sqlite3.Row
cursor = connection.cursor()
cursor.executescript(
    """
    CREATE TABLE positions_bitemporal(
      portfolio_id TEXT,
      instrument_id TEXT,
      canonical_instrument_id TEXT,
      currency TEXT,
      quantity REAL,
      market_value REAL,
      valid_time TEXT,
      knowledge_time TEXT,
      revision_id INTEGER
    );
    INSERT INTO positions_bitemporal VALUES
      ('P1','A','CAN-A','USD',8,800,
       '2026-07-23T10:00:00','2026-07-23T10:01:00',1),
      ('P1','A','CAN-A','USD',10,1000,
       '2026-07-24T10:00:00','2026-07-24T10:01:00',2),
      ('P1','A','CAN-A','USD',11,1100,
       '2026-07-24T10:00:00','2026-07-24T11:00:00',3),
      ('P1','B','CAN-B','EUR',5,500,
       '2026-07-24T09:00:00','2026-07-24T09:05:00',1);

    CREATE TABLE scenario_run_positions(
      run_id TEXT,
      portfolio_id TEXT,
      instrument_id TEXT,
      market_value REAL
    );
    CREATE TABLE scenario_results(
      run_id TEXT,
      portfolio_id TEXT,
      instrument_id TEXT,
      repricing_status TEXT
    );
    INSERT INTO scenario_run_positions VALUES
      ('R1','P1','A',900),
      ('R1','P1','B',100),
      ('R2','P1','A',500),
      ('R2','P1','B',NULL);
    INSERT INTO scenario_results VALUES
      ('R1','P1','A','SUCCESS'),
      ('R2','P1','A','SUCCESS'),
      ('R2','P1','B','SUCCESS');

    CREATE TABLE pnl_components(
      portfolio_id TEXT,
      business_date TEXT,
      base_currency TEXT,
      extract_basis TEXT,
      valuation_timestamp TEXT,
      market_pnl REAL,
      trading_pnl REAL,
      carry_pnl REAL,
      cash_flow_pnl REAL,
      financing_pnl REAL,
      fx_translation_pnl REAL,
      fees_pnl REAL
    );
    INSERT INTO pnl_components VALUES
      ('P1','2026-07-24','USD','OFFICIAL','2026-07-24T17:00:00',
       90,2,4,3,1,2,-1),
      ('P1','2026-07-24','USD','OFFICIAL','2026-07-24T17:00:00',
       10,0,1,0,0,0,-1);

    CREATE TABLE nav_pnl(
      portfolio_id TEXT,
      business_date TEXT,
      base_currency TEXT,
      extract_basis TEXT,
      valuation_timestamp TEXT,
      nav_version INTEGER,
      nav REAL,
      total_pnl REAL
    );
    INSERT INTO nav_pnl VALUES
      ('P1','2026-07-24','USD','OFFICIAL','2026-07-24T17:00:00',
       3,100000,111),
      ('P2','2026-07-24','EUR','OFFICIAL','2026-07-24T17:00:00',
       1,50000,5);
    """
)

output: dict[str, object] = {}

point_in_time_query = (
    SQL_DIR / "point_in_time_portfolio_state.sql"
).read_text(encoding="utf-8")
point_in_time_rows = [
    dict(row)
    for row in cursor.execute(
        point_in_time_query,
        {
            "decision_time": "2026-07-24T10:30:00",
            "knowledge_cutoff": "2026-07-24T10:30:00",
        },
    )
]
assert len(point_in_time_rows) == 2
asset_a = next(
    row for row in point_in_time_rows if row["instrument_id"] == "A"
)
assert asset_a["quantity"] == 10
assert asset_a["valid_time"] == "2026-07-24T10:00:00"
assert all(
    row["identity_currency_status"] == "PASS"
    for row in point_in_time_rows
)
output["point_in_time_portfolio_state.sql"] = point_in_time_rows

coverage_query = (SQL_DIR / "scenario_coverage.sql").read_text(
    encoding="utf-8"
)
coverage_rows = [
    dict(row)
    for row in cursor.execute(
        coverage_query,
        {"minimum_coverage": 0.99},
    )
]
run_one = next(row for row in coverage_rows if row["run_id"] == "R1")
run_two = next(row for row in coverage_rows if row["run_id"] == "R2")
assert run_one["market_value_coverage"] == 0.9
assert run_one["failed_instruments"] == 1
assert run_one["publication_status"] == "REJECT"
assert run_two["null_market_values"] == 1
assert run_two["publication_status"] == "INCOMPLETE"
output["scenario_coverage.sql"] = coverage_rows

attribution_query = (
    SQL_DIR / "attribution_reconciliation.sql"
).read_text(encoding="utf-8")
attribution_rows = [
    dict(row)
    for row in cursor.execute(
        attribution_query,
        {
            "absolute_tolerance": 1e-12,
            "nav_tolerance": 1e-12,
            "gross_tolerance": 1e-12,
        },
    )
]
portfolio_one = next(
    row for row in attribution_rows if row["portfolio_id"] == "P1"
)
portfolio_two = next(
    row for row in attribution_rows if row["portfolio_id"] == "P2"
)
assert abs(portfolio_one["residual"]) < 1e-12
assert portfolio_one["publication_status"] == "PASS"
assert portfolio_two["component_rows"] is None
assert portfolio_two["publication_status"] == "REJECT"
output["attribution_reconciliation.sql"] = attribution_rows

print(json.dumps({"status": "PASS", "queries": output}, indent=2))
