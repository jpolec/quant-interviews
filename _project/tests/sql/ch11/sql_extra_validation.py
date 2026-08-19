from __future__ import annotations

import hashlib
import math
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
    rows: list[tuple[Any, ...]] = []
    for statement in (
        part.strip()
        for part in (SQL / "idempotent_event_ingestion.sql")
        .read_text(encoding="utf-8")
        .split(";")
        if part.strip()
    ):
        cursor = connection.execute(statement, params)
        if cursor.description is not None:
            rows = cursor.fetchall()
    return rows


passed: list[str] = []
HIGH_DYNAMIC_ZSCORE_ABS_TOLERANCE = 1.0e-8

# A configured traversal budget produces a named blocking result, never silence.
rows = query(
    "dependency_cycle_control.sql",
    """
    CREATE TABLE dependencies(parent TEXT, child TEXT);
    INSERT INTO dependencies VALUES ('n0','n1'),('n1','n2'),('n2','n3'),('n3','n4');
    """,
    {"max_depth": 2},
)
assert rows and {row[-1] for row in rows} == {"REJECT_DEPTH_LIMIT"}
invalid_limit = query(
    "dependency_cycle_control.sql",
    "CREATE TABLE dependencies(parent TEXT, child TEXT);",
    {"max_depth": 0},
)
assert invalid_limit == [(None, None, "[]", 0, "REJECT_INVALID_MAX_DEPTH")]
passed.append("depth_limit_is_named")

# Exact-positive policy keeps a tiny positive denominator and stable centered variance.
rows = query(
    "grouped_calculations.sql",
    """
    CREATE TABLE prices(asset TEXT, timestamp TEXT, price REAL);
    INSERT INTO prices VALUES
      ('A','t0',1.0e-9),('A','t1',1.0),
      ('A','t2',70000001.5),('A','t3',4900000245000002.0);
    """,
)
assert rows[1][-1] == "PASS" and abs(rows[1][4] - 999999999.0) < 1e-6
assert rows[-1][5] is not None
high_dynamic_returns = [float(row[4]) for row in rows[1:]]
high_dynamic_mean = math.fsum(high_dynamic_returns) / len(high_dynamic_returns)
high_dynamic_variance = math.fsum(
    (value - high_dynamic_mean) ** 2 for value in high_dynamic_returns
) / (len(high_dynamic_returns) - 1)
expected_high_dynamic_zscore = (
    high_dynamic_returns[-1] - high_dynamic_mean
) / math.sqrt(high_dynamic_variance)
assert abs(rows[-1][5] - expected_high_dynamic_zscore) <= (
    HIGH_DYNAMIC_ZSCORE_ABS_TOLERANCE
)
duplicate_rows = query(
    "grouped_calculations.sql",
    """
    CREATE TABLE prices(asset TEXT, timestamp TEXT, price REAL);
    INSERT INTO prices VALUES ('A','t0',1.0),('A','t0',2.0),('A','t1',3.0);
    """,
)
assert all(row[-1] == "REJECT_DUPLICATE_KEY" for row in duplicate_rows)
assert all(row[4] is None and row[5] is None for row in duplicate_rows)
passed.append("stable_zscore_abs_tol_1e-8_and_exact_positive_domain")

# Duplicate decisions are preserved as two rejects, not collapsed to one output.
pit_setup = """
CREATE TABLE decisions(
  decision_row_id TEXT, decision_id TEXT, instrument_id TEXT,
  effective_at TEXT, knowledge_at TEXT
);
CREATE TABLE market_data(
  instrument_id TEXT, effective_at TEXT, known_at TEXT, version INTEGER,
  value REAL, source_id TEXT, row_hash TEXT
);
INSERT INTO decisions VALUES
  ('dr1','d1','X','2026-01-02','2026-01-02'),
  ('dr2','d1','X','2026-01-02','2026-01-02');
INSERT INTO market_data VALUES
  ('X','2026-01-01','2026-01-01',1,10.0,'feed','h1');
"""
rows = query("point_in_time_asof.sql", pit_setup)
assert len(rows) == 2
assert {row[-1] for row in rows} == {"REJECT_DUPLICATE_DECISION"}
passed.append("pit_duplicate_decisions_preserved")

# A duplicate full version blocks publication; NULL and missing lineage are distinct.
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
      ('dr1','d1','X','2026-01-02','2026-01-02'),
      ('dr2','d2','Y','2026-01-02','2026-01-02'),
      ('dr3','d3','Z','2026-01-02','2026-01-02');
    INSERT INTO market_data VALUES
      ('X','2026-01-01','2026-01-01',1,10.0,'feed','h1'),
      ('X','2026-01-01','2026-01-01',1,11.0,'feed','h2'),
      ('Y','2026-01-01','2026-01-01',1,NULL,'feed','hy'),
      ('Z','2026-01-01','2026-01-01',1,12.0,'feed','');
    """,
)
statuses = {row[1]: row[-1] for row in rows}
assert statuses == {
    "d1": "REJECT_DUPLICATE_VERSION",
    "d2": "REJECT_INVALID_VALUE",
    "d3": "REJECT_MISSING_LINEAGE",
}, statuses
assert all(row[5] is None for row in rows)
passed.append("pit_duplicate_null_lineage_gates")

# Feature NULL cannot be reported as PASS.
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
    INSERT INTO decisions VALUES ('dr1','d1','X','2026-01-02','2026-01-02');
    INSERT INTO feature_versions VALUES
      ('X','momentum',NULL,'2026-01-01','2026-01-01',1,'calc','h1');
    """,
)
assert rows[0][-1] == "REJECT_INVALID_VALUE" and rows[0][6] is None
passed.append("feature_null_fails_closed")

# Bitemporal duplicate version is surfaced despite a deterministic physical tie-break.
rows = query(
    "bitemporal_asof.sql",
    """
    CREATE TABLE instrument_values(
      instrument_id TEXT, valid_from TEXT, valid_to TEXT,
      knowledge_from TEXT, knowledge_to TEXT, value REAL,
      version_id INTEGER, source_id TEXT, row_hash TEXT, source_priority INTEGER
    );
    INSERT INTO instrument_values VALUES
      ('X','2026-01-01',NULL,'2026-01-01',NULL,10.0,1,'feed','h1',1),
      ('X','2026-01-01',NULL,'2026-01-01',NULL,11.0,1,'feed','h2',1);
    """,
    {"decision_time": "2026-01-02", "knowledge_cutoff": "2026-01-02"},
)
assert rows[0][1] is None and rows[0][-1] == "REJECT_DUPLICATE_VERSION"
passed.append("bitemporal_duplicate_version_gate")


def window_result(values: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(
            """CREATE TABLE prices(
            timestamp TEXT, asset TEXT, sequence_number INTEGER,
            version_id INTEGER, row_id TEXT, row_hash TEXT, price REAL)"""
        )
        connection.executemany("INSERT INTO prices VALUES (?,?,?,?,?,?,?)", values)
        return connection.execute(
            (SQL / "window_returns_ranks.sql").read_text(encoding="utf-8")
        ).fetchall()
    finally:
        connection.close()


ordered_values = [
    ("2026-01-01", "A", 1, 1, "r1", "h1", 100.0),
    ("2026-01-02", "A", 2, 1, "r2", "h2", 101.0),
    ("2026-01-02", "A", 3, 1, "r3", "h3", 102.0),
    ("2026-01-03", "A", 4, 1, "r4", "h4", 103.0),
]
assert window_result(ordered_values) == window_result(list(reversed(ordered_values)))
duplicate_revision = window_result(ordered_values + [
    ("2026-01-02", "A", 2, 1, "r2b", "h2b", 999.0)
])
assert all(row[-1] == "REJECT_DUPLICATE_REVISION" for row in duplicate_revision)
assert all(row[7] is None for row in duplicate_revision)
passed.append("window_total_order_shuffle_and_revision_gate")

# A late correction replaces an observation before the 20-row window is formed.
connection = sqlite3.connect(":memory:")
try:
    connection.execute(
        """CREATE TABLE daily_observations(
        instrument_id TEXT, observation_time TEXT, sequence_number INTEGER,
        version_id INTEGER, knowledge_time TEXT, source_id TEXT, row_hash TEXT,
        return_1d REAL, close_price REAL)"""
    )
    observations = [
        (
            "A", f"2026-01-{sequence:02d}", sequence, 1,
            f"2026-01-{sequence:02d}T12:00:00Z", "calc", f"h{sequence}",
            sequence / 1000.0, 100.0 + sequence,
        )
        for sequence in range(1, 22)
    ]
    observations.append(
        ("A", "2026-01-02", 2, 2, "2026-01-22T12:00:00Z", "calc", "h2-v2", 0.5, 102.0)
    )
    connection.executemany(
        "INSERT INTO daily_observations VALUES (?,?,?,?,?,?,?,?,?)", observations
    )
    rows = connection.execute(
        (SQL / "windowed_signal_features.sql").read_text(encoding="utf-8"),
        {"knowledge_cutoff": "2026-01-23T00:00:00Z"},
    ).fetchall()
    assert len(rows) == 21
    corrected = next(row for row in rows if row[2] == 2)
    assert corrected[3] == 2 and corrected[7] == 0.5
    assert all(row[-1] == "PASS" for row in rows)
finally:
    connection.close()
passed.append("window_feature_revision_selected_before_window")

# Position reconstruction emits one blocking status for duplicate sequence and gaps.
base_position_schema = """
CREATE TABLE position_events(
  event_id TEXT, account_id TEXT, instrument_id TEXT, event_time TEXT,
  knowledge_time TEXT, version_id INTEGER, sequence_number INTEGER,
  quantity_delta REAL, source_id TEXT, row_hash TEXT
);
"""
rows = query(
    "temporal_position_reconstruction.sql",
    base_position_schema + """
    INSERT INTO position_events VALUES
      ('e1','a','X','t1','k1',1,1,1.0,'oms','h1'),
      ('e2','a','X','t2','k2',1,2,1.0,'oms','h2'),
      ('e3','a','X','t3','k3',1,2,1.0,'oms','h3');
    """,
    {"as_of_time": "z", "knowledge_cutoff": "z"},
)
assert rows == [("a", "X", None, 2, "REJECT_DUPLICATE_SEQUENCE")]
gap_rows = query(
    "temporal_position_reconstruction.sql",
    base_position_schema + """
    INSERT INTO position_events VALUES
      ('e1','a','X','t1','k1',1,1,1.0,'oms','h1'),
      ('e2','a','X','t2','k2',1,3,1.0,'oms','h2');
    """,
    {"as_of_time": "z", "knowledge_cutoff": "z"},
)
assert gap_rows == [("a", "X", None, 3, "REJECT_SEQUENCE_GAP")]
passed.append("position_duplicate_and_gap_fail_closed")

# Reconciliation preserves NULL expected values and uses only PIT-eligible revisions.
reconciliation_schema = """
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
"""
rows = query(
    "window_reconciliation.sql",
    reconciliation_schema + """
    INSERT INTO cash_events VALUES
      ('e1','a','t1','k1',1,100,'USD','ledger','eh1');
    INSERT INTO account_snapshots VALUES
      ('a','t1','k1',1,NULL,'USD','custodian','sh1');
    """,
    {"knowledge_cutoff": "z", "tolerance_minor": 0, "tolerance_version": "v1"},
)
assert rows[0][-1] == "REJECT_MISSING_VALUE" and rows[0][5] is None
pit_rows = query(
    "window_reconciliation.sql",
    reconciliation_schema + """
    INSERT INTO cash_events VALUES
      ('e1','a','t1','k1',1,100,'USD','ledger','eh1');
    INSERT INTO account_snapshots VALUES
      ('a','t1','k1',1,100,'USD','custodian','sh1'),
      ('a','t1','z9',2,999,'USD','custodian','sh2');
    """,
    {"knowledge_cutoff": "k9", "tolerance_minor": 0, "tolerance_version": "v1"},
)
assert pit_rows[0][3] == 100 and pit_rows[0][-1] == "PASS"
passed.append("reconciliation_null_and_pit_revision")

# Malformed payload/registry values and hostile field names are classified, not executed as paths.
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
      ('bad_payload',1,0,'CURRENT','["x"]','h1'),
      ('bad_registry',1,0,'CURRENT','[','h2'),
      ('scalar_registry',1,0,'CURRENT','42','h3'),
      ('hostile_key',1,0,'CURRENT','["x\\\".[0]"]','h4');
    INSERT INTO event_log VALUES
      ('e1','bad_payload',1,0,'{','t1'),
      ('e2','bad_registry',1,0,'{}','t2'),
      ('e3','scalar_registry',1,0,'{}','t3'),
      ('e4','hostile_key',1,0,'{}','t4');
    """,
)
statuses = {row[0]: row[-1] for row in rows}
assert statuses == {
    "e1": "REJECT_INVALID_JSON",
    "e2": "REJECT_INVALID_REGISTRY",
    "e3": "REJECT_INVALID_REGISTRY",
    "e4": "REJECT_MISSING_REQUIRED_FIELD",
}, statuses
passed.append("malformed_json_is_classified")

# Ingestion computes its own digest and stores fixed-point integer effects.
connection = sqlite3.connect(":memory:")
try:
    base = {
        "consumer_name": "ledger",
        "message_id": "m1",
        "account_id": "a1",
        "amount_minor": 125,
        "currency": "USD",
        "scale": 2,
        "schema_version": 1,
    }
    assert execute_ingestion(connection, base)[0][0] == "NEWLY_PROCESSED"
    assert execute_ingestion(connection, {**base, "amount_minor": 126})[0][0] == (
        "REJECT_ID_REUSED_WITH_DIFFERENT_PAYLOAD"
    )
    invalid = execute_ingestion(
        connection, {**base, "message_id": "m2", "currency": "usd"}
    )
    assert invalid[0][0] == "REJECT_INVALID_INPUT"
    stored = connection.execute(
        "SELECT amount_minor, typeof(amount_minor), payload_hash "
        "FROM business_effects JOIN processed_messages USING (consumer_name, message_id)"
    ).fetchall()
    assert stored[0][0:2] == (125, "integer") and len(stored[0][2]) == 64
    assert connection.execute("SELECT COUNT(*) FROM business_effects").fetchone()[0] == 1
finally:
    connection.close()
passed.append("fixed_point_ingestion_hash_boundary")

# The plan gate checks the complete order key and rejects the known partial-sort shape.
connection = sqlite3.connect(":memory:")
try:
    connection.executescript(
        """
        CREATE TABLE prices(
          symbol TEXT, valid_time TEXT, knowledge_time TEXT, version INTEGER,
          source_id TEXT, row_hash TEXT, price REAL
        );
        """
    )
    connection.executemany(
        "INSERT INTO prices VALUES (?,?,?,?,?,?,?)",
        [
            (f"S{i % 7}", f"2026-01-{i % 28 + 1:02d}", "2026-02-01", i,
             "feed", f"h{i:04d}", 100.0 + i)
            for i in range(500)
        ],
    )
    statements = [
        part.strip()
        for part in (SQL / "query_plan_indexes.sql").read_text(encoding="utf-8").split(";")
        if part.strip()
    ]
    connection.execute(statements[0])
    plan = connection.execute(
        statements[1],
        {"valid_cutoff": "2026-12-31", "knowledge_cutoff": "2026-12-31"},
    ).fetchall()
    plan_text = "\n".join(str(row[-1]) for row in plan)
    assert "idx_prices_symbol_valid_knowledge_version" in plan_text
    assert "LAST TERM" not in plan_text
finally:
    connection.close()
passed.append("query_plan_shape_gate")

for name in passed:
    print(f"PASS {name}")
print(f"TOTAL_PASS={len(passed)}")
