-- Dialect: SQLite 3.35+ deterministic lineage fixture.
CREATE TABLE surface_run (
    run_id TEXT PRIMARY KEY,
    valid_at TEXT NOT NULL,
    knowledge_cutoff TEXT NOT NULL,
    quote_snapshot_hash TEXT NOT NULL CHECK (
        length(quote_snapshot_hash) = 64
        AND quote_snapshot_hash NOT GLOB '*[^0-9a-f]*'
    ),
    forward_curve_hash TEXT NOT NULL CHECK (
        length(forward_curve_hash) = 64
        AND forward_curve_hash NOT GLOB '*[^0-9a-f]*'
    ),
    dividend_version TEXT NOT NULL,
    solver_version TEXT NOT NULL,
    repair_policy_version TEXT NOT NULL,
    repair_owner TEXT,
    minimum_repair REAL NOT NULL CHECK (minimum_repair >= 0.0),
    maximum_signed_repair REAL NOT NULL CHECK (maximum_signed_repair >= 0.0),
    status TEXT NOT NULL CHECK (status IN ('PASS', 'CONDITIONAL_PASS', 'REJECT')),
    CHECK (minimum_repair <= maximum_signed_repair),
    CHECK (minimum_repair = 0.0 OR repair_owner IS NOT NULL)
);

INSERT INTO surface_run VALUES (
    'run-1',
    '2026-07-25T10:00:00.000000Z',
    '2026-07-25T10:00:05.000000Z',
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
    'dividend-v3', 'surface-lp-v0.10', 'surface-repair-policy-v3',
    'market-data-control', 0.0, 0.25, 'PASS'
);

CREATE TABLE surface_audit AS
SELECT
    run_id,
    0 AS missing_lineage,
    0 AS unsigned_repair,
    0 AS invalid_status,
    quote_snapshot_hash,
    forward_curve_hash,
    status
FROM surface_run;
