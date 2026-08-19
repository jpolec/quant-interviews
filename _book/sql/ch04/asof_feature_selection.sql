-- Dialect: SQLite 3.45
CREATE TABLE feature_value (
    feature_id TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    published_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    known_at TEXT NOT NULL,
    revision_id INTEGER NOT NULL,
    superseded_at TEXT,
    source_priority INTEGER NOT NULL,
    payload_hash TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (feature_id, valid_from, known_at, revision_id)
);

INSERT INTO feature_value VALUES
('eps','2025-01-01','2026-01-01','2026-01-20T07:55:00Z','2026-01-20T08:00:00Z','2026-01-20T08:00:00Z',1,'2026-02-15T08:00:00Z',10,'03d15aae003babe206af89e333552f94d7d2d42fab311bec66b6e1e7cb363309',4.20),
('eps','2025-01-01','2026-01-01','2026-02-15T07:55:00Z','2026-02-15T08:00:00Z','2026-02-15T08:00:00Z',2,NULL,10,'0e7ae77949700b46c67e53adfc29646a1e7c7a8f1d7dc2d6db96cb0cc2af5e57',3.90),
('volatility','2025-01-01','2026-01-01','2026-01-20T07:58:00Z','2026-01-20T08:02:00Z','2026-01-20T08:02:00Z',1,NULL,10,'dfd74760d0ba9a1bcec6a40f1ccf43f9593cf192d7ff4ea97991e9b85e7ca7a1',0.20),
('bad-known','2025-01-01','2026-01-01','2026-01-20T08:00:00Z','2026-01-20T08:05:00Z','2026-01-20T08:00:00Z',1,NULL,10,'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',1.0),
('expired','2024-01-01','2025-01-01','2024-12-01T08:00:00Z','2024-12-01T08:01:00Z','2024-12-01T08:01:00Z',1,NULL,10,'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',2.0);

CREATE TABLE decision_context (
    valuation_time TEXT NOT NULL,
    as_of_knowledge_time TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    schema_version TEXT NOT NULL
);

INSERT INTO decision_context VALUES
('2025-12-31T16:00:00Z','2026-02-01T09:00:00Z','ch04-snapshot-001','1.0');

CREATE TABLE selected_feature AS
WITH validated AS (
    SELECT *
    FROM feature_value
    WHERE (valid_to IS NULL OR valid_from < valid_to)
      AND published_at <= ingested_at
      AND known_at = MAX(published_at, ingested_at)
      AND LENGTH(payload_hash) = 64
      AND payload_hash NOT GLOB '*[^0-9a-f]*'
),
eligible AS (
    SELECT
        f.*,
        ROW_NUMBER() OVER (
            PARTITION BY f.feature_id
            ORDER BY
                f.valid_from DESC,
                f.known_at DESC,
                f.revision_id DESC,
                f.source_priority DESC,
                f.payload_hash DESC
        ) AS selection_rank
    FROM validated AS f
    CROSS JOIN decision_context AS d
    WHERE f.valid_from <= d.valuation_time
      AND (f.valid_to IS NULL OR d.valuation_time < f.valid_to)
      AND f.known_at <= d.as_of_knowledge_time
      AND (
          f.superseded_at IS NULL
          OR d.as_of_knowledge_time < f.superseded_at
      )
)
SELECT *
FROM eligible
WHERE selection_rank = 1;

CREATE TABLE feature_snapshot_manifest AS
SELECT
    d.snapshot_id,
    d.schema_version,
    COUNT(*) AS row_count,
    MIN(s.known_at) AS minimum_known_at,
    MAX(s.known_at) AS maximum_known_at,
    GROUP_CONCAT(
        s.feature_id || ':' || s.payload_hash,
        '|'
    ) AS ordered_payload_manifest
FROM decision_context AS d
JOIN (
    SELECT *
    FROM selected_feature
    ORDER BY feature_id, payload_hash
) AS s
GROUP BY d.snapshot_id, d.schema_version;
