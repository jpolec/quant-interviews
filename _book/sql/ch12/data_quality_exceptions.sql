-- Dialect: SQLite 3.45+
-- PIT selection occurs before freshness/missing checks; one row is emitted per issue.
WITH future_rows AS (
    SELECT o.*, 'future_knowledge' AS issue
    FROM observations AS o
    WHERE o.known_at > :decision_time
    UNION ALL
    SELECT o.*, 'future_observation' AS issue
    FROM observations AS o
    WHERE o.known_at <= :decision_time
      AND o.effective_at > :decision_time
),
eligible_ranked AS (
    SELECT
        o.*,
        ROW_NUMBER() OVER (
            PARTITION BY o.instrument_id, o.field_name
            ORDER BY
                o.effective_at DESC,
                o.known_at DESC,
                o.version_id DESC,
                o.source_id,
                o.observation_id
        ) AS rn
    FROM observations AS o
    WHERE o.known_at <= :decision_time
      AND o.effective_at <= :decision_time
),
selected AS (
    SELECT
        observation_id,
        instrument_id,
        field_name,
        effective_at,
        known_at,
        version_id,
        source_id,
        row_status,
        value
    FROM eligible_ranked
    WHERE rn = 1
),
selected_issues AS (
    SELECT s.*, 'missing' AS issue
    FROM selected AS s
    WHERE s.row_status = 'ACTIVE' AND s.value IS NULL
    UNION ALL
    SELECT s.*, 'non_finite' AS issue
    FROM selected AS s
    WHERE s.row_status = 'ACTIVE'
      AND s.value IS NOT NULL
      AND (typeof(s.value) NOT IN ('integer','real') OR ABS(s.value) > 1.0e308)
    UNION ALL
    SELECT s.*, 'inactive_has_value' AS issue
    FROM selected AS s
    WHERE s.row_status <> 'ACTIVE' AND s.value IS NOT NULL
    UNION ALL
    SELECT s.*, 'invalid_row_status' AS issue
    FROM selected AS s
    WHERE s.row_status NOT IN ('ACTIVE','TOMBSTONE') OR s.row_status IS NULL
    UNION ALL
    SELECT s.*, 'stale' AS issue
    FROM selected AS s
    WHERE s.row_status = 'ACTIVE'
      AND s.effective_at < :stale_cutoff
),
duplicates AS (
    SELECT o.*, 'duplicate_version' AS issue
    FROM observations AS o
    JOIN (
        SELECT
            instrument_id, field_name, effective_at, known_at, version_id, source_id
        FROM observations
        GROUP BY
            instrument_id, field_name, effective_at, known_at, version_id, source_id
        HAVING COUNT(*) > 1
    ) AS d
      ON d.instrument_id = o.instrument_id
     AND d.field_name = o.field_name
     AND d.effective_at = o.effective_at
     AND d.known_at = o.known_at
     AND d.version_id = o.version_id
     AND d.source_id = o.source_id
),
all_issues AS (
    SELECT * FROM future_rows
    UNION ALL
    SELECT * FROM selected_issues
    UNION ALL
    SELECT * FROM duplicates
)
SELECT
    observation_id,
    instrument_id,
    field_name,
    effective_at,
    known_at,
    version_id,
    source_id,
    value,
    issue
FROM all_issues
ORDER BY instrument_id, field_name, effective_at, known_at, issue, observation_id;
