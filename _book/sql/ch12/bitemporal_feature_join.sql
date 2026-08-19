-- Dialect: SQLite 3.45+
-- Semantic precedence is separated from the technical duplicate winner.
WITH eligible AS (
    SELECT
        d.decision_id,
        d.instrument_id,
        d.feature_key,
        d.market_time,
        d.knowledge_time,
        f.feature_id,
        f.value,
        f.row_status,
        f.valid_from,
        f.valid_to,
        f.known_from,
        f.known_to,
        f.version_id,
        f.source_id,
        f.source_priority,
        DENSE_RANK() OVER (
            PARTITION BY d.decision_id, d.instrument_id, d.feature_key
            ORDER BY
                f.source_priority DESC,
                f.valid_from DESC,
                f.known_from DESC
        ) AS semantic_precedence,
        ROW_NUMBER() OVER (
            PARTITION BY d.decision_id, d.instrument_id, d.feature_key
            ORDER BY
                f.source_priority DESC,
                f.valid_from DESC,
                f.known_from DESC,
                f.version_id DESC,
                f.source_id,
                f.feature_id
        ) AS technical_winner
    FROM decisions AS d
    JOIN features AS f
      ON f.instrument_id = d.instrument_id
     AND f.feature_key = d.feature_key
     AND f.valid_from <= d.market_time
     AND (f.valid_to IS NULL OR d.market_time < f.valid_to)
     AND f.known_from <= d.knowledge_time
     AND (f.known_to IS NULL OR d.knowledge_time < f.known_to)
),
top_summary AS (
    SELECT
        decision_id,
        instrument_id,
        feature_key,
        COUNT(DISTINCT (
            COALESCE(row_status, '<NULL>') || char(31) ||
            COALESCE(printf('%.17g', value), '<NULL>')
        )) AS semantic_choices
    FROM eligible
    WHERE semantic_precedence = 1
    GROUP BY decision_id, instrument_id, feature_key
),
winner AS (
    SELECT * FROM eligible WHERE technical_winner = 1
)
SELECT
    d.decision_id,
    d.instrument_id,
    d.feature_key,
    w.value,
    w.valid_from,
    w.valid_to,
    w.known_from,
    w.known_to,
    w.version_id,
    w.source_id,
    w.source_priority,
    CASE
        WHEN w.feature_id IS NULL THEN 'MISSING'
        WHEN s.semantic_choices > 1 THEN 'AMBIGUOUS'
        WHEN w.row_status = 'TOMBSTONE' THEN 'WITHDRAWN'
        WHEN w.row_status <> 'ACTIVE' OR w.row_status IS NULL THEN 'INVALID_STATUS'
        WHEN w.value IS NULL THEN 'INVALID_VALUE'
        ELSE 'SELECTED'
    END AS selection_status
FROM decisions AS d
LEFT JOIN winner AS w
  ON w.decision_id = d.decision_id
 AND w.instrument_id = d.instrument_id
 AND w.feature_key = d.feature_key
LEFT JOIN top_summary AS s
  ON s.decision_id = d.decision_id
 AND s.instrument_id = d.instrument_id
 AND s.feature_key = d.feature_key
ORDER BY d.decision_id, d.instrument_id, d.feature_key;
