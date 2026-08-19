-- Dialect: SQLite 3.45+
-- Emits ACTIVE, WITHDRAWN, AMBIGUOUS and EMPTY certificates per decision.
WITH eligible AS (
    SELECT
        d.decision_id,
        d.decision_date,
        d.knowledge_time,
        m.membership_id,
        m.instrument_id,
        m.row_status,
        m.valid_from,
        m.valid_to,
        m.known_from,
        m.known_to,
        m.version_id,
        m.source_id,
        DENSE_RANK() OVER (
            PARTITION BY d.decision_id, m.instrument_id
            ORDER BY m.valid_from DESC, m.known_from DESC
        ) AS semantic_precedence,
        ROW_NUMBER() OVER (
            PARTITION BY d.decision_id, m.instrument_id
            ORDER BY
                m.valid_from DESC,
                m.known_from DESC,
                m.version_id DESC,
                m.source_id,
                m.membership_id
        ) AS technical_winner
    FROM decision_dates AS d
    JOIN universe_membership AS m
      ON m.valid_from <= d.decision_date
     AND (m.valid_to IS NULL OR d.decision_date < m.valid_to)
     AND m.known_from <= d.knowledge_time
     AND (m.known_to IS NULL OR d.knowledge_time < m.known_to)
),
summary AS (
    SELECT
        decision_id,
        instrument_id,
        COUNT(DISTINCT row_status) AS semantic_choices
    FROM eligible
    WHERE semantic_precedence = 1
    GROUP BY decision_id, instrument_id
),
winners AS (
    SELECT * FROM eligible WHERE technical_winner = 1
),
certificates AS (
    SELECT
        d.decision_id,
        d.decision_date,
        d.knowledge_time,
        w.instrument_id,
        w.membership_id,
        w.valid_from,
        w.valid_to,
        w.known_from,
        w.known_to,
        w.version_id,
        w.source_id,
        CASE
            WHEN w.membership_id IS NULL THEN 'EMPTY'
            WHEN s.semantic_choices > 1 THEN 'AMBIGUOUS'
            WHEN w.row_status = 'TOMBSTONE' THEN 'WITHDRAWN'
            WHEN w.row_status = 'ACTIVE' THEN 'SELECTED'
            ELSE 'INVALID_STATUS'
        END AS selection_status
    FROM decision_dates AS d
    LEFT JOIN winners AS w ON w.decision_id = d.decision_id
    LEFT JOIN summary AS s
      ON s.decision_id = w.decision_id
     AND s.instrument_id = w.instrument_id
)
SELECT *
FROM certificates
ORDER BY decision_id, instrument_id;
