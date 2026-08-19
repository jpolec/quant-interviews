-- Dialect: SQLite 3.45+
-- One typed certificate is returned even when no mapping is eligible.
WITH request AS (
    SELECT
        :id_type AS id_type,
        :id_value AS id_value,
        :valid_at AS valid_at,
        :known_at AS known_at
),
eligible AS (
    SELECT
        i.*,
        DENSE_RANK() OVER (
            ORDER BY
                i.source_priority DESC,
                i.valid_from DESC,
                i.known_from DESC
        ) AS semantic_precedence,
        ROW_NUMBER() OVER (
            ORDER BY
                i.source_priority DESC,
                i.valid_from DESC,
                i.known_from DESC,
                i.version_id DESC,
                i.source_id,
                i.mapping_id
        ) AS technical_winner
    FROM instrument_identity AS i
    JOIN request AS r
      ON i.id_type = r.id_type
     AND i.id_value = r.id_value
     AND i.valid_from <= r.valid_at
     AND (i.valid_to IS NULL OR r.valid_at < i.valid_to)
     AND i.known_from <= r.known_at
     AND (i.known_to IS NULL OR r.known_at < i.known_to)
),
top_summary AS (
    SELECT COUNT(DISTINCT instrument_id) AS semantic_choices
    FROM eligible
    WHERE semantic_precedence = 1
),
winner AS (
    SELECT *
    FROM eligible
    WHERE technical_winner = 1
)
SELECT
    w.instrument_id,
    w.mapping_id,
    w.version_id,
    w.source_id,
    w.source_priority,
    w.valid_from,
    w.valid_to,
    w.known_from,
    w.known_to,
    w.override_id,
    CASE
        WHEN w.mapping_id IS NULL THEN 'MISSING'
        WHEN s.semantic_choices > 1 THEN 'AMBIGUOUS'
        ELSE 'RESOLVED'
    END AS resolution_status
FROM request AS r
CROSS JOIN top_summary AS s
LEFT JOIN winner AS w ON TRUE;
