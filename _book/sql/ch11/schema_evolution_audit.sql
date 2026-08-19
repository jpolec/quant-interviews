-- Dialect: SQLite 3.45 with JSON1; malformed input is data, never an exception.
WITH registry_physical AS (
    SELECT r.rowid AS registry_row_id, r.*
    FROM event_schema_registry AS r
), registry_ranked AS (
    SELECT
        *,
        COUNT(*) OVER (
            PARTITION BY event_type, schema_major, schema_minor
        ) AS registry_count,
        ROW_NUMBER() OVER (
            PARTITION BY event_type, schema_major, schema_minor
            ORDER BY registry_row_id
        ) AS registry_rank
    FROM registry_physical
), registry_selected AS (
    SELECT *
    FROM registry_ranked
    WHERE registry_rank = 1
), events_physical AS (
    SELECT
        e.rowid AS event_row_id,
        e.*,
        COUNT(*) OVER (PARTITION BY event_id) AS event_id_count
    FROM event_log AS e
), registered AS (
    SELECT
        e.*,
        r.compatibility_state,
        r.required_fields_json,
        r.semantic_schema_hash,
        r.registry_count,
        CASE
            WHEN json_valid(r.required_fields_json) AND
                 json_type(
                     CASE WHEN json_valid(r.required_fields_json)
                          THEN r.required_fields_json ELSE 'null' END
                 ) = 'array'
                THEN 1 ELSE 0
        END AS required_fields_is_array,
        CASE
            WHEN json_valid(e.payload_json) THEN e.payload_json ELSE '{}'
        END AS safe_payload,
        CASE
            WHEN json_valid(r.required_fields_json) AND
                 json_type(
                     CASE WHEN json_valid(r.required_fields_json)
                          THEN r.required_fields_json ELSE 'null' END
                 ) = 'array'
                THEN r.required_fields_json
            ELSE '[]'
        END AS safe_required_fields
    FROM events_physical AS e
    LEFT JOIN registry_selected AS r
      ON r.event_type = e.event_type
     AND r.schema_major = e.schema_major
     AND r.schema_minor = e.schema_minor
), required_field_audit AS (
    SELECT
        r.event_row_id,
        SUM(
            CASE
                WHEN required.type <> 'text' OR
                     TRIM(CAST(required.value AS TEXT)) = ''
                    THEN 1 ELSE 0
            END
        ) AS invalid_required_fields,
        SUM(
            CASE
                WHEN required.type = 'text' AND
                     NOT EXISTS (
                         SELECT 1
                         FROM json_each(r.safe_payload) AS actual
                         WHERE actual.key = CAST(required.value AS TEXT)
                     )
                    THEN 1 ELSE 0
            END
        ) AS missing_required_fields
    FROM registered AS r
    LEFT JOIN json_each(r.safe_required_fields) AS required
    GROUP BY r.event_row_id
), evaluated AS (
    SELECT
        r.*,
        COALESCE(a.invalid_required_fields, 0) AS invalid_required_fields,
        COALESCE(a.missing_required_fields, 0) AS missing_required_fields,
        CASE
            WHEN r.event_id_count <> 1 THEN 'REJECT_DUPLICATE_EVENT_ID'
            WHEN r.registry_count > 1 THEN 'REJECT_DUPLICATE_SCHEMA'
            WHEN r.compatibility_state IS NULL THEN 'REJECT_UNKNOWN_SCHEMA'
            WHEN r.compatibility_state NOT IN ('BACKWARD_COMPATIBLE', 'CURRENT')
                THEN 'REJECT_INCOMPATIBLE_SCHEMA'
            WHEN NOT json_valid(r.payload_json) THEN 'REJECT_INVALID_JSON'
            WHEN json_type(r.safe_payload) <> 'object'
                THEN 'REJECT_INVALID_PAYLOAD_SHAPE'
            WHEN NOT json_valid(r.required_fields_json) OR
                 r.required_fields_is_array <> 1 OR
                 COALESCE(a.invalid_required_fields, 0) > 0
                THEN 'REJECT_INVALID_REGISTRY'
            WHEN r.semantic_schema_hash IS NULL OR
                 TRIM(r.semantic_schema_hash) = ''
                THEN 'REJECT_MISSING_LINEAGE'
            WHEN COALESCE(a.missing_required_fields, 0) > 0
                THEN 'REJECT_MISSING_REQUIRED_FIELD'
            ELSE 'PASS'
        END AS status
    FROM registered AS r
    LEFT JOIN required_field_audit AS a USING (event_row_id)
)
SELECT
    event_id,
    event_type,
    schema_major,
    schema_minor,
    ingested_at,
    semantic_schema_hash,
    missing_required_fields,
    status
FROM evaluated
WHERE status <> 'PASS'
ORDER BY event_type, schema_major, schema_minor, event_id, event_row_id;
