-- Dialect: SQLite 3.45; fail-closed bitemporal selection with complete lineage.
WITH eligible_raw AS (
    SELECT
        iv.rowid AS physical_row_id,
        iv.*
    FROM instrument_values AS iv
    WHERE iv.valid_from <= :decision_time
      AND (iv.valid_to IS NULL OR :decision_time < iv.valid_to)
      AND iv.knowledge_from <= :knowledge_cutoff
      AND (iv.knowledge_to IS NULL OR :knowledge_cutoff < iv.knowledge_to)
), version_audit AS (
    SELECT
        physical_row_id,
        COUNT(*) OVER (
            PARTITION BY
                instrument_id, valid_from, COALESCE(valid_to, ''),
                knowledge_from, COALESCE(knowledge_to, ''), version_id,
                source_priority, source_id
        ) AS version_count
    FROM eligible_raw
), ranked AS (
    SELECT
        e.*,
        v.version_count,
        MAX(v.version_count) OVER (
            PARTITION BY e.instrument_id
        ) AS maximum_version_count,
        ROW_NUMBER() OVER (
            PARTITION BY e.instrument_id
            ORDER BY
                e.valid_from DESC,
                e.knowledge_from DESC,
                e.version_id DESC,
                e.source_priority ASC,
                e.source_id DESC,
                e.row_hash DESC,
                e.physical_row_id DESC
        ) AS rank_at_cutoff
    FROM eligible_raw AS e
    JOIN version_audit AS v USING (physical_row_id)
), evaluated AS (
    SELECT
        *,
        CASE
            WHEN maximum_version_count <> 1 THEN 'REJECT_DUPLICATE_VERSION'
            WHEN value IS NULL OR ABS(value) > 1.7976931348623157e308
                THEN 'REJECT_INVALID_VALUE'
            WHEN source_id IS NULL OR TRIM(source_id) = '' OR
                 row_hash IS NULL OR TRIM(row_hash) = ''
                THEN 'REJECT_MISSING_LINEAGE'
            ELSE 'PASS'
        END AS status
    FROM ranked
    WHERE rank_at_cutoff = 1
)
SELECT
    instrument_id,
    CASE WHEN status = 'PASS' THEN value END AS value,
    valid_from,
    valid_to,
    knowledge_from,
    knowledge_to,
    version_id,
    source_id,
    row_hash,
    status
FROM evaluated
ORDER BY instrument_id;
