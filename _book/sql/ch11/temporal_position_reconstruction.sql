-- Dialect: SQLite 3.45; PIT revision selection and exact sequence policy.
WITH eligible_physical AS (
    SELECT p.rowid AS physical_row_id, p.*
    FROM position_events AS p
    WHERE p.event_time <= :as_of_time
      AND p.knowledge_time <= :knowledge_cutoff
), revision_audit AS (
    SELECT
        *,
        COUNT(*) OVER (
            PARTITION BY event_id, knowledge_time, version_id, source_id
        ) AS revision_count,
        ROW_NUMBER() OVER (
            PARTITION BY event_id
            ORDER BY knowledge_time DESC, version_id DESC, source_id DESC,
                     row_hash DESC, physical_row_id DESC
        ) AS revision_rank
    FROM eligible_physical
), revision_scoped AS (
    SELECT
        *,
        MAX(revision_count) OVER (PARTITION BY account_id, instrument_id)
            AS scope_revision_count
    FROM revision_audit
), canonical AS (
    SELECT *
    FROM revision_scoped
    WHERE revision_rank = 1
), key_audit AS (
    SELECT
        account_id,
        instrument_id,
        MAX(scope_revision_count) AS maximum_revision_count,
        MAX(sequence_count) AS maximum_sequence_count,
        MIN(sequence_number) AS minimum_sequence,
        MAX(sequence_number) AS maximum_sequence,
        COUNT(*) AS event_count,
        MAX(CASE WHEN quantity_delta IS NULL OR
                      ABS(quantity_delta) > 1.7976931348623157e308
                 THEN 1 ELSE 0 END) AS invalid_quantity,
        MAX(CASE WHEN source_id IS NULL OR TRIM(source_id) = '' OR
                      row_hash IS NULL OR TRIM(row_hash) = ''
                 THEN 1 ELSE 0 END) AS missing_lineage
    FROM (
        SELECT
            c.*,
            COUNT(*) OVER (
                PARTITION BY account_id, instrument_id, sequence_number
            ) AS sequence_count
        FROM canonical AS c
    )
    GROUP BY account_id, instrument_id
), sequenced AS (
    SELECT
        c.*,
        a.*,
        SUM(c.quantity_delta) OVER (
            PARTITION BY c.account_id, c.instrument_id
            ORDER BY c.sequence_number, c.event_time, c.event_id,
                     c.version_id, c.source_id, c.row_hash
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_quantity,
        ROW_NUMBER() OVER (
            PARTITION BY c.account_id, c.instrument_id
            ORDER BY c.sequence_number DESC, c.event_time DESC, c.event_id DESC,
                     c.version_id DESC, c.source_id DESC, c.row_hash DESC
        ) AS final_rank
    FROM canonical AS c
    JOIN key_audit AS a USING (account_id, instrument_id)
), evaluated AS (
    SELECT
        *,
        CASE
            WHEN maximum_revision_count <> 1
                THEN 'REJECT_DUPLICATE_REVISION'
            WHEN maximum_sequence_count <> 1
                THEN 'REJECT_DUPLICATE_SEQUENCE'
            WHEN minimum_sequence <> 1 OR maximum_sequence <> event_count
                THEN 'REJECT_SEQUENCE_GAP'
            WHEN invalid_quantity <> 0 THEN 'REJECT_INVALID_QUANTITY'
            WHEN missing_lineage <> 0 THEN 'REJECT_MISSING_LINEAGE'
            ELSE 'PASS'
        END AS status
    FROM sequenced
    WHERE final_rank = 1
)
SELECT
    account_id,
    instrument_id,
    CASE WHEN status = 'PASS' THEN running_quantity END AS position,
    maximum_sequence AS last_sequence,
    status
FROM evaluated
ORDER BY account_id, instrument_id;
