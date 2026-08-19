-- Dialect: SQLite 3.45; preserve physical decisions and fail closed on PIT ambiguity.
WITH decisions_physical AS (
    SELECT d.rowid AS physical_decision_row, d.*
    FROM decisions AS d
), decision_audit AS (
    SELECT
        physical_decision_row,
        COUNT(*) OVER (PARTITION BY decision_id) AS decision_id_count,
        COUNT(*) OVER (PARTITION BY decision_row_id) AS decision_row_id_count
    FROM decisions_physical
), market_physical AS (
    SELECT m.rowid AS physical_market_row, m.*
    FROM market_data AS m
), market_version_audit AS (
    SELECT
        physical_market_row,
        COUNT(*) OVER (
            PARTITION BY
                instrument_id, effective_at, known_at, version, source_id
        ) AS version_count
    FROM market_physical
), candidates AS (
    SELECT
        d.physical_decision_row,
        d.decision_row_id,
        d.decision_id,
        d.instrument_id,
        d.effective_at AS decision_valid_time,
        d.knowledge_at AS decision_knowledge_cutoff,
        m.physical_market_row,
        m.value,
        m.effective_at AS selected_valid_time,
        m.known_at AS selected_knowledge_time,
        m.version,
        m.source_id,
        m.row_hash,
        a.decision_id_count,
        a.decision_row_id_count,
        COALESCE(v.version_count, 0) AS version_count,
        ROW_NUMBER() OVER (
            PARTITION BY d.physical_decision_row
            ORDER BY
                m.effective_at DESC,
                m.known_at DESC,
                m.version DESC,
                m.source_id DESC,
                m.row_hash DESC,
                m.physical_market_row DESC
        ) AS candidate_rank,
        MAX(COALESCE(v.version_count, 0)) OVER (
            PARTITION BY d.physical_decision_row
        ) AS maximum_version_count
    FROM decisions_physical AS d
    JOIN decision_audit AS a USING (physical_decision_row)
    LEFT JOIN market_physical AS m
      ON m.instrument_id = d.instrument_id
     AND m.effective_at <= d.effective_at
     AND m.known_at <= d.knowledge_at
    LEFT JOIN market_version_audit AS v USING (physical_market_row)
), evaluated AS (
    SELECT
        *,
        CASE
            WHEN decision_row_id IS NULL OR TRIM(decision_row_id) = ''
                THEN 'REJECT_MISSING_DECISION_ROW_ID'
            WHEN decision_id_count <> 1 OR decision_row_id_count <> 1
                THEN 'REJECT_DUPLICATE_DECISION'
            WHEN physical_market_row IS NULL THEN 'REJECT_MISSING_ROW'
            WHEN maximum_version_count <> 1 THEN 'REJECT_DUPLICATE_VERSION'
            WHEN value IS NULL OR ABS(value) > 1.7976931348623157e308
                THEN 'REJECT_INVALID_VALUE'
            WHEN source_id IS NULL OR TRIM(source_id) = '' OR
                 row_hash IS NULL OR TRIM(row_hash) = ''
                THEN 'REJECT_MISSING_LINEAGE'
            ELSE 'PASS'
        END AS status
    FROM candidates
    WHERE candidate_rank = 1
)
SELECT
    decision_row_id,
    decision_id,
    instrument_id,
    decision_valid_time,
    decision_knowledge_cutoff,
    CASE WHEN status = 'PASS' THEN value END AS value,
    selected_valid_time,
    selected_knowledge_time,
    version,
    source_id,
    row_hash,
    status
FROM evaluated
ORDER BY decision_id, decision_row_id, physical_decision_row;
