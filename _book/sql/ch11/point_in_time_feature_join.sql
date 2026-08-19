-- Dialect: SQLite 3.45; preserve physical decisions and complete feature lineage.
WITH decisions_physical AS (
    SELECT d.rowid AS physical_decision_row, d.*
    FROM decisions AS d
), decision_audit AS (
    SELECT
        physical_decision_row,
        COUNT(*) OVER (PARTITION BY decision_id) AS decision_id_count,
        COUNT(*) OVER (PARTITION BY decision_row_id) AS decision_row_id_count
    FROM decisions_physical
), features_physical AS (
    SELECT f.rowid AS physical_feature_row, f.*
    FROM feature_versions AS f
), feature_version_audit AS (
    SELECT
        physical_feature_row,
        COUNT(*) OVER (
            PARTITION BY
                instrument_id, feature_name, effective_from, knowledge_from,
                version_id, source_id
        ) AS version_count
    FROM features_physical
), candidates AS (
    SELECT
        d.physical_decision_row,
        d.decision_row_id,
        d.decision_id,
        d.instrument_id,
        d.event_time,
        d.decision_time,
        f.physical_feature_row,
        f.feature_name,
        f.feature_value,
        f.effective_from,
        f.knowledge_from,
        f.version_id,
        f.source_id,
        f.row_hash,
        a.decision_id_count,
        a.decision_row_id_count,
        COALESCE(v.version_count, 0) AS version_count,
        ROW_NUMBER() OVER (
            PARTITION BY d.physical_decision_row, f.feature_name
            ORDER BY
                f.effective_from DESC,
                f.knowledge_from DESC,
                f.version_id DESC,
                f.source_id DESC,
                f.row_hash DESC,
                f.physical_feature_row DESC
        ) AS candidate_rank,
        MAX(COALESCE(v.version_count, 0)) OVER (
            PARTITION BY d.physical_decision_row, f.feature_name
        ) AS maximum_version_count
    FROM decisions_physical AS d
    JOIN decision_audit AS a USING (physical_decision_row)
    LEFT JOIN features_physical AS f
      ON f.instrument_id = d.instrument_id
     AND f.effective_from <= d.event_time
     AND f.knowledge_from <= d.decision_time
    LEFT JOIN feature_version_audit AS v USING (physical_feature_row)
), evaluated AS (
    SELECT
        *,
        CASE
            WHEN decision_row_id IS NULL OR TRIM(decision_row_id) = ''
                THEN 'REJECT_MISSING_DECISION_ROW_ID'
            WHEN decision_id_count <> 1 OR decision_row_id_count <> 1
                THEN 'REJECT_DUPLICATE_DECISION'
            WHEN physical_feature_row IS NULL THEN 'REJECT_MISSING_ROW'
            WHEN maximum_version_count <> 1 THEN 'REJECT_DUPLICATE_VERSION'
            WHEN feature_value IS NULL OR
                 ABS(feature_value) > 1.7976931348623157e308
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
    event_time,
    decision_time,
    feature_name,
    CASE WHEN status = 'PASS' THEN feature_value END AS feature_value,
    effective_from,
    knowledge_from,
    version_id,
    source_id,
    row_hash,
    status
FROM evaluated
ORDER BY decision_id, decision_row_id, feature_name, physical_decision_row;
