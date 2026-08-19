-- Dialect: SQLite 3.45; PIT revision selection before a prior-only window.
WITH eligible_physical AS (
    SELECT d.rowid AS physical_row_id, d.*
    FROM daily_observations AS d
    WHERE d.knowledge_time <= :knowledge_cutoff
), audited AS (
    SELECT
        *,
        COUNT(*) OVER (
            PARTITION BY
                instrument_id, observation_time, sequence_number,
                knowledge_time, version_id, source_id
        ) AS revision_count,
        ROW_NUMBER() OVER (
            PARTITION BY instrument_id, observation_time, sequence_number
            ORDER BY knowledge_time DESC, version_id DESC, source_id DESC,
                     row_hash DESC, physical_row_id DESC
        ) AS revision_rank
    FROM eligible_physical
), revision_scoped AS (
    SELECT
        *,
        MAX(revision_count) OVER (PARTITION BY instrument_id)
            AS scope_revision_count
    FROM audited
), canonical AS (
    SELECT *
    FROM revision_scoped
    WHERE revision_rank = 1
), instrument_audit AS (
    SELECT
        instrument_id,
        MAX(scope_revision_count) AS maximum_revision_count,
        MAX(sequence_count) AS maximum_sequence_count,
        MAX(CASE WHEN source_id IS NULL OR TRIM(source_id) = '' OR
                      row_hash IS NULL OR TRIM(row_hash) = ''
                 THEN 1 ELSE 0 END) AS missing_lineage,
        MAX(CASE WHEN close_price IS NULL OR close_price <= 0.0 OR
                      ABS(close_price) > 1.7976931348623157e308 OR
                      (return_1d IS NOT NULL AND
                       ABS(return_1d) > 1.7976931348623157e308)
                 THEN 1 ELSE 0 END) AS invalid_numeric
    FROM (
        SELECT
            c.*,
            COUNT(*) OVER (
                PARTITION BY instrument_id, sequence_number
            ) AS sequence_count
        FROM canonical AS c
    )
    GROUP BY instrument_id
), windowed AS (
    SELECT
        c.*,
        a.maximum_revision_count,
        a.maximum_sequence_count,
        a.missing_lineage,
        a.invalid_numeric,
        COUNT(c.return_1d) OVER prior_window AS prior_count,
        AVG(c.return_1d) OVER prior_window AS mean_return_20d,
        AVG(c.return_1d * c.return_1d) OVER prior_window
            AS mean_square_return_20d,
        LAG(c.close_price, 1) OVER total_order AS prior_close
    FROM canonical AS c
    JOIN instrument_audit AS a USING (instrument_id)
    WINDOW
      total_order AS (
        PARTITION BY c.instrument_id
        ORDER BY c.observation_time, c.sequence_number, c.version_id,
                 c.source_id, c.row_hash
      ),
      prior_window AS (
        PARTITION BY c.instrument_id
        ORDER BY c.observation_time, c.sequence_number, c.version_id,
                 c.source_id, c.row_hash
        ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
      )
)
SELECT
    instrument_id,
    observation_time,
    sequence_number,
    version_id,
    knowledge_time,
    source_id,
    row_hash,
    return_1d,
    CASE
        WHEN maximum_revision_count = 1 AND maximum_sequence_count = 1 AND
             missing_lineage = 0 AND invalid_numeric = 0 AND prior_count = 20
            THEN mean_return_20d
    END AS mean_return_20d,
    CASE
        WHEN maximum_revision_count = 1 AND maximum_sequence_count = 1 AND
             missing_lineage = 0 AND invalid_numeric = 0 AND prior_count = 20
            THEN mean_square_return_20d
    END AS mean_square_return_20d,
    CASE
        WHEN maximum_revision_count = 1 AND maximum_sequence_count = 1 AND
             missing_lineage = 0 AND invalid_numeric = 0
            THEN prior_close
    END AS prior_close,
    CASE
        WHEN maximum_revision_count <> 1 THEN 'REJECT_DUPLICATE_REVISION'
        WHEN maximum_sequence_count <> 1 THEN 'REJECT_DUPLICATE_ORDER_KEY'
        WHEN missing_lineage <> 0 THEN 'REJECT_MISSING_LINEAGE'
        WHEN invalid_numeric <> 0 THEN 'REJECT_INVALID_NUMERIC'
        ELSE 'PASS'
    END AS status
FROM windowed
ORDER BY instrument_id, observation_time, sequence_number, version_id,
         source_id, row_hash;
