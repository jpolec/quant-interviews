-- Dialect: SQLite 3.45; PIT snapshot selection and fixed-point reconciliation.
WITH parameters AS (
    SELECT
        CASE
            WHEN typeof(:tolerance_minor) = 'integer' AND :tolerance_minor >= 0
                THEN :tolerance_minor
        END AS tolerance_minor,
        CASE
            WHEN typeof(:tolerance_version) = 'text' AND
                 TRIM(:tolerance_version) <> ''
                THEN :tolerance_version
        END AS tolerance_version
), events_eligible AS (
    SELECT e.rowid AS physical_event_row, e.*
    FROM cash_events AS e
    WHERE e.knowledge_time <= :knowledge_cutoff
), event_revisions AS (
    SELECT
        *,
        COUNT(*) OVER (
            PARTITION BY event_id, knowledge_time, version_id, source_id
        ) AS revision_count,
        ROW_NUMBER() OVER (
            PARTITION BY event_id
            ORDER BY knowledge_time DESC, version_id DESC, source_id DESC,
                     row_hash DESC, physical_event_row DESC
        ) AS revision_rank
    FROM events_eligible
), event_revision_scoped AS (
    SELECT
        *,
        MAX(revision_count) OVER (PARTITION BY account_id, currency)
            AS scope_revision_count
    FROM event_revisions
), canonical_events AS (
    SELECT *
    FROM event_revision_scoped
    WHERE revision_rank = 1
), snapshots_eligible AS (
    SELECT s.rowid AS physical_snapshot_row, s.*
    FROM account_snapshots AS s
    WHERE s.knowledge_time <= :knowledge_cutoff
), snapshot_revisions AS (
    SELECT
        *,
        COUNT(*) OVER (
            PARTITION BY
                account_id, snapshot_time, currency, knowledge_time,
                version_id, source_id
        ) AS revision_count,
        ROW_NUMBER() OVER (
            PARTITION BY account_id, snapshot_time, currency
            ORDER BY knowledge_time DESC, version_id DESC, source_id DESC,
                     row_hash DESC, physical_snapshot_row DESC
        ) AS revision_rank
    FROM snapshots_eligible
), snapshot_revision_scoped AS (
    SELECT
        *,
        MAX(revision_count) OVER (
            PARTITION BY account_id, snapshot_time, currency
        ) AS scope_revision_count
    FROM snapshot_revisions
), canonical_snapshots AS (
    SELECT *
    FROM snapshot_revision_scoped
    WHERE revision_rank = 1
), reconciled AS (
    SELECT
        s.account_id,
        s.snapshot_time,
        s.currency,
        s.expected_cash_minor,
        s.source_id AS snapshot_source_id,
        s.row_hash AS snapshot_row_hash,
        s.scope_revision_count AS snapshot_revision_count,
        COALESCE(SUM(
            CASE WHEN e.event_time <= s.snapshot_time THEN e.cash_delta_minor END
        ), 0) AS actual_cash_minor,
        COALESCE(MAX(e.scope_revision_count), 1)
            AS maximum_event_revision_count,
        COALESCE(MAX(CASE WHEN e.source_id IS NULL OR TRIM(e.source_id) = '' OR
                              e.row_hash IS NULL OR TRIM(e.row_hash) = ''
                         THEN 1 ELSE 0 END), 0) AS missing_event_lineage,
        COALESCE(MAX(CASE WHEN e.cash_delta_minor IS NULL OR
                              typeof(e.cash_delta_minor) <> 'integer'
                         THEN 1 ELSE 0 END), 0) AS invalid_event_value
    FROM canonical_snapshots AS s
    LEFT JOIN canonical_events AS e
      ON e.account_id = s.account_id
     AND e.currency = s.currency
     AND e.event_time <= s.snapshot_time
    GROUP BY
        s.physical_snapshot_row, s.account_id, s.snapshot_time, s.currency,
        s.expected_cash_minor, s.source_id, s.row_hash, s.revision_count
), evaluated AS (
    SELECT
        r.*,
        r.actual_cash_minor - r.expected_cash_minor AS residual_minor,
        p.tolerance_minor,
        p.tolerance_version,
        CASE
            WHEN p.tolerance_minor IS NULL OR p.tolerance_version IS NULL
                THEN 'REJECT_INVALID_TOLERANCE'
            WHEN r.snapshot_revision_count <> 1 OR
                 r.maximum_event_revision_count <> 1
                THEN 'REJECT_DUPLICATE_REVISION'
            WHEN r.expected_cash_minor IS NULL OR
                 typeof(r.expected_cash_minor) <> 'integer' OR
                 r.invalid_event_value <> 0
                THEN 'REJECT_MISSING_VALUE'
            WHEN r.snapshot_source_id IS NULL OR
                 TRIM(r.snapshot_source_id) = '' OR
                 r.snapshot_row_hash IS NULL OR
                 TRIM(r.snapshot_row_hash) = '' OR
                 r.missing_event_lineage <> 0
                THEN 'REJECT_MISSING_LINEAGE'
            WHEN ABS(r.actual_cash_minor - r.expected_cash_minor) >
                 p.tolerance_minor
                THEN 'REJECT_BREAK'
            ELSE 'PASS'
        END AS status
    FROM reconciled AS r
    CROSS JOIN parameters AS p
)
SELECT
    account_id,
    snapshot_time,
    currency,
    expected_cash_minor,
    actual_cash_minor,
    residual_minor,
    tolerance_minor,
    tolerance_version,
    snapshot_source_id,
    snapshot_row_hash,
    status
FROM evaluated
ORDER BY account_id, snapshot_time, currency;
