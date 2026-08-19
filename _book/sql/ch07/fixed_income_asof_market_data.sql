-- Dialect: SQLite 3.35+
-- Governed bitemporal curve-quote and fixing selection.
-- Required inputs include immutable 64-hex row_hash values and an
-- expected_market_data(record_type, economic_key) completeness registry.
WITH all_candidates AS (
    SELECT
        'QUOTE' AS record_type,
        q.instrument_id || '|' || q.curve_id || '|' || q.tenor
            || '|' || q.quote_type AS economic_key,
        q.quote_value AS numeric_value,
        q.valid_from,
        q.valid_to,
        q.known_from,
        q.known_to,
        q.revision_id,
        q.source_priority,
        q.is_cancelled,
        q.row_hash
    FROM market_quote_history AS q
    WHERE q.valid_from <= :valuation_time
      AND (q.valid_to IS NULL OR :valuation_time < q.valid_to)
      AND q.known_from <= :knowledge_cutoff
      AND (q.known_to IS NULL OR :knowledge_cutoff < q.known_to)
    UNION ALL
    SELECT
        'FIXING' AS record_type,
        f.index_id || '|' || f.fixing_date AS economic_key,
        f.fixing_value AS numeric_value,
        f.valid_from,
        f.valid_to,
        f.known_from,
        f.known_to,
        f.revision_id,
        f.source_priority,
        f.is_cancelled,
        f.row_hash
    FROM fixing_history AS f
    WHERE f.valid_from <= :valuation_time
      AND (f.valid_to IS NULL OR :valuation_time < f.valid_to)
      AND f.known_from <= :knowledge_cutoff
      AND (f.known_to IS NULL OR :knowledge_cutoff < f.known_to)
), ranked_candidates AS (
    SELECT
        c.*,
        DENSE_RANK() OVER (
            PARTITION BY c.record_type, c.economic_key
            ORDER BY
                c.known_from DESC,
                c.valid_from DESC,
                c.source_priority DESC,
                c.revision_id DESC
        ) AS business_rank,
        ROW_NUMBER() OVER (
            PARTITION BY c.record_type, c.economic_key
            ORDER BY
                c.known_from DESC,
                c.valid_from DESC,
                c.source_priority DESC,
                c.revision_id DESC,
                c.row_hash DESC,
                printf('%.17g', c.numeric_value) DESC
        ) AS deterministic_rank
    FROM all_candidates AS c
), candidate_diagnostics AS (
    SELECT
        record_type,
        economic_key,
        SUM(CASE WHEN business_rank = 1 THEN 1 ELSE 0 END)
            AS top_rank_tie_count,
        COUNT(DISTINCT CASE
            WHEN business_rank = 1 THEN row_hash
        END) AS top_rank_distinct_hash_count,
        COUNT(DISTINCT CASE
            WHEN business_rank = 1 THEN printf('%.17g', numeric_value)
        END) AS top_rank_distinct_value_count
    FROM ranked_candidates
    GROUP BY record_type, economic_key
), selected_candidates AS (
    SELECT *
    FROM ranked_candidates
    WHERE deterministic_rank = 1
), expected_keys AS (
    SELECT
        record_type,
        economic_key,
        COUNT(*) AS expected_key_count
    FROM expected_market_data
    GROUP BY record_type, economic_key
), key_universe AS (
    SELECT record_type, economic_key FROM expected_keys
    UNION
    SELECT record_type, economic_key FROM candidate_diagnostics
), governed_rows AS (
    SELECT
        u.record_type,
        u.economic_key,
        s.numeric_value,
        s.valid_from,
        s.known_from,
        s.revision_id,
        s.source_priority,
        s.is_cancelled,
        s.row_hash,
        COALESCE(d.top_rank_tie_count, 0) AS top_rank_tie_count,
        COALESCE(d.top_rank_distinct_hash_count, 0)
            AS top_rank_distinct_hash_count,
        COALESCE(d.top_rank_distinct_value_count, 0)
            AS top_rank_distinct_value_count,
        COALESCE(e.expected_key_count, 0) AS expected_key_count,
        CASE
            WHEN e.economic_key IS NULL THEN 'REJECT_UNEXPECTED_KEY'
            WHEN e.expected_key_count <> 1 THEN 'REJECT_DUPLICATE_EXPECTATION'
            WHEN s.economic_key IS NULL THEN 'REJECT_MISSING_RECORD'
            WHEN d.top_rank_tie_count <> 1 THEN 'REJECT_TOP_RANK_TIE'
            WHEN d.top_rank_distinct_hash_count <> 1
              OR d.top_rank_distinct_value_count <> 1
            THEN 'REJECT_CONFLICTING_PAYLOAD'
            WHEN s.is_cancelled <> 0 THEN 'REJECT_CANCELLED_WINNER'
            WHEN s.numeric_value IS NULL THEN 'REJECT_NULL_VALUE'
            WHEN s.row_hash IS NULL
              OR LENGTH(s.row_hash) <> 64
              OR LOWER(s.row_hash) <> s.row_hash
              OR s.row_hash GLOB '*[^0-9a-f]*'
            THEN 'REJECT_INVALID_ROW_HASH'
            ELSE 'PASS'
        END AS selection_status
    FROM key_universe AS u
    LEFT JOIN expected_keys AS e
      ON e.record_type = u.record_type
     AND e.economic_key = u.economic_key
    LEFT JOIN selected_candidates AS s
      ON s.record_type = u.record_type
     AND s.economic_key = u.economic_key
    LEFT JOIN candidate_diagnostics AS d
      ON d.record_type = u.record_type
     AND d.economic_key = u.economic_key
), evidence_rows AS (
    SELECT
        *,
        record_type || '|' || economic_key || '|'
            || COALESCE(printf('%.17g', numeric_value), '<NULL>') || '|'
            || COALESCE(valid_from, '<NULL>') || '|'
            || COALESCE(known_from, '<NULL>') || '|'
            || COALESCE(CAST(revision_id AS TEXT), '<NULL>') || '|'
            || COALESCE(CAST(source_priority AS TEXT), '<NULL>') || '|'
            || COALESCE(CAST(is_cancelled AS TEXT), '<NULL>') || '|'
            || COALESCE(row_hash, '<NULL>') || '|'
            || selection_status AS manifest_row
    FROM governed_rows
), snapshot_manifest AS (
    SELECT
        COUNT(*) AS snapshot_row_count,
        MIN(known_from) AS minimum_known_from,
        MAX(known_from) AS maximum_known_from,
        (
            SELECT GROUP_CONCAT(manifest_row, ';')
            FROM (
                SELECT manifest_row
                FROM evidence_rows
                ORDER BY record_type, economic_key
            ) AS ordered_rows
        ) AS snapshot_manifest_payload
    FROM evidence_rows
)
SELECT
    r.record_type,
    r.economic_key,
    r.numeric_value,
    r.valid_from,
    r.known_from,
    r.revision_id,
    m.snapshot_row_count,
    m.minimum_known_from,
    m.maximum_known_from,
    m.snapshot_manifest_payload,
    'fixed-income-asof-v0.10' AS snapshot_schema_version,
    r.row_hash,
    r.top_rank_tie_count,
    r.expected_key_count,
    r.is_cancelled,
    r.selection_status
FROM evidence_rows AS r
CROSS JOIN snapshot_manifest AS m
ORDER BY r.record_type, r.economic_key;
