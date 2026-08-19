-- Dialect: PostgreSQL 16+
-- One fail-closed result per (prediction_id, feature_name), including tombstones.
WITH request_ranked AS (
    SELECT
        p.prediction_id,
        p.instrument_id,
        p.feature_name,
        p.feature_cutoff_time,
        p.decision_time,
        ROW_NUMBER() OVER (
            PARTITION BY p.prediction_id, p.feature_name
            ORDER BY
                p.decision_time DESC,
                p.feature_cutoff_time DESC,
                p.instrument_id DESC
        ) AS request_rank,
        COUNT(*) OVER (
            PARTITION BY p.prediction_id, p.feature_name
        ) AS request_count
    FROM prediction_request AS p
),
request_key AS (
    SELECT
        prediction_id,
        instrument_id,
        feature_name,
        feature_cutoff_time,
        decision_time,
        request_count
    FROM request_ranked
    WHERE request_rank = 1
),
eligible_revision AS (
    SELECT
        p.prediction_id,
        p.instrument_id,
        p.feature_name,
        p.feature_cutoff_time,
        p.decision_time,
        p.request_count,
        f.feature_value,
        f.effective_time,
        f.knowledge_time,
        f.version_id,
        f.source_id,
        f.row_hash,
        f.is_tombstone,
        ROW_NUMBER() OVER (
            PARTITION BY p.prediction_id, p.feature_name
            ORDER BY
                f.effective_time DESC NULLS LAST,
                f.knowledge_time DESC NULLS LAST,
                f.version_id DESC NULLS LAST,
                f.source_id DESC NULLS LAST,
                f.row_hash DESC NULLS LAST
        ) AS row_rank,
        COUNT(f.row_hash) OVER (
            PARTITION BY
                p.prediction_id,
                p.feature_name,
                f.effective_time,
                f.knowledge_time,
                f.version_id,
                f.source_id
        ) AS exact_tie_count
    FROM request_key AS p
    LEFT JOIN feature_value AS f
      ON f.instrument_id = p.instrument_id
     AND f.feature_name = p.feature_name
     AND f.effective_time <= p.feature_cutoff_time
     AND f.knowledge_time <= p.decision_time
),
winner AS (
    SELECT
        *,
        CASE
            WHEN feature_cutoff_time > decision_time THEN 'INVALID_REQUEST_TIME'
            WHEN request_count <> 1 THEN 'AMBIGUOUS_REQUEST'
            WHEN row_hash IS NULL THEN 'MISSING'
            WHEN exact_tie_count > 1 THEN 'AMBIGUOUS_REVISION'
            WHEN is_tombstone THEN 'TOMBSTONED'
            WHEN feature_value IS NULL THEN 'INVALID_NULL'
            WHEN feature_value IN (
                'NaN'::double precision,
                'Infinity'::double precision,
                '-Infinity'::double precision
            ) THEN 'INVALID_NONFINITE'
            ELSE 'SELECTED'
        END AS selection_status
    FROM eligible_revision
    WHERE row_rank = 1
)
SELECT
    prediction_id,
    instrument_id,
    feature_name,
    feature_cutoff_time,
    decision_time,
    CASE WHEN selection_status = 'SELECTED' THEN feature_value END AS feature_value,
    effective_time,
    knowledge_time,
    version_id,
    source_id,
    row_hash,
    request_count,
    exact_tie_count,
    selection_status
FROM winner
ORDER BY prediction_id, feature_name;
