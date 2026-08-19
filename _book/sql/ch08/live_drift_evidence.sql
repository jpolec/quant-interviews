-- Dialect: PostgreSQL 16+
-- One-to-one PIT score/outcome/P&L evidence; invalid rows remain visible.
WITH ranked_score AS (
    SELECT
        score.*,
        ROW_NUMBER() OVER (
            PARTITION BY score.prediction_id, score.model_version
            ORDER BY
                score.knowledge_time DESC,
                score.version_id DESC,
                score.source_id DESC,
                score.row_hash DESC
        ) AS row_rank,
        COUNT(*) OVER (
            PARTITION BY
                score.prediction_id,
                score.model_version,
                score.knowledge_time,
                score.version_id,
                score.source_id
        ) AS exact_tie_count
    FROM model_score AS score
    WHERE score.knowledge_time <= :monitoring_cutoff
),
score_winner AS (
    SELECT
        prediction_id,
        model_version,
        score_date,
        probability,
        is_tombstone,
        exact_tie_count
    FROM ranked_score
    WHERE row_rank = 1
),
ranked_outcome AS (
    SELECT
        outcome.*,
        ROW_NUMBER() OVER (
            PARTITION BY outcome.prediction_id
            ORDER BY
                outcome.label_known_time DESC,
                outcome.version_id DESC,
                outcome.source_id DESC,
                outcome.row_hash DESC
        ) AS row_rank,
        COUNT(*) OVER (
            PARTITION BY
                outcome.prediction_id,
                outcome.label_known_time,
                outcome.version_id,
                outcome.source_id
        ) AS exact_tie_count
    FROM outcome AS outcome
    WHERE outcome.label_known_time <= :monitoring_cutoff
),
outcome_winner AS (
    SELECT
        prediction_id,
        label,
        is_tombstone,
        exact_tie_count
    FROM ranked_outcome
    WHERE row_rank = 1
),
ranked_pnl AS (
    SELECT
        pnl.*,
        ROW_NUMBER() OVER (
            PARTITION BY pnl.prediction_id, pnl.pnl_run_id
            ORDER BY
                pnl.knowledge_time DESC,
                pnl.version_id DESC,
                pnl.source_id DESC,
                pnl.row_hash DESC
        ) AS row_rank,
        COUNT(*) OVER (
            PARTITION BY
                pnl.prediction_id,
                pnl.pnl_run_id,
                pnl.knowledge_time,
                pnl.version_id,
                pnl.source_id
        ) AS exact_tie_count
    FROM strategy_pnl AS pnl
    WHERE pnl.pnl_run_id = :pnl_run_id
      AND pnl.knowledge_time <= :monitoring_cutoff
),
pnl_winner AS (
    SELECT
        prediction_id,
        pnl_run_id,
        net_pnl,
        turnover,
        is_tombstone,
        exact_tie_count
    FROM ranked_pnl
    WHERE row_rank = 1
),
one_to_one AS (
    SELECT
        score.model_version,
        score.score_date,
        score.prediction_id,
        score.probability,
        outcome.label,
        pnl.net_pnl,
        pnl.turnover,
        score.is_tombstone AS score_tombstone,
        outcome.is_tombstone AS outcome_tombstone,
        pnl.is_tombstone AS pnl_tombstone,
        score.exact_tie_count AS score_tie_count,
        outcome.exact_tie_count AS outcome_tie_count,
        pnl.exact_tie_count AS pnl_tie_count,
        CASE WHEN outcome.prediction_id IS NULL THEN 0 ELSE 1 END AS has_label_row,
        CASE WHEN pnl.prediction_id IS NULL THEN 0 ELSE 1 END AS has_pnl_row
    FROM score_winner AS score
    LEFT JOIN outcome_winner AS outcome
      ON outcome.prediction_id = score.prediction_id
    LEFT JOIN pnl_winner AS pnl
      ON pnl.prediction_id = score.prediction_id
     AND pnl.pnl_run_id = :pnl_run_id
),
daily AS (
    SELECT
        model_version,
        score_date,
        COUNT(*) AS expected_prediction_keys,
        COUNT(*) FILTER (
            WHERE NOT score_tombstone AND score_tie_count = 1
        ) AS scored_rows,
        SUM(has_label_row) FILTER (
            WHERE NOT outcome_tombstone AND outcome_tie_count = 1
        ) AS labelled_rows,
        SUM(has_pnl_row) FILTER (
            WHERE NOT pnl_tombstone AND pnl_tie_count = 1
        ) AS pnl_rows,
        COUNT(*) FILTER (
            WHERE probability IS NULL
               OR probability < 0.0
               OR probability > 1.0
               OR probability IN (
                    'NaN'::double precision,
                    'Infinity'::double precision,
                    '-Infinity'::double precision
               )
        ) AS invalid_probability_count,
        COUNT(*) FILTER (WHERE label IS NOT NULL AND label NOT IN (0, 1))
            AS invalid_label_count,
        COUNT(*) FILTER (
            WHERE score_tie_count > 1 OR outcome_tie_count > 1 OR pnl_tie_count > 1
        ) AS ambiguous_revision_count,
        AVG(probability) FILTER (WHERE probability BETWEEN 0.0 AND 1.0)
            AS mean_probability,
        AVG(label) FILTER (WHERE label IN (0, 1)) AS event_rate,
        AVG((label - probability) * (label - probability)) FILTER (
            WHERE label IN (0, 1) AND probability BETWEEN 0.0 AND 1.0
        ) AS brier_score,
        SUM(net_pnl) FILTER (
            WHERE NOT pnl_tombstone AND pnl_tie_count = 1
        ) AS net_pnl,
        SUM(turnover) FILTER (
            WHERE NOT pnl_tombstone AND pnl_tie_count = 1
        ) AS turnover
    FROM one_to_one
    GROUP BY model_version, score_date
)
SELECT
    model_version,
    score_date,
    expected_prediction_keys,
    scored_rows,
    labelled_rows,
    pnl_rows,
    invalid_probability_count,
    invalid_label_count,
    ambiguous_revision_count,
    mean_probability,
    event_rate,
    brier_score,
    net_pnl,
    turnover,
    labelled_rows::double precision / NULLIF(expected_prediction_keys, 0) AS label_coverage,
    pnl_rows::double precision / NULLIF(expected_prediction_keys, 0) AS pnl_coverage,
    CASE
        WHEN invalid_probability_count > 0
          OR invalid_label_count > 0
          OR ambiguous_revision_count > 0
        THEN 'REJECT_DATA_QUALITY'
        ELSE 'EVIDENCE_READY'
    END AS evidence_status
FROM daily
ORDER BY score_date, model_version;
