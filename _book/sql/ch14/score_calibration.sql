-- Dialect: SQLite 3.45+
-- Calibration is emitted only from unique, in-domain observations and a single
-- authoritative anchor version (or an explicit leave-one-out peer reference).
WITH duplicate_scores AS (
    SELECT interviewer_id, interview_id, COUNT(*) AS duplicate_count
    FROM interview_scores
    GROUP BY interviewer_id, interview_id
    HAVING COUNT(*) > 1
), anchor_conflicts AS (
    SELECT interview_id, COUNT(*) AS anchor_count,
           COUNT(DISTINCT anchor_version || ':' || anchor_score) AS anchor_variants
    FROM benchmark_case_anchor
    GROUP BY interview_id
), case_totals AS (
    SELECT interview_id, SUM(interviewer_score) AS score_sum, COUNT(*) AS score_count
    FROM interview_scores
    WHERE interviewer_score BETWEEN 0.0 AND 5.0
    GROUP BY interview_id
), residuals AS (
    SELECT
        score.interviewer_id,
        score.interview_id,
        score.interviewer_score,
        anchor.anchor_score,
        anchor.anchor_version,
        CASE
            WHEN duplicate.duplicate_count IS NOT NULL THEN 'INVALID_DUPLICATE_SCORE'
            WHEN score.interviewer_score NOT BETWEEN 0.0 AND 5.0 THEN 'INVALID_SCORE_DOMAIN'
            WHEN conflict.anchor_variants > 1 THEN 'INVALID_ANCHOR_CONFLICT'
            WHEN anchor.anchor_score IS NOT NULL
                 AND anchor.anchor_score NOT BETWEEN 0.0 AND 5.0 THEN 'INVALID_ANCHOR_DOMAIN'
            WHEN anchor.anchor_score IS NOT NULL THEN 'PASS'
            WHEN totals.score_count > 1 THEN 'PASS'
            ELSE 'INSUFFICIENT_REFERENCE'
        END AS row_status,
        COALESCE(
            anchor.anchor_score,
            CASE WHEN totals.score_count > 1
                 THEN (totals.score_sum - score.interviewer_score)
                      / (totals.score_count - 1) END
        ) AS reference_score,
        CASE WHEN anchor.anchor_score IS NOT NULL
             THEN 'independent_anchor:' || anchor.anchor_version
             ELSE 'leave_one_out_panel' END AS reference_type
    FROM interview_scores AS score
    LEFT JOIN duplicate_scores AS duplicate
      ON duplicate.interviewer_id = score.interviewer_id
     AND duplicate.interview_id = score.interview_id
    LEFT JOIN anchor_conflicts AS conflict USING (interview_id)
    LEFT JOIN benchmark_case_anchor AS anchor USING (interview_id)
    LEFT JOIN case_totals AS totals USING (interview_id)
), interviewer_controls AS (
    SELECT
        interviewer_id,
        COUNT(*) AS input_rows,
        SUM(CASE WHEN row_status = 'PASS' THEN 1 ELSE 0 END) AS valid_observations,
        SUM(CASE WHEN row_status <> 'PASS' THEN 1 ELSE 0 END) AS invalid_observations,
        AVG(CASE WHEN row_status = 'PASS'
                 THEN interviewer_score - reference_score END) AS mean_signed_difference,
        SQRT(AVG(CASE WHEN row_status = 'PASS'
                      THEN (interviewer_score - reference_score)
                           * (interviewer_score - reference_score) END))
            AS root_mean_squared_difference,
        MIN(CASE WHEN row_status = 'PASS' THEN reference_type END) AS reference_type,
        MIN(CASE WHEN row_status <> 'PASS' THEN row_status END) AS failure_status
    FROM residuals
    GROUP BY interviewer_id
)
SELECT
    interviewer_id,
    input_rows,
    valid_observations,
    mean_signed_difference,
    root_mean_squared_difference,
    reference_type,
    CASE
        WHEN invalid_observations > 0 THEN failure_status
        WHEN valid_observations < 2 THEN 'INSUFFICIENT_EVIDENCE'
        ELSE 'PASS'
    END AS calibration_status
FROM interviewer_controls
ORDER BY interviewer_id;
