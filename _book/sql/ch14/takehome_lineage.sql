-- Dialect: SQLite 3.45+
-- Latest deterministic reproduction with strict manifest/hash/metric controls.
WITH reproduction_key_conflicts AS (
    SELECT submission_id, COUNT(*) AS conflict_count
    FROM (
        SELECT submission_id, executed_at, reproduction_id
        FROM takehome_reproduction
        GROUP BY submission_id, executed_at, reproduction_id
        HAVING COUNT(*) > 1
    )
    GROUP BY submission_id
), ranked_reproduction AS (
    SELECT
        reproduction.*,
        ROW_NUMBER() OVER (
            PARTITION BY submission_id
            ORDER BY executed_at DESC, reproduction_id DESC
        ) AS row_number
    FROM takehome_reproduction AS reproduction
    WHERE executed_at IS NOT NULL
      AND strftime('%Y-%m-%dT%H:%M:%SZ', executed_at) = executed_at
), joined AS (
    SELECT
        submission.*,
        reproduction.reproduced_code_commit,
        reproduction.reproduced_data_hash,
        reproduction.reproduced_config_hash,
        reproduction.reproduced_environment_hash,
        reproduction.reproduced_result_hash,
        reproduction.tolerance_status,
        reproduction.metric_residual,
        reproduction.metric_tolerance,
        reproduction.failure_reason,
        reproduction.executed_at,
        COALESCE(conflict.conflict_count, 0) AS conflict_count,
        CASE WHEN submission.schema_version = '1.0'
               AND length(submission.code_commit) IN (40, 64)
               AND submission.code_commit NOT GLOB '*[^0-9a-f]*'
               AND length(submission.data_hash) = 64
               AND submission.data_hash NOT GLOB '*[^0-9a-f]*'
               AND length(submission.config_hash) = 64
               AND submission.config_hash NOT GLOB '*[^0-9a-f]*'
               AND length(submission.environment_hash) = 64
               AND submission.environment_hash NOT GLOB '*[^0-9a-f]*'
               AND length(submission.result_hash) = 64
               AND submission.result_hash NOT GLOB '*[^0-9a-f]*'
             THEN 1 ELSE 0 END AS manifest_valid
    FROM takehome_submission AS submission
    LEFT JOIN ranked_reproduction AS reproduction
      ON reproduction.submission_id = submission.submission_id
     AND reproduction.row_number = 1
    LEFT JOIN reproduction_key_conflicts AS conflict
      ON conflict.submission_id = submission.submission_id
)
SELECT
    submission_id,
    candidate_id,
    schema_version,
    code_commit,
    data_hash,
    config_hash,
    environment_hash,
    result_hash AS candidate_result_hash,
    reproduced_code_commit AS reviewer_code_commit,
    reproduced_data_hash AS reviewer_data_hash,
    reproduced_config_hash AS reviewer_config_hash,
    reproduced_environment_hash AS reviewer_environment_hash,
    reproduced_result_hash AS reviewer_result_hash,
    tolerance_status,
    metric_residual,
    metric_tolerance,
    failure_reason,
    executed_at,
    CASE
        WHEN manifest_valid = 0 THEN 'invalid_manifest'
        WHEN conflict_count > 0 THEN 'invalid_reproduction_history'
        WHEN reproduced_result_hash IS NULL THEN 'not_reproduced'
        WHEN reproduced_code_commit <> code_commit
          OR reproduced_data_hash <> data_hash
          OR reproduced_config_hash <> config_hash
          OR reproduced_environment_hash <> environment_hash
          THEN 'input_commitment_mismatch'
        WHEN length(reproduced_result_hash) <> 64
          OR reproduced_result_hash GLOB '*[^0-9a-f]*' THEN 'invalid_reproduction_hash'
        WHEN reproduced_result_hash = result_hash THEN 'bitwise_reproduced'
        WHEN tolerance_status = 'SCHEMA_COMPLETE_PASS'
          AND metric_residual IS NOT NULL
          AND metric_tolerance IS NOT NULL
          AND metric_tolerance >= 0.0
          AND abs(metric_residual) <= metric_tolerance THEN 'numerically_reproduced'
        ELSE 'mismatch'
    END AS reproducibility_status
FROM joined
ORDER BY submission_id;
