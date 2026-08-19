-- Dialect: PostgreSQL
-- Input schema and deterministic positive/negative fixtures live in the CH02
-- SQL validator. This query is reusable against the canonical covariance_run.
WITH run_rollup AS (
    SELECT
        run_id,
        COUNT(*) AS record_count,
        SUM(
            CASE
                WHEN snapshot_hash IS NULL OR TRIM(snapshot_hash) = ''
                  OR estimator_configuration_hash IS NULL
                  OR TRIM(estimator_configuration_hash) = ''
                THEN 1 ELSE 0
            END
        ) AS missing_hash_count,
        SUM(
            CASE
                WHEN estimator IS NULL OR TRIM(estimator) = ''
                  OR estimator_version IS NULL OR TRIM(estimator_version) = ''
                THEN 1 ELSE 0
            END
        ) AS missing_estimator_count,
        SUM(
            CASE
                WHEN window_start IS NULL OR window_end IS NULL
                  OR window_start >= window_end
                  OR knowledge_cutoff IS NULL
                  OR window_end > knowledge_cutoff
                THEN 1 ELSE 0
            END
        ) AS invalid_window_count,
        SUM(
            CASE
                WHEN maximum_observation_knowledge_time IS NULL
                  OR knowledge_cutoff IS NULL
                  OR maximum_observation_knowledge_time > knowledge_cutoff
                  OR created_at IS NULL
                  OR knowledge_cutoff > created_at
                THEN 1 ELSE 0
            END
        ) AS point_in_time_breach_count,
        SUM(CASE WHEN published <> 1 OR published IS NULL THEN 1 ELSE 0 END)
            AS unpublished_count,
        MIN(stored_decision_state) AS minimum_stored_decision_state,
        MAX(stored_decision_state) AS maximum_stored_decision_state
    FROM covariance_run
    GROUP BY run_id
), evidence_decision AS (
    SELECT
        run_id,
        record_count,
        missing_hash_count,
        missing_estimator_count,
        invalid_window_count,
        point_in_time_breach_count,
        unpublished_count,
        minimum_stored_decision_state,
        maximum_stored_decision_state,
        CASE
            WHEN record_count <> 1 THEN 'DUPLICATE_RUN_ID'
            WHEN missing_hash_count > 0 THEN 'MISSING_LINEAGE_HASH'
            WHEN missing_estimator_count > 0 THEN 'MISSING_ESTIMATOR_LINEAGE'
            WHEN invalid_window_count > 0 THEN 'INVALID_ESTIMATION_WINDOW'
            WHEN point_in_time_breach_count > 0 THEN 'FUTURE_KNOWLEDGE'
            WHEN unpublished_count > 0 THEN 'RUN_NOT_PUBLISHED'
            ELSE 'COMPLETE_POINT_IN_TIME_LINEAGE'
        END AS evidence_reason,
        CASE
            WHEN record_count = 1
             AND missing_hash_count = 0
             AND missing_estimator_count = 0
             AND invalid_window_count = 0
             AND point_in_time_breach_count = 0
             AND unpublished_count = 0
            THEN 'PASS'
            ELSE 'REJECT'
        END AS computed_decision_state
    FROM run_rollup
)
SELECT
    run_id,
    record_count,
    missing_hash_count,
    missing_estimator_count,
    invalid_window_count,
    point_in_time_breach_count,
    unpublished_count,
    minimum_stored_decision_state AS stored_decision_state,
    computed_decision_state,
    CASE
        WHEN minimum_stored_decision_state IS NULL
          OR minimum_stored_decision_state <> maximum_stored_decision_state
          OR minimum_stored_decision_state <> computed_decision_state
        THEN 'STORED_DECISION_MISMATCH'
        ELSE evidence_reason
    END AS violation_reason
FROM evidence_decision;
