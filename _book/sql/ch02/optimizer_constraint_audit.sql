-- Dialect: PostgreSQL
-- Input schema and fixtures live in the CH02 SQL validator. The query joins the
-- canonical run, weights, constraint coefficients and signed KKT evidence.
WITH constraint_evaluation AS (
SELECT
    coefficient.run_id,
    coefficient.constraint_name,
    coefficient.constraint_kind,
    MIN(coefficient.target_value) AS target_value,
    MIN(coefficient.tolerance) AS tolerance,
    SUM(coefficient.coefficient * output.weight) AS realized_value,
    ABS(
        SUM(coefficient.coefficient * output.weight)
        - MIN(coefficient.target_value)
    ) AS absolute_residual,
    COUNT(output.asset) AS joined_asset_count
FROM optimizer_constraint_coefficient AS coefficient
LEFT JOIN optimizer_output AS output
  ON output.run_id = coefficient.run_id
 AND output.asset = coefficient.asset
GROUP BY
    coefficient.run_id,
    coefficient.constraint_name,
    coefficient.constraint_kind
), run_rollup AS (
    SELECT
        run.run_id,
        COUNT(DISTINCT run.record_id) AS run_record_count,
        MIN(run.covariance_hash) AS covariance_hash,
        MIN(run.constraints_hash) AS constraints_hash,
        MIN(run.targets_hash) AS targets_hash,
        MIN(run.solver_version) AS solver_version,
        MIN(run.configuration_hash) AS configuration_hash,
        MIN(run.decision_time) AS decision_time,
        MIN(run.covariance_maximum_knowledge_time)
            AS covariance_maximum_knowledge_time,
        MIN(run.constraints_knowledge_time) AS constraints_knowledge_time,
        MIN(run.created_at) AS created_at,
        MIN(run.expected_asset_count) AS expected_asset_count,
        MIN(run.expected_constraint_count) AS expected_constraint_count,
        MIN(run.primal_tolerance) AS primal_tolerance,
        MIN(run.stationarity_tolerance) AS stationarity_tolerance,
        MIN(run.dual_tolerance) AS dual_tolerance,
        MIN(run.complementarity_tolerance) AS complementarity_tolerance
    FROM optimizer_run AS run
    GROUP BY run.run_id
), output_rollup AS (
    SELECT
        run_id,
        COUNT(*) AS output_asset_count,
        SUM(weight) AS budget,
        SUM(weight * expected_return) AS portfolio_return
    FROM optimizer_output
    GROUP BY run_id
), constraint_rollup AS (
    SELECT
        run_id,
        COUNT(*) AS evaluated_constraint_count,
        MAX(absolute_residual) AS maximum_constraint_residual,
        SUM(
            CASE WHEN absolute_residual > tolerance THEN 1 ELSE 0 END
        ) AS constraint_breach_count,
        SUM(
            CASE
                WHEN constraint_kind = 'FACTOR_NEUTRALITY'
                 AND absolute_residual > tolerance
                THEN 1 ELSE 0
            END
        ) AS factor_constraint_breach_count,
        MIN(joined_asset_count) AS minimum_joined_asset_count
    FROM constraint_evaluation
    GROUP BY run_id
), kkt_rollup AS (
    SELECT
        run_id,
        COUNT(*) AS kkt_evidence_count,
        MIN(covariance_hash) AS kkt_covariance_hash,
        MIN(constraints_hash) AS kkt_constraints_hash,
        MIN(configuration_hash) AS kkt_configuration_hash,
        MIN(solver_version) AS kkt_solver_version,
        MIN(multipliers_json) AS multipliers_json,
        MIN(active_constraints_json) AS active_constraints_json,
        MAX(primal_residual) AS primal_residual,
        MAX(stationarity_residual) AS stationarity_residual,
        MAX(dual_feasibility_violation) AS dual_feasibility_violation,
        MAX(complementarity_residual) AS complementarity_residual,
        MIN(decision_state) AS minimum_stored_decision_state,
        MAX(decision_state) AS maximum_stored_decision_state
    FROM optimizer_kkt_evidence
    GROUP BY run_id
), joined_evidence AS (
    SELECT
        run.*,
        COALESCE(output.output_asset_count, 0) AS output_asset_count,
        output.budget,
        output.portfolio_return,
        COALESCE(constraint_evidence.evaluated_constraint_count, 0)
            AS evaluated_constraint_count,
        constraint_evidence.maximum_constraint_residual,
        COALESCE(constraint_evidence.constraint_breach_count, 0)
            AS constraint_breach_count,
        COALESCE(constraint_evidence.factor_constraint_breach_count, 0)
            AS factor_constraint_breach_count,
        COALESCE(constraint_evidence.minimum_joined_asset_count, 0)
            AS minimum_joined_asset_count,
        COALESCE(kkt.kkt_evidence_count, 0) AS kkt_evidence_count,
        kkt.kkt_covariance_hash,
        kkt.kkt_constraints_hash,
        kkt.kkt_configuration_hash,
        kkt.kkt_solver_version,
        kkt.multipliers_json,
        kkt.active_constraints_json,
        kkt.primal_residual,
        kkt.stationarity_residual,
        kkt.dual_feasibility_violation,
        kkt.complementarity_residual,
        kkt.minimum_stored_decision_state,
        kkt.maximum_stored_decision_state
    FROM run_rollup AS run
    LEFT JOIN output_rollup AS output ON output.run_id = run.run_id
    LEFT JOIN constraint_rollup AS constraint_evidence
      ON constraint_evidence.run_id = run.run_id
    LEFT JOIN kkt_rollup AS kkt ON kkt.run_id = run.run_id
), evaluated AS (
    SELECT
        joined_evidence.*,
        CASE
            WHEN run_record_count <> 1 THEN 'DUPLICATE_RUN_ID'
            WHEN covariance_hash IS NULL OR TRIM(covariance_hash) = ''
              OR constraints_hash IS NULL OR TRIM(constraints_hash) = ''
              OR targets_hash IS NULL OR TRIM(targets_hash) = ''
              OR configuration_hash IS NULL OR TRIM(configuration_hash) = ''
              OR solver_version IS NULL OR TRIM(solver_version) = ''
            THEN 'MISSING_LINEAGE_HASH_OR_VERSION'
            WHEN decision_time IS NULL
              OR covariance_maximum_knowledge_time IS NULL
              OR constraints_knowledge_time IS NULL
              OR created_at IS NULL
              OR covariance_maximum_knowledge_time > decision_time
              OR constraints_knowledge_time > decision_time
              OR decision_time > created_at
            THEN 'FUTURE_KNOWLEDGE'
            WHEN output_asset_count <> expected_asset_count
            THEN 'INCOMPLETE_OPTIMIZER_OUTPUT'
            WHEN kkt_evidence_count <> 1
              OR multipliers_json IS NULL OR TRIM(multipliers_json) = ''
              OR active_constraints_json IS NULL
              OR TRIM(active_constraints_json) = ''
              OR kkt_covariance_hash IS NULL
              OR kkt_constraints_hash IS NULL
              OR kkt_configuration_hash IS NULL
              OR kkt_solver_version IS NULL
            THEN 'INCOMPLETE_KKT_EVIDENCE'
            WHEN kkt_covariance_hash <> covariance_hash
              OR kkt_constraints_hash <> constraints_hash
              OR kkt_configuration_hash <> configuration_hash
              OR kkt_solver_version <> solver_version
            THEN 'LINEAGE_MISMATCH'
            WHEN evaluated_constraint_count <> expected_constraint_count
              OR minimum_joined_asset_count <> expected_asset_count
            THEN 'INCOMPLETE_CONSTRAINT_EVIDENCE'
            WHEN factor_constraint_breach_count > 0
            THEN 'FACTOR_NEUTRALITY_BREACH'
            WHEN constraint_breach_count > 0
            THEN 'CONSTRAINT_RESIDUAL_LIMIT_EXCEEDED'
            WHEN primal_residual IS NULL
              OR stationarity_residual IS NULL
              OR dual_feasibility_violation IS NULL
              OR complementarity_residual IS NULL
              OR primal_residual > primal_tolerance
              OR stationarity_residual > stationarity_tolerance
              OR dual_feasibility_violation > dual_tolerance
              OR complementarity_residual > complementarity_tolerance
            THEN 'KKT_RESIDUAL_LIMIT_EXCEEDED'
            ELSE 'ALL_SIGNED_GATES_PASS'
        END AS evidence_reason
    FROM joined_evidence
), decision AS (
    SELECT
        evaluated.*,
        CASE
            WHEN evidence_reason = 'ALL_SIGNED_GATES_PASS' THEN 'PASS'
            ELSE 'REJECT'
        END AS computed_decision_state
    FROM evaluated
)
SELECT
    run_id,
    covariance_hash,
    constraints_hash,
    targets_hash,
    solver_version,
    configuration_hash,
    decision_time,
    covariance_maximum_knowledge_time,
    constraints_knowledge_time,
    output_asset_count,
    expected_asset_count,
    budget,
    portfolio_return,
    evaluated_constraint_count,
    expected_constraint_count,
    maximum_constraint_residual,
    constraint_breach_count,
    factor_constraint_breach_count,
    primal_residual,
    stationarity_residual,
    dual_feasibility_violation,
    complementarity_residual,
    minimum_stored_decision_state AS stored_decision_state,
    computed_decision_state,
    CASE
        WHEN minimum_stored_decision_state IS NULL
          OR minimum_stored_decision_state <> maximum_stored_decision_state
          OR minimum_stored_decision_state <> computed_decision_state
        THEN 'STORED_DECISION_MISMATCH'
        ELSE evidence_reason
    END AS violation_reason
FROM decision;
