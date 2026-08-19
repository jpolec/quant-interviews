-- Dialect: SQLite 3.45+
-- Recompute the expected canonical decision from point-in-time, role-scoped
-- evidence and compare it with the recorded debrief decision.
WITH ranked_evidence AS (
    SELECT
        decision.interview_id,
        requirement.competency,
        requirement.weight,
        requirement.minimum_quality,
        requirement.hard_floor,
        evidence.quality,
        evidence.score,
        ROW_NUMBER() OVER (
            PARTITION BY decision.interview_id, requirement.competency
            ORDER BY evidence.observed_at DESC, evidence.evidence_id DESC
        ) AS row_number
    FROM interview_decision AS decision
    JOIN required_competency AS requirement
      ON requirement.role_id = decision.role_id
     AND requirement.policy_version = decision.policy_version
    LEFT JOIN competency_evidence AS evidence
      ON evidence.interview_id = decision.interview_id
     AND evidence.competency = requirement.competency
     AND evidence.observed_at <= decision.decision_time
), current_evidence AS (
    SELECT * FROM ranked_evidence WHERE row_number = 1 OR row_number IS NULL
), controls AS (
    SELECT
        decision.interview_id,
        decision.final_decision,
        decision.final_score,
        decision.completeness_threshold,
        SUM(current.weight) AS required_weight,
        SUM(CASE WHEN current.quality >= current.minimum_quality
                 THEN current.weight ELSE 0.0 END) AS observed_weight,
        SUM(CASE WHEN current.score < current.hard_floor THEN 1 ELSE 0 END)
            AS hard_floor_violations,
        SUM(CASE WHEN current.score IS NULL THEN 1 ELSE 0 END) AS missing_competencies
    FROM interview_decision AS decision
    JOIN current_evidence AS current USING (interview_id)
    GROUP BY decision.interview_id, decision.final_decision,
             decision.final_score, decision.completeness_threshold
), expected AS (
    SELECT *,
        observed_weight / NULLIF(required_weight, 0.0) AS weighted_completeness,
        CASE
            WHEN required_weight IS NULL OR required_weight <= 0.0
              OR missing_competencies > 0
              OR observed_weight / NULLIF(required_weight, 0.0)
                   < completeness_threshold THEN 'INSUFFICIENT_EVIDENCE'
            WHEN hard_floor_violations > 0 THEN 'REJECT'
            WHEN final_score >= 4.0 THEN 'ACCEPT'
            WHEN final_score >= 3.0 THEN 'CONDITIONAL_ACCEPT'
            ELSE 'REJECT'
        END AS expected_decision
    FROM controls
)
SELECT
    interview_id,
    final_decision,
    final_score,
    weighted_completeness,
    missing_competencies,
    hard_floor_violations,
    expected_decision,
    CASE WHEN final_decision = expected_decision
         THEN 'consistent' ELSE 'decision_conflict' END AS control_status
FROM expected
ORDER BY interview_id;
