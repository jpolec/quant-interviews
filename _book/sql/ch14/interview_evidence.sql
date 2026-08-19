-- Dialect: SQLite 3.45+
-- Point-in-time evidence snapshot.  Any malformed row or identity conflict makes
-- the candidate snapshot fail closed (zero rows) rather than silently ranking it.
WITH parameters AS (
    SELECT :decision_time AS decision_time
), input_validation AS (
    SELECT
        SUM(CASE WHEN candidate_id IS NULL OR trim(candidate_id) = ''
                   OR competency IS NULL OR trim(competency) = ''
                   OR source_id IS NULL OR trim(source_id) = ''
                   OR schema_version <> '1.0'
                   OR score NOT BETWEEN 0.0 AND 5.0
                   OR confidence NOT BETWEEN 0.0 AND 1.0
                   OR observed_at IS NULL
                   OR strftime('%Y-%m-%dT%H:%M:%SZ', observed_at) <> observed_at
                 THEN 1 ELSE 0 END) AS invalid_rows,
        SUM(CASE WHEN evidence_id IS NULL THEN 1 ELSE 0 END) AS missing_ids
    FROM interview_evidence
), identity_validation AS (
    SELECT COUNT(*) AS conflicting_ids
    FROM (
        SELECT evidence_id
        FROM interview_evidence
        GROUP BY evidence_id
        HAVING COUNT(*) > 1
    )
), ranked AS (
    SELECT
        evidence.candidate_id,
        evidence.competency,
        evidence.score,
        evidence.confidence,
        evidence.observed_at,
        evidence.evidence_id,
        evidence.source_id,
        evidence.schema_version,
        ROW_NUMBER() OVER (
            PARTITION BY evidence.candidate_id, evidence.competency
            ORDER BY evidence.observed_at DESC, evidence.evidence_id DESC
        ) AS row_number
    FROM interview_evidence AS evidence
    CROSS JOIN parameters
    CROSS JOIN input_validation
    CROSS JOIN identity_validation
    WHERE input_validation.invalid_rows = 0
      AND input_validation.missing_ids = 0
      AND identity_validation.conflicting_ids = 0
      AND strftime('%Y-%m-%dT%H:%M:%SZ', parameters.decision_time)
            = parameters.decision_time
      AND evidence.observed_at <= parameters.decision_time
)
SELECT
    candidate_id,
    competency,
    score,
    confidence,
    observed_at,
    evidence_id,
    source_id,
    schema_version
FROM ranked
WHERE row_number = 1
ORDER BY candidate_id, competency;
