-- Dialect: SQLite 3.45+
SELECT
    break_type,
    entity_key,
    internal_value,
    external_value,
    tolerance,
    CASE
        WHEN tolerance IS NULL OR tolerance < 0 THEN 'INVALID_TOLERANCE'
        WHEN internal_value IS NULL THEN 'MISSING_INTERNAL'
        WHEN external_value IS NULL THEN 'MISSING_EXTERNAL'
        WHEN ABS(internal_value - external_value) > tolerance THEN 'MATERIAL'
        ELSE 'MATCH'
    END AS break_state,
    CASE
        WHEN internal_value IS NULL OR external_value IS NULL THEN NULL
        ELSE internal_value - external_value
    END AS difference
FROM reconciliation_results
WHERE run_id = :run_id
  AND (
      tolerance IS NULL
      OR tolerance < 0
      OR internal_value IS NULL
      OR external_value IS NULL
      OR ABS(internal_value - external_value) > tolerance
  )
ORDER BY
    CASE
        WHEN tolerance IS NULL OR tolerance < 0 THEN 0
        WHEN internal_value IS NULL OR external_value IS NULL THEN 1
        ELSE 2
    END,
    ABS(COALESCE(internal_value - external_value, 0)) DESC,
    entity_key;
