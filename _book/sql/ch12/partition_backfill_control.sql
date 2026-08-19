-- Dialect: SQLite 3.45+
SELECT
    p.job_name,
    p.partition_key,
    p.run_id,
    p.semantic_run_hash,
    p.status,
    p.input_hash,
    p.output_hash,
    p.attempts,
    p.maximum_attempts,
    p.last_error_code,
    CASE
        WHEN p.status NOT IN ('PENDING','RUNNING','FAILED','SUCCEEDED')
          OR p.attempts < 0
          OR p.maximum_attempts <= 0
          OR p.attempts > p.maximum_attempts
          OR length(p.semantic_run_hash) <> 64
          OR p.semantic_run_hash GLOB '*[^0-9a-f]*'
          OR length(p.input_hash) <> 64
          OR p.input_hash GLOB '*[^0-9a-f]*'
            THEN 'INVALID_METADATA'
        WHEN p.status = 'SUCCEEDED'
         AND p.output_hash IS NOT NULL
         AND length(p.output_hash) = 64
         AND p.output_hash NOT GLOB '*[^0-9a-f]*' THEN 'COMPLETE'
        WHEN p.status = 'SUCCEEDED' THEN 'INVALID_METADATA'
        WHEN p.status = 'RUNNING' THEN 'IN_PROGRESS'
        WHEN p.attempts >= p.maximum_attempts THEN 'EXHAUSTED'
        ELSE 'ELIGIBLE_FOR_RETRY'
    END AS retry_state
FROM pipeline_partitions AS p
WHERE p.job_name = :job_name
  AND p.partition_key BETWEEN :start_key AND :end_key
ORDER BY p.partition_key;
