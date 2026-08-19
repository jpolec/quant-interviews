-- Dialect: SQLite 3.45+
BEGIN IMMEDIATE;

INSERT INTO research_runs(
    run_id,
    semantic_run_hash,
    code_hash,
    data_hash,
    config_hash,
    seed,
    output_hash,
    created_at,
    status
)
SELECT
    :run_id,
    :semantic_run_hash,
    :code_hash,
    :data_hash,
    :config_hash,
    :seed,
    :output_hash,
    CURRENT_TIMESTAMP,
    :status
WHERE length(:run_id) > 0
  AND :status IN ('SUCCEEDED','FAILED')
  AND length(:semantic_run_hash) = 64
  AND :semantic_run_hash NOT GLOB '*[^0-9a-f]*'
  AND length(:code_hash) = 64 AND :code_hash NOT GLOB '*[^0-9a-f]*'
  AND length(:data_hash) = 64 AND :data_hash NOT GLOB '*[^0-9a-f]*'
  AND length(:config_hash) = 64 AND :config_hash NOT GLOB '*[^0-9a-f]*'
  AND length(:output_hash) = 64 AND :output_hash NOT GLOB '*[^0-9a-f]*'
ON CONFLICT(run_id) DO NOTHING;

SELECT
    COALESCE(r.run_id, :run_id) AS run_id,
    COALESCE(r.semantic_run_hash, :semantic_run_hash) AS semantic_run_hash,
    COALESCE(r.code_hash, :code_hash) AS code_hash,
    COALESCE(r.data_hash, :data_hash) AS data_hash,
    COALESCE(r.config_hash, :config_hash) AS config_hash,
    COALESCE(r.seed, :seed) AS seed,
    COALESCE(r.output_hash, :output_hash) AS output_hash,
    COALESCE(r.status, :status) AS status,
    CASE
        WHEN length(:run_id) = 0 OR :status NOT IN ('SUCCEEDED','FAILED')
          OR length(:semantic_run_hash) <> 64
          OR :semantic_run_hash GLOB '*[^0-9a-f]*'
          OR length(:code_hash) <> 64 OR :code_hash GLOB '*[^0-9a-f]*'
          OR length(:data_hash) <> 64 OR :data_hash GLOB '*[^0-9a-f]*'
          OR length(:config_hash) <> 64 OR :config_hash GLOB '*[^0-9a-f]*'
          OR length(:output_hash) <> 64 OR :output_hash GLOB '*[^0-9a-f]*'
            THEN 'INVALID_INPUT'
        WHEN changes() = 1 THEN 'REGISTERED'
        WHEN r.run_id IS NOT NULL
         AND r.semantic_run_hash = :semantic_run_hash
         AND r.code_hash = :code_hash
         AND r.data_hash = :data_hash
         AND r.config_hash = :config_hash
         AND r.seed = :seed
         AND r.output_hash = :output_hash
         AND r.status = :status THEN 'DUPLICATE_SAME_RUN'
        ELSE 'CONFLICT'
    END AS registry_status
FROM (SELECT 1) AS singleton
LEFT JOIN research_runs AS r ON r.run_id = :run_id;

COMMIT;
