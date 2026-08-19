-- Dialect: SQLite 3.45+
SELECT
    request_id,
    idempotency_key,
    actor_id,
    actor_role,
    tenant_id,
    portfolio_id,
    action_name,
    tool_name,
    tool_version,
    policy_version,
    input_schema_version,
    output_schema_version,
    expected_state_hash,
    input_hash,
    output_hash,
    approval_id,
    approver_id,
    precondition_status,
    postcondition_status,
    evidence_hash,
    started_at,
    completed_at,
    status,
    CASE
        WHEN request_id IS NULL OR idempotency_key IS NULL OR actor_id IS NULL
          OR tenant_id IS NULL OR portfolio_id IS NULL OR tool_name IS NULL
          OR tool_version IS NULL OR policy_version IS NULL THEN 'INVALID_SCOPE'
        WHEN status NOT IN ('SUCCEEDED','REJECTED','FAILED') THEN 'INVALID_STATUS'
        WHEN completed_at < started_at THEN 'INVALID_CLOCK_ORDER'
        WHEN length(input_hash) <> 64 OR input_hash GLOB '*[^0-9a-f]*'
          OR (output_hash IS NOT NULL AND (
              length(output_hash) <> 64 OR output_hash GLOB '*[^0-9a-f]*'
          ))
          OR length(evidence_hash) <> 64 OR evidence_hash GLOB '*[^0-9a-f]*'
            THEN 'INVALID_HASH'
        WHEN action_name IN ('publish','write','execute','cancel')
         AND (
             approval_id IS NULL OR approver_id IS NULL
             OR expected_state_hash IS NULL
             OR length(expected_state_hash) <> 64
             OR expected_state_hash GLOB '*[^0-9a-f]*'
         ) THEN 'MISSING_MUTATION_AUTHORIZATION'
        WHEN status = 'SUCCEEDED'
         AND (precondition_status <> 'PASS' OR postcondition_status <> 'PASS')
            THEN 'INVALID_SUCCESS_EVIDENCE'
        ELSE 'EVIDENCE_COMPLETE'
    END AS audit_status
FROM agent_tool_calls
WHERE started_at >= :from_time
  AND started_at < :to_time
ORDER BY started_at, request_id;
