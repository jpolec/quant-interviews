-- Dialect: SQLite 3.45+
SELECT
    request_id,
    order_id,
    actor_id,
    actor_role,
    tenant_id,
    portfolio_id,
    rule_version,
    model_version,
    before_snapshot_id,
    after_snapshot_id,
    input_hash,
    decision,
    reason,
    approval_id,
    approver_id,
    created_at,
    CASE
        WHEN decision NOT IN ('PASS','REQUIRES_APPROVAL','BLOCK','INVALID_OR_STALE_INPUT')
            THEN 'INVALID_DECISION'
        WHEN request_id IS NULL OR order_id IS NULL OR tenant_id IS NULL
          OR portfolio_id IS NULL OR rule_version IS NULL OR model_version IS NULL
            THEN 'INVALID_SCOPE'
        WHEN length(input_hash) <> 64 OR input_hash GLOB '*[^0-9a-f]*'
            THEN 'INVALID_HASH'
        WHEN (approval_id IS NULL) <> (approver_id IS NULL)
            THEN 'INVALID_APPROVAL_BINDING'
        WHEN decision = 'REQUIRES_APPROVAL' AND approval_id IS NULL
            THEN 'PENDING_APPROVAL'
        ELSE 'EVIDENCE_COMPLETE'
    END AS audit_status
FROM pretrade_approvals
WHERE order_id = :order_id
ORDER BY created_at, request_id;
