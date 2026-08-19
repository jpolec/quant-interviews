-- Dialect: SQLite
-- Point-in-time implementation-shortfall reconciliation with explicit data-
-- quality gates. Zero fills are valid and retain a NULL average fill price.
WITH request AS (
    SELECT
        :order_id AS order_id,
        :knowledge_cutoff AS knowledge_cutoff,
        :end_time AS end_time
),
order_candidates AS (
    SELECT o.*
    FROM parent_orders AS o
    JOIN request AS r ON r.order_id = o.order_id
),
order_diagnostics AS (
    SELECT
        COUNT(*) AS order_count,
        COALESCE(SUM(CASE
            WHEN side NOT IN ('BUY', 'SELL')
              OR target_quantity IS NULL OR target_quantity <= 0
              OR decision_price IS NULL OR decision_price <= 0
              OR arrival_price IS NULL OR arrival_price <= 0
              OR decision_time IS NULL OR arrival_time IS NULL
              OR decision_time > arrival_time
            THEN 1 ELSE 0
        END), 0) AS invalid_order_count
    FROM order_candidates
),
order_base AS (
    SELECT *
    FROM order_candidates
    LIMIT 1
),
ranked_fills AS (
    SELECT
        f.*,
        ROW_NUMBER() OVER (
            PARTITION BY f.fill_id
            ORDER BY f.knowledge_time DESC, f.revision_id DESC
        ) AS revision_rank,
        COUNT(*) OVER (
            PARTITION BY f.fill_id, f.knowledge_time, f.revision_id
        ) AS revision_tie_count
    FROM child_fills AS f
    JOIN order_base AS b ON b.order_id = f.order_id
    JOIN request AS r
    WHERE f.knowledge_time <= r.knowledge_cutoff
      AND f.fill_time BETWEEN b.arrival_time AND r.end_time
),
latest_fills AS (
    SELECT *
    FROM ranked_fills
    WHERE revision_rank = 1
),
fill_diagnostics AS (
    SELECT
        COALESCE(SUM(CASE WHEN revision_tie_count > 1 THEN 1 ELSE 0 END), 0)
            AS ambiguous_fill_revision_count,
        COALESCE(SUM(CASE
            WHEN is_cancelled NOT IN (0, 1)
              OR fill_quantity IS NULL OR fill_quantity <= 0
              OR fill_price IS NULL OR fill_price <= 0
              OR fill_time IS NULL OR knowledge_time IS NULL
            THEN 1 ELSE 0
        END), 0) AS invalid_fill_count
    FROM latest_fills
),
fills AS (
    SELECT
        COALESCE(SUM(CASE WHEN is_cancelled = 0 THEN fill_quantity ELSE 0 END), 0)
            AS executed_quantity,
        SUM(CASE WHEN is_cancelled = 0 THEN fill_quantity * fill_price END)
            / NULLIF(SUM(CASE WHEN is_cancelled = 0 THEN fill_quantity END), 0)
            AS average_fill_price
    FROM latest_fills
),
ranked_marks AS (
    SELECT
        m.*,
        ROW_NUMBER() OVER (
            PARTITION BY m.symbol
            ORDER BY m.valid_time DESC, m.knowledge_time DESC, m.revision_id DESC
        ) AS revision_rank,
        COUNT(*) OVER (
            PARTITION BY m.symbol, m.valid_time, m.knowledge_time, m.revision_id
        ) AS revision_tie_count
    FROM market_marks AS m
    JOIN order_base AS b ON b.symbol = m.symbol
    JOIN request AS r
    WHERE m.valid_time <= r.end_time
      AND m.knowledge_time <= r.knowledge_cutoff
),
close_mark AS (
    SELECT market_price, revision_tie_count
    FROM ranked_marks
    WHERE revision_rank = 1
),
components AS (
    SELECT
        r.order_id AS requested_order_id,
        b.*,
        f.executed_quantity,
        f.average_fill_price,
        c.market_price AS close_mark,
        CASE WHEN b.side = 'BUY' THEN 1.0
             WHEN b.side = 'SELL' THEN -1.0 END AS side_sign,
        od.order_count,
        od.invalid_order_count,
        fd.ambiguous_fill_revision_count,
        fd.invalid_fill_count,
        COALESCE(c.revision_tie_count, 0) AS ambiguous_mark_count
    FROM request AS r
    CROSS JOIN order_diagnostics AS od
    CROSS JOIN fill_diagnostics AS fd
    CROSS JOIN fills AS f
    LEFT JOIN order_base AS b ON 1 = 1
    LEFT JOIN close_mark AS c ON 1 = 1
),
assessment AS (
    SELECT
        *,
        CASE
            WHEN requested_order_id IS NULL OR TRIM(requested_order_id) = ''
                THEN 'REJECT_INVALID_PARAMETER'
            WHEN order_count = 0 THEN 'REJECT_MISSING_ORDER'
            WHEN order_count > 1 THEN 'REJECT_DUPLICATE_ORDER'
            WHEN invalid_order_count > 0 OR arrival_time > (SELECT end_time FROM request)
                THEN 'REJECT_INVALID_ORDER'
            WHEN ambiguous_fill_revision_count > 0
                THEN 'REJECT_AMBIGUOUS_FILL_REVISION'
            WHEN invalid_fill_count > 0 THEN 'REJECT_INVALID_FILL'
            WHEN executed_quantity > target_quantity THEN 'REJECT_OVERFILL'
            WHEN ambiguous_mark_count > 1 THEN 'REJECT_AMBIGUOUS_CLOSE_MARK'
            WHEN close_mark IS NULL THEN 'REJECT_MISSING_CLOSE_MARK'
            WHEN close_mark <= 0 THEN 'REJECT_INVALID_CLOSE_MARK'
            ELSE 'PASS'
        END AS status
    FROM components
)
SELECT
    requested_order_id AS order_id,
    target_quantity,
    CASE WHEN status = 'PASS' THEN executed_quantity END AS executed_quantity,
    CASE WHEN status = 'PASS' THEN average_fill_price END AS average_fill_price,
    close_mark,
    side_sign,
    CASE WHEN status = 'PASS'
        THEN side_sign * target_quantity * (arrival_price - decision_price) END
        AS delay_cost,
    CASE WHEN status = 'PASS' AND executed_quantity = 0 THEN 0.0
         WHEN status = 'PASS'
        THEN side_sign * executed_quantity
            * (average_fill_price - arrival_price) END AS trading_cost,
    CASE WHEN status = 'PASS'
        THEN side_sign * (target_quantity - executed_quantity)
            * (close_mark - arrival_price) END AS opportunity_cost,
    CASE WHEN status = 'PASS'
        THEN side_sign * (
            target_quantity * (arrival_price - decision_price)
            + CASE WHEN executed_quantity = 0 THEN 0.0
                   ELSE executed_quantity * (average_fill_price - arrival_price) END
            + (target_quantity - executed_quantity) * (close_mark - arrival_price)
        ) END AS implementation_shortfall,
    ambiguous_fill_revision_count,
    invalid_fill_count,
    status
FROM assessment;
