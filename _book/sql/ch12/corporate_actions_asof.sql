-- Dialect: SQLite 3.45+
-- Returns the selected revision with an explicit eligibility certificate.
WITH eligible AS (
    SELECT
        c.*,
        DENSE_RANK() OVER (
            PARTITION BY c.action_id
            ORDER BY c.known_from DESC
        ) AS semantic_precedence,
        ROW_NUMBER() OVER (
            PARTITION BY c.action_id
            ORDER BY c.known_from DESC, c.version_id DESC, c.source_id
        ) AS technical_winner
    FROM corporate_actions AS c
    WHERE c.ex_date <= :market_date
      AND c.announced_at <= :knowledge_time
      AND c.known_from <= :knowledge_time
      AND (c.known_to IS NULL OR :knowledge_time < c.known_to)
),
summary AS (
    SELECT
        action_id,
        COUNT(DISTINCT (
            COALESCE(action_type, '<NULL>') || char(31) ||
            COALESCE(action_status, '<NULL>') || char(31) ||
            COALESCE(printf('%.17g', ratio), '<NULL>') || char(31) ||
            COALESCE(printf('%.17g', cash_amount), '<NULL>') || char(31) ||
            COALESCE(child_instrument_id, '<NULL>') || char(31) ||
            COALESCE(successor_instrument_id, '<NULL>')
        )) AS semantic_choices
    FROM eligible
    WHERE semantic_precedence = 1
    GROUP BY action_id
)
SELECT
    e.action_id,
    e.version_id,
    e.source_id,
    e.instrument_id,
    e.action_type,
    e.announced_at,
    e.ex_date,
    e.record_date,
    e.payable_date,
    e.known_from,
    e.known_to,
    e.action_status,
    e.ratio,
    e.cash_amount,
    e.currency,
    e.child_instrument_id,
    e.successor_instrument_id,
    CASE
        WHEN s.semantic_choices > 1 THEN 'AMBIGUOUS'
        WHEN e.action_type NOT IN (
            'SPLIT','CASH_DIVIDEND','SPINOFF','RIGHTS','SYMBOL_CHANGE','DELISTING'
        ) THEN 'INVALID_ACTION_TYPE'
        WHEN e.action_status NOT IN ('ANNOUNCED','CONFIRMED','CANCELLED','TOMBSTONE')
            THEN 'INVALID_ACTION_STATUS'
        WHEN e.record_date IS NOT NULL AND e.record_date < e.ex_date
            THEN 'INVALID_DATE_ORDER'
        WHEN e.payable_date IS NOT NULL
         AND e.record_date IS NOT NULL
         AND e.payable_date < e.record_date THEN 'INVALID_DATE_ORDER'
        WHEN e.action_status IN ('CANCELLED','TOMBSTONE') THEN 'WITHDRAWN'
        WHEN e.action_status <> 'CONFIRMED' THEN 'NOT_CONFIRMED'
        WHEN e.ratio IS NULL OR e.ratio <= 0 THEN 'INVALID_RATIO'
        ELSE 'SELECTED'
    END AS selection_status
FROM eligible AS e
JOIN summary AS s USING (action_id)
WHERE e.technical_winner = 1
ORDER BY e.ex_date, e.action_id;
