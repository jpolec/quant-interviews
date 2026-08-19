-- Dialect: SQLite 3.x
-- Latest point-in-time holdings at the decision and knowledge cutoffs.
WITH eligible AS (
  SELECT
    p.*,
    COUNT(*) OVER (
      PARTITION BY
        p.portfolio_id,
        p.instrument_id,
        p.valid_time,
        p.knowledge_time,
        p.revision_id
    ) AS ordering_tie_count,
    ROW_NUMBER() OVER (
      PARTITION BY p.portfolio_id, p.instrument_id
      ORDER BY
        p.valid_time DESC,
        p.knowledge_time DESC,
        p.revision_id DESC
    ) AS row_number
  FROM positions_bitemporal AS p
  WHERE p.valid_time <= :decision_time
    AND p.knowledge_time <= :knowledge_cutoff
),
selected AS (
  SELECT *
  FROM eligible
  WHERE row_number = 1
)
SELECT
  portfolio_id,
  instrument_id,
  canonical_instrument_id,
  currency,
  quantity,
  market_value,
  valid_time,
  knowledge_time,
  revision_id,
  ordering_tie_count,
  CASE
    WHEN ordering_tie_count <> 1 THEN 'REJECT'
    WHEN canonical_instrument_id IS NULL OR currency IS NULL THEN 'REJECT'
    WHEN quantity IS NULL OR market_value IS NULL THEN 'REJECT'
    ELSE 'PASS'
  END AS identity_currency_status
FROM selected
ORDER BY portfolio_id, instrument_id;
