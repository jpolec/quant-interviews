-- Dialect: SQLite
-- Utility-regret gate for simulated or structurally scored first-price bids.
WITH assessed AS (
  SELECT
    *,
    COUNT(*) OVER (PARTITION BY observation_id) AS observation_copies,
    best_deviation_utility - candidate_utility AS utility_regret,
    CASE
      WHEN observation_id IS NULL OR TRIM(observation_id) = ''
        OR auction_id IS NULL OR TRIM(auction_id) = ''
        OR bidder_id IS NULL OR TRIM(bidder_id) = ''
        OR value_source NOT IN ('SIMULATED', 'STRUCTURAL_MODEL')
        OR model_version IS NULL OR TRIM(model_version) = ''
        OR tie_rule NOT IN ('uniform_among_highest_bids', 'lowest_bidder_id')
        OR as_of_utc IS NULL OR TRIM(as_of_utc) = ''
        OR bidders IS NULL OR bidders != CAST(bidders AS INTEGER) OR bidders < 2
        OR risk_aversion IS NULL OR risk_aversion < 0.0 OR risk_aversion >= 1.0
        OR value IS NULL OR value < 0.0 OR value > 1.0
        OR bid IS NULL OR bid < 0.0 OR bid > value
        OR candidate_utility IS NULL OR candidate_utility < 0.0
        OR best_deviation_utility IS NULL
        OR best_deviation_utility < candidate_utility
        OR regret_tolerance IS NULL OR regret_tolerance < 0.0
      THEN 1 ELSE 0
    END AS invalid_row
  FROM structurally_scored_auction_bids
), summary AS (
  SELECT
    bidders,
    risk_aversion,
    value_source,
    model_version,
    tie_rule,
    COUNT(*) AS rows_seen,
    COUNT(DISTINCT observation_id) AS observations,
    SUM(invalid_row) AS invalid_rows,
    SUM(CASE WHEN observation_copies > 1 THEN 1 ELSE 0 END) AS duplicate_rows,
    COUNT(DISTINCT regret_tolerance) AS tolerance_versions,
    AVG(CASE WHEN invalid_row = 0 THEN utility_regret END) AS mean_utility_regret,
    MAX(CASE WHEN invalid_row = 0 THEN utility_regret END) AS maximum_utility_regret,
    MIN(regret_tolerance) AS regret_tolerance
  FROM assessed
  GROUP BY bidders, risk_aversion, value_source, model_version, tie_rule
)
SELECT
  *,
  CASE
    WHEN invalid_rows = 0
      AND duplicate_rows = 0
      AND rows_seen = observations
      AND tolerance_versions = 1
      AND maximum_utility_regret IS NOT NULL
      AND maximum_utility_regret <= regret_tolerance
    THEN 'PASS'
    ELSE 'REJECT'
  END AS status
FROM summary
ORDER BY model_version, bidders, risk_aversion, tie_rule;
