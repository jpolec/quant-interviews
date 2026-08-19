-- Dialect: SQLite 3.x
-- Fail-closed scenario coverage from an immutable, unduplicated universe.
WITH universe_keys AS (
  SELECT
    run_id,
    portfolio_id,
    instrument_id,
    COUNT(*) AS universe_rows,
    MAX(market_value) AS market_value
  FROM scenario_run_positions
  GROUP BY run_id, portfolio_id, instrument_id
),
result_keys AS (
  SELECT
    run_id,
    portfolio_id,
    instrument_id,
    COUNT(*) AS result_rows,
    SUM(CASE WHEN repricing_status = 'SUCCESS' THEN 1 ELSE 0 END)
      AS success_rows
  FROM scenario_results
  GROUP BY run_id, portfolio_id, instrument_id
),
joined AS (
  SELECT
    u.*,
    COALESCE(r.result_rows, 0) AS result_rows,
    COALESCE(r.success_rows, 0) AS success_rows
  FROM universe_keys AS u
  LEFT JOIN result_keys AS r
    ON r.run_id = u.run_id
   AND r.portfolio_id = u.portfolio_id
   AND r.instrument_id = u.instrument_id
),
coverage AS (
  SELECT
    run_id,
    SUM(CASE WHEN market_value IS NOT NULL THEN ABS(market_value) ELSE 0.0 END)
      AS gross_market_value,
    SUM(
      CASE
        WHEN universe_rows = 1
          AND result_rows = 1
          AND success_rows = 1
          AND market_value IS NOT NULL
          THEN ABS(market_value)
        ELSE 0.0
      END
    ) AS covered_market_value,
    SUM(CASE WHEN result_rows = 0 OR success_rows <> 1 THEN 1 ELSE 0 END)
      AS failed_instruments,
    SUM(CASE WHEN result_rows > 1 THEN 1 ELSE 0 END) AS duplicate_results,
    SUM(CASE WHEN universe_rows > 1 THEN 1 ELSE 0 END) AS duplicate_universe_rows,
    SUM(CASE WHEN market_value IS NULL THEN 1 ELSE 0 END) AS null_market_values,
    COUNT(*) AS universe_instruments
  FROM joined
  GROUP BY run_id
)
SELECT
  *,
  CASE
    WHEN gross_market_value > 0.0
      THEN covered_market_value / gross_market_value
  END AS market_value_coverage,
  CASE
    WHEN :minimum_coverage IS NULL
      OR :minimum_coverage < 0.0
      OR :minimum_coverage > 1.0 THEN 'REJECT'
    WHEN null_market_values > 0 OR gross_market_value <= 0.0 THEN 'INCOMPLETE'
    WHEN duplicate_universe_rows > 0 OR duplicate_results > 0 THEN 'REJECT'
    WHEN failed_instruments > 0 THEN 'REJECT'
    WHEN covered_market_value / gross_market_value < :minimum_coverage
      THEN 'REJECT'
    ELSE 'PASS'
  END AS publication_status
FROM coverage
ORDER BY run_id;
