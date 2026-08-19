-- Dialect: SQLite 3.x
-- Fail-closed accounting P&L reconciliation at the canonical NAV grain.
WITH component_rollup AS (
  SELECT
    portfolio_id,
    business_date,
    base_currency,
    extract_basis,
    valuation_timestamp,
    SUM(market_pnl) AS market_pnl,
    SUM(trading_pnl) AS trading_pnl,
    SUM(carry_pnl) AS carry_pnl,
    SUM(cash_flow_pnl) AS cash_flow_pnl,
    SUM(financing_pnl) AS financing_pnl,
    SUM(fx_translation_pnl) AS fx_translation_pnl,
    SUM(fees_pnl) AS fees_pnl,
    COUNT(*) AS component_rows,
    SUM(
      CASE
        WHEN market_pnl IS NULL
          OR trading_pnl IS NULL
          OR carry_pnl IS NULL
          OR cash_flow_pnl IS NULL
          OR financing_pnl IS NULL
          OR fx_translation_pnl IS NULL
          OR fees_pnl IS NULL
          THEN 1
        ELSE 0
      END
    ) AS invalid_component_rows
  FROM pnl_components
  GROUP BY
    portfolio_id,
    business_date,
    base_currency,
    extract_basis,
    valuation_timestamp
),
nav_grain AS (
  SELECT
    portfolio_id,
    business_date,
    base_currency,
    extract_basis,
    valuation_timestamp,
    MAX(nav_version) AS nav_version,
    COUNT(*) AS nav_rows,
    MAX(nav) AS nav,
    MAX(total_pnl) AS total_pnl,
    SUM(CASE WHEN nav IS NULL OR total_pnl IS NULL THEN 1 ELSE 0 END)
      AS invalid_nav_rows
  FROM nav_pnl
  GROUP BY
    portfolio_id,
    business_date,
    base_currency,
    extract_basis,
    valuation_timestamp
),
reconciled AS (
  SELECT
    n.*,
    c.market_pnl,
    c.trading_pnl,
    c.carry_pnl,
    c.cash_flow_pnl,
    c.financing_pnl,
    c.fx_translation_pnl,
    c.fees_pnl,
    c.component_rows,
    c.invalid_component_rows,
    n.total_pnl - (
      c.market_pnl
      + c.trading_pnl
      + c.carry_pnl
      + c.cash_flow_pnl
      + c.financing_pnl
      + c.fx_translation_pnl
      + c.fees_pnl
    ) AS residual,
    (
      ABS(c.market_pnl)
      + ABS(c.trading_pnl)
      + ABS(c.carry_pnl)
      + ABS(c.cash_flow_pnl)
      + ABS(c.financing_pnl)
      + ABS(c.fx_translation_pnl)
      + ABS(c.fees_pnl)
    ) AS gross_attribution
  FROM nav_grain AS n
  LEFT JOIN component_rollup AS c
    ON c.portfolio_id = n.portfolio_id
   AND c.business_date = n.business_date
   AND c.base_currency = n.base_currency
   AND c.extract_basis = n.extract_basis
   AND c.valuation_timestamp = n.valuation_timestamp
)
SELECT
  *,
  CASE WHEN nav > 0.0 THEN ABS(residual) / nav END AS residual_to_nav,
  CASE
    WHEN gross_attribution > 0.0 THEN ABS(residual) / gross_attribution
    WHEN gross_attribution = 0.0 AND residual = 0.0 THEN 0.0
  END AS residual_to_gross_attribution,
  CASE
    WHEN :absolute_tolerance IS NULL OR :absolute_tolerance < 0.0
      OR :nav_tolerance IS NULL OR :nav_tolerance < 0.0
      OR :gross_tolerance IS NULL OR :gross_tolerance < 0.0
      THEN 'REJECT'
    WHEN nav_rows <> 1 OR invalid_nav_rows <> 0 THEN 'REJECT'
    WHEN nav IS NULL OR nav <= 0.0 OR total_pnl IS NULL THEN 'REJECT'
    WHEN component_rows IS NULL OR invalid_component_rows <> 0 THEN 'REJECT'
    WHEN residual IS NULL OR gross_attribution IS NULL THEN 'REJECT'
    WHEN gross_attribution = 0.0 AND residual <> 0.0 THEN 'REJECT'
    WHEN ABS(residual) > :absolute_tolerance THEN 'REJECT'
    WHEN ABS(residual) / nav > :nav_tolerance THEN 'REJECT'
    WHEN gross_attribution > 0.0
      AND ABS(residual) / gross_attribution > :gross_tolerance THEN 'REJECT'
    ELSE 'PASS'
  END AS publication_status
FROM reconciled
ORDER BY portfolio_id, business_date, base_currency, extract_basis;
