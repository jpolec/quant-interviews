-- Dialect: SQLite
-- Parameter: :reconciliation_tolerance (finite, non-negative absolute amount)
WITH parameters AS (
    SELECT
        CAST(:reconciliation_tolerance AS REAL) AS tolerance,
        CASE
            WHEN :reconciliation_tolerance IS NOT NULL
             AND CAST(:reconciliation_tolerance AS REAL) >= 0.0
             AND CAST(:reconciliation_tolerance AS REAL) < 1.0e308
                THEN 1
            ELSE 0
        END AS tolerance_is_valid
),
event_ledger AS (
    SELECT
        strategy_id,
        trade_date,
        event_id,
        sequence_number,
        option_pnl,
        delta_hedge_pnl,
        financing_pnl,
        dividend_pnl,
        execution_cost,
        accounting_pnl,
        payload_hash,
        accounting_pnl - (
            option_pnl
            + delta_hedge_pnl
            + financing_pnl
            + dividend_pnl
            - execution_cost
        ) AS event_residual,
        CASE
            WHEN strategy_id IS NULL
              OR trade_date IS NULL
              OR event_id IS NULL
              OR sequence_number IS NULL
              OR option_pnl IS NULL
              OR delta_hedge_pnl IS NULL
              OR financing_pnl IS NULL
              OR dividend_pnl IS NULL
              OR execution_cost IS NULL
              OR accounting_pnl IS NULL
              OR payload_hash IS NULL
                THEN 1
            ELSE 0
        END AS has_required_null,
        CASE WHEN execution_cost < 0.0 THEN 1 ELSE 0 END AS has_negative_cost
    FROM hedge_events
),
reconciled AS (
    SELECT
        strategy_id,
        trade_date,
        COUNT(*) AS event_count,
        COUNT(DISTINCT event_id) AS distinct_event_count,
        COUNT(DISTINCT sequence_number) AS distinct_sequence_count,
        MIN(sequence_number) AS first_sequence,
        MAX(sequence_number) AS last_sequence,
        SUM(has_required_null) AS required_null_count,
        SUM(has_negative_cost) AS negative_cost_count,
        SUM(option_pnl) AS option_pnl,
        SUM(delta_hedge_pnl) AS delta_hedge_pnl,
        SUM(financing_pnl) AS financing_pnl,
        SUM(dividend_pnl) AS dividend_pnl,
        SUM(execution_cost) AS execution_cost,
        SUM(
            option_pnl
            + delta_hedge_pnl
            + financing_pnl
            + dividend_pnl
            - execution_cost
        ) AS attributed_pnl,
        SUM(accounting_pnl) AS accounting_pnl,
        SUM(event_residual) AS residual,
        MAX(ABS(event_residual)) AS maximum_absolute_event_residual
    FROM event_ledger
    GROUP BY strategy_id, trade_date
)
SELECT
    reconciled.*,
    parameters.tolerance,
    CASE
        WHEN parameters.tolerance_is_valid = 0
            THEN 'REJECT_INVALID_TOLERANCE'
        WHEN required_null_count > 0
            THEN 'REJECT_NULL'
        WHEN negative_cost_count > 0
            THEN 'REJECT_NEGATIVE_COST'
        WHEN distinct_event_count <> event_count
            THEN 'REJECT_DUPLICATE_EVENT'
        WHEN distinct_sequence_count <> event_count
            THEN 'REJECT_DUPLICATE_SEQUENCE'
        WHEN first_sequence <> 1
          OR last_sequence - first_sequence + 1 <> event_count
            THEN 'REJECT_SEQUENCE_GAP'
        WHEN maximum_absolute_event_residual > parameters.tolerance
            THEN 'REJECT_EVENT_RESIDUAL'
        WHEN ABS(residual) > parameters.tolerance
            THEN 'REJECT_AGGREGATE_RESIDUAL'
        ELSE 'PASS'
    END AS status
FROM reconciled
CROSS JOIN parameters
ORDER BY strategy_id, trade_date;
