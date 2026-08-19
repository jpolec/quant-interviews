-- Dialect: SQLite 3.35+
-- Governed fixed-income P&L reconciliation.
-- The contract requires immutable risk_run_id/market_run_id columns on risk
-- evidence and expected_gamma_pair_count on each position.  Every mandatory
-- scalar, bucket, gamma pair and run/state dependency is fail-closed.
WITH position_rows AS (
    SELECT
        p.*,
        COUNT(*) OVER (
            PARTITION BY valuation_date, book_id, instrument_id
        ) AS position_source_row_count
    FROM position_pnl AS p
), risk_rows AS (
    SELECT
        valuation_date,
        book_id,
        instrument_id,
        state_id,
        risk_run_id,
        curve_id,
        key_rate_tenor,
        MAX(dv01) AS dv01,
        COUNT(*) AS source_row_count,
        SUM(
            CASE WHEN dv01 IS NULL OR ABS(dv01) > 1.0e308
                 THEN 1 ELSE 0 END
        ) AS null_value_count
    FROM curve_risk_bucket
    GROUP BY
        valuation_date,
        book_id,
        instrument_id,
        state_id,
        risk_run_id,
        curve_id,
        key_rate_tenor
), move_rows AS (
    SELECT
        valuation_date,
        state_id,
        market_run_id,
        curve_id,
        key_rate_tenor,
        MAX(curve_move_bp) AS curve_move_bp,
        COUNT(*) AS source_row_count,
        SUM(
            CASE
                WHEN curve_move_bp IS NULL OR ABS(curve_move_bp) > 1.0e308
                THEN 1 ELSE 0
            END
        ) AS null_value_count
    FROM curve_market_move
    GROUP BY
        valuation_date,
        state_id,
        market_run_id,
        curve_id,
        key_rate_tenor
), gamma_normalized AS (
    SELECT
        valuation_date,
        book_id,
        instrument_id,
        state_id,
        risk_run_id,
        CASE
            WHEN curve_id_1 || '|' || key_rate_tenor_1
                 <= curve_id_2 || '|' || key_rate_tenor_2
            THEN curve_id_1 ELSE curve_id_2
        END AS curve_id_lo,
        CASE
            WHEN curve_id_1 || '|' || key_rate_tenor_1
                 <= curve_id_2 || '|' || key_rate_tenor_2
            THEN key_rate_tenor_1 ELSE key_rate_tenor_2
        END AS tenor_lo,
        CASE
            WHEN curve_id_1 || '|' || key_rate_tenor_1
                 <= curve_id_2 || '|' || key_rate_tenor_2
            THEN curve_id_2 ELSE curve_id_1
        END AS curve_id_hi,
        CASE
            WHEN curve_id_1 || '|' || key_rate_tenor_1
                 <= curve_id_2 || '|' || key_rate_tenor_2
            THEN key_rate_tenor_2 ELSE key_rate_tenor_1
        END AS tenor_hi,
        cross_gamma
    FROM curve_cross_gamma
), gamma_rows AS (
    SELECT
        valuation_date,
        book_id,
        instrument_id,
        state_id,
        risk_run_id,
        curve_id_lo,
        tenor_lo,
        curve_id_hi,
        tenor_hi,
        MAX(cross_gamma) AS cross_gamma,
        COUNT(*) AS source_row_count,
        SUM(
            CASE
                WHEN cross_gamma IS NULL OR ABS(cross_gamma) > 1.0e308
                THEN 1 ELSE 0
            END
        ) AS null_value_count
    FROM gamma_normalized
    GROUP BY
        valuation_date,
        book_id,
        instrument_id,
        state_id,
        risk_run_id,
        curve_id_lo,
        tenor_lo,
        curve_id_hi,
        tenor_hi
), curve_explain AS (
    SELECT
        p.valuation_date,
        p.book_id,
        p.instrument_id,
        p.state_id,
        p.position_run_id,
        p.risk_run_id,
        p.market_run_id,
        COUNT(r.key_rate_tenor) AS matched_curve_bucket_count,
        COALESCE(SUM(-r.dv01 * m.curve_move_bp), 0.0)
            AS curve_delta_pnl,
        COALESCE(SUM(
            CASE
                WHEN r.key_rate_tenor IS NOT NULL
                 AND (m.key_rate_tenor IS NULL
                      OR r.null_value_count > 0
                      OR m.null_value_count > 0)
                THEN 1 ELSE 0
            END
        ), 0) AS missing_curve_evidence_count,
        COALESCE(SUM(
            CASE
                WHEN r.source_row_count > 1
                THEN r.source_row_count - 1 ELSE 0
            END
        ), 0) AS duplicate_risk_count,
        COALESCE(SUM(
            CASE
                WHEN m.source_row_count > 1
                THEN m.source_row_count - 1 ELSE 0
            END
        ), 0) AS duplicate_move_count
    FROM position_rows AS p
    LEFT JOIN risk_rows AS r
      ON r.valuation_date = p.valuation_date
     AND r.book_id = p.book_id
     AND r.instrument_id = p.instrument_id
     AND r.state_id = p.state_id
     AND r.risk_run_id = p.risk_run_id
    LEFT JOIN move_rows AS m
      ON m.valuation_date = r.valuation_date
     AND m.state_id = p.state_id
     AND m.market_run_id = p.market_run_id
     AND m.curve_id = r.curve_id
     AND m.key_rate_tenor = r.key_rate_tenor
    GROUP BY
        p.valuation_date,
        p.book_id,
        p.instrument_id,
        p.state_id,
        p.position_run_id,
        p.risk_run_id,
        p.market_run_id
), gamma_explain AS (
    SELECT
        p.valuation_date,
        p.book_id,
        p.instrument_id,
        p.state_id,
        p.position_run_id,
        p.risk_run_id,
        p.market_run_id,
        COUNT(g.curve_id_lo) AS matched_gamma_pair_count,
        COALESCE(SUM(
            CASE
                WHEN g.curve_id_lo = g.curve_id_hi
                 AND g.tenor_lo = g.tenor_hi
                THEN 0.5 ELSE 1.0
            END * g.cross_gamma
                * m1.curve_move_bp * m2.curve_move_bp
        ), 0.0) AS curve_cross_gamma_pnl,
        COALESCE(SUM(
            CASE
                WHEN g.curve_id_lo IS NOT NULL
                 AND (m1.key_rate_tenor IS NULL
                      OR m2.key_rate_tenor IS NULL
                      OR g.null_value_count > 0
                      OR m1.null_value_count > 0
                      OR m2.null_value_count > 0)
                THEN 1 ELSE 0
            END
        ), 0) AS missing_gamma_evidence_count,
        COALESCE(SUM(
            CASE
                WHEN g.source_row_count > 1
                THEN g.source_row_count - 1 ELSE 0
            END
        ), 0) AS duplicate_gamma_count
    FROM position_rows AS p
    LEFT JOIN gamma_rows AS g
      ON g.valuation_date = p.valuation_date
     AND g.book_id = p.book_id
     AND g.instrument_id = p.instrument_id
     AND g.state_id = p.state_id
     AND g.risk_run_id = p.risk_run_id
    LEFT JOIN move_rows AS m1
      ON m1.valuation_date = g.valuation_date
     AND m1.state_id = p.state_id
     AND m1.market_run_id = p.market_run_id
     AND m1.curve_id = g.curve_id_lo
     AND m1.key_rate_tenor = g.tenor_lo
    LEFT JOIN move_rows AS m2
      ON m2.valuation_date = g.valuation_date
     AND m2.state_id = p.state_id
     AND m2.market_run_id = p.market_run_id
     AND m2.curve_id = g.curve_id_hi
     AND m2.key_rate_tenor = g.tenor_hi
    GROUP BY
        p.valuation_date,
        p.book_id,
        p.instrument_id,
        p.state_id,
        p.position_run_id,
        p.risk_run_id,
        p.market_run_id
), evidence_alignment AS (
    SELECT
        p.valuation_date,
        p.book_id,
        p.instrument_id,
        p.state_id,
        p.position_run_id,
        p.risk_run_id,
        p.market_run_id,
        CASE WHEN NOT EXISTS (
            SELECT 1
            FROM curve_risk_bucket AS exact_risk
            WHERE exact_risk.valuation_date = p.valuation_date
              AND exact_risk.book_id = p.book_id
              AND exact_risk.instrument_id = p.instrument_id
              AND exact_risk.state_id = p.state_id
              AND exact_risk.risk_run_id = p.risk_run_id
        ) THEN (
            SELECT COUNT(*)
            FROM curve_risk_bucket AS r
            WHERE r.valuation_date = p.valuation_date
              AND r.book_id = p.book_id
              AND r.instrument_id = p.instrument_id
              AND (r.state_id <> p.state_id
                   OR r.risk_run_id <> p.risk_run_id)
        ) ELSE 0 END + CASE WHEN NOT EXISTS (
            SELECT 1
            FROM curve_cross_gamma AS exact_gamma
            WHERE exact_gamma.valuation_date = p.valuation_date
              AND exact_gamma.book_id = p.book_id
              AND exact_gamma.instrument_id = p.instrument_id
              AND exact_gamma.state_id = p.state_id
              AND exact_gamma.risk_run_id = p.risk_run_id
        ) THEN (
            SELECT COUNT(*)
            FROM curve_cross_gamma AS g
            WHERE g.valuation_date = p.valuation_date
              AND g.book_id = p.book_id
              AND g.instrument_id = p.instrument_id
              AND (g.state_id <> p.state_id
                   OR g.risk_run_id <> p.risk_run_id)
        ) ELSE 0 END AS misaligned_evidence_count
    FROM position_rows AS p
), explained AS (
    SELECT
        p.*,
        c.curve_delta_pnl,
        g.curve_cross_gamma_pnl,
        -p.cs01 * p.credit_spread_move_bp AS credit_delta_pnl,
        0.5 * p.credit_convexity
            * p.credit_spread_move_bp * p.credit_spread_move_bp
            AS credit_convexity_pnl,
        c.matched_curve_bucket_count,
        g.matched_gamma_pair_count,
        c.missing_curve_evidence_count
            + g.missing_gamma_evidence_count AS missing_evidence_count,
        c.duplicate_risk_count + c.duplicate_move_count
            + g.duplicate_gamma_count
            + MAX(p.position_source_row_count - 1, 0)
            AS duplicate_evidence_count,
        a.misaligned_evidence_count,
        (CASE WHEN p.actual_pnl IS NULL OR ABS(p.actual_pnl) > 1.0e308 THEN 1 ELSE 0 END
         + CASE WHEN p.carry_pnl IS NULL OR ABS(p.carry_pnl) > 1.0e308 THEN 1 ELSE 0 END
         + CASE WHEN p.roll_down_pnl IS NULL OR ABS(p.roll_down_pnl) > 1.0e308 THEN 1 ELSE 0 END
         + CASE WHEN p.basis_pnl IS NULL OR ABS(p.basis_pnl) > 1.0e308 THEN 1 ELSE 0 END
         + CASE WHEN p.cs01 IS NULL OR ABS(p.cs01) > 1.0e308 THEN 1 ELSE 0 END
         + CASE WHEN p.credit_spread_move_bp IS NULL OR ABS(p.credit_spread_move_bp) > 1.0e308 THEN 1 ELSE 0 END
         + CASE WHEN p.credit_convexity IS NULL OR ABS(p.credit_convexity) > 1.0e308 THEN 1 ELSE 0 END
         + CASE WHEN p.volatility_pnl IS NULL OR ABS(p.volatility_pnl) > 1.0e308 THEN 1 ELSE 0 END
         + CASE WHEN p.cash_and_fixings_pnl IS NULL OR ABS(p.cash_and_fixings_pnl) > 1.0e308 THEN 1 ELSE 0 END
         + CASE WHEN p.exercise_pnl IS NULL OR ABS(p.exercise_pnl) > 1.0e308 THEN 1 ELSE 0 END
         + CASE WHEN p.model_change_pnl IS NULL OR ABS(p.model_change_pnl) > 1.0e308 THEN 1 ELSE 0 END
         + CASE WHEN p.data_correction_pnl IS NULL OR ABS(p.data_correction_pnl) > 1.0e308 THEN 1 ELSE 0 END
         + CASE WHEN p.expected_curve_bucket_count IS NULL THEN 1 ELSE 0 END
         + CASE WHEN p.expected_gamma_pair_count IS NULL THEN 1 ELSE 0 END
         + CASE WHEN p.stale_input_count IS NULL THEN 1 ELSE 0 END)
            AS missing_scalar_count
    FROM position_rows AS p
    LEFT JOIN curve_explain AS c
      ON c.valuation_date = p.valuation_date
     AND c.book_id = p.book_id
     AND c.instrument_id = p.instrument_id
     AND c.state_id = p.state_id
     AND c.position_run_id = p.position_run_id
     AND c.risk_run_id = p.risk_run_id
     AND c.market_run_id = p.market_run_id
    LEFT JOIN gamma_explain AS g
      ON g.valuation_date = p.valuation_date
     AND g.book_id = p.book_id
     AND g.instrument_id = p.instrument_id
     AND g.state_id = p.state_id
     AND g.position_run_id = p.position_run_id
     AND g.risk_run_id = p.risk_run_id
     AND g.market_run_id = p.market_run_id
    LEFT JOIN evidence_alignment AS a
      ON a.valuation_date = p.valuation_date
     AND a.book_id = p.book_id
     AND a.instrument_id = p.instrument_id
     AND a.state_id = p.state_id
     AND a.position_run_id = p.position_run_id
     AND a.risk_run_id = p.risk_run_id
     AND a.market_run_id = p.market_run_id
), reconciled AS (
    SELECT
        *,
        carry_pnl + roll_down_pnl + curve_delta_pnl
        + curve_cross_gamma_pnl + basis_pnl
        + credit_delta_pnl + credit_convexity_pnl
        + volatility_pnl + cash_and_fixings_pnl + exercise_pnl
        + model_change_pnl + data_correction_pnl AS explained_pnl
    FROM explained
)
SELECT
    valuation_date,
    book_id,
    instrument_id,
    state_id,
    position_run_id,
    risk_run_id,
    market_run_id,
    actual_pnl,
    explained_pnl,
    actual_pnl - explained_pnl AS residual_pnl,
    ABS(actual_pnl - explained_pnl)
        / MAX(ABS(actual_pnl), :pnl_floor) AS residual_fraction,
    matched_curve_bucket_count,
    missing_evidence_count + missing_scalar_count AS missing_evidence_count,
    duplicate_evidence_count,
    matched_gamma_pair_count,
    misaligned_evidence_count,
    missing_scalar_count,
    CASE
        WHEN :absolute_residual_limit IS NULL
          OR :relative_residual_limit IS NULL
          OR :pnl_floor IS NULL
          OR :absolute_residual_limit < 0.0
          OR :relative_residual_limit < 0.0
          OR :pnl_floor <= 0.0
          OR ABS(:absolute_residual_limit) > 1.0e308
          OR ABS(:relative_residual_limit) > 1.0e308
          OR ABS(:pnl_floor) > 1.0e308
          OR missing_scalar_count > 0
          OR explained_pnl IS NULL
          OR actual_pnl IS NULL
          OR missing_evidence_count > 0
          OR duplicate_evidence_count > 0
          OR misaligned_evidence_count > 0
          OR matched_curve_bucket_count <> expected_curve_bucket_count
          OR matched_gamma_pair_count <> expected_gamma_pair_count
          OR stale_input_count > 0
        THEN 'REJECT'
        WHEN ABS(actual_pnl - explained_pnl) > :absolute_residual_limit
          OR ABS(actual_pnl - explained_pnl)
             / MAX(ABS(actual_pnl), :pnl_floor) > :relative_residual_limit
        THEN 'REJECT'
        ELSE 'PASS'
    END AS reconciliation_status
FROM reconciled
ORDER BY ABS(actual_pnl - explained_pnl) DESC;
