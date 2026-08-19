-- Dialect: PostgreSQL 16+
-- Complete point-in-time RV against an authoritative expected-bar schedule.
WITH expected_bar AS (
    SELECT
        schedule.instrument_id,
        schedule.trade_date,
        schedule.bar_end_time
    FROM trading_bar_schedule AS schedule
    WHERE schedule.bar_end_time <= :forecast_cutoff
),
eligible_bar_revision AS (
    SELECT
        bar.instrument_id,
        bar.trade_date,
        bar.bar_end_time,
        bar.log_return,
        bar.knowledge_time,
        bar.version_id,
        bar.source_id,
        bar.row_hash,
        bar.is_tombstone,
        ROW_NUMBER() OVER (
            PARTITION BY bar.instrument_id, bar.trade_date, bar.bar_end_time
            ORDER BY
                bar.knowledge_time DESC,
                bar.version_id DESC,
                bar.source_id DESC,
                bar.row_hash DESC
        ) AS row_rank,
        COUNT(*) OVER (
            PARTITION BY
                bar.instrument_id,
                bar.trade_date,
                bar.bar_end_time,
                bar.knowledge_time,
                bar.version_id,
                bar.source_id
        ) AS exact_tie_count
    FROM intraday_return AS bar
    WHERE bar.bar_end_time <= :forecast_cutoff
      AND bar.knowledge_time <= :decision_time
),
winning_bar AS (
    SELECT
        instrument_id,
        trade_date,
        bar_end_time,
        log_return,
        knowledge_time,
        version_id,
        source_id,
        row_hash,
        is_tombstone,
        exact_tie_count
    FROM eligible_bar_revision
    WHERE row_rank = 1
),
bar_evidence AS (
    SELECT
        expected.instrument_id,
        expected.trade_date,
        expected.bar_end_time,
        winner.log_return,
        winner.knowledge_time,
        winner.row_hash,
        CASE
            WHEN winner.row_hash IS NULL THEN 'MISSING_BAR'
            WHEN winner.exact_tie_count > 1 THEN 'AMBIGUOUS_BAR'
            WHEN winner.is_tombstone THEN 'TOMBSTONED_BAR'
            WHEN winner.log_return IS NULL THEN 'INVALID_NULL_BAR'
            WHEN winner.log_return IN (
                'NaN'::double precision,
                'Infinity'::double precision,
                '-Infinity'::double precision
            ) THEN 'INVALID_NONFINITE_BAR'
            ELSE 'SELECTED'
        END AS bar_status
    FROM expected_bar AS expected
    LEFT JOIN winning_bar AS winner
      ON winner.instrument_id = expected.instrument_id
     AND winner.trade_date = expected.trade_date
     AND winner.bar_end_time = expected.bar_end_time
),
daily_intraday AS (
    SELECT
        instrument_id,
        trade_date,
        COUNT(*) AS expected_bars,
        COUNT(*) FILTER (WHERE bar_status = 'SELECTED') AS selected_bars,
        COUNT(*) FILTER (WHERE bar_status = 'MISSING_BAR') AS missing_bars,
        COUNT(*) FILTER (WHERE bar_status = 'AMBIGUOUS_BAR') AS ambiguous_bars,
        COUNT(*) FILTER (WHERE bar_status = 'TOMBSTONED_BAR') AS tombstoned_bars,
        COUNT(*) FILTER (
            WHERE bar_status IN ('INVALID_NULL_BAR', 'INVALID_NONFINITE_BAR')
        ) AS invalid_bars,
        SUM(log_return * log_return) FILTER (WHERE bar_status = 'SELECTED') AS intraday_rv,
        MAX(knowledge_time) FILTER (WHERE bar_status = 'SELECTED') AS maximum_bar_knowledge_time,
        ARRAY_AGG(row_hash ORDER BY bar_end_time)
            FILTER (WHERE bar_status = 'SELECTED') AS contributing_bar_hashes
    FROM bar_evidence
    GROUP BY instrument_id, trade_date
),
eligible_overnight_revision AS (
    SELECT
        overnight.instrument_id,
        overnight.trade_date,
        overnight.overnight_log_return,
        overnight.knowledge_time,
        overnight.version_id,
        overnight.source_id,
        overnight.row_hash,
        overnight.is_tombstone,
        ROW_NUMBER() OVER (
            PARTITION BY overnight.instrument_id, overnight.trade_date
            ORDER BY
                overnight.knowledge_time DESC,
                overnight.version_id DESC,
                overnight.source_id DESC,
                overnight.row_hash DESC
        ) AS row_rank,
        COUNT(*) OVER (
            PARTITION BY
                overnight.instrument_id,
                overnight.trade_date,
                overnight.knowledge_time,
                overnight.version_id,
                overnight.source_id
        ) AS exact_tie_count
    FROM overnight_return AS overnight
    WHERE overnight.knowledge_time <= :decision_time
),
winning_overnight AS (
    SELECT
        instrument_id,
        trade_date,
        overnight_log_return,
        knowledge_time,
        version_id,
        source_id,
        row_hash,
        is_tombstone,
        exact_tie_count
    FROM eligible_overnight_revision
    WHERE row_rank = 1
),
classified AS (
    SELECT
        daily.*,
        overnight.overnight_log_return,
        overnight.knowledge_time AS overnight_knowledge_time,
        overnight.version_id AS overnight_version_id,
        overnight.source_id AS overnight_source_id,
        overnight.row_hash AS overnight_row_hash,
        CASE
            WHEN overnight.row_hash IS NULL THEN 'MISSING_OVERNIGHT'
            WHEN overnight.exact_tie_count > 1 THEN 'AMBIGUOUS_OVERNIGHT'
            WHEN overnight.is_tombstone THEN 'TOMBSTONED_OVERNIGHT'
            WHEN overnight.overnight_log_return IS NULL THEN 'INVALID_NULL_OVERNIGHT'
            WHEN overnight.overnight_log_return IN (
                'NaN'::double precision,
                'Infinity'::double precision,
                '-Infinity'::double precision
            ) THEN 'INVALID_NONFINITE_OVERNIGHT'
            ELSE 'SELECTED'
        END AS overnight_status
    FROM daily_intraday AS daily
    LEFT JOIN winning_overnight AS overnight
      ON overnight.instrument_id = daily.instrument_id
     AND overnight.trade_date = daily.trade_date
)
SELECT
    instrument_id,
    trade_date,
    CASE
        WHEN selected_bars = expected_bars AND overnight_status = 'SELECTED'
        THEN intraday_rv
    END AS intraday_rv,
    CASE
        WHEN selected_bars = expected_bars AND overnight_status = 'SELECTED'
        THEN overnight_log_return * overnight_log_return
    END AS overnight_rv,
    CASE
        WHEN selected_bars = expected_bars AND overnight_status = 'SELECTED'
        THEN intraday_rv + overnight_log_return * overnight_log_return
    END AS total_rv,
    expected_bars,
    selected_bars,
    missing_bars,
    ambiguous_bars,
    tombstoned_bars,
    invalid_bars,
    maximum_bar_knowledge_time,
    contributing_bar_hashes,
    overnight_knowledge_time,
    overnight_version_id,
    overnight_source_id,
    overnight_row_hash,
    overnight_status,
    GREATEST(maximum_bar_knowledge_time, overnight_knowledge_time) AS target_available_time,
    CASE
        WHEN selected_bars = expected_bars AND overnight_status = 'SELECTED'
        THEN 'PUBLISHABLE'
        ELSE 'REJECT_INCOMPLETE'
    END AS rv_status
FROM classified
ORDER BY instrument_id, trade_date;
