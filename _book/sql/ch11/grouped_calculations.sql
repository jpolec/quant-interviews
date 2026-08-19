-- Dialect: SQLite 3.45; exact-positive price domain and stable three-return z-score.
WITH physical AS (
    SELECT p.rowid AS physical_row_id, p.*
    FROM prices AS p
), asset_audit AS (
    SELECT
        asset,
        MAX(key_count) AS maximum_key_count,
        MAX(invalid_price) AS has_invalid_price
    FROM (
        SELECT
            asset,
            COUNT(*) OVER (PARTITION BY asset, timestamp) AS key_count,
            CASE
                WHEN price IS NULL OR price <= 0.0 OR
                     ABS(price) > 1.7976931348623157e308
                    THEN 1 ELSE 0
            END AS invalid_price
        FROM physical
    )
    GROUP BY asset
), ordered AS (
    SELECT
        p.*,
        a.maximum_key_count,
        a.has_invalid_price,
        LAG(price) OVER (
            PARTITION BY p.asset
            ORDER BY p.timestamp, p.physical_row_id
        ) AS previous_price
    FROM physical AS p
    JOIN asset_audit AS a USING (asset)
), returns AS (
    SELECT
        *,
        CASE
            WHEN maximum_key_count <> 1 OR has_invalid_price <> 0
                THEN NULL
            WHEN previous_price IS NULL THEN NULL
            ELSE price / previous_price - 1.0
        END AS return_1
    FROM ordered
), lagged_returns AS (
    SELECT
        *,
        LAG(return_1, 1) OVER feature_order AS prior_return_1,
        LAG(return_1, 2) OVER feature_order AS prior_return_2
    FROM returns
    WINDOW feature_order AS (
        PARTITION BY asset
        ORDER BY timestamp, physical_row_id
    )
), centered AS (
    SELECT
        *,
        CASE
            WHEN return_1 IS NULL OR prior_return_1 IS NULL OR
                 prior_return_2 IS NULL
                THEN NULL
            ELSE (return_1 + prior_return_1 + prior_return_2) / 3.0
        END AS rolling_mean
    FROM lagged_returns
), stable_variance AS (
    SELECT
        *,
        CASE
            WHEN rolling_mean IS NULL THEN NULL
            ELSE (
                (return_1 - rolling_mean) * (return_1 - rolling_mean) +
                (prior_return_1 - rolling_mean) *
                    (prior_return_1 - rolling_mean) +
                (prior_return_2 - rolling_mean) *
                    (prior_return_2 - rolling_mean)
            ) / 2.0
        END AS sample_variance
    FROM centered
)
SELECT
    asset,
    timestamp,
    price,
    previous_price,
    return_1,
    CASE
        WHEN sample_variance IS NULL OR sample_variance <= 1.0e-24 THEN NULL
        ELSE (return_1 - rolling_mean) / SQRT(sample_variance)
    END AS return_zscore_3,
    CASE
        WHEN maximum_key_count <> 1 THEN 'REJECT_DUPLICATE_KEY'
        WHEN has_invalid_price <> 0 THEN 'REJECT_INVALID_PRICE'
        ELSE 'PASS'
    END AS status
FROM stable_variance
ORDER BY asset, timestamp, physical_row_id;
