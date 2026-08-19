-- Dialect: SQLite 3.45+
WITH parameters AS (
    SELECT :as_of_date AS as_of_date, :tolerance AS tolerance
),
keys AS (
    SELECT account_id, currency
    FROM cash_balances
    WHERE as_of_date = :as_of_date
    UNION
    SELECT account_id, currency
    FROM custodian_cash_balances
    WHERE as_of_date = :as_of_date
),
compared AS (
    SELECT
        k.account_id,
        k.currency,
        (SELECT SUM(amount) FROM cash_balances AS i
         WHERE i.as_of_date = :as_of_date
           AND i.account_id = k.account_id AND i.currency = k.currency) AS internal_amount,
        (SELECT SUM(amount) FROM custodian_cash_balances AS e
         WHERE e.as_of_date = :as_of_date
           AND e.account_id = k.account_id AND e.currency = k.currency) AS external_amount
    FROM keys AS k
),
with_empty_certificate AS (
    SELECT * FROM compared
    UNION ALL
    SELECT NULL, NULL, NULL, NULL
    WHERE NOT EXISTS (SELECT 1 FROM compared)
)
SELECT
    c.account_id,
    c.currency,
    c.internal_amount,
    c.external_amount,
    CASE
        WHEN p.tolerance IS NULL OR p.tolerance < 0 THEN 'INVALID_TOLERANCE'
        WHEN c.account_id IS NULL THEN 'EMPTY_UNIVERSE'
        WHEN c.internal_amount IS NULL THEN 'MISSING_INTERNAL'
        WHEN c.external_amount IS NULL THEN 'MISSING_EXTERNAL'
        WHEN ABS(c.internal_amount - c.external_amount) > p.tolerance THEN 'MATERIAL'
        ELSE 'MATCH'
    END AS reconciliation_status,
    CASE
        WHEN c.internal_amount IS NULL OR c.external_amount IS NULL THEN NULL
        ELSE c.internal_amount - c.external_amount
    END AS difference
FROM with_empty_certificate AS c
CROSS JOIN parameters AS p
WHERE p.tolerance IS NULL
   OR p.tolerance < 0
   OR c.account_id IS NULL
   OR c.internal_amount IS NULL
   OR c.external_amount IS NULL
   OR ABS(c.internal_amount - c.external_amount) > p.tolerance
ORDER BY c.account_id, c.currency;
