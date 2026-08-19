-- Dialect: SQLite 3.45. Full selection order is covered by the plan-gate index.
CREATE INDEX IF NOT EXISTS idx_prices_symbol_valid_knowledge_version
ON prices(
    symbol,
    valid_time DESC,
    knowledge_time DESC,
    version DESC,
    source_id DESC,
    row_hash DESC
);

EXPLAIN QUERY PLAN
WITH eligible AS (
    SELECT
        symbol,
        price,
        valid_time,
        knowledge_time,
        version,
        source_id,
        row_hash,
        ROW_NUMBER() OVER (
            PARTITION BY symbol
            ORDER BY
                valid_time DESC,
                knowledge_time DESC,
                version DESC,
                source_id DESC,
                row_hash DESC
        ) AS priority
    FROM prices INDEXED BY idx_prices_symbol_valid_knowledge_version
    WHERE valid_time <= :valid_cutoff
      AND knowledge_time <= :knowledge_cutoff
)
SELECT symbol, price, valid_time, knowledge_time, version, source_id, row_hash
FROM eligible
WHERE priority = 1
ORDER BY symbol;
