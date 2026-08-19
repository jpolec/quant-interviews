-- Dialect: SQLite 3.45; revision selection and one total order for every window.
WITH physical AS (
    SELECT p.rowid AS physical_row_id, p.*
    FROM prices AS p
), audited AS (
    SELECT
        *,
        COUNT(*) OVER (
            PARTITION BY asset, timestamp, sequence_number, version_id
        ) AS revision_count,
        COUNT(*) OVER (PARTITION BY row_id) AS row_id_count,
        ROW_NUMBER() OVER (
            PARTITION BY asset, timestamp, sequence_number
            ORDER BY version_id DESC, row_id DESC, row_hash DESC,
                     physical_row_id DESC
        ) AS revision_rank
    FROM physical
), revision_scoped AS (
    SELECT
        *,
        MAX(revision_count) OVER (PARTITION BY asset)
            AS scope_revision_count,
        MAX(row_id_count) OVER (PARTITION BY asset)
            AS scope_row_id_count
    FROM audited
), canonical AS (
    SELECT *
    FROM revision_scoped
    WHERE revision_rank = 1
), asset_audit AS (
    SELECT
        asset,
        MAX(scope_revision_count) AS maximum_revision_count,
        MAX(scope_row_id_count) AS maximum_row_id_count,
        MAX(CASE WHEN row_id IS NULL OR TRIM(row_id) = '' OR
                      row_hash IS NULL OR TRIM(row_hash) = ''
                 THEN 1 ELSE 0 END) AS missing_lineage,
        MAX(CASE WHEN price IS NULL OR price <= 0.0 OR
                      ABS(price) > 1.7976931348623157e308
                 THEN 1 ELSE 0 END) AS invalid_price,
        MAX(sequence_count) AS maximum_sequence_count
    FROM (
        SELECT
            c.*,
            COUNT(*) OVER (
                PARTITION BY asset, sequence_number
            ) AS sequence_count
        FROM canonical AS c
    )
    GROUP BY asset
), ordered AS (
    SELECT
        c.*,
        a.*,
        LAG(c.price) OVER total_order AS previous_price
    FROM canonical AS c
    JOIN asset_audit AS a USING (asset)
    WINDOW total_order AS (
        PARTITION BY c.asset
        ORDER BY c.timestamp, c.sequence_number, c.version_id, c.row_id
    )
), returns AS (
    SELECT
        *,
        CASE
            WHEN maximum_revision_count <> 1 OR maximum_row_id_count <> 1 OR
                 maximum_sequence_count <> 1 OR missing_lineage <> 0 OR
                 invalid_price <> 0 OR previous_price IS NULL
                THEN NULL
            ELSE price / previous_price - 1.0
        END AS return_1
    FROM ordered
), ranked AS (
    SELECT
        *,
        COUNT(return_1) OVER (PARTITION BY timestamp) AS valid_count,
        RANK() OVER (
            PARTITION BY timestamp
            ORDER BY CASE WHEN return_1 IS NULL THEN 1 ELSE 0 END, return_1
        ) AS sparse_rank,
        COUNT(return_1) OVER trailing_window AS trailing_count,
        AVG(return_1) OVER trailing_window AS trailing_mean
    FROM returns
    WINDOW trailing_window AS (
        PARTITION BY asset
        ORDER BY timestamp, sequence_number, version_id, row_id
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    )
)
SELECT
    timestamp,
    asset,
    sequence_number,
    version_id,
    row_id,
    row_hash,
    price,
    return_1,
    CASE
        WHEN return_1 IS NULL THEN NULL
        WHEN valid_count <= 1 THEN 0.0
        ELSE (sparse_rank - 1.0) / (valid_count - 1.0)
    END AS cross_sectional_percent_rank,
    CASE WHEN trailing_count = 5 THEN trailing_mean END AS trailing_mean_5,
    CASE
        WHEN maximum_revision_count <> 1 THEN 'REJECT_DUPLICATE_REVISION'
        WHEN maximum_row_id_count <> 1 OR maximum_sequence_count <> 1
            THEN 'REJECT_DUPLICATE_ORDER_KEY'
        WHEN missing_lineage <> 0 THEN 'REJECT_MISSING_LINEAGE'
        WHEN invalid_price <> 0 THEN 'REJECT_INVALID_PRICE'
        ELSE 'PASS'
    END AS status
FROM ranked
ORDER BY timestamp, asset, sequence_number, version_id, row_id;
