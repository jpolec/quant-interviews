-- Dialect: SQLite
-- Parameters:
--   :valuation_time
--   :as_of_knowledge_time
--
-- A revision is available only after it is known, published and ingested.
-- The ordered window aggregate makes snapshot_hash_material deterministic.
WITH available_revisions AS (
    SELECT
        underlying_id,
        expiry,
        strike,
        option_type,
        quote_convention,
        bid_vol,
        ask_vol,
        valid_from,
        valid_to,
        known_at,
        published_at,
        ingested_at,
        MAX(known_at, published_at, ingested_at) AS available_at,
        revision_id,
        superseded_at,
        source_priority,
        payload_hash
    FROM implied_vol_versions
    WHERE known_at IS NOT NULL
      AND published_at IS NOT NULL
      AND ingested_at IS NOT NULL
),
eligible AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY
                underlying_id,
                expiry,
                strike,
                option_type,
                quote_convention
            ORDER BY
                available_at DESC,
                known_at DESC,
                revision_id DESC,
                source_priority DESC,
                payload_hash DESC
        ) AS revision_rank
    FROM available_revisions
    WHERE valid_from <= :valuation_time
      AND :valuation_time < valid_to
      AND available_at <= :as_of_knowledge_time
      AND (
          superseded_at IS NULL
          OR superseded_at > :as_of_knowledge_time
      )
),
selected AS (
    SELECT *
    FROM eligible
    WHERE revision_rank = 1
),
manifest_window AS (
    SELECT
        'ch06-option-surface-v0.10' AS schema_version,
        COUNT(*) OVER () AS row_count,
        MIN(available_at) OVER () AS minimum_available_at,
        MAX(available_at) OVER () AS maximum_available_at,
        GROUP_CONCAT(payload_hash, '|') OVER (
            ORDER BY
                underlying_id,
                expiry,
                strike,
                option_type,
                quote_convention
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS snapshot_hash_material,
        ROW_NUMBER() OVER (
            ORDER BY
                underlying_id,
                expiry,
                strike,
                option_type,
                quote_convention
        ) AS manifest_rank
    FROM selected
),
manifest AS (
    SELECT
        schema_version,
        row_count,
        minimum_available_at,
        maximum_available_at,
        snapshot_hash_material
    FROM manifest_window
    WHERE manifest_rank = 1
)
SELECT
    selected.*,
    manifest.schema_version,
    manifest.row_count,
    manifest.minimum_available_at,
    manifest.maximum_available_at,
    manifest.snapshot_hash_material
FROM selected
CROSS JOIN manifest
ORDER BY
    underlying_id,
    expiry,
    strike,
    option_type,
    quote_convention;
