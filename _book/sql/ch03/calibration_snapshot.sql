-- Dialect: PostgreSQL
-- Requires pgcrypto and UTC timestamptz source columns. Bind :market_as_of and
-- :knowledge_as_of as timestamptz and :snapshot_id as text. Canonical numeric
-- fields use eight decimal places; timestamps use UTC with six fractional
-- digits. The SQLite fixture registers compatible PostgreSQL functions.
WITH run_parameters AS (
    SELECT
        :market_as_of AS market_as_of,
        :knowledge_as_of AS knowledge_as_of,
        :snapshot_id AS snapshot_id
),
eligible_versions AS (
    SELECT
        q.*,
        p.snapshot_id,
        p.market_as_of,
        p.knowledge_as_of,
        ROW_NUMBER() OVER (
            PARTITION BY
                q.instrument_id,
                q.expiry,
                q.strike,
                q.option_type,
                q.quote_convention
            ORDER BY
                q.known_at DESC,
                q.revision_id DESC,
                q.quote_payload_hash DESC,
                q.bid DESC,
                q.ask DESC
        ) AS version_rank
    FROM calibration_quotes AS q
    CROSS JOIN run_parameters AS p
    WHERE q.valid_from <= p.market_as_of
      AND (q.valid_to > p.market_as_of OR q.valid_to IS NULL)
      AND q.known_at <= p.knowledge_as_of
      AND (q.superseded_at > p.knowledge_as_of OR q.superseded_at IS NULL)
),
duplicate_version_groups AS (
    SELECT
        instrument_id,
        expiry,
        strike,
        option_type,
        quote_convention,
        COUNT(*) AS copy_count
    FROM eligible_versions
    GROUP BY
        instrument_id,
        expiry,
        strike,
        option_type,
        quote_convention
    HAVING COUNT(*) > 1
),
eligibility_audit AS (
    SELECT COUNT(*) AS duplicate_version_group_count
    FROM duplicate_version_groups
),
selected_quotes AS (
    SELECT
        snapshot_id,
        market_as_of,
        knowledge_as_of,
        instrument_id,
        expiry,
        strike,
        option_type,
        quote_convention,
        bid,
        ask,
        valid_from,
        valid_to,
        known_at,
        superseded_at,
        revision_id,
        quote_payload_hash
    FROM eligible_versions
    WHERE version_rank = 1
),
canonical_values AS (
    SELECT
        q.*,
        TO_CHAR(expiry, 'YYYY-MM-DD') AS canonical_expiry,
        TO_CHAR(strike, 'FM999999999999999990.00000000')
            AS canonical_strike,
        TO_CHAR(bid, 'FM999999999999999990.00000000') AS canonical_bid,
        TO_CHAR(ask, 'FM999999999999999990.00000000') AS canonical_ask,
        TO_CHAR(
            TIMEZONE('UTC', valid_from),
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
        ) AS canonical_valid_from,
        CASE
            WHEN valid_to IS NULL THEN NULL
            ELSE TO_CHAR(
                TIMEZONE('UTC', valid_to),
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
            )
        END AS canonical_valid_to,
        TO_CHAR(
            TIMEZONE('UTC', known_at),
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
        ) AS canonical_known_at,
        CASE
            WHEN superseded_at IS NULL THEN NULL
            ELSE TO_CHAR(
                TIMEZONE('UTC', superseded_at),
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
            )
        END AS canonical_superseded_at,
        CAST(revision_id AS TEXT) AS canonical_revision_id
    FROM selected_quotes AS q
),
serialized_payloads AS (
    SELECT
        q.*,
        'bid='
        || CASE
            WHEN canonical_bid IS NULL THEN '-1:'
            ELSE CAST(LENGTH(canonical_bid) AS TEXT)
                || ':' || canonical_bid
        END
        || ';ask='
        || CASE
            WHEN canonical_ask IS NULL THEN '-1:'
            ELSE CAST(LENGTH(canonical_ask) AS TEXT)
                || ':' || canonical_ask
        END AS canonical_quote_payload
    FROM canonical_values AS q
),
verified_payloads AS (
    SELECT
        q.*,
        LOWER(ENCODE(DIGEST(canonical_quote_payload, 'sha256'), 'hex'))
            AS computed_quote_payload_hash
    FROM serialized_payloads AS q
),
canonical_rows AS (
    SELECT
        q.*,
        CASE
            WHEN bid IS NOT NULL
             AND ask IS NOT NULL
             AND bid <= ask
             AND quote_payload_hash IS NOT NULL
             AND quote_payload_hash = LOWER(quote_payload_hash)
             AND LENGTH(quote_payload_hash) = 64
             AND TRANSLATE(quote_payload_hash, '0123456789abcdef', '') = ''
             AND quote_payload_hash = computed_quote_payload_hash
            THEN 1
            ELSE 0
        END AS payload_hash_valid,
        'instrument_id='
        || CASE
            WHEN instrument_id IS NULL THEN '-1:'
            ELSE CAST(LENGTH(CAST(instrument_id AS TEXT)) AS TEXT)
                || ':' || CAST(instrument_id AS TEXT)
        END
        || ';expiry='
        || CASE
            WHEN canonical_expiry IS NULL THEN '-1:'
            ELSE CAST(LENGTH(canonical_expiry) AS TEXT)
                || ':' || canonical_expiry
        END
        || ';strike='
        || CASE
            WHEN canonical_strike IS NULL THEN '-1:'
            ELSE CAST(LENGTH(canonical_strike) AS TEXT)
                || ':' || canonical_strike
        END
        || ';option_type='
        || CASE
            WHEN option_type IS NULL THEN '-1:'
            ELSE CAST(LENGTH(CAST(option_type AS TEXT)) AS TEXT)
                || ':' || CAST(option_type AS TEXT)
        END
        || ';quote_convention='
        || CASE
            WHEN quote_convention IS NULL THEN '-1:'
            ELSE CAST(LENGTH(CAST(quote_convention AS TEXT)) AS TEXT)
                || ':' || CAST(quote_convention AS TEXT)
        END
        || ';bid='
        || CASE
            WHEN canonical_bid IS NULL THEN '-1:'
            ELSE CAST(LENGTH(canonical_bid) AS TEXT)
                || ':' || canonical_bid
        END
        || ';ask='
        || CASE
            WHEN canonical_ask IS NULL THEN '-1:'
            ELSE CAST(LENGTH(canonical_ask) AS TEXT)
                || ':' || canonical_ask
        END
        || ';valid_from='
        || CASE
            WHEN canonical_valid_from IS NULL THEN '-1:'
            ELSE CAST(LENGTH(canonical_valid_from) AS TEXT)
                || ':' || canonical_valid_from
        END
        || ';valid_to='
        || CASE
            WHEN canonical_valid_to IS NULL THEN '-1:'
            ELSE CAST(LENGTH(canonical_valid_to) AS TEXT)
                || ':' || canonical_valid_to
        END
        || ';known_at='
        || CASE
            WHEN canonical_known_at IS NULL THEN '-1:'
            ELSE CAST(LENGTH(canonical_known_at) AS TEXT)
                || ':' || canonical_known_at
        END
        || ';superseded_at='
        || CASE
            WHEN canonical_superseded_at IS NULL THEN '-1:'
            ELSE CAST(LENGTH(canonical_superseded_at) AS TEXT)
                || ':' || canonical_superseded_at
        END
        || ';revision_id='
        || CASE
            WHEN canonical_revision_id IS NULL THEN '-1:'
            ELSE CAST(LENGTH(canonical_revision_id) AS TEXT)
                || ':' || canonical_revision_id
        END
        || ';quote_payload_hash='
        || CASE
            WHEN quote_payload_hash IS NULL THEN '-1:'
            ELSE CAST(LENGTH(CAST(quote_payload_hash AS TEXT)) AS TEXT)
                || ':' || CAST(quote_payload_hash AS TEXT)
        END AS canonical_row
    FROM verified_payloads AS q
),
snapshot_manifest AS (
    SELECT
        p.snapshot_id,
        p.market_as_of,
        p.knowledge_as_of,
        LOWER(ENCODE(DIGEST(
            COALESCE(
                STRING_AGG(
                    q.canonical_row,
                    '' ORDER BY
                        q.instrument_id COLLATE "C",
                        q.expiry,
                        q.strike,
                        q.option_type COLLATE "C",
                        q.quote_convention COLLATE "C",
                        q.revision_id
                ),
                ''
            ),
            'sha256'
        ), 'hex')) AS snapshot_hash,
        COUNT(q.canonical_row) AS row_count,
        COALESCE(SUM(CASE WHEN q.payload_hash_valid = 0 THEN 1 ELSE 0 END), 0)
            AS invalid_payload_hash_count,
        a.duplicate_version_group_count,
        MIN(q.known_at) AS minimum_known_at,
        MAX(q.known_at) AS maximum_known_at,
        'sha256' AS hash_algorithm,
        'length-prefixed-utf8-v1' AS serializer_version,
        'calibration_snapshot_v2' AS schema_version,
        CASE
            WHEN a.duplicate_version_group_count > 0 THEN 'REJECT'
            WHEN COUNT(q.canonical_row) = 0 THEN 'REJECT'
            WHEN SUM(CASE WHEN q.payload_hash_valid = 0 THEN 1 ELSE 0 END) > 0
                THEN 'REJECT'
            ELSE 'PASS'
        END AS snapshot_status,
        CASE
            WHEN a.duplicate_version_group_count > 0
                THEN 'DUPLICATE_ELIGIBLE_VERSION'
            WHEN COUNT(q.canonical_row) = 0 THEN 'EMPTY_SNAPSHOT'
            WHEN SUM(CASE WHEN q.payload_hash_valid = 0 THEN 1 ELSE 0 END) > 0
                THEN 'MISSING_INVALID_OR_MISMATCHED_PAYLOAD_HASH'
            ELSE 'COMPLETE_POINT_IN_TIME_SNAPSHOT'
        END AS snapshot_reason
    FROM run_parameters AS p
    CROSS JOIN eligibility_audit AS a
    LEFT JOIN canonical_rows AS q ON 1 = 1
    GROUP BY
        p.snapshot_id,
        p.market_as_of,
        p.knowledge_as_of,
        a.duplicate_version_group_count
)
SELECT
    m.snapshot_id,
    m.market_as_of,
    m.knowledge_as_of,
    m.snapshot_hash,
    m.row_count,
    m.invalid_payload_hash_count,
    m.duplicate_version_group_count,
    m.minimum_known_at,
    m.maximum_known_at,
    m.hash_algorithm,
    m.serializer_version,
    m.schema_version,
    m.snapshot_status,
    m.snapshot_reason,
    q.instrument_id,
    q.expiry,
    q.strike,
    q.option_type,
    q.quote_convention,
    q.bid,
    q.ask,
    q.valid_from,
    q.valid_to,
    q.known_at,
    q.superseded_at,
    q.revision_id,
    q.quote_payload_hash,
    q.computed_quote_payload_hash,
    q.payload_hash_valid,
    q.canonical_quote_payload,
    q.canonical_row
FROM snapshot_manifest AS m
LEFT JOIN canonical_rows AS q ON 1 = 1
ORDER BY
    q.instrument_id COLLATE "C",
    q.expiry,
    q.strike,
    q.option_type COLLATE "C",
    q.quote_convention COLLATE "C",
    q.revision_id;
