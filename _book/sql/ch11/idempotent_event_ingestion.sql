-- Dialect: SQLite 3.45 with trusted sha256(text) UDF at the ingestion boundary.
-- Canonical payload construction and hashing happen here. No caller hash is trusted.
CREATE TABLE IF NOT EXISTS processed_messages (
    consumer_name TEXT NOT NULL,
    message_id TEXT NOT NULL,
    canonical_payload TEXT NOT NULL,
    payload_hash TEXT NOT NULL CHECK (
        length(payload_hash) = 64 AND
        payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (consumer_name, message_id)
);

CREATE TABLE IF NOT EXISTS business_effects (
    consumer_name TEXT NOT NULL,
    message_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    amount_minor INTEGER NOT NULL,
    currency TEXT NOT NULL CHECK (
        length(currency) = 3 AND currency GLOB '[A-Z][A-Z][A-Z]'
    ),
    scale INTEGER NOT NULL CHECK (scale BETWEEN 0 AND 9),
    schema_version INTEGER NOT NULL CHECK (schema_version >= 1),
    PRIMARY KEY (consumer_name, message_id)
);

CREATE TEMP TABLE IF NOT EXISTS ingestion_attempt (
    canonical_payload TEXT,
    computed_hash TEXT,
    input_is_valid INTEGER NOT NULL
);
CREATE TEMP TABLE IF NOT EXISTS ingestion_outcome(inserted INTEGER NOT NULL);

BEGIN IMMEDIATE;
DELETE FROM ingestion_attempt;
DELETE FROM ingestion_outcome;

INSERT INTO ingestion_attempt(canonical_payload, computed_hash, input_is_valid)
SELECT
    canonical_payload,
    CASE WHEN input_is_valid = 1 THEN sha256(canonical_payload) END,
    input_is_valid
FROM (
    SELECT
        json_object(
            'schema_version', :schema_version,
            'account_id', :account_id,
            'amount_minor', :amount_minor,
            'currency', :currency,
            'scale', :scale
        ) AS canonical_payload,
        CASE
            WHEN typeof(:consumer_name) = 'text' AND TRIM(:consumer_name) <> '' AND
                 typeof(:message_id) = 'text' AND TRIM(:message_id) <> '' AND
                 typeof(:account_id) = 'text' AND TRIM(:account_id) <> '' AND
                 typeof(:amount_minor) = 'integer' AND
                 typeof(:currency) = 'text' AND length(:currency) = 3 AND
                 :currency GLOB '[A-Z][A-Z][A-Z]' AND
                 typeof(:scale) = 'integer' AND :scale BETWEEN 0 AND 9 AND
                 typeof(:schema_version) = 'integer' AND :schema_version >= 1
                THEN 1 ELSE 0
        END AS input_is_valid
);

INSERT INTO processed_messages(
    consumer_name, message_id, canonical_payload, payload_hash
)
SELECT :consumer_name, :message_id, canonical_payload, computed_hash
FROM ingestion_attempt
WHERE input_is_valid = 1
ON CONFLICT (consumer_name, message_id) DO NOTHING;

INSERT INTO ingestion_outcome(inserted) VALUES (changes());

INSERT INTO business_effects(
    consumer_name, message_id, account_id, amount_minor, currency, scale,
    schema_version
)
SELECT
    :consumer_name, :message_id, :account_id, :amount_minor, :currency, :scale,
    :schema_version
FROM ingestion_attempt
WHERE input_is_valid = 1
  AND (SELECT inserted FROM ingestion_outcome) = 1;
COMMIT;

SELECT
    CASE
        WHEN attempt.input_is_valid <> 1 THEN 'REJECT_INVALID_INPUT'
        WHEN stored.payload_hash <> attempt.computed_hash OR
             stored.canonical_payload <> attempt.canonical_payload
            THEN 'REJECT_ID_REUSED_WITH_DIFFERENT_PAYLOAD'
        WHEN outcome.inserted = 1 THEN 'NEWLY_PROCESSED'
        ELSE 'DUPLICATE_SAME_PAYLOAD'
    END AS status,
    attempt.computed_hash AS payload_hash
FROM ingestion_attempt AS attempt
CROSS JOIN ingestion_outcome AS outcome
LEFT JOIN processed_messages AS stored
  ON stored.consumer_name = :consumer_name
 AND stored.message_id = :message_id;
