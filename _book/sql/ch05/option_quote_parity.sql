-- Dialect: SQLite 3.35+ deterministic fixture; timestamps are canonical UTC ISO-8601 text.
PRAGMA foreign_keys = ON;

CREATE TABLE parity_audit_config (
    audit_id TEXT PRIMARY KEY,
    underlying TEXT NOT NULL,
    maturity TEXT NOT NULL,
    strike REAL NOT NULL CHECK (strike > 0.0),
    valid_at TEXT NOT NULL,
    knowledge_cutoff TEXT NOT NULL,
    maximum_market_event_skew_seconds INTEGER NOT NULL CHECK (maximum_market_event_skew_seconds >= 0)
);

INSERT INTO parity_audit_config VALUES (
    'parity-20260725-100', 'XYZ', '2026-12-18', 100.0,
    '2026-07-25T10:00:00.500000Z', '2026-07-25T10:00:05.000000Z', 2
);

CREATE TABLE option_quote (
    quote_id TEXT PRIMARY KEY,
    revision_id INTEGER NOT NULL CHECK (revision_id >= 1),
    source_priority INTEGER NOT NULL,
    underlying TEXT NOT NULL,
    maturity TEXT NOT NULL,
    strike REAL NOT NULL CHECK (strike > 0.0),
    option_type TEXT NOT NULL CHECK (option_type IN ('C', 'P')),
    bid REAL NOT NULL CHECK (bid >= 0.0),
    ask REAL NOT NULL CHECK (ask >= bid),
    market_event_time TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    known_at TEXT NOT NULL,
    superseded_at TEXT,
    payload_hash TEXT NOT NULL CHECK (
        length(payload_hash) = 64
        AND payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (valid_to IS NULL OR valid_to > valid_from),
    CHECK (superseded_at IS NULL OR superseded_at > known_at)
);

INSERT INTO option_quote VALUES
(
    'call-r1', 1, 10, 'XYZ', '2026-12-18', 100.0, 'C', 8.80, 9.20,
    '2026-07-25T10:00:00.000000Z',
    '2026-07-25T09:59:59.000000Z', NULL,
    '2026-07-25T10:00:01.000000Z', '2026-07-25T10:00:03.000000Z',
    '1111111111111111111111111111111111111111111111111111111111111111'
),
(
    'call-r2', 2, 10, 'XYZ', '2026-12-18', 100.0, 'C', 8.90, 9.10,
    '2026-07-25T10:00:00.000000Z',
    '2026-07-25T10:00:00.000000Z', NULL,
    '2026-07-25T10:00:03.000000Z', NULL,
    '2222222222222222222222222222222222222222222222222222222222222222'
),
(
    'call-r3-future', 3, 10, 'XYZ', '2026-12-18', 100.0, 'C', 9.00, 9.20,
    '2026-07-25T10:00:00.000000Z',
    '2026-07-25T10:00:00.000000Z', NULL,
    '2026-07-25T10:00:06.000000Z', NULL,
    '3333333333333333333333333333333333333333333333333333333333333333'
),
(
    'put-r1', 1, 10, 'XYZ', '2026-12-18', 100.0, 'P', 6.10, 6.30,
    '2026-07-25T10:00:01.000000Z',
    '2026-07-25T09:59:58.000000Z', NULL,
    '2026-07-25T10:00:02.000000Z', NULL,
    '4444444444444444444444444444444444444444444444444444444444444444'
);

CREATE TABLE parity_snapshot AS
WITH eligible AS (
    SELECT
        config.audit_id,
        quote.*,
        ROW_NUMBER() OVER (
            PARTITION BY config.audit_id, quote.underlying, quote.maturity,
                         quote.strike, quote.option_type
            ORDER BY quote.known_at DESC, quote.revision_id DESC,
                     quote.source_priority DESC, quote.quote_id DESC
        ) AS selected_rank
    FROM parity_audit_config AS config
    JOIN option_quote AS quote
      ON quote.underlying = config.underlying
     AND quote.maturity = config.maturity
     AND quote.strike = config.strike
     AND quote.valid_from <= config.valid_at
     AND (quote.valid_to IS NULL OR quote.valid_to > config.valid_at)
     AND quote.known_at <= config.knowledge_cutoff
     AND (quote.superseded_at IS NULL OR quote.superseded_at > config.knowledge_cutoff)
),
selected AS (
    SELECT * FROM eligible WHERE selected_rank = 1
)
SELECT
    config.audit_id,
    call.quote_id AS call_quote_id,
    put.quote_id AS put_quote_id,
    call.revision_id AS call_revision_id,
    put.revision_id AS put_revision_id,
    config.valid_at,
    config.knowledge_cutoff,
    call.market_event_time AS call_market_event_time,
    put.market_event_time AS put_market_event_time,
    call.known_at AS call_known_at,
    put.known_at AS put_known_at,
    call.bid AS call_bid,
    call.ask AS call_ask,
    put.bid AS put_bid,
    put.ask AS put_ask,
    call.bid - put.ask AS executable_call_minus_put_lower,
    call.ask - put.bid AS executable_call_minus_put_upper,
    ABS((JULIANDAY(call.market_event_time) - JULIANDAY(put.market_event_time)) * 86400.0)
        AS market_event_skew_seconds,
    ABS((JULIANDAY(call.known_at) - JULIANDAY(put.known_at)) * 86400.0)
        AS knowledge_time_skew_seconds,
    CASE
        WHEN ABS((JULIANDAY(call.market_event_time) - JULIANDAY(put.market_event_time)) * 86400.0)
             > config.maximum_market_event_skew_seconds
        THEN 'REJECT_MARKET_EVENT_SKEW'
        ELSE 'PASS'
    END AS snapshot_status,
    call.payload_hash || ':' || put.payload_hash AS selected_payload_hashes
FROM parity_audit_config AS config
JOIN selected AS call
  ON call.audit_id = config.audit_id AND call.option_type = 'C'
JOIN selected AS put
  ON put.audit_id = config.audit_id AND put.option_type = 'P';
