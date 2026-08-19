-- Dialect: SQLite
-- Fail-closed, point-in-time NBBO. Venue state is ranked before cancellation,
-- eligibility and freshness filters, so an old quote cannot be resurrected.
WITH request AS (
    SELECT
        :symbol AS symbol,
        :decision_time AS decision_time,
        :knowledge_cutoff AS knowledge_cutoff,
        :max_quote_age AS max_quote_age
),
ranked AS (
    SELECT
        q.*,
        ROW_NUMBER() OVER (
            PARTITION BY q.symbol, q.venue
            ORDER BY q.authoritative_sequence DESC,
                     q.knowledge_time DESC,
                     q.revision_id DESC
        ) AS recency_rank,
        COUNT(*) OVER (
            PARTITION BY q.symbol, q.venue, q.authoritative_sequence,
                         q.knowledge_time, q.revision_id
        ) AS recency_tie_count
    FROM quote_history AS q
    JOIN request AS r ON r.symbol = q.symbol
    WHERE q.valid_time <= r.decision_time
      AND q.knowledge_time <= r.knowledge_cutoff
),
latest AS (
    SELECT *
    FROM ranked
    WHERE recency_rank = 1
),
diagnostics AS (
    SELECT
        COUNT(*) AS venue_state_count,
        COALESCE(SUM(CASE WHEN recency_tie_count > 1 THEN 1 ELSE 0 END), 0)
            AS ambiguous_recency_count,
        COALESCE(SUM(CASE
            WHEN authoritative_sequence IS NULL
              OR authoritative_sequence < 0
              OR authoritative_sequence <> CAST(authoritative_sequence AS INTEGER)
              OR bid_price IS NULL OR ask_price IS NULL
              OR bid_price <= 0 OR ask_price <= 0
              OR bid_price >= ask_price
              OR bid_size IS NULL OR ask_size IS NULL
              OR bid_size < 0 OR ask_size < 0
              OR is_cancelled NOT IN (0, 1)
              OR is_eligible NOT IN (0, 1)
            THEN 1 ELSE 0
        END), 0) AS invalid_quote_count
    FROM latest
),
eligible AS (
    SELECT l.*
    FROM latest AS l
    CROSS JOIN request AS r
    WHERE l.is_cancelled = 0
      AND l.is_eligible = 1
      AND r.decision_time - l.valid_time BETWEEN 0 AND r.max_quote_age
),
prices AS (
    SELECT
        MAX(bid_price) AS nbbo_bid,
        MIN(ask_price) AS nbbo_ask,
        COUNT(*) AS eligible_venue_count
    FROM eligible
),
sizes AS (
    SELECT
        COALESCE(SUM(CASE WHEN e.bid_price = p.nbbo_bid THEN e.bid_size ELSE 0 END), 0)
            AS nbbo_bid_size,
        COALESCE(SUM(CASE WHEN e.ask_price = p.nbbo_ask THEN e.ask_size ELSE 0 END), 0)
            AS nbbo_ask_size
    FROM eligible AS e
    CROSS JOIN prices AS p
)
SELECT
    r.symbol,
    p.nbbo_bid,
    p.nbbo_ask,
    s.nbbo_bid_size,
    s.nbbo_ask_size,
    p.eligible_venue_count,
    d.venue_state_count,
    d.ambiguous_recency_count,
    d.invalid_quote_count,
    CASE
        WHEN r.symbol IS NULL OR TRIM(r.symbol) = ''
          OR r.decision_time IS NULL OR r.knowledge_cutoff IS NULL
          OR r.max_quote_age IS NULL OR r.max_quote_age < 0
            THEN 'REJECT_INVALID_PARAMETER'
        WHEN d.ambiguous_recency_count > 0 THEN 'REJECT_AMBIGUOUS_RECENCY'
        WHEN d.invalid_quote_count > 0 THEN 'REJECT_INVALID_QUOTE'
        WHEN p.eligible_venue_count = 0 THEN 'REJECT_MISSING_SIDE'
        WHEN p.nbbo_bid IS NULL OR p.nbbo_ask IS NULL THEN 'REJECT_MISSING_SIDE'
        WHEN p.nbbo_bid >= p.nbbo_ask THEN 'REJECT_LOCKED_OR_CROSSED'
        ELSE 'PASS'
    END AS status
FROM request AS r
CROSS JOIN diagnostics AS d
CROSS JOIN prices AS p
CROSS JOIN sizes AS s;
