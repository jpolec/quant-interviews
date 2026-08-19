-- Dialect: SQLite
-- Point-in-time event reconstruction with revision, sequence, payload and
-- lifecycle validation. The query deliberately returns one diagnostic row.
WITH RECURSIVE
request AS (
    SELECT
        :order_id AS order_id,
        :knowledge_cutoff AS knowledge_cutoff,
        :expected_initial_sequence AS expected_initial_sequence
),
ranked_revisions AS (
    SELECT
        e.*,
        ROW_NUMBER() OVER (
            PARTITION BY e.event_id
            ORDER BY e.knowledge_time DESC, e.revision_id DESC
        ) AS revision_rank,
        COUNT(*) OVER (
            PARTITION BY e.event_id, e.knowledge_time, e.revision_id
        ) AS revision_tie_count
    FROM order_events AS e
    JOIN request AS r ON r.order_id = e.order_id
    WHERE e.knowledge_time <= r.knowledge_cutoff
),
latest_revisions AS (
    SELECT *
    FROM ranked_revisions
    WHERE revision_rank = 1
),
revision_diagnostics AS (
    SELECT
        COALESCE(SUM(CASE WHEN revision_tie_count > 1 THEN 1 ELSE 0 END), 0)
            AS ambiguous_revision_count
    FROM latest_revisions
),
events AS (
    SELECT *
    FROM latest_revisions
    WHERE is_cancelled = 0
),
sequenced AS (
    SELECT
        e.*,
        ROW_NUMBER() OVER (
            ORDER BY e.sequence_number, e.event_id
        ) AS event_position,
        LAG(e.sequence_number) OVER (
            ORDER BY e.sequence_number, e.event_id
        ) AS previous_sequence,
        LAG(e.event_time) OVER (
            ORDER BY e.sequence_number, e.event_id
        ) AS previous_event_time,
        COUNT(*) OVER (
            PARTITION BY e.sequence_number
        ) AS sequence_use_count
    FROM events AS e
),
diagnostics AS (
    SELECT
        COUNT(*) AS event_count,
        (SELECT ambiguous_revision_count FROM revision_diagnostics)
            AS ambiguous_revision_count,
        COALESCE(SUM(CASE WHEN sequence_use_count > 1 THEN 1 ELSE 0 END), 0)
            AS duplicate_sequence_count,
        COALESCE(SUM(CASE
            WHEN previous_sequence IS NULL
             AND sequence_number <> (SELECT expected_initial_sequence FROM request)
                THEN 1
            WHEN previous_sequence IS NOT NULL
             AND sequence_number <> previous_sequence + 1
                THEN 1
            ELSE 0
        END), 0) AS sequence_gap_count,
        COALESCE(SUM(CASE WHEN event_type = 'SUBMIT' THEN 1 ELSE 0 END), 0)
            AS submit_count,
        COALESCE(SUM(CASE
            WHEN event_type NOT IN (
                'SUBMIT', 'ACK', 'PARTIAL_FILL', 'FILL',
                'CANCEL_REQUEST', 'CANCEL_ACK', 'REJECT'
            ) THEN 1
            WHEN event_type = 'SUBMIT'
             AND (quantity IS NULL OR quantity <= 0 OR quantity <> CAST(quantity AS INTEGER))
                THEN 1
            WHEN event_type IN ('PARTIAL_FILL', 'FILL')
             AND (
                 quantity IS NULL OR quantity <= 0
                 OR quantity <> CAST(quantity AS INTEGER)
                 OR price IS NULL OR price <= 0
             ) THEN 1
            WHEN event_type NOT IN ('SUBMIT', 'PARTIAL_FILL', 'FILL')
             AND (quantity IS NOT NULL OR price IS NOT NULL)
                THEN 1
            WHEN is_cancelled NOT IN (0, 1) THEN 1
            ELSE 0
        END), 0) AS invalid_payload_count,
        COALESCE(SUM(CASE
            WHEN previous_event_time IS NOT NULL
             AND event_time < previous_event_time THEN 1
            ELSE 0
        END), 0) AS event_time_regression_count
    FROM sequenced
),
state (
    event_position, status, submitted_quantity, filled_quantity,
    fill_notional, cancel_requested, invalid_transition_count
) AS (
    SELECT 0, 'INIT', NULL, 0.0, 0.0, 0, 0
    UNION ALL
    SELECT
        s.event_position,
        CASE
            WHEN s.event_type = 'SUBMIT' AND st.status = 'INIT'
                THEN 'PENDING'
            WHEN s.event_type = 'ACK' AND st.status = 'PENDING'
                THEN 'WORKING'
            WHEN s.event_type = 'PARTIAL_FILL'
             AND st.status IN (
                 'WORKING', 'PARTIALLY_FILLED', 'CANCEL_PENDING',
                 'PARTIALLY_FILLED_CANCEL_PENDING'
             )
             AND st.filled_quantity + s.quantity < st.submitted_quantity
                THEN CASE
                    WHEN st.cancel_requested = 1
                        THEN 'PARTIALLY_FILLED_CANCEL_PENDING'
                    ELSE 'PARTIALLY_FILLED'
                END
            WHEN s.event_type = 'FILL'
             AND st.status IN (
                 'WORKING', 'PARTIALLY_FILLED', 'CANCEL_PENDING',
                 'PARTIALLY_FILLED_CANCEL_PENDING'
             )
             AND st.filled_quantity + s.quantity = st.submitted_quantity
                THEN 'FILLED'
            WHEN s.event_type = 'CANCEL_REQUEST'
             AND st.status IN ('WORKING', 'PARTIALLY_FILLED')
                THEN CASE
                    WHEN st.filled_quantity > 0
                        THEN 'PARTIALLY_FILLED_CANCEL_PENDING'
                    ELSE 'CANCEL_PENDING'
                END
            WHEN s.event_type = 'CANCEL_ACK'
             AND st.status IN ('CANCEL_PENDING', 'PARTIALLY_FILLED_CANCEL_PENDING')
                THEN CASE
                    WHEN st.filled_quantity > 0
                        THEN 'PARTIALLY_FILLED_CANCELLED'
                    ELSE 'CANCELLED'
                END
            WHEN s.event_type = 'REJECT' AND st.status = 'PENDING'
                THEN 'REJECTED'
            ELSE 'INVALID'
        END,
        CASE WHEN s.event_type = 'SUBMIT' THEN s.quantity
             ELSE st.submitted_quantity END,
        st.filled_quantity + CASE
            WHEN s.event_type IN ('PARTIAL_FILL', 'FILL')
                THEN COALESCE(s.quantity, 0)
            ELSE 0
        END,
        st.fill_notional + CASE
            WHEN s.event_type IN ('PARTIAL_FILL', 'FILL')
                THEN COALESCE(s.quantity * s.price, 0)
            ELSE 0
        END,
        CASE
            WHEN s.event_type = 'CANCEL_REQUEST' THEN 1
            WHEN s.event_type IN ('CANCEL_ACK', 'FILL') THEN 0
            ELSE st.cancel_requested
        END,
        st.invalid_transition_count + CASE
            WHEN
                (s.event_type = 'SUBMIT' AND st.status = 'INIT')
                OR (s.event_type = 'ACK' AND st.status = 'PENDING')
                OR (
                    s.event_type = 'PARTIAL_FILL'
                    AND st.status IN (
                        'WORKING', 'PARTIALLY_FILLED', 'CANCEL_PENDING',
                        'PARTIALLY_FILLED_CANCEL_PENDING'
                    )
                    AND st.filled_quantity + s.quantity < st.submitted_quantity
                )
                OR (
                    s.event_type = 'FILL'
                    AND st.status IN (
                        'WORKING', 'PARTIALLY_FILLED', 'CANCEL_PENDING',
                        'PARTIALLY_FILLED_CANCEL_PENDING'
                    )
                    AND st.filled_quantity + s.quantity = st.submitted_quantity
                )
                OR (
                    s.event_type = 'CANCEL_REQUEST'
                    AND st.status IN ('WORKING', 'PARTIALLY_FILLED')
                )
                OR (
                    s.event_type = 'CANCEL_ACK'
                    AND st.status IN (
                        'CANCEL_PENDING', 'PARTIALLY_FILLED_CANCEL_PENDING'
                    )
                )
                OR (s.event_type = 'REJECT' AND st.status = 'PENDING')
            THEN 0 ELSE 1
        END
    FROM state AS st
    JOIN sequenced AS s ON s.event_position = st.event_position + 1
),
final_state AS (
    SELECT *
    FROM state
    ORDER BY event_position DESC
    LIMIT 1
)
SELECT
    r.order_id,
    f.event_position AS event_count,
    f.submitted_quantity,
    CASE WHEN d.event_count > 0
          AND d.ambiguous_revision_count = 0
          AND d.duplicate_sequence_count = 0
          AND d.sequence_gap_count = 0
          AND d.submit_count = 1
          AND d.invalid_payload_count = 0
          AND d.event_time_regression_count = 0
          AND f.invalid_transition_count = 0
          AND f.filled_quantity <= f.submitted_quantity
        THEN f.filled_quantity END AS filled_quantity,
    CASE WHEN f.filled_quantity > 0
          AND d.ambiguous_revision_count = 0
          AND d.duplicate_sequence_count = 0
          AND d.sequence_gap_count = 0
          AND d.submit_count = 1
          AND d.invalid_payload_count = 0
          AND d.event_time_regression_count = 0
          AND f.invalid_transition_count = 0
          AND f.filled_quantity <= f.submitted_quantity
        THEN f.fill_notional / f.filled_quantity END AS average_fill_price,
    d.sequence_gap_count,
    d.duplicate_sequence_count,
    d.ambiguous_revision_count,
    d.invalid_payload_count,
    f.invalid_transition_count,
    f.status AS reconstructed_state,
    CASE
        WHEN r.expected_initial_sequence IS NULL
          OR r.expected_initial_sequence <= 0
          OR r.expected_initial_sequence <> CAST(r.expected_initial_sequence AS INTEGER)
            THEN 'REJECT_INVALID_PARAMETER'
        WHEN d.event_count = 0 THEN 'REJECT_MISSING_ORDER'
        WHEN d.ambiguous_revision_count > 0 THEN 'REJECT_AMBIGUOUS_REVISION'
        WHEN d.duplicate_sequence_count > 0 THEN 'REJECT_DUPLICATE_SEQUENCE'
        WHEN d.sequence_gap_count > 0 THEN 'REJECT_SEQUENCE_GAP'
        WHEN d.submit_count <> 1 THEN 'REJECT_SUBMIT_CARDINALITY'
        WHEN d.invalid_payload_count > 0 THEN 'REJECT_INVALID_PAYLOAD'
        WHEN d.event_time_regression_count > 0 THEN 'REJECT_EVENT_TIME_REGRESSION'
        WHEN f.invalid_transition_count > 0 THEN 'REJECT_INVALID_TRANSITION'
        WHEN f.filled_quantity > f.submitted_quantity THEN 'REJECT_OVERFILL'
        ELSE 'PASS'
    END AS status
FROM request AS r
CROSS JOIN diagnostics AS d
CROSS JOIN final_state AS f;
