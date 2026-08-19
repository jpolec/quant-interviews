-- Dialect: SQLite 3.45+
-- Run inside one transaction. The database checks hash shape, full duplicate
-- content and aggregate sequence continuity. A trusted ingest service must
-- recompute SHA-256(payload) before binding :payload_hash.
BEGIN IMMEDIATE;

INSERT INTO event_log(
    event_id,
    aggregate_id,
    sequence_no,
    event_type,
    payload,
    payload_hash,
    recorded_at
)
SELECT
    :event_id,
    :aggregate_id,
    :sequence_no,
    :event_type,
    :payload,
    :payload_hash,
    CURRENT_TIMESTAMP
WHERE length(:payload_hash) = 64
  AND :payload_hash NOT GLOB '*[^0-9a-f]*'
  AND length(:event_id) > 0
  AND length(:aggregate_id) > 0
  AND length(:event_type) > 0
  AND :sequence_no > 0
  AND (
      EXISTS (SELECT 1 FROM event_log WHERE event_id = :event_id)
      OR :sequence_no = COALESCE(
          (SELECT MAX(sequence_no) + 1
           FROM event_log
           WHERE aggregate_id = :aggregate_id),
          1
      )
  )
ON CONFLICT DO NOTHING;

SELECT
    COALESCE(e.event_id, :event_id) AS event_id,
    COALESCE(e.aggregate_id, :aggregate_id) AS aggregate_id,
    COALESCE(e.sequence_no, :sequence_no) AS sequence_no,
    COALESCE(e.payload_hash, :payload_hash) AS payload_hash,
    CASE
        WHEN length(:payload_hash) <> 64
          OR :payload_hash GLOB '*[^0-9a-f]*' THEN 'INVALID_HASH'
        WHEN length(:event_id) = 0
          OR length(:aggregate_id) = 0
          OR length(:event_type) = 0
          OR :sequence_no <= 0 THEN 'INVALID_INPUT'
        WHEN changes() = 1 THEN 'APPENDED'
        WHEN e.event_id IS NOT NULL
         AND e.aggregate_id = :aggregate_id
         AND e.sequence_no = :sequence_no
         AND e.event_type = :event_type
         AND e.payload = :payload
         AND e.payload_hash = :payload_hash
            THEN 'DUPLICATE_SAME_CONTENT'
        WHEN e.event_id IS NOT NULL THEN 'CONFLICT'
        WHEN EXISTS (
            SELECT 1 FROM event_log
            WHERE aggregate_id = :aggregate_id AND sequence_no = :sequence_no
        ) THEN 'SEQUENCE_CONFLICT'
        ELSE 'SEQUENCE_GAP'
    END AS ingestion_status
FROM (SELECT 1) AS singleton
LEFT JOIN event_log AS e ON e.event_id = :event_id;

COMMIT;
