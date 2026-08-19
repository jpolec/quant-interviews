-- Dialect: SQLite 3
-- Point-in-time outcome selection for a frozen forecast-evaluation run.
CREATE TABLE evaluation_context(
    evaluation_id TEXT PRIMARY KEY,
    evaluation_knowledge_cutoff TEXT NOT NULL
);

CREATE TABLE forecast(
    forecast_id TEXT PRIMARY KEY,
    decision_time TEXT NOT NULL,
    probability REAL NOT NULL,
    model_version TEXT NOT NULL,
    forecast_source_id TEXT NOT NULL,
    forecast_row_hash TEXT NOT NULL
);

CREATE TABLE outcome_revision(
    forecast_id TEXT NOT NULL,
    label REAL NOT NULL,
    effective_at TEXT NOT NULL,
    known_at TEXT NOT NULL,
    revision INTEGER NOT NULL,
    outcome_source_id TEXT NOT NULL,
    outcome_row_hash TEXT NOT NULL,
    PRIMARY KEY(
        forecast_id,
        known_at,
        revision,
        outcome_source_id,
        outcome_row_hash
    )
);

-- Frozen fixtures include a future correction and a deterministic tie.
INSERT INTO evaluation_context VALUES
    ('E1', '2026-01-04T00:00:00');

INSERT INTO forecast VALUES
    ('P1', '2026-01-02T10:00:00', 0.70, 'model-v3', 'forecast-service', 'f-hash-p1'),
    ('P2', '2026-01-02T11:00:00', 0.35, 'model-v3', 'forecast-service', 'f-hash-p2');

INSERT INTO outcome_revision VALUES
    ('P1', 1.0, '2026-01-03T09:00:00', '2026-01-03T10:00:00', 1, 'vendor-a', 'o-hash-p1-r1'),
    ('P1', 0.0, '2026-01-03T09:00:00', '2026-01-05T10:00:00', 2, 'vendor-a', 'o-hash-p1-r2-future'),
    ('P2', 0.0, '2026-01-03T09:30:00', '2026-01-03T12:00:00', 1, 'vendor-a', 'o-hash-p2-a'),
    ('P2', 1.0, '2026-01-03T09:30:00', '2026-01-03T12:00:00', 1, 'vendor-b', 'o-hash-p2-b');

CREATE TABLE selected_outcome AS
WITH eligible_revision AS (
    SELECT
        c.evaluation_id,
        c.evaluation_knowledge_cutoff,
        f.forecast_id,
        f.decision_time,
        f.probability,
        f.model_version,
        f.forecast_source_id,
        f.forecast_row_hash,
        o.label,
        o.effective_at,
        o.known_at,
        o.revision,
        o.outcome_source_id,
        o.outcome_row_hash,
        ROW_NUMBER() OVER (
            PARTITION BY c.evaluation_id, f.forecast_id
            ORDER BY
                o.known_at DESC,
                o.revision DESC,
                o.outcome_source_id DESC,
                o.outcome_row_hash DESC
        ) AS revision_rank
    FROM evaluation_context AS c
    JOIN forecast AS f
      ON f.decision_time <= c.evaluation_knowledge_cutoff
    JOIN outcome_revision AS o
      ON o.forecast_id = f.forecast_id
     AND o.known_at <= c.evaluation_knowledge_cutoff
)
SELECT
    evaluation_id,
    evaluation_knowledge_cutoff,
    forecast_id,
    decision_time,
    probability,
    model_version,
    forecast_source_id,
    forecast_row_hash,
    label,
    effective_at,
    known_at,
    revision,
    outcome_source_id,
    outcome_row_hash,
    'SELECTED_AS_OF_CUTOFF' AS selection_status
FROM eligible_revision
WHERE revision_rank = 1;
