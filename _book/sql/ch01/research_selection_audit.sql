-- Dialect: SQLite 3
-- Audit whether every pre-registered research family was selected completely.
CREATE TABLE research_family(
    family_id TEXT PRIMARY KEY,
    evaluation_knowledge_cutoff TEXT NOT NULL,
    declared_trial_count INTEGER NOT NULL
);

CREATE TABLE research_trial(
    trial_id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL,
    score REAL,
    selected INTEGER NOT NULL CHECK(selected IN (0, 1)),
    known_at TEXT NOT NULL,
    code_version TEXT NOT NULL,
    trial_row_hash TEXT NOT NULL
);

INSERT INTO research_family VALUES
    ('F_PASS', '2026-01-02T00:00:00', 3),
    ('F_NONE_SELECTED', '2026-01-02T00:00:00', 2),
    ('F_MULTI_SELECTED', '2026-01-02T00:00:00', 2),
    ('F_MISSING_SCORE', '2026-01-02T00:00:00', 2),
    ('F_INCOMPLETE_AS_OF', '2026-01-02T00:00:00', 2);

INSERT INTO research_trial VALUES
    ('T1', 'F_PASS', 1.1, 0, '2026-01-01T10:00:00', 'git-a1', 'hash-t1'),
    ('T2', 'F_PASS', 2.3, 1, '2026-01-01T10:01:00', 'git-a1', 'hash-t2'),
    ('T3', 'F_PASS', 0.7, 0, '2026-01-01T10:02:00', 'git-a1', 'hash-t3'),
    ('T4', 'F_NONE_SELECTED', 0.4, 0, '2026-01-01T11:00:00', 'git-a1', 'hash-t4'),
    ('T5', 'F_NONE_SELECTED', 0.5, 0, '2026-01-01T11:01:00', 'git-a1', 'hash-t5'),
    ('T6', 'F_MULTI_SELECTED', 0.9, 1, '2026-01-01T12:00:00', 'git-a1', 'hash-t6'),
    ('T7', 'F_MULTI_SELECTED', 1.0, 1, '2026-01-01T12:01:00', 'git-a1', 'hash-t7'),
    ('T8', 'F_MISSING_SCORE', NULL, 0, '2026-01-01T13:00:00', 'git-a1', 'hash-t8'),
    ('T9', 'F_MISSING_SCORE', 1.2, 1, '2026-01-01T13:01:00', 'git-a1', 'hash-t9'),
    ('T10', 'F_INCOMPLETE_AS_OF', 1.4, 1, '2026-01-01T14:00:00', 'git-a1', 'hash-t10'),
    ('T11', 'F_INCOMPLETE_AS_OF', 1.5, 0, '2026-01-03T14:00:00', 'git-a2', 'hash-t11-future');

CREATE TABLE selection_audit AS
WITH family_counts AS (
    SELECT
        f.family_id,
        f.evaluation_knowledge_cutoff,
        f.declared_trial_count,
        COUNT(t.trial_id) FILTER (
            WHERE t.known_at <= f.evaluation_knowledge_cutoff
        ) AS visible_trial_count,
        SUM(CASE
            WHEN t.known_at <= f.evaluation_knowledge_cutoff
             AND t.selected = 1 THEN 1
            ELSE 0
        END) AS selected_count,
        SUM(CASE
            WHEN t.known_at <= f.evaluation_knowledge_cutoff
             AND t.score IS NULL THEN 1
            ELSE 0
        END) AS missing_score_count,
        SUM(CASE
            WHEN t.known_at > f.evaluation_knowledge_cutoff THEN 1
            ELSE 0
        END) AS future_knowledge_count,
        MAX(CASE
            WHEN t.known_at <= f.evaluation_knowledge_cutoff THEN t.score
            ELSE NULL
        END) AS best_visible_score,
        MAX(CASE
            WHEN t.known_at <= f.evaluation_knowledge_cutoff
             AND t.selected = 1 THEN t.trial_id
            ELSE NULL
        END) AS selected_trial_id,
        MAX(CASE
            WHEN t.known_at <= f.evaluation_knowledge_cutoff
             AND t.selected = 1 THEN t.code_version
            ELSE NULL
        END) AS selected_code_version,
        MAX(CASE
            WHEN t.known_at <= f.evaluation_knowledge_cutoff
             AND t.selected = 1 THEN t.trial_row_hash
            ELSE NULL
        END) AS selected_trial_row_hash
    FROM research_family AS f
    LEFT JOIN research_trial AS t
      ON t.family_id = f.family_id
    GROUP BY
        f.family_id,
        f.evaluation_knowledge_cutoff,
        f.declared_trial_count
)
SELECT
    family_id,
    evaluation_knowledge_cutoff,
    declared_trial_count,
    visible_trial_count,
    selected_count,
    missing_score_count,
    future_knowledge_count,
    best_visible_score,
    selected_trial_id,
    selected_code_version,
    selected_trial_row_hash,
    CASE
        WHEN visible_trial_count <> declared_trial_count THEN 'REJECT'
        WHEN selected_count <> 1 THEN 'REJECT'
        WHEN missing_score_count <> 0 THEN 'REJECT'
        ELSE 'PASS'
    END AS audit_status,
    CASE
        WHEN visible_trial_count <> declared_trial_count THEN 'INCOMPLETE_AS_OF_CUTOFF'
        WHEN selected_count = 0 THEN 'NO_SELECTED_TRIAL'
        WHEN selected_count > 1 THEN 'MULTIPLE_SELECTED_TRIALS'
        WHEN missing_score_count <> 0 THEN 'MISSING_VISIBLE_SCORE'
        ELSE 'COMPLETE_SINGLE_SELECTION'
    END AS audit_reason
FROM family_counts;
