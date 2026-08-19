-- Dialect: SQLite
-- Exact nine-state reconciliation of a hash-bound coordination certificate.
WITH assessed AS (
  SELECT
    *,
    COUNT(*) OVER (PARTITION BY trial_id) AS trial_copies,
    COUNT(*) OVER (
      PARTITION BY strategy, enumeration_hash, alice_forbidden, bob_forbidden
    ) AS state_copies,
    CASE
      WHEN strategy IS NULL OR TRIM(strategy) = ''
        OR strategy_version IS NULL OR TRIM(strategy_version) = ''
        OR trial_id IS NULL OR TRIM(trial_id) = ''
        OR enumeration_hash IS NULL OR LENGTH(enumeration_hash) != 64
        OR enumeration_hash GLOB '*[^0-9a-f]*'
        OR manifest_enumeration_hash IS NULL
        OR enumeration_hash != manifest_enumeration_hash
        OR alice_forbidden NOT BETWEEN 0 AND 2
        OR bob_forbidden NOT BETWEEN 0 AND 2
        OR success IS NULL OR success < 0.0 OR success > 1.0
        OR expected_success IS NULL OR expected_success < 0.0 OR expected_success > 1.0
        OR regret IS NULL OR regret < 0.0
        OR regret_tolerance IS NULL OR regret_tolerance < 0.0
      THEN 1 ELSE 0
    END AS invalid_row
  FROM coordination_trials
), summary AS (
  SELECT
    strategy,
    strategy_version,
    enumeration_hash,
    manifest_enumeration_hash,
    COUNT(*) AS rows_seen,
    COUNT(DISTINCT trial_id) AS trials,
    COUNT(DISTINCT CAST(alice_forbidden AS TEXT) || ':' || CAST(bob_forbidden AS TEXT))
      AS represented_states,
    SUM(invalid_row) AS invalid_rows,
    SUM(CASE WHEN trial_copies > 1 OR state_copies > 1 THEN 1 ELSE 0 END)
      AS duplicate_rows,
    COUNT(DISTINCT regret_tolerance) AS tolerance_versions,
    AVG(success) AS success_rate,
    AVG(regret) AS mean_regret,
    MAX(regret) AS worst_regret,
    MAX(ABS(success - expected_success)) AS maximum_success_residual,
    MIN(regret_tolerance) AS regret_tolerance,
    GROUP_CONCAT(
      CAST(alice_forbidden AS TEXT) || ':' || CAST(bob_forbidden AS TEXT) || '=' ||
      printf('%.12g', success), '|'
    ) AS statewise_success
  FROM assessed
  GROUP BY strategy, strategy_version, enumeration_hash, manifest_enumeration_hash
)
SELECT
  *,
  CASE
    WHEN rows_seen = 9
      AND trials = 9
      AND represented_states = 9
      AND invalid_rows = 0
      AND duplicate_rows = 0
      AND tolerance_versions = 1
      AND maximum_success_residual <= regret_tolerance
      AND worst_regret <= regret_tolerance
    THEN 'PASS'
    ELSE 'REJECT'
  END AS status
FROM summary
ORDER BY mean_regret, strategy, enumeration_hash;
