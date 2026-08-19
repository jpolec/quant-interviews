-- Dialect: SQLite
-- Fail-closed posterior reconciliation for complete program-level trials.
WITH assessed AS (
  SELECT
    *,
    COUNT(*) OVER (PARTITION BY trial_id) AS trial_copies,
    CASE
      WHEN trial_id IS NULL OR TRIM(trial_id) = ''
        OR protocol IS NULL OR TRIM(protocol) = ''
        OR protocol_version IS NULL OR TRIM(protocol_version) = ''
        OR prior_definition IS NULL OR TRIM(prior_definition) = ''
        OR prior_program IS NULL OR prior_program <= 0.0 OR prior_program >= 1.0
        OR candidates IS NULL OR candidates != CAST(candidates AS INTEGER) OR candidates < 1
        OR alpha IS NULL OR alpha <= 0.0 OR alpha >= 1.0
        OR power IS NULL OR power <= 0.0 OR power >= 1.0
        OR conditional_independence != 1
        OR complete_trial != 1
        OR reported NOT IN (0, 1)
        OR signal_world NOT IN (0, 1)
        OR minimum_reports IS NULL OR minimum_reports != CAST(minimum_reports AS INTEGER)
        OR minimum_reports < 30
        OR reconciliation_tolerance IS NULL
        OR reconciliation_tolerance < 0.0 OR reconciliation_tolerance >= 1.0
      THEN 1 ELSE 0
    END AS invalid_row
  FROM selection_events
), summary AS (
  SELECT
    protocol,
    protocol_version,
    prior_definition,
    prior_program,
    candidates,
    alpha,
    power,
    minimum_reports,
    reconciliation_tolerance,
    COUNT(*) AS rows_seen,
    COUNT(DISTINCT trial_id) AS trials,
    SUM(CASE WHEN reported = 1 THEN 1 ELSE 0 END) AS reports,
    SUM(CASE WHEN reported = 1 AND signal_world = 1 THEN 1 ELSE 0 END) AS true_reports,
    SUM(invalid_row) AS invalid_rows,
    SUM(CASE WHEN trial_copies > 1 THEN 1 ELSE 0 END) AS duplicate_rows,
    COUNT(DISTINCT minimum_reports) AS minimum_report_versions,
    COUNT(DISTINCT reconciliation_tolerance) AS tolerance_versions
  FROM assessed
  GROUP BY protocol, protocol_version, prior_definition, prior_program,
           candidates, alpha, power, minimum_reports, reconciliation_tolerance
), model AS (
  SELECT
    *,
    1.0 - (1.0 - power) * pow(1.0 - alpha, candidates - 1)
      AS event_given_signal,
    1.0 - pow(1.0 - alpha, candidates) AS event_given_null,
    CASE WHEN reports = 0 THEN NULL ELSE 1.0 * true_reports / reports END
      AS empirical_posterior
  FROM summary
), scored AS (
  SELECT
    *,
    prior_program * event_given_signal /
      (prior_program * event_given_signal +
       (1.0 - prior_program) * event_given_null) AS analytic_posterior
  FROM model
)
SELECT
  *,
  CASE
    WHEN empirical_posterior IS NULL THEN NULL
    ELSE empirical_posterior - analytic_posterior
  END AS posterior_residual,
  CASE
    WHEN invalid_rows = 0
      AND duplicate_rows = 0
      AND rows_seen = trials
      AND minimum_report_versions = 1
      AND tolerance_versions = 1
      AND reports >= minimum_reports
      AND empirical_posterior IS NOT NULL
      AND ABS(empirical_posterior - analytic_posterior) <= reconciliation_tolerance
    THEN 'PASS'
    ELSE 'REJECT'
  END AS status
FROM scored
ORDER BY protocol, candidates;
