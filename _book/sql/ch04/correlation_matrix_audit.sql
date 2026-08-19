-- Dialect: SQLite 3.45
-- Scope: complete structural/PSD gate for 2x2 only. Higher dimensions must
-- pass the hash-identical Python spectral gate before use.
CREATE TABLE correlation_entry (
    matrix_id TEXT NOT NULL,
    row_id INTEGER NOT NULL,
    col_id INTEGER NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (matrix_id, row_id, col_id)
);

INSERT INTO correlation_entry VALUES
('SPD',1,1,1.0),('SPD',1,2,0.6),('SPD',2,1,0.6),('SPD',2,2,1.0),
('SINGULAR',1,1,1.0),('SINGULAR',1,2,1.0),('SINGULAR',2,1,1.0),('SINGULAR',2,2,1.0),
('INVALID',1,1,1.0),('INVALID',1,2,1.1),('INVALID',2,1,1.1),('INVALID',2,2,1.0),
('BAD3',1,1,1.0),('BAD3',1,2,-0.6),('BAD3',1,3,-0.6),
('BAD3',2,1,-0.6),('BAD3',2,2,1.0),('BAD3',2,3,-0.6),
('BAD3',3,1,-0.6),('BAD3',3,2,-0.6),('BAD3',3,3,1.0),
('BAD4',1,1,1.0),('BAD4',1,2,-0.34),('BAD4',1,3,-0.34),('BAD4',1,4,-0.34),
('BAD4',2,1,-0.34),('BAD4',2,2,1.0),('BAD4',2,3,-0.34),('BAD4',2,4,-0.34),
('BAD4',3,1,-0.34),('BAD4',3,2,-0.34),('BAD4',3,3,1.0),('BAD4',3,4,-0.34),
('BAD4',4,1,-0.34),('BAD4',4,2,-0.34),('BAD4',4,3,-0.34),('BAD4',4,4,1.0);

CREATE TABLE correlation_matrix_audit AS
WITH dimensions AS (
    SELECT
        matrix_id,
        COUNT(DISTINCT row_id) AS row_count,
        COUNT(DISTINCT col_id) AS column_count,
        COUNT(*) AS entry_count
    FROM correlation_entry
    GROUP BY matrix_id
),
symmetry AS (
    SELECT
        a.matrix_id,
        MAX(CASE WHEN b.value IS NULL THEN 1e9 ELSE ABS(a.value - b.value) END)
            AS maximum_symmetry_error
    FROM correlation_entry AS a
    LEFT JOIN correlation_entry AS b
      ON b.matrix_id = a.matrix_id
     AND b.row_id = a.col_id
     AND b.col_id = a.row_id
    GROUP BY a.matrix_id
),
entry_checks AS (
    SELECT
        matrix_id,
        MAX(CASE WHEN row_id = col_id THEN ABS(value - 1.0) ELSE 0.0 END)
            AS maximum_diagonal_error,
        SUM(CASE WHEN row_id <> col_id AND ABS(value) > 1.0 THEN 1 ELSE 0 END)
            AS out_of_bounds_entries
    FROM correlation_entry
    GROUP BY matrix_id
),
two_by_two AS (
    SELECT
        matrix_id,
        MAX(CASE WHEN row_id = 1 AND col_id = 2 THEN value END) AS rho12
    FROM correlation_entry
    GROUP BY matrix_id
)
SELECT
    d.matrix_id,
    d.row_count,
    d.column_count,
    d.entry_count,
    s.maximum_symmetry_error,
    e.maximum_diagonal_error,
    e.out_of_bounds_entries,
    CASE WHEN d.row_count = 2 THEN 1.0 - t.rho12 * t.rho12 END
        AS determinant_2x2,
    CASE
        WHEN d.row_count <> d.column_count
          OR d.entry_count <> d.row_count * d.column_count
          OR s.maximum_symmetry_error > 1e-12
          OR e.maximum_diagonal_error > 1e-12
          OR e.out_of_bounds_entries > 0
        THEN 'REJECT_STRUCTURAL'
        WHEN d.row_count <> 2
        THEN 'REJECT_UNSUPPORTED_DIMENSION'
        WHEN 1.0 - t.rho12 * t.rho12 < -1e-12
        THEN 'REJECT_NOT_PSD'
        WHEN ABS(1.0 - t.rho12 * t.rho12) <= 1e-12
        THEN 'PASS_SINGULAR_PSD_2X2'
        ELSE 'PASS_SPD_2X2'
    END AS structural_status,
    CASE WHEN d.row_count = 2 THEN 0 ELSE 1 END AS python_spectral_gate_required
FROM dimensions AS d
JOIN symmetry AS s USING (matrix_id)
JOIN entry_checks AS e USING (matrix_id)
JOIN two_by_two AS t USING (matrix_id);
