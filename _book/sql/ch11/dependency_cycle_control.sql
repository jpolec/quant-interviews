-- Dialect: SQLite 3.45 with JSON1; explicit resource limit, never silent truncation.
WITH RECURSIVE
parameters AS (
    SELECT
        CASE
            WHEN typeof(:max_depth) = 'integer' AND :max_depth >= 1
                THEN :max_depth
        END AS max_depth
), edges AS (
    SELECT DISTINCT parent, child
    FROM dependencies
    WHERE parent IS NOT NULL AND child IS NOT NULL
), graph_limit AS (
    SELECT COUNT(*) AS node_count
    FROM (
        SELECT parent AS node FROM edges
        UNION
        SELECT child AS node FROM edges
    )
), walk(root, node, path_json, is_cycle, depth) AS (
    SELECT parent, child, json_array(parent, child), parent = child, 1
    FROM edges
    CROSS JOIN parameters
    WHERE parameters.max_depth IS NOT NULL
    UNION ALL
    SELECT
        walk.root,
        edge.child,
        json_insert(walk.path_json, '$[#]', edge.child),
        EXISTS (
            SELECT 1
            FROM json_each(walk.path_json)
            WHERE value = edge.child
        ),
        walk.depth + 1
    FROM walk
    JOIN edges AS edge ON edge.parent = walk.node
    CROSS JOIN parameters
    CROSS JOIN graph_limit
    WHERE walk.is_cycle = 0
      AND walk.depth < parameters.max_depth
      AND walk.depth < graph_limit.node_count
), findings AS (
    SELECT
        root,
        node,
        path_json,
        depth,
        'REJECT_CYCLE' AS status
    FROM walk
    WHERE is_cycle = 1
    UNION ALL
    SELECT DISTINCT
        walk.root,
        walk.node,
        walk.path_json,
        walk.depth,
        'REJECT_DEPTH_LIMIT' AS status
    FROM walk
    JOIN edges AS edge ON edge.parent = walk.node
    CROSS JOIN parameters
    CROSS JOIN graph_limit
    WHERE walk.is_cycle = 0
      AND walk.depth = parameters.max_depth
      AND parameters.max_depth < graph_limit.node_count
    UNION ALL
    SELECT
        NULL,
        NULL,
        json_array(),
        0,
        'REJECT_INVALID_MAX_DEPTH'
    FROM parameters
    WHERE max_depth IS NULL
)
SELECT DISTINCT root, node, path_json, depth, status
FROM findings
ORDER BY status, root, path_json;
