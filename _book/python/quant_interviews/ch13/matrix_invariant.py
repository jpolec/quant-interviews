from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.integer]


@dataclass(frozen=True)
class ReductionCertificate:
    operations: tuple[tuple[int, int, int, int], ...]
    row_parity_before: tuple[int, ...]
    column_parity_before: tuple[int, ...]
    row_parity_after: tuple[int, ...]
    column_parity_after: tuple[int, ...]
    replay_is_zero: bool
    status: str


def row_parity(matrix: IntArray) -> NDArray[np.int64]:
    values = _validated_copy(matrix)
    return np.sum(values, axis=1, dtype=np.int64) % 2


def column_parity(matrix: IntArray) -> NDArray[np.int64]:
    values = _validated_copy(matrix)
    return np.sum(values, axis=0, dtype=np.int64) % 2


def toggle_rectangle_corners(
    matrix: IntArray,
    r1: int,
    r2: int,
    c1: int,
    c2: int,
) -> NDArray[np.uint8]:
    result = _validated_copy(matrix)
    if not (
        0 <= r1 < r2 < result.shape[0]
        and 0 <= c1 < c2 < result.shape[1]
    ):
        raise ValueError("invalid rectangle")
    for row, column in (
        (r1, c1),
        (r1, c2),
        (r2, c1),
        (r2, c2),
    ):
        result[row, column] ^= np.uint8(1)
    return result


def reducible_to_zero(matrix: IntArray) -> bool:
    return bool(
        np.all(row_parity(matrix) == 0)
        and np.all(column_parity(matrix) == 0)
    )


def reduction_operations(
    matrix: IntArray,
) -> list[tuple[int, int, int, int]]:
    work = _validated_copy(matrix)
    if not reducible_to_zero(work):
        raise ValueError("row or column parity invariant blocks reduction")
    rows, columns = work.shape
    operations: list[tuple[int, int, int, int]] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            if work[row, column]:
                operation = (row, rows - 1, column, columns - 1)
                work = toggle_rectangle_corners(work, *operation)
                operations.append(operation)
    if np.any(work):
        raise RuntimeError("constructive proof failed")
    return operations


def reduction_certificate(matrix: IntArray) -> ReductionCertificate:
    start = _validated_copy(matrix)
    operations = reduction_operations(start)
    replay = start
    for operation in operations:
        replay = toggle_rectangle_corners(replay, *operation)
    before_rows = tuple(int(value) for value in row_parity(start))
    before_columns = tuple(int(value) for value in column_parity(start))
    after_rows = tuple(int(value) for value in row_parity(replay))
    after_columns = tuple(int(value) for value in column_parity(replay))
    is_zero = bool(not np.any(replay))
    return ReductionCertificate(
        operations=tuple(operations),
        row_parity_before=before_rows,
        column_parity_before=before_columns,
        row_parity_after=after_rows,
        column_parity_after=after_columns,
        replay_is_zero=is_zero,
        status="PASS" if is_zero else "REJECT",
    )


def independent_constraint_count(rows: int, columns: int) -> int:
    if rows < 2 or columns < 2:
        raise ValueError("matrix game requires at least 2x2")
    return rows + columns - 1


def reachable_subspace_dimension(rows: int, columns: int) -> int:
    if rows < 2 or columns < 2:
        raise ValueError("matrix game requires at least 2x2")
    return (rows - 1) * (columns - 1)


def _validated_copy(matrix: IntArray) -> NDArray[np.uint8]:
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("matrix must be at least 2x2")
    if values.dtype.kind not in {"b", "i", "u"}:
        raise TypeError("matrix must have boolean or integer dtype")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("matrix must be binary")
    return values.astype(np.uint8, copy=True)
