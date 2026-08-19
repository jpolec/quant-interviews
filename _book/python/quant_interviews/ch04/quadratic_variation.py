from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QuadraticVariationRow:
    steps: int
    mesh: float
    mean: float
    standard_deviation: float
    rmse: float
    theoretical_standard_deviation: float


@dataclass(frozen=True)
class QuadraticVariationStudy:
    rows: tuple[QuadraticVariationRow, ...]
    rmse_mesh_slope: float
    status: str


def brownian_paths(
    *,
    paths: int,
    steps: int,
    horizon: float = 1.0,
    seed: int = 7,
) -> NDArray[np.float64]:
    if paths < 2 or steps < 1 or horizon <= 0.0:
        raise ValueError("paths >= 2, steps >= 1 and positive horizon required")
    rng = np.random.default_rng(seed)
    dt = horizon / steps
    increments = np.sqrt(dt) * rng.standard_normal((paths, steps))
    return np.c_[np.zeros(paths), np.cumsum(increments, axis=1)]


def realized_quadratic_variation(paths: NDArray[np.float64]) -> NDArray[np.float64]:
    if (
        paths.ndim != 2
        or paths.shape[0] < 1
        or paths.shape[1] < 2
        or not np.all(np.isfinite(paths))
    ):
        raise ValueError("finite paths with at least one increment required")
    increments = np.diff(paths, axis=1)
    return np.sum(increments * increments, axis=1)


def quadratic_variation_convergence(
    *,
    partition_sizes: tuple[int, ...] = (32, 64, 128, 256),
    paths: int = 20_000,
    horizon: float = 1.0,
    seed: int = 7,
) -> QuadraticVariationStudy:
    """Audit the narrowing distribution of Brownian quadratic variation."""

    if (
        paths < 2
        or horizon <= 0.0
        or len(partition_sizes) < 3
        or any(size < 2 for size in partition_sizes)
        or tuple(sorted(set(partition_sizes))) != partition_sizes
    ):
        raise ValueError("strictly increasing partitions and valid simulation inputs required")
    finest = partition_sizes[-1]
    if any(finest % size != 0 for size in partition_sizes):
        raise ValueError("partition sizes must be nested divisors of the finest grid")

    rng = np.random.default_rng(seed)
    fine_dw = np.sqrt(horizon / finest) * rng.standard_normal((paths, finest))
    rows: list[QuadraticVariationRow] = []
    for steps in partition_sizes:
        block = finest // steps
        increments = fine_dw.reshape(paths, steps, block).sum(axis=2)
        quadratic_variation = np.sum(increments * increments, axis=1)
        rows.append(
            QuadraticVariationRow(
                steps=steps,
                mesh=horizon / steps,
                mean=float(np.mean(quadratic_variation)),
                standard_deviation=float(np.std(quadratic_variation, ddof=1)),
                rmse=float(
                    np.sqrt(np.mean((quadratic_variation - horizon) ** 2))
                ),
                theoretical_standard_deviation=(
                    horizon * np.sqrt(2.0 / steps)
                ),
            )
        )

    slope = float(
        np.polyfit(
            np.log([row.mesh for row in rows]),
            np.log([row.rmse for row in rows]),
            1,
        )[0]
    )
    status = "PASS" if 0.4 <= slope <= 0.6 else "REJECT"
    return QuadraticVariationStudy(tuple(rows), slope, status)
