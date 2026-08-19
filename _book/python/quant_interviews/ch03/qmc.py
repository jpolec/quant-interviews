"""Monte Carlo and independently scrambled Sobol evidence for Chapter 3."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import qmc


def _evaluate(
    function: Callable[[np.ndarray], ArrayLike],
    points: np.ndarray,
) -> np.ndarray:
    values = np.asarray(function(points), dtype=float)
    if values.shape != (points.shape[0],) or not np.all(np.isfinite(values)):
        raise ValueError("integrand must return one finite value per point")
    return values


def mc_integral(
    unit_cube_function: Callable[[np.ndarray], ArrayLike],
    dimension: int,
    samples: int,
    seed: int = 0,
) -> float:
    """Estimate a unit-cube integral with pseudorandom sampling."""

    if dimension <= 0 or samples <= 0:
        raise ValueError("positive dimension and sample count required")
    points = np.random.default_rng(seed).random((samples, dimension))
    return float(np.mean(_evaluate(unit_cube_function, points)))


def randomized_sobol_integral(
    unit_cube_function: Callable[[np.ndarray], ArrayLike],
    dimension: int,
    power: int,
    seed: int = 0,
) -> float:
    """Estimate a unit-cube integral with one independent Sobol scramble."""

    if dimension <= 0 or power <= 0:
        raise ValueError("positive dimension and Sobol power required")
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    points = sampler.random_base2(power)
    return float(np.mean(_evaluate(unit_cube_function, points)))


def randomized_sobol_replicates(
    unit_cube_function: Callable[[np.ndarray], ArrayLike],
    dimension: int,
    power: int,
    *,
    scrambles: int,
    seed: int,
) -> np.ndarray:
    """Run exactly one independently seeded Sobol scramble per replicate."""

    if scrambles < 4:
        raise ValueError("at least four independent scrambles required")
    return np.array(
        [
            randomized_sobol_integral(
                unit_cube_function,
                dimension,
                power,
                seed + replicate,
            )
            for replicate in range(scrambles)
        ]
    )


def replicate_summary(
    estimates: ArrayLike,
    exact: float,
) -> dict[str, float]:
    """Summarize an estimator from between-replicate uncertainty."""

    values = np.asarray(estimates, dtype=float)
    if (
        values.ndim != 1
        or values.size < 4
        or not np.all(np.isfinite(values))
        or not math.isfinite(exact)
    ):
        raise ValueError("four finite replicates and finite reference required")
    return {
        "mean": float(values.mean()),
        "replicate_se": float(values.std(ddof=1) / math.sqrt(values.size)),
        "absolute_error": abs(float(values.mean()) - exact),
    }


def randomized_qmc_study(
    unit_cube_function: Callable[[np.ndarray], ArrayLike],
    dimension: int,
    powers: Sequence[int],
    *,
    scrambles: int,
    seed: int,
    exact: float,
) -> list[dict[str, float | int | np.ndarray | None]]:
    """Compare MC and RQMC uncertainty across independent replications."""

    if (
        scrambles < 4
        or not powers
        or any(not isinstance(power, int) or power <= 0 for power in powers)
        or len(set(powers)) != len(powers)
        or any(
            current <= previous
            for previous, current in zip(powers, powers[1:], strict=False)
        )
    ):
        raise ValueError("at least four scrambles and one power required")
    rows: list[dict[str, float | int | np.ndarray | None]] = []
    previous_mc_error: float | None = None
    previous_rqmc_error: float | None = None
    previous_power: int | None = None
    for power in powers:
        samples = 2**power
        started = perf_counter()
        mc_replicates = np.array(
            [
                mc_integral(
                    unit_cube_function,
                    dimension,
                    samples,
                    seed + replicate,
                )
                for replicate in range(scrambles)
            ]
        )
        mc_seconds = perf_counter() - started
        started = perf_counter()
        rqmc_replicates = randomized_sobol_replicates(
            unit_cube_function,
            dimension,
            power,
            scrambles=scrambles,
            seed=seed + 10_000,
        )
        rqmc_seconds = perf_counter() - started
        mc_summary = replicate_summary(mc_replicates, exact)
        rqmc_summary = replicate_summary(rqmc_replicates, exact)
        mc_error = mc_summary["absolute_error"]
        rqmc_error = rqmc_summary["absolute_error"]
        rows.append(
            {
                "samples": samples,
                "sample_power": power,
                "scrambles": scrambles,
                "seed": seed,
                "mc_mean": mc_summary["mean"],
                "mc_replicate_se": mc_summary["replicate_se"],
                "rqmc_mean": rqmc_summary["mean"],
                "rqmc_replicate_se": rqmc_summary["replicate_se"],
                "mc_absolute_error": mc_error,
                "rqmc_absolute_error": rqmc_error,
                "mc_seconds": mc_seconds,
                "rqmc_seconds": rqmc_seconds,
                "mc_observed_slope": (
                    None
                    if (
                        previous_mc_error in (None, 0.0)
                        or mc_error == 0.0
                        or previous_power is None
                    )
                    else math.log(previous_mc_error / mc_error, 2.0)
                    / (power - previous_power)
                ),
                "rqmc_observed_slope": (
                    None
                    if (
                        previous_rqmc_error in (None, 0.0)
                        or rqmc_error == 0.0
                        or previous_power is None
                    )
                    else math.log(previous_rqmc_error / rqmc_error, 2.0)
                    / (power - previous_power)
                ),
                "mc_replicates": mc_replicates,
                "rqmc_replicates": rqmc_replicates,
            }
        )
        previous_mc_error = mc_error
        previous_rqmc_error = rqmc_error
        previous_power = power
    return rows
