from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class PowerEstimate:
    rho: float
    sample_size: int
    break_size: float
    rejection_rate: float
    simulations: int
    standard_error: float
    confidence_interval_95: tuple[float, float]
    significance: float
    seed: int


def simulate_ar1(
    rho: float,
    sample_size: int,
    *,
    seed: int,
    break_size: float = 0.0,
) -> np.ndarray:
    if not 0.0 <= rho <= 1.05 or sample_size < 50:
        raise ValueError("Invalid rho or sample size.")
    rng = np.random.default_rng(seed)
    burn = 100
    x = np.zeros(sample_size + burn)
    for t in range(1, x.size):
        x[t] = rho * x[t - 1] + rng.normal()
    out = x[burn:]
    if break_size != 0.0:
        out[out.size // 2 :] += break_size
    return out


def adf_rejection_rate(
    rho: float,
    sample_size: int,
    *,
    simulations: int = 200,
    significance: float = 0.05,
    break_size: float = 0.0,
    seed: int = 0,
) -> PowerEstimate:
    if simulations <= 0 or not 0.0 < significance < 1.0:
        raise ValueError("Invalid simulation settings.")
    rejected = 0
    for i in range(simulations):
        x = simulate_ar1(rho, sample_size, seed=seed + i, break_size=break_size)
        pvalue = float(adfuller(x, maxlag=1, regression="c", autolag=None)[1])
        rejected += pvalue < significance
    rate = rejected / simulations
    standard_error = sqrt(rate * (1.0 - rate) / simulations)
    # Wilson interval remains finite and informative at rates zero and one.
    z = 1.959963984540054
    denominator = 1.0 + z * z / simulations
    centre = (rate + z * z / (2.0 * simulations)) / denominator
    radius = (
        z
        * sqrt(
            rate * (1.0 - rate) / simulations
            + z * z / (4.0 * simulations * simulations)
        )
        / denominator
    )
    return PowerEstimate(
        rho=rho,
        sample_size=sample_size,
        break_size=break_size,
        rejection_rate=rate,
        simulations=simulations,
        standard_error=standard_error,
        confidence_interval_95=(
            0.0 if rejected == 0 else max(0.0, centre - radius),
            1.0 if rejected == simulations else min(1.0, centre + radius),
        ),
        significance=significance,
        seed=seed,
    )
