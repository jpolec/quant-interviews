from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ou_conditional_moments(
    x_t: float | NDArray[np.float64],
    *,
    kappa: float,
    theta: float,
    sigma: float,
    delta: float,
) -> tuple[float | NDArray[np.float64], float]:
    state = np.asarray(x_t, dtype=float)
    values = np.asarray([kappa, theta, sigma, delta], dtype=float)
    if (
        state.ndim > 1
        or state.size == 0
        or not np.all(np.isfinite(state))
        or not np.all(np.isfinite(values))
        or kappa <= 0.0
        or sigma < 0.0
        or delta < 0.0
    ):
        raise ValueError("finite scalar/vector state and valid OU parameters required")
    decay = np.exp(-kappa * delta)
    mean = theta + (state - theta) * decay
    variance = sigma**2 * (-np.expm1(-2.0 * kappa * delta)) / (2.0 * kappa)
    if state.ndim == 0:
        return float(mean), float(variance)
    return np.asarray(mean, dtype=float), float(variance)


def simulate_ou_exact(
    x0: float,
    *,
    kappa: float,
    theta: float,
    sigma: float,
    dt: float,
    steps: int,
    paths: int,
    seed: int = 17,
) -> NDArray[np.float64]:
    controls = np.asarray([x0, kappa, theta, sigma, dt], dtype=float)
    if (
        not np.all(np.isfinite(controls))
        or dt <= 0.0
        or not isinstance(steps, int)
        or isinstance(steps, bool)
        or steps < 1
        or not isinstance(paths, int)
        or isinstance(paths, bool)
        or paths < 2
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise ValueError("finite state/parameters and integer simulation dimensions required")
    # Validate the remaining OU controls even before allocation.
    ou_conditional_moments(x0, kappa=kappa, theta=theta, sigma=sigma, delta=dt)
    rng = np.random.default_rng(seed)
    result = np.empty((paths, steps + 1), dtype=float)
    result[:, 0] = x0
    for index in range(steps):
        mean, variance = ou_conditional_moments(
            result[:, index], kappa=kappa, theta=theta, sigma=sigma, delta=dt
        )
        result[:, index + 1] = mean + np.sqrt(variance) * rng.standard_normal(paths)
    return result
