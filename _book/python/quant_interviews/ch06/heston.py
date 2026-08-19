from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np


@dataclass(frozen=True)
class HestonRefinementRow:
    steps: int
    empirical_variance_mean: float
    theoretical_variance_mean: float
    variance_mean_standard_error: float
    variance_mean_z_score: float
    forward_martingale_error: float
    negative_pre_truncation_fraction: float
    zero_terminal_fraction: float
    variance_mean_refinement_change: float | None


@dataclass(frozen=True)
class HestonDiagnostics:
    feller_condition: bool
    drift_convention: str
    rows: tuple[HestonRefinementRow, ...]
    variance_z_limit: float
    martingale_limit: float
    negative_fraction_limit: float
    zero_terminal_fraction_limit: float
    refinement_change_limit: float
    require_feller: bool
    status: str


def heston_variance_mean(v0: float, kappa: float, theta: float, t: float) -> float:
    if (
        not all(isfinite(value) for value in (v0, kappa, theta, t))
        or v0 < 0.0
        or kappa <= 0.0
        or theta < 0.0
        or t < 0.0
    ):
        raise ValueError("invalid Heston variance-mean inputs")
    return float(theta + (v0 - theta) * np.exp(-kappa * t))


def heston_feller(kappa: float, theta: float, xi: float) -> bool:
    if (
        not all(isfinite(value) for value in (kappa, theta, xi))
        or kappa <= 0.0
        or theta < 0.0
        or xi < 0.0
    ):
        raise ValueError("invalid Heston Feller inputs")
    return 2.0 * kappa * theta >= xi * xi


def _simulate_heston_terminal(
    *,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    horizon: float,
    steps: int,
    paths: int,
    seed: int,
    spot_shocks: np.ndarray | None = None,
    independent_shocks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if (
        not all(
            isfinite(value)
            for value in (v0, kappa, theta, xi, rho, horizon)
        )
        or v0 < 0.0
        or kappa <= 0.0
        or theta < 0.0
        or xi < 0.0
        or abs(rho) > 1.0
        or horizon <= 0.0
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps <= 0
        or isinstance(paths, bool)
        or not isinstance(paths, int)
        or paths < 2
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("invalid Heston simulation inputs")
    if (spot_shocks is None) != (independent_shocks is None):
        raise ValueError("both common-random-number shock arrays are required")
    if spot_shocks is None:
        rng = np.random.default_rng(seed)
        spot_shocks = rng.standard_normal((steps, paths))
        independent_shocks = rng.standard_normal((steps, paths))
    if (
        spot_shocks.shape != (steps, paths)
        or independent_shocks is None
        or independent_shocks.shape != (steps, paths)
        or not np.isfinite(spot_shocks).all()
        or not np.isfinite(independent_shocks).all()
    ):
        raise ValueError("Heston shock arrays have an invalid shape or value")
    dt = horizon / steps
    variance = np.full(paths, v0, dtype=float)
    centred_log_forward = np.zeros(paths, dtype=float)
    negative_count = 0
    total_updates = steps * paths
    for step_index in range(steps):
        spot_shock = spot_shocks[step_index]
        variance_shock = (
            rho * spot_shock
            + sqrt(1.0 - rho * rho) * independent_shocks[step_index]
        )
        used_variance = np.maximum(variance, 0.0)
        centred_log_forward += (
            -0.5 * used_variance * dt
            + np.sqrt(used_variance * dt) * spot_shock
        )
        candidate = (
            variance
            + kappa * (theta - used_variance) * dt
            + xi * np.sqrt(used_variance * dt) * variance_shock
        )
        negative_count += int(np.sum(candidate < 0.0))
        variance = np.maximum(candidate, 0.0)
    return (
        centred_log_forward,
        variance,
        negative_count / total_updates,
        float(np.mean(variance == 0.0)),
    )


def heston_diagnostics(
    *,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    horizon: float,
    step_counts: tuple[int, ...] = (32, 64, 128),
    paths: int = 40_000,
    seed: int = 0,
    variance_z_limit: float = 4.0,
    martingale_limit: float = 0.01,
    negative_fraction_limit: float = 0.05,
    zero_terminal_fraction_limit: float = 0.05,
    refinement_change_limit: float = 0.005,
    require_feller: bool = False,
) -> HestonDiagnostics:
    values = (
        v0,
        kappa,
        theta,
        xi,
        rho,
        horizon,
        variance_z_limit,
        martingale_limit,
        negative_fraction_limit,
        zero_terminal_fraction_limit,
        refinement_change_limit,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("Heston inputs must be finite")
    if (
        v0 < 0.0
        or kappa <= 0.0
        or theta < 0.0
        or xi < 0.0
        or abs(rho) > 1.0
        or horizon <= 0.0
    ):
        raise ValueError("invalid Heston parameters")
    if (
        isinstance(paths, bool)
        or not isinstance(paths, int)
        or paths < 2
        or not step_counts
        or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in step_counts
        )
    ):
        raise ValueError("simulation dimensions must be positive")
    if tuple(sorted(set(step_counts))) != step_counts:
        raise ValueError("step_counts must be unique and increasing")
    finest_steps = step_counts[-1]
    if any(finest_steps % steps != 0 for steps in step_counts):
        raise ValueError("each step count must divide the finest step count")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or any(
            limit < 0.0
            for limit in (
                variance_z_limit,
                martingale_limit,
                negative_fraction_limit,
                zero_terminal_fraction_limit,
                refinement_change_limit,
            )
        )
    ):
        raise ValueError("seed and diagnostic limits are invalid")

    rng = np.random.default_rng(seed)
    finest_spot_shocks = rng.standard_normal((finest_steps, paths))
    finest_independent_shocks = rng.standard_normal((finest_steps, paths))

    theoretical_mean = heston_variance_mean(v0, kappa, theta, horizon)
    rows: list[HestonRefinementRow] = []
    previous_empirical_mean: float | None = None
    for steps in step_counts:
        aggregation_ratio = finest_steps // steps
        spot_shocks = finest_spot_shocks.reshape(
            steps, aggregation_ratio, paths
        ).sum(axis=1) / sqrt(aggregation_ratio)
        independent_shocks = finest_independent_shocks.reshape(
            steps, aggregation_ratio, paths
        ).sum(axis=1) / sqrt(aggregation_ratio)
        log_forward, terminal_variance, negative_fraction, zero_fraction = (
            _simulate_heston_terminal(
                v0=v0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
                horizon=horizon,
                steps=steps,
                paths=paths,
                seed=seed,
                spot_shocks=spot_shocks,
                independent_shocks=independent_shocks,
            )
        )
        empirical_mean = float(np.mean(terminal_variance))
        standard_error = float(np.std(terminal_variance, ddof=1) / sqrt(paths))
        z_score = abs(empirical_mean - theoretical_mean) / max(
            standard_error, 1e-16
        )
        rows.append(
            HestonRefinementRow(
                steps=steps,
                empirical_variance_mean=empirical_mean,
                theoretical_variance_mean=theoretical_mean,
                variance_mean_standard_error=standard_error,
                variance_mean_z_score=z_score,
                forward_martingale_error=abs(float(np.mean(np.exp(log_forward))) - 1.0),
                negative_pre_truncation_fraction=negative_fraction,
                zero_terminal_fraction=zero_fraction,
                variance_mean_refinement_change=(
                    None
                    if previous_empirical_mean is None
                    else abs(empirical_mean - previous_empirical_mean)
                ),
            )
        )
        previous_empirical_mean = empirical_mean
    numerical_passed = (
        rows[-1].variance_mean_z_score <= variance_z_limit
        and rows[-1].forward_martingale_error <= martingale_limit
        and max(row.negative_pre_truncation_fraction for row in rows)
        <= negative_fraction_limit
        and max(row.zero_terminal_fraction for row in rows)
        <= zero_terminal_fraction_limit
        and (
            rows[-1].variance_mean_refinement_change is None
            or rows[-1].variance_mean_refinement_change
            <= refinement_change_limit
        )
    )
    feller_condition = heston_feller(kappa, theta, xi)
    if not numerical_passed or (require_feller and not feller_condition):
        status = "REJECT"
    elif not feller_condition:
        status = "CONDITIONAL_PASS"
    else:
        status = "PASS"
    return HestonDiagnostics(
        feller_condition=feller_condition,
        drift_convention="centred log-forward under the pricing measure",
        rows=tuple(rows),
        variance_z_limit=variance_z_limit,
        martingale_limit=martingale_limit,
        negative_fraction_limit=negative_fraction_limit,
        zero_terminal_fraction_limit=zero_terminal_fraction_limit,
        refinement_change_limit=refinement_change_limit,
        require_feller=require_feller,
        status=status,
    )


def heston_euler(
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    horizon: float,
    steps: int,
    paths: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    log_forward, terminal_variance, _, _ = _simulate_heston_terminal(
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        horizon=horizon,
        steps=steps,
        paths=paths,
        seed=seed,
    )
    return log_forward, terminal_variance
