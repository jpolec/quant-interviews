from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

Drift = Callable[[NDArray[np.float64]], NDArray[np.float64]]
Diffusion = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class SchemeConvergenceRow:
    steps: int
    step_size: float
    euler_strong_rmse: float
    milstein_strong_rmse: float
    euler_weak_bias: float
    milstein_weak_bias: float
    weak_standard_error: float
    analytic_euler_weak_bias: float
    weak_reconciliation_z: float


@dataclass(frozen=True)
class SchemeConvergenceStudy:
    rows: tuple[SchemeConvergenceRow, ...]
    euler_strong_slope: float
    milstein_strong_slope: float
    euler_weak_slope: float | None
    weak_order_status: str
    status: str


def _validate_scheme_inputs(
    x0: float,
    d_w: NDArray[np.float64],
    dt: float,
) -> None:
    if (
        not np.isfinite(x0)
        or d_w.ndim != 2
        or d_w.shape[0] < 1
        or d_w.shape[1] < 1
        or not np.all(np.isfinite(d_w))
        or dt <= 0.0
    ):
        raise ValueError("finite state, increment matrix and positive dt required")


def euler_maruyama(
    x0: float,
    d_w: NDArray[np.float64],
    dt: float,
    drift: Drift,
    diffusion: Diffusion,
) -> NDArray[np.float64]:
    _validate_scheme_inputs(x0, d_w, dt)
    paths, steps = d_w.shape
    x = np.empty((paths, steps + 1), dtype=float)
    x[:, 0] = x0
    for i in range(steps):
        x[:, i + 1] = x[:, i] + drift(x[:, i]) * dt + diffusion(x[:, i]) * d_w[:, i]
    return x


def milstein(
    x0: float,
    d_w: NDArray[np.float64],
    dt: float,
    drift: Drift,
    diffusion: Diffusion,
    diffusion_derivative: Diffusion,
) -> NDArray[np.float64]:
    _validate_scheme_inputs(x0, d_w, dt)
    paths, steps = d_w.shape
    x = np.empty((paths, steps + 1), dtype=float)
    x[:, 0] = x0
    for i in range(steps):
        state = x[:, i]
        b = diffusion(state)
        x[:, i + 1] = (
            state
            + drift(state) * dt
            + b * d_w[:, i]
            + 0.5 * b * diffusion_derivative(state) * (d_w[:, i] ** 2 - dt)
        )
    return x


def gbm_scheme_convergence(
    *,
    step_counts: tuple[int, ...] = (8, 16, 32, 64, 128),
    paths: int = 40_000,
    initial_state: float = 1.0,
    drift_rate: float = 0.03,
    volatility: float = 0.2,
    horizon: float = 1.0,
    seed: int = 37,
) -> SchemeConvergenceStudy:
    """Compare Euler and Milstein on coupled nested GBM increments."""

    values = np.asarray(
        [initial_state, drift_rate, volatility, horizon],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(values))
        or initial_state <= 0.0
        or volatility < 0.0
        or horizon <= 0.0
        or not isinstance(paths, int)
        or isinstance(paths, bool)
        or paths < 2
        or len(step_counts) < 3
        or any(not isinstance(count, int) or isinstance(count, bool) for count in step_counts)
        or tuple(sorted(set(step_counts))) != step_counts
    ):
        raise ValueError("valid GBM inputs and increasing step counts required")
    finest = step_counts[-1]
    if any(count < 2 or finest % count != 0 for count in step_counts):
        raise ValueError("step counts must be nested divisors of the finest grid")
    rng = np.random.default_rng(seed)
    fine_d_w = np.sqrt(horizon / finest) * rng.standard_normal((paths, finest))
    terminal_brownian = np.sum(fine_d_w, axis=1)
    exact = initial_state * np.exp(
        (drift_rate - 0.5 * volatility**2) * horizon
        + volatility * terminal_brownian
    )
    exact_mean = initial_state * np.exp(drift_rate * horizon)
    rows: list[SchemeConvergenceRow] = []
    drift = lambda state: drift_rate * state
    diffusion = lambda state: volatility * state
    derivative = lambda state: np.full_like(state, volatility)
    for steps in step_counts:
        block = finest // steps
        d_w = fine_d_w.reshape(paths, steps, block).sum(axis=2)
        dt = horizon / steps
        euler = euler_maruyama(
            initial_state,
            d_w,
            dt,
            drift,
            diffusion,
        )[:, -1]
        milstein_terminal = milstein(
            initial_state,
            d_w,
            dt,
            drift,
            diffusion,
            derivative,
        )[:, -1]
        euler_difference = euler - exact
        weak_se = float(np.std(euler_difference, ddof=1) / np.sqrt(paths))
        empirical_euler_bias = float(np.mean(euler_difference))
        analytic_euler_bias = float(
            initial_state * (1.0 + drift_rate * dt) ** steps - exact_mean
        )
        reconciliation_difference = abs(empirical_euler_bias - analytic_euler_bias)
        deterministic_roundoff = 1e-12 * max(abs(exact_mean), 1.0)
        reconciliation_z = (
            0.0
            if weak_se <= np.finfo(float).eps
            and reconciliation_difference <= deterministic_roundoff
            else reconciliation_difference / max(weak_se, np.finfo(float).tiny)
        )
        rows.append(
            SchemeConvergenceRow(
                steps=steps,
                step_size=dt,
                euler_strong_rmse=float(
                    np.sqrt(np.mean((euler - exact) ** 2))
                ),
                milstein_strong_rmse=float(
                    np.sqrt(np.mean((milstein_terminal - exact) ** 2))
                ),
                euler_weak_bias=empirical_euler_bias,
                milstein_weak_bias=float(np.mean(milstein_terminal - exact)),
                weak_standard_error=weak_se,
                analytic_euler_weak_bias=analytic_euler_bias,
                weak_reconciliation_z=reconciliation_z,
            )
        )

    log_h = np.log([row.step_size for row in rows])
    euler_strong_slope = float(
        np.polyfit(log_h, np.log([row.euler_strong_rmse for row in rows]), 1)[0]
    )
    milstein_strong_slope = float(
        np.polyfit(
            log_h,
            np.log([row.milstein_strong_rmse for row in rows]),
            1,
        )[0]
    )
    analytic_bias = np.abs([row.analytic_euler_weak_bias for row in rows])
    empirical_bias = np.abs([row.euler_weak_bias for row in rows])
    if np.all(analytic_bias <= 64.0 * np.finfo(float).eps * max(abs(exact_mean), 1.0)):
        euler_weak_slope = None
        weak_order_status = "EXACT_FOR_TEST_FUNCTION"
    else:
        significant = np.array(
            [
                bias > 2.0 * row.weak_standard_error
                for bias, row in zip(empirical_bias, rows, strict=True)
            ],
            dtype=bool,
        )
        if int(np.sum(significant)) >= 3:
            euler_weak_slope = float(
                np.polyfit(log_h[significant], np.log(empirical_bias[significant]), 1)[0]
            )
        else:
            # The analytic bias is reported explicitly; insufficient Monte Carlo signal
            # is not mislabeled as an empirical order estimate.
            euler_weak_slope = None
        weak_order_status = (
            "PASS"
            if all(row.weak_reconciliation_z <= 5.0 for row in rows)
            and (
                euler_weak_slope is None
                or 0.75 <= euler_weak_slope <= 1.25
            )
            else "REJECT"
        )
    deterministic_limit = volatility == 0.0
    status = (
        "PASS"
        if (
            (0.4 <= euler_strong_slope <= 0.65 and 0.85 <= milstein_strong_slope <= 1.15)
            or (
                deterministic_limit
                and 0.85 <= euler_strong_slope <= 1.15
                and 0.85 <= milstein_strong_slope <= 1.15
            )
        )
        and weak_order_status in {"PASS", "EXACT_FOR_TEST_FUNCTION"}
        else "REJECT"
    )
    return SchemeConvergenceStudy(
        rows=tuple(rows),
        euler_strong_slope=euler_strong_slope,
        milstein_strong_slope=milstein_strong_slope,
        euler_weak_slope=euler_weak_slope,
        weak_order_status=weak_order_status,
        status=status,
    )
