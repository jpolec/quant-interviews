from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np
from numpy.typing import NDArray

from .evidence import (
    ConventionSchema,
    EvidenceCertificate,
    InvariantResult,
    TolerancePolicy,
    Violation,
    decision_from_invariants,
    stable_manifest_hash,
)


def exact_ou_transition(
    state: NDArray[np.float64],
    *,
    mean_reversion: float,
    volatility: float,
    dt: float,
    normal_shock: NDArray[np.float64],
) -> NDArray[np.float64]:
    """One exact OU transition with cancellation-safe conditional variance."""

    state_array = np.asarray(state, dtype=float)
    shocks = np.asarray(normal_shock, dtype=float)
    if (
        not np.isfinite(mean_reversion)
        or not np.isfinite(volatility)
        or not np.isfinite(dt)
        or mean_reversion <= 0.0
        or volatility < 0.0
        or dt <= 0.0
        or state_array.shape != shocks.shape
        or not np.all(np.isfinite(state_array))
        or not np.all(np.isfinite(shocks))
    ):
        raise ValueError("invalid exact OU transition inputs")
    decay = exp(-mean_reversion * dt)
    variance = (
        volatility**2
        * (-np.expm1(-2.0 * mean_reversion * dt))
        / (2.0 * mean_reversion)
    )
    return decay * state_array + sqrt(variance) * shocks


def exact_ou_terminal_moments(
    x0: float,
    *,
    mean_reversion: float,
    volatility: float,
    horizon: float,
) -> tuple[float, float]:
    if (
        not np.isfinite(x0)
        or not np.isfinite(mean_reversion)
        or not np.isfinite(volatility)
        or not np.isfinite(horizon)
        or mean_reversion <= 0.0
        or volatility < 0.0
        or horizon <= 0.0
    ):
        raise ValueError("invalid exact OU moment inputs")
    decay = exp(-mean_reversion * horizon)
    variance = (
        volatility**2
        * (-np.expm1(-2.0 * mean_reversion * horizon))
        / (2.0 * mean_reversion)
    )
    return x0 * decay, variance


@dataclass(frozen=True)
class GaussianMeanReversion:
    """Exact discretization of dx = -a x dt + sigma dW."""

    mean_reversion: float
    volatility: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.mean_reversion)
            or not np.isfinite(self.volatility)
            or self.mean_reversion <= 0.0
            or self.volatility < 0.0
        ):
            raise ValueError(
                "mean_reversion must be positive and volatility non-negative"
            )

    def transition(
        self,
        state: NDArray[np.float64],
        dt: float,
        z: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return exact_ou_transition(
            state,
            mean_reversion=self.mean_reversion,
            volatility=self.volatility,
            dt=dt,
            normal_shock=z,
        )

    def terminal_distribution(
        self,
        x0: float,
        horizon: float,
    ) -> tuple[float, float]:
        return exact_ou_terminal_moments(
            x0,
            mean_reversion=self.mean_reversion,
            volatility=self.volatility,
            horizon=horizon,
        )

    def simulate_terminal(
        self,
        x0: float,
        horizon: float,
        steps: int,
        paths: int,
        seed: int = 7,
    ) -> NDArray[np.float64]:
        if (
            not np.isfinite(x0)
            or not np.isfinite(horizon)
            or horizon <= 0.0
            or not isinstance(steps, int)
            or not isinstance(paths, int)
            or steps <= 0
            or paths <= 1
            or not isinstance(seed, int)
        ):
            raise ValueError("horizon, steps and paths must be positive")
        rng = np.random.default_rng(seed)
        state = np.full(paths, x0, dtype=float)
        dt = horizon / steps
        for _ in range(steps):
            state = self.transition(
                state,
                dt,
                rng.standard_normal(paths),
            )
        return state


@dataclass(frozen=True)
class HullWhiteCurveFitResult:
    curve_times: NDArray[np.float64]
    market_discount_factors: NDArray[np.float64]
    model_discount_factors: NDArray[np.float64]
    maximum_curve_fit_residual: float
    phi_times: NDArray[np.float64]
    phi_values: NDArray[np.float64]
    sample_terminal_mean: float
    theoretical_terminal_mean: float
    mean_z_score: float
    sample_terminal_variance: float
    theoretical_terminal_variance: float
    variance_z_score: float


def _integrated_ou_variance(
    maturity: float,
    mean_reversion: float,
    volatility: float,
) -> float:
    a = mean_reversion
    integral = (
        maturity
        - 2.0 * (-np.expm1(-a * maturity)) / a
        + (-np.expm1(-2.0 * a * maturity)) / (2.0 * a)
    )
    return volatility**2 * integral / a**2


def hull_white_curve_fit_audit(
    *,
    curve_times: NDArray[np.float64],
    market_discount_factors: NDArray[np.float64],
    mean_reversion: float,
    volatility: float,
    x0: float = 0.0,
    simulation_horizon: float = 5.0,
    simulation_steps: int = 20,
    paths: int = 50_000,
    seed: int = 707,
    integration_points: int = 4_001,
    z_limit: float = 4.0,
    tolerance: TolerancePolicy = TolerancePolicy(5.0e-6, 5.0e-6),
) -> EvidenceCertificate[HullWhiteCurveFitResult]:
    times = np.asarray(curve_times, dtype=float)
    market_dfs = np.asarray(market_discount_factors, dtype=float)
    if (
        times.ndim != 1
        or market_dfs.shape != times.shape
        or times.size < 2
        or np.any(times <= 0.0)
        or np.any(np.diff(times) <= 0.0)
        or np.any(market_dfs <= 0.0)
        or not np.all(np.isfinite(times))
        or not np.all(np.isfinite(market_dfs))
    ):
        raise ValueError("invalid initial discount curve")
    if (
        not np.isfinite(x0)
        or not np.isfinite(simulation_horizon)
        or not np.isfinite(z_limit)
        or not isinstance(simulation_steps, int)
        or not isinstance(paths, int)
        or not isinstance(seed, int)
        or not isinstance(integration_points, int)
        or simulation_horizon <= 0.0
        or simulation_steps <= 0
        or integration_points < 101
        or paths <= 1
        or z_limit <= 0.0
    ):
        raise ValueError("invalid audit configuration")
    model = GaussianMeanReversion(mean_reversion, volatility)
    maximum_maturity = float(times[-1])
    phi_times = np.linspace(0.0, maximum_maturity, integration_points)
    cumulative_forward = -np.log(
        np.interp(
            phi_times,
            np.r_[0.0, times],
            np.r_[1.0, market_dfs],
        )
    )
    initial_forwards = np.gradient(cumulative_forward, phi_times)
    correction = (
        volatility**2
        * (1.0 - np.exp(-mean_reversion * phi_times)) ** 2
        / (2.0 * mean_reversion**2)
    )
    # With r(t)=x(t)+phi(t) and E[x(t)|x(0)=x0]=x0*exp(-a*t),
    # the deterministic shift must remove the initial OU state.  The bond
    # exponent below then includes the matching -x0*B(0,T) term.
    phi = (
        initial_forwards
        + correction
        - x0 * np.exp(-mean_reversion * phi_times)
    )
    cumulative_phi = np.r_[
        0.0,
        np.cumsum(
            0.5 * (phi[1:] + phi[:-1]) * np.diff(phi_times)
        ),
    ]
    model_dfs = np.empty_like(market_dfs)
    for index, maturity in enumerate(times):
        integrated_phi = float(np.interp(maturity, phi_times, cumulative_phi))
        integrated_variance = _integrated_ou_variance(
            float(maturity),
            mean_reversion,
            volatility,
        )
        b_zero_maturity = (
            -np.expm1(-mean_reversion * float(maturity))
            / mean_reversion
        )
        model_dfs[index] = exp(
            -integrated_phi
            - x0 * b_zero_maturity
            + 0.5 * integrated_variance
        )
    curve_error = float(np.max(np.abs(model_dfs - market_dfs)))

    sample = model.simulate_terminal(
        x0,
        simulation_horizon,
        simulation_steps,
        paths,
        seed,
    )
    theoretical_mean, theoretical_variance = model.terminal_distribution(
        x0,
        simulation_horizon,
    )
    sample_mean = float(np.mean(sample))
    sample_variance = float(np.var(sample, ddof=1))
    mean_standard_error = sqrt(theoretical_variance / paths)
    variance_standard_error = theoretical_variance * sqrt(2.0 / (paths - 1))
    mean_z = abs(sample_mean - theoretical_mean) / max(
        mean_standard_error,
        np.finfo(float).tiny,
    )
    variance_z = abs(sample_variance - theoretical_variance) / max(
        variance_standard_error,
        np.finfo(float).tiny,
    )
    curve_limit = tolerance.limit(float(np.max(market_dfs)))
    invariants = (
        InvariantResult(
            "initial_curve_recovery",
            curve_error,
            0.0,
            curve_limit,
            curve_error <= curve_limit,
        ),
        InvariantResult(
            "terminal_mean_sampling_z",
            mean_z,
            0.0,
            z_limit,
            mean_z <= z_limit,
        ),
        InvariantResult(
            "terminal_variance_sampling_z",
            variance_z,
            0.0,
            z_limit,
            variance_z <= z_limit,
        ),
    )
    violations = tuple(
        Violation(
            code=f"HW_{invariant.name.upper()}",
            message=f"Hull-White invariant failed: {invariant.name}",
            observed=str(invariant.observed),
            limit=str(invariant.tolerance),
        )
        for invariant in invariants
        if not invariant.passed
    )
    result = HullWhiteCurveFitResult(
        curve_times=times,
        market_discount_factors=market_dfs,
        model_discount_factors=model_dfs,
        maximum_curve_fit_residual=curve_error,
        phi_times=phi_times,
        phi_values=phi,
        sample_terminal_mean=sample_mean,
        theoretical_terminal_mean=theoretical_mean,
        mean_z_score=mean_z,
        sample_terminal_variance=sample_variance,
        theoretical_terminal_variance=theoretical_variance,
        variance_z_score=variance_z,
    )
    return EvidenceCertificate(
        status=decision_from_invariants(invariants),
        result=result,
        invariants=invariants,
        violations=violations,
        conventions=ConventionSchema(
            currency="rate units",
            rate_compounding="continuous",
            day_count="model year fractions",
            discounting_curve="initial market discount curve",
        ),
        tolerances=tolerance,
        input_manifest_hash=stable_manifest_hash(
            {
                "curve_times": times,
                "market_discount_factors": market_dfs,
                "mean_reversion": mean_reversion,
                "volatility": volatility,
                "x0": x0,
                "simulation_horizon": simulation_horizon,
                "simulation_steps": simulation_steps,
                "paths": paths,
                "seed": seed,
                "integration_points": integration_points,
                "z_limit": z_limit,
                "tolerance": tolerance,
            }
        ),
        implementation_version="ch07.hull_white_exact.v0.10",
    )


if __name__ == "__main__":
    audit = hull_white_curve_fit_audit(
        curve_times=np.arange(1.0, 6.0),
        market_discount_factors=np.exp(-0.04 * np.arange(1.0, 6.0)),
        mean_reversion=0.60,
        volatility=0.012,
    )
    print("status=", audit.status)
    print("curve residual=", audit.result.maximum_curve_fit_residual)
    print("mean z=", audit.result.mean_z_score)
    print("variance z=", audit.result.variance_z_score)
