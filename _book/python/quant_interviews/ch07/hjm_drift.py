from __future__ import annotations

from dataclasses import dataclass

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


def exponential_hjm_volatility(
    time_to_maturity: NDArray[np.float64],
    sigma0: float,
    decay: float,
) -> NDArray[np.float64]:
    tau = np.asarray(time_to_maturity, dtype=float)
    if (
        not np.isfinite(sigma0)
        or not np.isfinite(decay)
        or sigma0 < 0.0
        or decay <= 0.0
        or np.any(tau < 0.0)
        or not np.all(np.isfinite(tau))
    ):
        raise ValueError("invalid HJM parameters")
    return sigma0 * np.exp(-decay * tau)


def one_factor_hjm_drift(
    time_to_maturity: NDArray[np.float64],
    sigma0: float,
    decay: float,
) -> NDArray[np.float64]:
    """Risk-neutral HJM drift for sigma(t,T)=sigma0 exp(-decay*(T-t))."""

    tau = np.asarray(time_to_maturity, dtype=float)
    sigma = exponential_hjm_volatility(tau, sigma0, decay)
    integrated_sigma = sigma0 * (-np.expm1(-decay * tau)) / decay
    return sigma * integrated_sigma


@dataclass(frozen=True)
class HJMMartingaleResult:
    maturities: NDArray[np.float64]
    initial_discount_factors: NDArray[np.float64]
    estimated_discounted_bond_values: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    z_scores: NDArray[np.float64]
    coarse_estimates: NDArray[np.float64]
    refinement_differences: NDArray[np.float64]
    refinement_standard_errors: NDArray[np.float64]
    nested_brownian_paths: bool
    maximum_absolute_z_score: float


def _integral_to_tau(
    cumulative_integral: NDArray[np.float64],
    tau_grid: NDArray[np.float64],
    target_tau: float,
) -> NDArray[np.float64]:
    upper = int(np.searchsorted(tau_grid, target_tau, side="right"))
    upper = min(max(upper, 1), tau_grid.size - 1)
    lower = upper - 1
    weight = (
        (target_tau - tau_grid[lower])
        / (tau_grid[upper] - tau_grid[lower])
    )
    return (
        (1.0 - weight) * cumulative_integral[:, lower]
        + weight * cumulative_integral[:, upper]
    )


def _simulate_discounted_bonds(
    *,
    maturities: NDArray[np.float64],
    initial_forward_rate: float,
    sigma0: float,
    decay: float,
    audit_horizon: float,
    time_steps: int,
    maturity_steps: int,
    paths: int,
    seed: int,
    brownian_increments: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    maximum_tau = float(np.max(maturities))
    tau_grid = np.linspace(0.0, maximum_tau, maturity_steps + 1)
    dt = audit_horizon / time_steps
    forward_curves = np.full(
        (paths, tau_grid.size),
        initial_forward_rate,
        dtype=float,
    )
    money_market = np.ones(paths)
    volatility = exponential_hjm_volatility(tau_grid, sigma0, decay)
    drift = one_factor_hjm_drift(tau_grid, sigma0, decay)
    if brownian_increments is None:
        rng = np.random.default_rng(seed)
        increments = rng.normal(
            scale=np.sqrt(dt),
            size=(time_steps, paths),
        )
    else:
        increments = np.asarray(brownian_increments, dtype=float)
        if (
            increments.shape != (time_steps, paths)
            or not np.all(np.isfinite(increments))
        ):
            raise ValueError("brownian_increments have invalid shape or values")
    for step in range(time_steps):
        money_market *= np.exp(forward_curves[:, 0] * dt)
        maturity_derivative = np.gradient(
            forward_curves,
            tau_grid,
            axis=1,
            edge_order=2,
        )
        brownian_increment = increments[step, :, None]
        forward_curves += (
            (maturity_derivative + drift[None, :]) * dt
            + volatility[None, :] * brownian_increment
        )
    cumulative = np.zeros_like(forward_curves)
    cumulative[:, 1:] = np.cumsum(
        0.5
        * (forward_curves[:, 1:] + forward_curves[:, :-1])
        * np.diff(tau_grid)[None, :],
        axis=1,
    )
    discounted_bonds = np.empty((paths, maturities.size))
    for index, maturity in enumerate(maturities):
        remaining = float(maturity - audit_horizon)
        integrated_forward = _integral_to_tau(
            cumulative,
            tau_grid,
            remaining,
        )
        discounted_bonds[:, index] = (
            np.exp(-integrated_forward) / money_market
        )
    return discounted_bonds


def nested_brownian_increments(
    *,
    time_steps: int,
    paths: int,
    audit_horizon: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return coarse increments and their exactly nested fine components."""

    if (
        not isinstance(time_steps, int)
        or not isinstance(paths, int)
        or not isinstance(seed, int)
        or time_steps <= 0
        or paths <= 0
        or not np.isfinite(audit_horizon)
        or audit_horizon <= 0.0
    ):
        raise ValueError("invalid Brownian-grid configuration")
    rng = np.random.default_rng(seed)
    fine = rng.normal(
        scale=np.sqrt(audit_horizon / (2 * time_steps)),
        size=(2 * time_steps, paths),
    )
    coarse = fine.reshape(time_steps, 2, paths).sum(axis=1)
    return coarse, fine


def hjm_martingale_audit(
    *,
    maturities: NDArray[np.float64],
    initial_forward_rate: float,
    sigma0: float,
    decay: float,
    audit_horizon: float = 0.5,
    time_steps: int = 32,
    maturity_steps: int = 96,
    paths: int = 6_000,
    seed: int = 707,
    z_limit: float = 4.5,
    refinement_limit: float = 8.0e-4,
    tolerance: TolerancePolicy = TolerancePolicy(1.0e-5, 1.0e-5),
) -> EvidenceCertificate[HJMMartingaleResult]:
    maturity_array = np.asarray(maturities, dtype=float)
    if (
        maturity_array.ndim != 1
        or maturity_array.size == 0
        or not np.all(np.isfinite(maturity_array))
        or np.any(np.diff(maturity_array) <= 0.0)
        or np.any(maturity_array <= audit_horizon)
        or not np.isfinite(initial_forward_rate)
        or not np.isfinite(audit_horizon)
        or audit_horizon <= 0.0
        or not isinstance(time_steps, int)
        or not isinstance(maturity_steps, int)
        or not isinstance(paths, int)
        or not isinstance(seed, int)
        or time_steps < 4
        or maturity_steps < 16
        or paths < 100
        or not np.isfinite(z_limit)
        or not np.isfinite(refinement_limit)
        or z_limit <= 0.0
        or refinement_limit <= 0.0
    ):
        raise ValueError("invalid HJM martingale-audit configuration")
    coarse_increments, fine_increments = nested_brownian_increments(
        time_steps=time_steps,
        paths=paths,
        audit_horizon=audit_horizon,
        seed=seed,
    )
    coarse = _simulate_discounted_bonds(
        maturities=maturity_array,
        initial_forward_rate=initial_forward_rate,
        sigma0=sigma0,
        decay=decay,
        audit_horizon=audit_horizon,
        time_steps=time_steps,
        maturity_steps=maturity_steps,
        paths=paths,
        seed=seed,
        brownian_increments=coarse_increments,
    )
    fine = _simulate_discounted_bonds(
        maturities=maturity_array,
        initial_forward_rate=initial_forward_rate,
        sigma0=sigma0,
        decay=decay,
        audit_horizon=audit_horizon,
        time_steps=2 * time_steps,
        maturity_steps=2 * maturity_steps,
        paths=paths,
        seed=seed,
        brownian_increments=fine_increments,
    )
    initial_discount_factors = np.exp(
        -initial_forward_rate * maturity_array
    )
    estimates = np.mean(fine, axis=0)
    sampling_standard_errors = np.std(fine, axis=0, ddof=1) / np.sqrt(paths)
    deterministic_error_floors = np.array(
        [tolerance.limit(float(value)) for value in initial_discount_factors]
    )
    standard_errors = np.maximum(
        sampling_standard_errors,
        deterministic_error_floors,
    )
    z_scores = (
        estimates - initial_discount_factors
    ) / np.maximum(standard_errors, np.finfo(float).tiny)
    coarse_estimates = np.mean(coarse, axis=0)
    paired_refinement = fine - coarse
    refinement_differences = np.abs(np.mean(paired_refinement, axis=0))
    refinement_standard_errors = (
        np.std(paired_refinement, axis=0, ddof=1) / np.sqrt(paths)
    )
    maximum_z = float(np.max(np.abs(z_scores)))
    maximum_refinement = float(np.max(refinement_differences))
    invariants = (
        InvariantResult(
            "discounted_bond_martingale_z",
            maximum_z,
            0.0,
            z_limit,
            maximum_z <= z_limit,
        ),
        InvariantResult(
            "time_maturity_grid_refinement",
            maximum_refinement,
            0.0,
            refinement_limit,
            maximum_refinement <= refinement_limit,
        ),
    )
    violations = tuple(
        Violation(
            code=f"HJM_{invariant.name.upper()}",
            message=f"HJM invariant failed: {invariant.name}",
            observed=str(invariant.observed),
            limit=str(invariant.tolerance),
        )
        for invariant in invariants
        if not invariant.passed
    )
    result = HJMMartingaleResult(
        maturities=maturity_array,
        initial_discount_factors=initial_discount_factors,
        estimated_discounted_bond_values=estimates,
        standard_errors=standard_errors,
        z_scores=z_scores,
        coarse_estimates=coarse_estimates,
        refinement_differences=refinement_differences,
        refinement_standard_errors=refinement_standard_errors,
        nested_brownian_paths=True,
        maximum_absolute_z_score=maximum_z,
    )
    return EvidenceCertificate(
        status=decision_from_invariants(invariants),
        result=result,
        invariants=invariants,
        violations=violations,
        conventions=ConventionSchema(
            currency="discounted bond-price units",
            rate_compounding="continuous instantaneous forwards",
            day_count="model year fractions",
            discounting_curve="money-market numeraire",
        ),
        tolerances=tolerance,
        input_manifest_hash=stable_manifest_hash(
            {
                "maturities": maturity_array,
                "initial_forward_rate": initial_forward_rate,
                "sigma0": sigma0,
                "decay": decay,
                "audit_horizon": audit_horizon,
                "time_steps": time_steps,
                "maturity_steps": maturity_steps,
                "paths": paths,
                "seed": seed,
                "z_limit": z_limit,
                "refinement_limit": refinement_limit,
                "tolerance": tolerance,
            }
        ),
        implementation_version="ch07.hjm_drift.v0.10",
    )


if __name__ == "__main__":
    certificate = hjm_martingale_audit(
        maturities=np.array([1.0, 2.0, 5.0]),
        initial_forward_rate=0.035,
        sigma0=0.008,
        decay=0.30,
    )
    print("status=", certificate.status)
    print("z_scores=", certificate.result.z_scores)
    print("refinement=", certificate.result.refinement_differences)
