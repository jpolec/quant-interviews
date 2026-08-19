from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DensityProcessDiagnostic:
    mean_terminal_density: float | None
    confidence_interval: tuple[float, float] | None
    effective_sample_size_fraction: float
    coefficient_of_variation: float | None
    maximum_normalized_weight: float | None
    log_weight_range: float | None
    novikov_finite_sample_proxy: float | None
    integrand_timing: str
    reason: str
    status: str


def _validate_inputs(
    theta: NDArray[np.float64],
    d_w: NDArray[np.float64],
    dt: float,
    integrand_timing: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    theta_values = np.asarray(theta, dtype=float)
    increments = np.asarray(d_w, dtype=float)
    if theta_values.shape != increments.shape:
        raise ValueError("theta and dW must have the same shape")
    if theta_values.ndim != 2 or theta_values.shape[0] < 2 or theta_values.shape[1] < 1:
        raise ValueError("theta and dW must be nonempty path-by-step matrices")
    if not np.all(np.isfinite(theta_values)) or not np.all(np.isfinite(increments)):
        raise ValueError("theta and dW must contain only finite values")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if integrand_timing != "left_endpoint_predictable":
        raise ValueError("integrand_timing must certify left-endpoint predictability")
    return theta_values, increments


def stochastic_exponential(
    theta: NDArray[np.float64],
    d_w: NDArray[np.float64],
    dt: float,
    *,
    integrand_timing: str,
) -> NDArray[np.float64]:
    values, increments = _validate_inputs(theta, d_w, dt, integrand_timing)
    log_density = np.cumsum(values * increments - 0.5 * values * values * dt, axis=1)
    if np.max(log_density) > math.log(np.finfo(float).max) or np.min(log_density) < math.log(
        np.nextafter(0.0, 1.0)
    ):
        raise FloatingPointError("density path leaves the representable positive range")
    return np.exp(log_density)


def novikov_proxy(theta: NDArray[np.float64], dt: float) -> float | None:
    """Finite-sample diagnostic only; it is not a proof of Novikov's condition."""

    values = np.asarray(theta, dtype=float)
    if (
        values.ndim not in (1, 2)
        or values.size == 0
        or not np.all(np.isfinite(values))
        or not math.isfinite(dt)
        or dt <= 0.0
    ):
        raise ValueError("finite deterministic vector or path matrix required")
    time_integrals = (
        np.array([np.sum(values**2) * dt])
        if values.ndim == 1
        else np.sum(values**2, axis=1) * dt
    )
    exponents = 0.5 * time_integrals
    maximum = float(np.max(exponents))
    shifted_mean = float(np.mean(np.exp(exponents - maximum)))
    log_proxy = maximum + math.log(shifted_mean)
    if log_proxy > math.log(np.finfo(float).max):
        return None
    return math.exp(log_proxy)


def density_process_diagnostic(
    theta: NDArray[np.float64],
    d_w: NDArray[np.float64],
    *,
    dt: float,
    integrand_timing: str,
    confidence_z: float = 1.96,
    minimum_ess_fraction: float = 0.1,
) -> DensityProcessDiagnostic:
    values, increments = _validate_inputs(theta, d_w, dt, integrand_timing)
    if (
        not math.isfinite(confidence_z)
        or confidence_z <= 0.0
        or not math.isfinite(minimum_ess_fraction)
        or not 0.0 < minimum_ess_fraction <= 1.0
    ):
        raise ValueError("confidence and ESS policies must be finite and positive")
    terminal_log = np.sum(values * increments - 0.5 * values * values * dt, axis=1)
    maximum = float(np.max(terminal_log))
    shifted = np.exp(terminal_log - maximum)
    shifted_sum = float(np.sum(shifted))
    log_mean = maximum + math.log(shifted_sum / terminal_log.size)
    normalized = shifted / shifted_sum
    ess_fraction = 1.0 / float(np.sum(normalized**2)) / terminal_log.size
    novikov = novikov_proxy(values, dt)
    if (
        log_mean > math.log(np.finfo(float).max)
        or log_mean < math.log(np.nextafter(0.0, 1.0))
        or novikov is None
    ):
        return DensityProcessDiagnostic(
            mean_terminal_density=None,
            confidence_interval=None,
            effective_sample_size_fraction=ess_fraction,
            coefficient_of_variation=None,
            maximum_normalized_weight=float(np.max(normalized)),
            log_weight_range=float(np.max(terminal_log) - np.min(terminal_log)),
            novikov_finite_sample_proxy=novikov,
            integrand_timing=integrand_timing,
            reason="NONREPRESENTABLE_OR_NONFINITE_DIAGNOSTIC",
            status="REJECT",
        )
    terminal = np.exp(terminal_log)
    mean = math.exp(log_mean)
    standard_deviation = float(np.std(terminal, ddof=1))
    standard_error = standard_deviation / math.sqrt(terminal.size)
    interval = (
        mean - confidence_z * standard_error,
        mean + confidence_z * standard_error,
    )
    coefficient = standard_deviation / mean
    diagnostics = (
        mean,
        standard_error,
        coefficient,
        ess_fraction,
        float(np.max(normalized)),
        float(np.max(terminal_log) - np.min(terminal_log)),
        novikov,
    )
    finite = all(math.isfinite(value) for value in diagnostics)
    mean_pass = interval[0] <= 1.0 <= interval[1]
    status = (
        "PASS"
        if finite and mean_pass and ess_fraction >= minimum_ess_fraction
        else "CONDITIONAL_PASS"
        if finite and mean_pass and ess_fraction > 0.0
        else "REJECT"
    )
    return DensityProcessDiagnostic(
        mean_terminal_density=mean,
        confidence_interval=interval,
        effective_sample_size_fraction=ess_fraction,
        coefficient_of_variation=coefficient,
        maximum_normalized_weight=float(np.max(normalized)),
        log_weight_range=float(np.max(terminal_log) - np.min(terminal_log)),
        novikov_finite_sample_proxy=novikov,
        integrand_timing=integrand_timing,
        reason="FINITE_SAMPLE_PROXY_ONLY",
        status=status,
    )
