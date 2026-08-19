from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IntegralDiagnostic:
    strategy: str
    mean_terminal: float
    standard_error: float
    second_moment: float
    isometry_target: float
    relative_isometry_error: float


@dataclass(frozen=True)
class MartingaleIntegralStudy:
    predictable: IntegralDiagnostic
    anticipative: IntegralDiagnostic
    status: str


def deterministic_ito_moments(
    integrand: NDArray[np.float64],
    dt: float,
) -> tuple[float, float]:
    if (
        integrand.ndim != 1
        or integrand.size == 0
        or not np.all(np.isfinite(integrand))
        or dt <= 0.0
    ):
        raise ValueError("finite one-dimensional integrand and positive dt required")
    variance = float(np.sum(integrand * integrand) * dt)
    return 0.0, variance


def simulate_deterministic_ito(
    integrand: NDArray[np.float64],
    *,
    dt: float,
    paths: int = 50_000,
    seed: int = 11,
) -> NDArray[np.float64]:
    if paths < 2:
        raise ValueError("paths must be at least two")
    deterministic_ito_moments(integrand, dt)
    rng = np.random.default_rng(seed)
    d_w = np.sqrt(dt) * rng.standard_normal((paths, integrand.size))
    return d_w @ integrand


def stochastic_integral_martingale_study(
    *,
    steps: int = 128,
    paths: int = 40_000,
    horizon: float = 1.0,
    seed: int = 11,
) -> MartingaleIntegralStudy:
    """Contrast a predictable Ito sum with an anticipative construction."""

    if steps < 2 or paths < 2 or horizon <= 0.0:
        raise ValueError("steps, paths and horizon must define a valid simulation")
    dt = horizon / steps
    rng = np.random.default_rng(seed)
    d_w = np.sqrt(dt) * rng.standard_normal((paths, steps))
    left_state = np.c_[
        np.zeros(paths),
        np.cumsum(d_w[:, :-1], axis=1),
    ]

    predictable_terms = left_state * d_w
    predictable_terminal = np.sum(predictable_terms, axis=1)
    predictable_isometry = float(np.mean(np.sum(left_state**2 * dt, axis=1)))

    anticipative_integrand = np.sign(d_w)
    anticipative_terminal = np.sum(anticipative_integrand * d_w, axis=1)
    anticipative_isometry = float(
        np.mean(np.sum(anticipative_integrand**2 * dt, axis=1))
    )

    def diagnostic(
        name: str,
        terminal: NDArray[np.float64],
        isometry_target: float,
    ) -> IntegralDiagnostic:
        second_moment = float(np.mean(terminal**2))
        return IntegralDiagnostic(
            strategy=name,
            mean_terminal=float(np.mean(terminal)),
            standard_error=float(np.std(terminal, ddof=1) / np.sqrt(paths)),
            second_moment=second_moment,
            isometry_target=isometry_target,
            relative_isometry_error=abs(second_moment - isometry_target)
            / max(isometry_target, np.finfo(float).tiny),
        )

    predictable = diagnostic(
        "predictable_left_endpoint",
        predictable_terminal,
        predictable_isometry,
    )
    anticipative = diagnostic(
        "anticipative_sign_of_next_increment",
        anticipative_terminal,
        anticipative_isometry,
    )
    predictable_pass = (
        abs(predictable.mean_terminal) <= 5.0 * predictable.standard_error
        and predictable.relative_isometry_error <= 0.05
    )
    anticipative_rejected = (
        anticipative.mean_terminal
        > 20.0 * anticipative.standard_error
    )
    return MartingaleIntegralStudy(
        predictable=predictable,
        anticipative=anticipative,
        status=(
            "PASS"
            if predictable_pass and anticipative_rejected
            else "REJECT"
        ),
    )
