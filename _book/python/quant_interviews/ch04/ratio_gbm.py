from __future__ import annotations

from dataclasses import dataclass
import math
from math import sqrt


@dataclass(frozen=True)
class RatioDiffusionParameters:
    drift: float
    raw_variance: float
    volatility: float | None
    has_brownian_driver: bool
    status: str


def ratio_gbm_parameters(
    *,
    mu_x: float,
    mu_y: float,
    sigma_x: float,
    sigma_y: float,
    correlation: float,
) -> tuple[float, float]:
    result = ratio_gbm_parameter_audit(
        mu_x=mu_x,
        mu_y=mu_y,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        correlation=correlation,
    )
    if result.volatility is None:
        raise ValueError("ratio variance failed the governed numerical gate")
    return result.drift, result.volatility


def ratio_gbm_parameter_audit(
    *,
    mu_x: float,
    mu_y: float,
    sigma_x: float,
    sigma_y: float,
    correlation: float,
    absolute_tolerance: float = 1e-14,
    relative_tolerance: float = 1e-12,
) -> RatioDiffusionParameters:
    values = (
        mu_x,
        mu_y,
        sigma_x,
        sigma_y,
        correlation,
        absolute_tolerance,
        relative_tolerance,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("ratio inputs must be finite")
    if sigma_x < 0.0 or sigma_y < 0.0:
        raise ValueError("volatilities must be nonnegative")
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("tolerances must be nonnegative")
    drift = mu_x - mu_y + sigma_y**2 - correlation * sigma_x * sigma_y
    variance = sigma_x**2 + sigma_y**2 - 2.0 * correlation * sigma_x * sigma_y
    scale = max(sigma_x**2 + sigma_y**2, 1.0)
    floor = absolute_tolerance + relative_tolerance * scale
    if variance < -floor:
        return RatioDiffusionParameters(
            drift=drift,
            raw_variance=variance,
            volatility=None,
            has_brownian_driver=False,
            status="REJECT",
        )
    used_variance = max(variance, 0.0)
    return RatioDiffusionParameters(
        drift=drift,
        raw_variance=variance,
        volatility=sqrt(used_variance),
        has_brownian_driver=used_variance > floor,
        status="PASS" if variance >= 0.0 else "CONDITIONAL_PASS",
    )
