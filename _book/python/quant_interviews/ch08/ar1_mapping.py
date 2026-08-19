from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log


@dataclass(frozen=True)
class PersistenceMapping:
    phi: float
    sampling_interval: float
    half_life_periods: float | None
    ou_mean_reversion: float | None
    oscillatory: bool


def map_ar1_persistence(phi: float, *, sampling_interval: float = 1.0) -> PersistenceMapping:
    """Map a discrete AR(1) coefficient to half-life and OU mean reversion.

    The mapping is meaningful for 0 < phi < 1. Negative coefficients create
    oscillatory decay and require a different interpretation.
    """
    if not isinstance(phi, (int, float)) or not isinstance(
        sampling_interval, (int, float)
    ):
        raise TypeError("phi and sampling_interval must be real scalars.")
    if not isfinite(phi) or not isfinite(sampling_interval):
        raise ValueError("phi and sampling_interval must be finite.")
    if sampling_interval <= 0.0:
        raise ValueError("sampling_interval must be positive.")
    if not -1.0 < phi <= 1.0:
        raise ValueError("phi must lie in (-1, 1].")
    if phi == 0.0:
        return PersistenceMapping(phi, sampling_interval, 0.0, None, False)
    if abs(phi) == 1.0:
        return PersistenceMapping(phi, sampling_interval, None, None, phi < 0.0)
    half_life = log(0.5) / log(abs(phi))
    kappa = -log(phi) / sampling_interval if phi > 0.0 else None
    return PersistenceMapping(
        phi,
        sampling_interval,
        half_life,
        kappa,
        phi < 0.0,
    )
