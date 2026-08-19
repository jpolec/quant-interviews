from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import isfinite

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DupireDiagnostic:
    local_variance: NDArray[np.float64]
    call_time_derivative: NDArray[np.float64]
    call_strike_derivative: NDArray[np.float64]
    call_density: NDArray[np.float64]
    minimum_density: float
    density_floor: float
    denominator_floor_hits: int
    negative_fraction: float
    nonfinite_fraction: float
    median_interior_local_variance: float | None
    maximum_finite_local_variance: float | None
    surface_sha256: str
    call_bound_violation_count: int
    strike_slope_violation_count: int
    upstream_static_arbitrage_status: str
    median_recovery_error: float | None
    maximum_recovery_error: float | None
    status: str


def dupire_surface_sha256(
    *,
    strike: NDArray[np.float64],
    maturity: NDArray[np.float64],
    call_surface: NDArray[np.float64],
    forwards: NDArray[np.float64],
    discount_factors: NDArray[np.float64],
    rate: float,
    dividend_yield: float,
) -> str:
    """Return the canonical hash used to link surface and Dupire evidence."""

    digest = hashlib.sha256()
    for value in (
        strike,
        maturity,
        call_surface,
        forwards,
        discount_factors,
    ):
        contiguous = np.ascontiguousarray(value, dtype="<f8")
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    digest.update(f"{rate:.17g}|{dividend_yield:.17g}".encode("ascii"))
    return digest.hexdigest()


def dupire_local_variance_diagnostic(
    *,
    strike: NDArray[np.float64],
    maturity: NDArray[np.float64],
    call_surface: NDArray[np.float64],
    rate: float,
    dividend_yield: float,
    forwards: NDArray[np.float64],
    discount_factors: NDArray[np.float64],
    upstream_surface_sha256: str,
    upstream_static_arbitrage_status: str,
    expected_local_variance: float | None = None,
    median_recovery_tolerance: float = 0.003,
    maximum_recovery_tolerance: float = 0.01,
    static_arbitrage_tolerance: float = 1e-8,
    density_absolute_tolerance: float = 1e-10,
    density_relative_tolerance: float = 1e-6,
) -> DupireDiagnostic:
    strike = np.asarray(strike, dtype=float)
    maturity = np.asarray(maturity, dtype=float)
    surface = np.asarray(call_surface, dtype=float)
    forwards = np.asarray(forwards, dtype=float)
    discount_factors = np.asarray(discount_factors, dtype=float)
    controls = (
        rate,
        dividend_yield,
        density_absolute_tolerance,
        density_relative_tolerance,
        median_recovery_tolerance,
        maximum_recovery_tolerance,
        static_arbitrage_tolerance,
    )
    if (
        strike.ndim != 1
        or maturity.ndim != 1
        or surface.shape != (maturity.size, strike.size)
        or forwards.shape != maturity.shape
        or discount_factors.shape != maturity.shape
        or strike.size < 5
        or maturity.size < 4
    ):
        raise ValueError("Dupire surface has inconsistent shape")
    if (
        not np.all(np.isfinite(strike))
        or not np.all(np.isfinite(maturity))
        or not np.all(np.isfinite(surface))
        or not np.all(np.isfinite(forwards))
        or not np.all(np.isfinite(discount_factors))
        or not all(isfinite(value) for value in controls)
    ):
        raise ValueError("Dupire inputs must be finite")
    if (
        np.any(np.diff(strike) <= 0.0)
        or np.any(np.diff(maturity) <= 0.0)
        or np.any(strike <= 0.0)
        or np.any(maturity <= 0.0)
        or np.any(surface < 0.0)
        or np.any(forwards <= 0.0)
        or np.any(discount_factors <= 0.0)
        or density_absolute_tolerance < 0.0
        or density_relative_tolerance < 0.0
        or median_recovery_tolerance < 0.0
        or maximum_recovery_tolerance < 0.0
        or static_arbitrage_tolerance < 0.0
        or upstream_static_arbitrage_status not in {"PASS", "REJECT"}
        or not isinstance(upstream_surface_sha256, str)
        or len(upstream_surface_sha256) != 64
        or any(character not in "0123456789abcdef" for character in upstream_surface_sha256)
        or (
            expected_local_variance is not None
            and (
                not isfinite(expected_local_variance)
                or expected_local_variance < 0.0
            )
        )
    ):
        raise ValueError("invalid Dupire coordinates")

    surface_sha256 = dupire_surface_sha256(
        strike=strike,
        maturity=maturity,
        call_surface=surface,
        forwards=forwards,
        discount_factors=discount_factors,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    lineage_match = surface_sha256 == upstream_surface_sha256

    lower = np.maximum(
        discount_factors[:, None] * (forwards[:, None] - strike[None, :]),
        0.0,
    )
    upper = discount_factors[:, None] * forwards[:, None]
    bound_violations = (surface < lower - static_arbitrage_tolerance) | (
        surface > upper + static_arbitrage_tolerance
    )
    slopes = np.diff(surface, axis=1) / np.diff(strike)[None, :]
    slope_violations = slopes > static_arbitrage_tolerance

    d_t = np.gradient(surface, maturity, axis=0, edge_order=2)
    d_k = np.gradient(surface, strike, axis=1, edge_order=2)
    d_kk = np.gradient(d_k, strike, axis=1, edge_order=2)
    numerator = (
        d_t
        + (rate - dividend_yield) * strike[None, :] * d_k
        + dividend_yield * surface
    )
    scale = max(float(np.nanmax(np.abs(d_kk))), 1e-16)
    density_floor = (
        density_absolute_tolerance + density_relative_tolerance * scale
    )
    denominator = 0.5 * strike[None, :] ** 2 * d_kk
    floor_mask = d_kk <= density_floor
    local_variance = np.full_like(surface, np.nan)
    valid = ~floor_mask
    local_variance[valid] = numerator[valid] / denominator[valid]

    interior = local_variance[1:-1, 2:-2]
    finite = np.isfinite(interior)
    negative = finite & (interior < 0.0)
    negative_fraction = float(np.mean(negative))
    nonfinite_fraction = float(np.mean(~finite))
    finite_values = interior[finite]
    median = (
        float(np.median(finite_values))
        if finite_values.size
        else None
    )
    maximum_finite = (
        float(np.max(finite_values)) if finite_values.size else None
    )
    recovery_exclusion = max(2, int(np.ceil(0.1 * strike.size)))
    recovery = local_variance[2:-2, recovery_exclusion:-recovery_exclusion]
    recovery_values = recovery[np.isfinite(recovery)]
    if expected_local_variance is None or recovery_values.size == 0:
        median_error = None
        maximum_error = None
        recovery_pass = recovery_values.size > 0
    else:
        recovery_median = float(np.median(recovery_values))
        median_error = abs(recovery_median - expected_local_variance)
        maximum_error = float(
            np.max(np.abs(recovery_values - expected_local_variance))
        )
        recovery_pass = (
            median_error <= median_recovery_tolerance
            and maximum_error <= maximum_recovery_tolerance
        )
    passed = (
        upstream_static_arbitrage_status == "PASS"
        and lineage_match
        and not np.any(bound_violations)
        and not np.any(slope_violations)
        and negative_fraction == 0.0
        and nonfinite_fraction == 0.0
        and recovery_pass
    )
    return DupireDiagnostic(
        local_variance=local_variance,
        call_time_derivative=d_t,
        call_strike_derivative=d_k,
        call_density=d_kk,
        minimum_density=float(np.min(d_kk[1:-1, 2:-2])),
        density_floor=density_floor,
        denominator_floor_hits=int(np.count_nonzero(floor_mask[1:-1, 2:-2])),
        negative_fraction=negative_fraction,
        nonfinite_fraction=nonfinite_fraction,
        median_interior_local_variance=median,
        maximum_finite_local_variance=maximum_finite,
        surface_sha256=surface_sha256,
        call_bound_violation_count=int(np.count_nonzero(bound_violations)),
        strike_slope_violation_count=int(np.count_nonzero(slope_violations)),
        upstream_static_arbitrage_status=upstream_static_arbitrage_status,
        median_recovery_error=median_error,
        maximum_recovery_error=maximum_error,
        status="PASS" if passed else "REJECT",
    )


def dupire_local_variance(
    *,
    strike: NDArray[np.float64],
    maturity: NDArray[np.float64],
    call_surface: NDArray[np.float64],
    rate: float,
    dividend_yield: float,
    forwards: NDArray[np.float64],
    discount_factors: NDArray[np.float64],
    upstream_surface_sha256: str,
    upstream_static_arbitrage_status: str,
) -> NDArray[np.float64]:
    """Compatibility wrapper returning the governed local-variance grid."""

    return dupire_local_variance_diagnostic(
        strike=strike,
        maturity=maturity,
        call_surface=call_surface,
        rate=rate,
        dividend_yield=dividend_yield,
        forwards=forwards,
        discount_factors=discount_factors,
        upstream_surface_sha256=upstream_surface_sha256,
        upstream_static_arbitrage_status=upstream_static_arbitrage_status,
    ).local_variance
