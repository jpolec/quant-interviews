from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HedgeSpanningCertificate:
    asset_count: int
    factor_count: int
    factor_rank: int
    holdings_nullity: int
    condition_number: float | None
    hedge_solution_policy: str
    holdings_unique: bool
    market_complete: bool
    claim_spanned: bool
    hedge_positions: NDArray[np.float64]
    replicated_factor_exposure: NDArray[np.float64]
    residual_factor_exposure: NDArray[np.float64]
    residual_norm: float
    replicated_variance: float
    residual_variance: float
    status: str


def least_squares_hedge(
    basis: NDArray[np.float64],
    continuation: NDArray[np.float64],
) -> NDArray[np.float64]:
    design = np.asarray(basis, dtype=float)
    target = np.asarray(continuation, dtype=float)
    if (
        design.ndim != 2
        or target.ndim != 1
        or design.shape[0] != target.size
        or design.shape[0] < 1
        or design.shape[1] < 1
        or not np.all(np.isfinite(design))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("finite compatible regression shapes required")
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    return beta


def hedge_residual(
    basis: NDArray[np.float64],
    continuation: NDArray[np.float64],
    beta: NDArray[np.float64],
) -> NDArray[np.float64]:
    design = np.asarray(basis, dtype=float)
    target = np.asarray(continuation, dtype=float)
    coefficients = np.asarray(beta, dtype=float)
    if (
        design.ndim != 2
        or target.ndim != 1
        or coefficients.ndim != 1
        or design.shape != (target.size, coefficients.size)
        or not np.all(np.isfinite(design))
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError("finite compatible hedge residual inputs required")
    return target - design @ coefficients


def spanning_hedge_certificate(
    traded_factor_exposures: NDArray[np.float64],
    claim_factor_exposure: NDArray[np.float64],
    *,
    factor_covariance: NDArray[np.float64] | None = None,
    tolerance: float = 1e-10,
) -> HedgeSpanningCertificate:
    """Separate market completeness, claim spanning and holdings uniqueness."""

    exposures = np.asarray(traded_factor_exposures, dtype=float)
    claim = np.asarray(claim_factor_exposure, dtype=float)
    if (
        exposures.ndim != 2
        or exposures.shape[0] < 1
        or exposures.shape[1] < 1
        or claim.ndim != 1
        or exposures.shape[1] != claim.size
        or not np.all(np.isfinite(exposures))
        or not np.all(np.isfinite(claim))
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("finite traded-by-factor exposures and positive tolerance required")
    covariance = (
        np.eye(claim.size)
        if factor_covariance is None
        else np.asarray(factor_covariance, dtype=float)
    )
    if (
        covariance.shape != (claim.size, claim.size)
        or not np.all(np.isfinite(covariance))
        or not np.allclose(covariance, covariance.T, atol=tolerance, rtol=0.0)
        or np.min(np.linalg.eigvalsh(covariance)) < -tolerance
    ):
        raise ValueError("factor covariance must be finite symmetric PSD")

    positions, *_ = np.linalg.lstsq(exposures.T, claim, rcond=tolerance)
    replicated = positions @ exposures
    residual = claim - replicated
    replicated_variance = float(replicated @ covariance @ replicated)
    residual_variance = float(residual @ covariance @ residual)
    residual_norm = float(np.linalg.norm(residual))
    scale = max(float(claim @ covariance @ claim), np.finfo(float).tiny)
    singular_values = np.linalg.svd(exposures, compute_uv=False)
    rank = int(np.sum(singular_values > tolerance * max(singular_values[0], 1.0)))
    market_complete = rank == claim.size
    claim_spanned = residual_variance <= tolerance * scale
    asset_count = exposures.shape[0]
    nullity = asset_count - rank
    holdings_unique = claim_spanned and nullity == 0
    condition = (
        float(singular_values[0] / singular_values[rank - 1]) if rank > 0 else None
    )
    return HedgeSpanningCertificate(
        asset_count=asset_count,
        factor_count=claim.size,
        factor_rank=rank,
        holdings_nullity=nullity,
        condition_number=condition,
        hedge_solution_policy="minimum_norm",
        holdings_unique=holdings_unique,
        market_complete=market_complete,
        claim_spanned=claim_spanned,
        hedge_positions=positions,
        replicated_factor_exposure=replicated,
        residual_factor_exposure=residual,
        residual_norm=residual_norm,
        replicated_variance=replicated_variance,
        residual_variance=residual_variance,
        status="COMPLETE" if market_complete and claim_spanned else "INCOMPLETE",
    )
