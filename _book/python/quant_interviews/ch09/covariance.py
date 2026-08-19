from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PerturbationDiagnostics:
    condition_number: float
    participation_ratio_effective_rank: float
    weight_change_l2: float
    normalization: str


def _validate_psd(matrix: FloatArray, name: str) -> FloatArray:
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    scale = max(float(np.linalg.norm(value, ord=2)), 1.0)
    if not np.allclose(value, value.T, atol=1e-12 * scale, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    if float(np.linalg.eigvalsh(value).min()) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return (value + value.T) / 2.0


def participation_ratio_effective_rank(cov: FloatArray) -> float:
    matrix = _validate_psd(cov, "covariance")
    eigenvalues = np.maximum(np.linalg.eigvalsh(matrix), 0.0)
    squared_sum = float(eigenvalues @ eigenvalues)
    if squared_sum == 0.0:
        return 0.0
    return float(eigenvalues.sum() ** 2 / squared_sum)


def effective_rank(cov: FloatArray) -> float:
    """Compatibility alias for participation-ratio effective rank."""
    return participation_ratio_effective_rank(cov)


def shrink_covariance(
    sample: FloatArray,
    delta: float,
    target: FloatArray | None = None,
) -> FloatArray:
    sample_matrix = _validate_psd(sample, "sample covariance")
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be in [0, 1]")
    if target is None:
        target_matrix = (
            np.eye(sample_matrix.shape[0])
            * np.trace(sample_matrix)
            / sample_matrix.shape[0]
        )
    else:
        target_matrix = _validate_psd(target, "shrinkage target")
    if target_matrix.shape != sample_matrix.shape:
        raise ValueError("target shape mismatch")
    shrunk = (1.0 - delta) * sample_matrix + delta * target_matrix
    return (shrunk + shrunk.T) / 2.0


def precision_portfolio(
    signal: FloatArray,
    cov: FloatArray,
    normalization: Literal["net", "gross"] = "net",
) -> FloatArray:
    vector = np.asarray(signal, dtype=float)
    matrix = _validate_psd(cov, "covariance")
    if vector.ndim != 1 or matrix.shape != (vector.size, vector.size):
        raise ValueError("shape mismatch")
    if not np.isfinite(vector).all():
        raise ValueError("signal must be finite")
    if float(np.linalg.eigvalsh(matrix).min()) <= 0.0:
        raise ValueError("precision portfolio requires positive definite covariance")
    raw = np.linalg.solve(matrix, vector)
    denominator = (
        float(raw.sum())
        if normalization == "net"
        else float(np.abs(raw).sum())
    )
    if normalization not in {"net", "gross"}:
        raise ValueError("normalization must be 'net' or 'gross'")
    threshold = 100.0 * np.finfo(float).eps * max(
        float(np.linalg.norm(raw)),
        1.0,
    )
    if abs(denominator) <= threshold:
        raise ValueError("portfolio normalization is ill-conditioned")
    return raw / denominator


def perturbation_diagnostics(
    signal: FloatArray,
    cov: FloatArray,
    perturbed: FloatArray,
    normalization: Literal["net", "gross"] = "net",
) -> PerturbationDiagnostics:
    base = _validate_psd(cov, "covariance")
    changed = _validate_psd(perturbed, "perturbed covariance")
    if base.shape != changed.shape:
        raise ValueError("covariance shapes differ")
    before = precision_portfolio(signal, base, normalization)
    after = precision_portfolio(signal, changed, normalization)
    return PerturbationDiagnostics(
        condition_number=float(np.linalg.cond(base)),
        participation_ratio_effective_rank=participation_ratio_effective_rank(
            base
        ),
        weight_change_l2=float(np.linalg.norm(after - before)),
        normalization=normalization,
    )
