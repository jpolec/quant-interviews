from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from numpy.typing import NDArray

from quant_interviews.ch02.regularization import nearest_correlation


@dataclass(frozen=True)
class CorrelationGateResult:
    raw_correlation: NDArray[np.float64]
    used_correlation: NDArray[np.float64] | None
    raw_eigenvalues: NDArray[np.float64]
    used_eigenvalues: NDArray[np.float64]
    raw_rank: int
    used_rank: int
    factor: NDArray[np.float64] | None
    factorization_method: str
    symmetry_error: float
    diagonal_error: float
    rank_cutoff: float
    repair_frobenius_norm: float | None
    reconstruction_residual: float | None
    raw_sha256: str
    used_sha256: str | None
    approval_owner: str | None
    policy_id: str
    reason: str
    status: str

    @property
    def correlation(self) -> NDArray[np.float64] | None:
        """Backward-compatible alias for the governed matrix actually used."""

        return self.used_correlation


@dataclass(frozen=True)
class CorrelatedNormalResult:
    draws: NDArray[np.float64]
    gate: CorrelationGateResult


def _matrix_hash(matrix: NDArray[np.float64]) -> str:
    payload = json.dumps(
        np.asarray(matrix, dtype=float).tolist(),
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _empty_rejection(
    source: NDArray[np.float64],
    *,
    symmetry_error: float,
    diagonal_error: float,
    raw_hash: str,
    policy_id: str,
    reason: str,
) -> CorrelationGateResult:
    return CorrelationGateResult(
        raw_correlation=source,
        used_correlation=None,
        raw_eigenvalues=np.array([], dtype=float),
        used_eigenvalues=np.array([], dtype=float),
        raw_rank=0,
        used_rank=0,
        factor=None,
        factorization_method="none",
        symmetry_error=symmetry_error,
        diagonal_error=diagonal_error,
        rank_cutoff=0.0,
        repair_frobenius_norm=None,
        reconstruction_residual=None,
        raw_sha256=raw_hash,
        used_sha256=None,
        approval_owner=None,
        policy_id=policy_id,
        reason=reason,
        status="REJECT",
    )


def correlation_input_gate(
    matrix: NDArray[np.float64],
    *,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-10,
    maximum_repair_norm: float = 0.02,
    approval_owner: str | None = None,
    policy_id: str = "ch04-correlation-policy-v0.10",
) -> CorrelationGateResult:
    """Return internally consistent raw/used correlation and fail closed on repair."""

    source = np.asarray(matrix, dtype=float)
    controls = np.asarray(
        [absolute_tolerance, relative_tolerance, maximum_repair_norm], dtype=float
    )
    if (
        source.ndim != 2
        or source.shape[0] != source.shape[1]
        or source.shape[0] < 2
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(controls))
        or absolute_tolerance < 0.0
        or relative_tolerance < 0.0
        or maximum_repair_norm < 0.0
        or not policy_id
        or (approval_owner is not None and not approval_owner.strip())
    ):
        raise ValueError("finite square matrix and finite nonnegative policies required")
    raw_hash = _matrix_hash(source)
    matrix_scale = max(float(np.max(np.abs(source))), 1.0)
    structural_limit = absolute_tolerance + relative_tolerance * matrix_scale
    symmetry_error = float(np.max(np.abs(source - source.T)))
    diagonal_error = float(np.max(np.abs(np.diag(source) - 1.0)))
    if symmetry_error > structural_limit or diagonal_error > structural_limit:
        return _empty_rejection(
            source,
            symmetry_error=symmetry_error,
            diagonal_error=diagonal_error,
            raw_hash=raw_hash,
            policy_id=policy_id,
            reason="STRUCTURAL_INPUT_FAILURE",
        )

    raw = 0.5 * (source + source.T)
    raw_values, raw_vectors = np.linalg.eigh(raw)
    spectral_scale = max(float(np.max(np.abs(raw_values))), np.finfo(float).tiny)
    rank_cutoff = absolute_tolerance + relative_tolerance * spectral_scale
    raw_rank = int(np.sum(raw_values > rank_cutoff))
    roundoff_floor = 10.0 * np.finfo(float).eps * spectral_scale

    if raw_values[0] > rank_cutoff:
        used = raw
        factor = np.linalg.cholesky(used)
        method = "cholesky"
        reason = "VALID_SPD"
        status = "PASS"
    elif raw_values[0] >= -roundoff_floor:
        used = raw
        positive = raw_values > rank_cutoff
        factor = raw_vectors[:, positive] * np.sqrt(raw_values[positive])
        method = "eigen_rank_preserving"
        reason = "VALID_SINGULAR_PSD"
        status = "PASS"
    elif raw_values[0] >= -rank_cutoff:
        clipped = np.maximum(raw_values, 0.0)
        preliminary = (raw_vectors * clipped) @ raw_vectors.T
        diagonal = np.sqrt(np.maximum(np.diag(preliminary), np.finfo(float).tiny))
        used = preliminary / np.outer(diagonal, diagonal)
        used = 0.5 * (used + used.T)
        used_values, used_vectors = np.linalg.eigh(used)
        positive = used_values > rank_cutoff
        factor = used_vectors[:, positive] * np.sqrt(used_values[positive])
        method = "eigen_roundoff_projection"
        reason = "NUMERICAL_FLOOR_REPAIR_RECORDED"
        status = "PASS_NUMERICAL"
    else:
        repaired = nearest_correlation(
            raw,
            tolerance=max(absolute_tolerance, relative_tolerance),
            max_iterations=300,
        )
        used = np.asarray(repaired["correlation"], dtype=float)
        repair_norm = float(np.linalg.norm(used - raw, ord="fro"))
        if (
            not bool(repaired["passed"])
            or not math.isfinite(repair_norm)
            or repair_norm > maximum_repair_norm
        ):
            return CorrelationGateResult(
                raw_correlation=raw,
                used_correlation=None,
                raw_eigenvalues=raw_values,
                used_eigenvalues=np.array([], dtype=float),
                raw_rank=raw_rank,
                used_rank=0,
                factor=None,
                factorization_method="none",
                symmetry_error=symmetry_error,
                diagonal_error=diagonal_error,
                rank_cutoff=rank_cutoff,
                repair_frobenius_norm=repair_norm,
                reconstruction_residual=None,
                raw_sha256=raw_hash,
                used_sha256=None,
                approval_owner=approval_owner,
                policy_id=policy_id,
                reason="REPAIR_LIMIT_OR_SOLVER_FAILURE",
                status="REJECT",
            )
        if approval_owner is None:
            return CorrelationGateResult(
                raw_correlation=raw,
                used_correlation=None,
                raw_eigenvalues=raw_values,
                used_eigenvalues=np.array([], dtype=float),
                raw_rank=raw_rank,
                used_rank=0,
                factor=None,
                factorization_method="none",
                symmetry_error=symmetry_error,
                diagonal_error=diagonal_error,
                rank_cutoff=rank_cutoff,
                repair_frobenius_norm=repair_norm,
                reconstruction_residual=None,
                raw_sha256=raw_hash,
                used_sha256=None,
                approval_owner=None,
                policy_id=policy_id,
                reason="REPAIR_APPROVAL_MISSING",
                status="REJECT",
            )
        used_values, used_vectors = np.linalg.eigh(used)
        positive = used_values > rank_cutoff
        factor = used_vectors[:, positive] * np.sqrt(used_values[positive])
        method = "dykstra_eigen"
        reason = "OWNER_APPROVED_MODEL_REPAIR"
        status = "CONDITIONAL_PASS"

    used_values = np.linalg.eigvalsh(used)
    used_rank = int(np.sum(used_values > rank_cutoff))
    repair_norm = float(np.linalg.norm(used - raw, ord="fro"))
    reconstruction = float(np.linalg.norm(factor @ factor.T - used, ord="fro"))
    reconstruction_limit = absolute_tolerance + relative_tolerance * max(
        float(np.linalg.norm(used, ord="fro")), 1.0
    )
    if reconstruction > reconstruction_limit or used_values[0] < -rank_cutoff:
        status = "REJECT"
        reason = "INTERNAL_FACTORIZATION_INCONSISTENCY"
        factor = None
    return CorrelationGateResult(
        raw_correlation=raw,
        used_correlation=used if factor is not None else None,
        raw_eigenvalues=raw_values,
        used_eigenvalues=used_values,
        raw_rank=raw_rank,
        used_rank=used_rank if factor is not None else 0,
        factor=factor,
        factorization_method=method if factor is not None else "none",
        symmetry_error=symmetry_error,
        diagonal_error=diagonal_error,
        rank_cutoff=rank_cutoff,
        repair_frobenius_norm=repair_norm,
        reconstruction_residual=reconstruction,
        raw_sha256=raw_hash,
        used_sha256=_matrix_hash(used) if factor is not None else None,
        approval_owner=approval_owner,
        policy_id=policy_id,
        reason=reason,
        status=status,
    )


def nearest_psd_correlation(
    matrix: NDArray[np.float64],
    floor: float = 1e-10,
    *,
    approval_owner: str,
) -> CorrelationGateResult:
    """Compatibility helper that returns, rather than hides, all repair evidence."""

    result = correlation_input_gate(
        matrix,
        absolute_tolerance=floor,
        relative_tolerance=floor,
        maximum_repair_norm=float(np.linalg.norm(matrix, ord="fro")) + 1.0,
        approval_owner=approval_owner,
    )
    if result.used_correlation is None:
        raise ValueError("correlation input failed the governed repair gate")
    return result


def correlated_normals(
    correlation: NDArray[np.float64],
    *,
    draws: int,
    seed: int = 31,
    approved_gate: CorrelationGateResult | None = None,
) -> CorrelatedNormalResult:
    if (
        not isinstance(draws, int)
        or isinstance(draws, bool)
        or draws < 2
        or not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise ValueError("draws must be an integer of at least two and seed an integer")
    gate = approved_gate if approved_gate is not None else correlation_input_gate(correlation)
    if gate.factor is None or gate.status not in {
        "PASS",
        "PASS_NUMERICAL",
        "CONDITIONAL_PASS",
    }:
        raise ValueError("correlation input failed the governed gate")
    if gate.status == "CONDITIONAL_PASS" and (
        approved_gate is None or approved_gate.approval_owner is None
    ):
        raise ValueError("conditional repair requires an explicitly approved gate")
    if _matrix_hash(np.asarray(correlation, dtype=float)) != gate.raw_sha256:
        raise ValueError("approved gate does not match the raw matrix")
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((draws, gate.factor.shape[1]))
    return CorrelatedNormalResult(draws=latent @ gate.factor.T, gate=gate)
