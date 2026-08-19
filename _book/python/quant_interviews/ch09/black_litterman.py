from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _timestamp(value: str, name: str) -> datetime:
    if not value or value.endswith("Z") is False:
        raise ValueError(f"{name} must be a nonempty UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    return parsed


@dataclass(frozen=True)
class ViewLineage:
    view_id: str
    author: str
    valid_time: str
    knowledge_time: str
    horizon: str
    units: str
    source_sha256: str
    approver: str
    approval_state: str = "APPROVED"

    def validate(self, decision_time: str) -> None:
        text_fields = (
            self.view_id,
            self.author,
            self.horizon,
            self.units,
            self.approver,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("view lineage text fields must be nonempty")
        knowledge = _timestamp(self.knowledge_time, "knowledge_time")
        _timestamp(self.valid_time, "valid_time")
        decision = _timestamp(decision_time, "decision_time")
        if knowledge > decision:
            raise ValueError("view knowledge_time exceeds decision_time")
        if not _SHA256.fullmatch(self.source_sha256):
            raise ValueError("view source_sha256 must be a lowercase SHA-256")
        if self.approval_state != "APPROVED":
            raise ValueError("every production view must be APPROVED")


@dataclass(frozen=True)
class BLResult:
    prior: FloatArray
    posterior_mean: FloatArray
    posterior_mean_covariance: FloatArray
    gain: FloatArray
    prior_view_surprise: FloatArray
    posterior_view_residual: FloatArray
    prior_predictive_covariance: FloatArray
    surprise_statistic: float
    surprise_degrees_of_freedom: int
    view_lineage: tuple[ViewLineage, ...]
    status: str

    @property
    def posterior_covariance(self) -> FloatArray:
        """Deprecated compatibility alias."""
        return self.posterior_mean_covariance

    @property
    def view_residual(self) -> FloatArray:
        """Compatibility alias for the posterior fitted-view residual."""
        return self.posterior_view_residual

    @property
    def view_ids(self) -> tuple[str, ...]:
        return tuple(record.view_id for record in self.view_lineage)


def _symmetric_positive_definite(
    value: FloatArray,
    name: str,
) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    if float(np.linalg.eigvalsh(matrix).min()) <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    return (matrix + matrix.T) / 2.0


def implied_equilibrium_returns(
    cov: FloatArray,
    market_weights: FloatArray,
    risk_aversion: float,
) -> FloatArray:
    matrix = _symmetric_positive_definite(cov, "covariance")
    weights = np.asarray(market_weights, dtype=float)
    if weights.shape != (matrix.shape[0],) or not np.isfinite(weights).all():
        raise ValueError("market-weight shape or values are invalid")
    if not np.isfinite(risk_aversion) or risk_aversion <= 0.0:
        raise ValueError("risk aversion must be positive")
    return risk_aversion * matrix @ weights


def posterior(
    cov: FloatArray,
    prior: FloatArray,
    tau: float,
    views: FloatArray,
    view_returns: FloatArray,
    omega: FloatArray,
    view_lineage: tuple[ViewLineage, ...],
    *,
    decision_time: str,
) -> BLResult:
    covariance = _symmetric_positive_definite(cov, "covariance")
    prior_mean = np.asarray(prior, dtype=float)
    mapping = np.asarray(views, dtype=float)
    targets = np.asarray(view_returns, dtype=float)
    uncertainty = _symmetric_positive_definite(omega, "Omega")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive")
    view_count = mapping.shape[0] if mapping.ndim == 2 else -1
    if (
        prior_mean.shape != (covariance.shape[0],)
        or mapping.shape != (view_count, covariance.shape[0])
        or targets.shape != (view_count,)
        or uncertainty.shape != (view_count, view_count)
    ):
        raise ValueError("Black-Litterman dimension mismatch")
    if not (
        np.isfinite(prior_mean).all()
        and np.isfinite(mapping).all()
        and np.isfinite(targets).all()
    ):
        raise ValueError("Black-Litterman inputs must be finite")
    lineage = tuple(view_lineage)
    if len(lineage) != view_count:
        raise ValueError("view lineage must contain one record per view")
    for record in lineage:
        record.validate(decision_time)
    identifiers = tuple(record.view_id for record in lineage)
    if len(set(identifiers)) != view_count:
        raise ValueError("view IDs must be unique")

    prior_uncertainty = tau * covariance
    middle = mapping @ prior_uncertainty @ mapping.T + uncertainty
    cross_covariance = prior_uncertainty @ mapping.T
    gain = np.linalg.solve(middle.T, cross_covariance.T).T
    prior_surprise = targets - mapping @ prior_mean
    mean = prior_mean + gain @ prior_surprise
    posterior_residual = targets - mapping @ mean
    posterior_covariance = (
        prior_uncertainty - gain @ mapping @ prior_uncertainty
    )
    posterior_covariance = (
        posterior_covariance + posterior_covariance.T
    ) / 2.0
    minimum = float(np.linalg.eigvalsh(posterior_covariance).min())
    if minimum < -1e-10 * max(float(np.linalg.norm(prior_uncertainty)), 1.0):
        raise RuntimeError("posterior mean covariance is not PSD")
    surprise = float(prior_surprise @ np.linalg.solve(middle, prior_surprise))
    degrees_of_freedom = int(np.linalg.matrix_rank(middle))
    return BLResult(
        prior=prior_mean,
        posterior_mean=mean,
        posterior_mean_covariance=posterior_covariance,
        gain=gain,
        prior_view_surprise=prior_surprise,
        posterior_view_residual=posterior_residual,
        prior_predictive_covariance=middle,
        surprise_statistic=surprise,
        surprise_degrees_of_freedom=degrees_of_freedom,
        view_lineage=lineage,
        status="PASS",
    )
