from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

from .calibration import calibration_metrics


@dataclass(frozen=True)
class DriftPolicy:
    policy_id: str
    bandwidth: float
    mmd_threshold: float
    brier_threshold: float
    minimum_persistence: int
    estimator: Literal["unbiased_u_statistic"] = "unbiased_u_statistic"

    @property
    def policy_hash(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class DriftWindowRecord:
    record_id: str
    window_id: str
    window_start: str
    window_end: str
    model_hash: str
    data_hash: str
    policy_hash: str
    evidence_id: str
    magnitude_breach: bool
    previous_record_id: str | None


@dataclass(frozen=True)
class DriftEvidence:
    reference_observations: int
    live_observations: int
    mean_shift: float
    std_ratio: float | None
    scale_status: Literal["OK", "DEGENERATE_REFERENCE"]
    mmd2: float
    bandwidth: float
    mmd_threshold: float
    mmd_estimator: str
    brier_change: float | None
    consecutive_breaches: int
    minimum_persistence: int
    evidence_id: str
    magnitude_breach: bool
    actionable: bool
    current_record: DriftWindowRecord

    @property
    def alert(self) -> bool:
        return self.actionable


def _digest(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _probability_sample(values: ArrayLike, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (
        result.ndim != 1
        or result.size < 5
        or not np.all(np.isfinite(result))
        or np.any((result < 0.0) | (result > 1.0))
    ):
        raise ValueError(f"{name} must be at least five finite probabilities.")
    return result


def reference_bandwidth(reference: ArrayLike) -> float:
    """Fit the kernel bandwidth using reference data only."""

    values = _probability_sample(reference, name="reference")
    distances = np.abs(values[:, None] - values[None, :])
    positive = distances[distances > 0.0]
    if positive.size == 0:
        raise ValueError("Reference scores are degenerate; bandwidth is undefined.")
    bandwidth = float(np.median(positive))
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("Reference bandwidth must be finite and positive.")
    return bandwidth


def rbf_mmd2(
    reference: ArrayLike,
    live: ArrayLike,
    bandwidth: float,
) -> float:
    """Unbiased two-sample MMD squared (U-statistic, diagonals excluded)."""

    x = _probability_sample(reference, name="reference").reshape(-1, 1)
    y = _probability_sample(live, name="live").reshape(-1, 1)
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive.")

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.exp(-((left - right.T) ** 2) / (2.0 * bandwidth**2))

    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    x_term = (float(kxx.sum()) - float(np.trace(kxx))) / (len(x) * (len(x) - 1))
    y_term = (float(kyy.sum()) - float(np.trace(kyy))) / (len(y) * (len(y) - 1))
    return float(x_term + y_term - 2.0 * float(kxy.mean()))


def calibrate_mmd_policy(
    reference_scores: ArrayLike,
    *,
    policy_id: str,
    minimum_persistence: int,
    brier_threshold: float = 0.03,
    false_alarm_probability: float = 0.01,
    bootstrap_draws: int = 300,
    seed: int = 0,
) -> DriftPolicy:
    """Calibrate bandwidth and threshold from a frozen reference sample."""

    reference = _probability_sample(reference_scores, name="reference_scores")
    if (
        not policy_id.strip()
        or minimum_persistence <= 0
        or not np.isfinite(brier_threshold)
        or brier_threshold < 0.0
        or not 0.0 < false_alarm_probability < 0.5
        or bootstrap_draws < 50
    ):
        raise ValueError("Invalid drift-policy calibration settings.")
    bandwidth = reference_bandwidth(reference)
    generator = np.random.default_rng(seed)
    split_size = reference.size // 2
    if split_size < 5:
        raise ValueError("Reference sample is too short for calibration splits.")
    null_statistics = np.empty(bootstrap_draws)
    for draw in range(bootstrap_draws):
        first = generator.choice(reference, size=split_size, replace=True)
        second = generator.choice(reference, size=split_size, replace=True)
        null_statistics[draw] = rbf_mmd2(first, second, bandwidth)
    threshold = float(
        np.quantile(null_statistics, 1.0 - false_alarm_probability, method="higher")
    )
    return DriftPolicy(
        policy_id=policy_id.strip(),
        bandwidth=bandwidth,
        mmd_threshold=max(threshold, 0.0),
        brier_threshold=float(brier_threshold),
        minimum_persistence=minimum_persistence,
    )


def _timestamp(value: str | pd.Timestamp, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware.")
    return timestamp.tz_convert("UTC")


def _validate_history(
    history: Sequence[DriftWindowRecord],
    *,
    current_start: pd.Timestamp,
    model_hash: str,
    policy_hash: str,
) -> None:
    record_ids: set[str] = set()
    previous: DriftWindowRecord | None = None
    for record in history:
        if record.record_id in record_ids:
            raise ValueError("Drift history contains a duplicate record ID.")
        record_ids.add(record.record_id)
        start = _timestamp(record.window_start, name="history window_start")
        end = _timestamp(record.window_end, name="history window_end")
        if start >= end:
            raise ValueError("Every drift history window requires start < end.")
        if record.model_hash != model_hash or record.policy_hash != policy_hash:
            raise ValueError("Drift history mixes model or policy versions.")
        if previous is not None:
            previous_end = _timestamp(previous.window_end, name="history window_end")
            if start != previous_end or record.previous_record_id != previous.record_id:
                raise ValueError("Drift history is not contiguous or hash-linked.")
        elif record.previous_record_id is not None:
            raise ValueError("The first supplied history record must start its chain.")
        previous = record
    if previous is not None:
        previous_end = _timestamp(previous.window_end, name="history window_end")
        if previous_end != current_start:
            raise ValueError("Current drift window is not contiguous with history.")


def drift_evidence(
    reference_scores: ArrayLike,
    live_scores: ArrayLike,
    *,
    policy: DriftPolicy,
    window_id: str,
    window_start: str | pd.Timestamp,
    window_end: str | pd.Timestamp,
    model_hash: str,
    data_hash: str,
    evidence_id: str,
    history: Sequence[DriftWindowRecord] = (),
    reference_labels: ArrayLike | None = None,
    live_labels: ArrayLike | None = None,
) -> DriftEvidence:
    """Compute persistence from an ordered immutable record chain."""

    reference = _probability_sample(reference_scores, name="reference_scores")
    live = _probability_sample(live_scores, name="live_scores")
    start = _timestamp(window_start, name="window_start")
    end = _timestamp(window_end, name="window_end")
    if start >= end:
        raise ValueError("window_start must precede window_end.")
    for name, value in {
        "window_id": window_id,
        "model_hash": model_hash,
        "data_hash": data_hash,
        "evidence_id": evidence_id,
    }.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be nonempty.")
    if (
        not np.isfinite(policy.bandwidth)
        or policy.bandwidth <= 0.0
        or not np.isfinite(policy.mmd_threshold)
        or policy.mmd_threshold < 0.0
        or not np.isfinite(policy.brier_threshold)
        or policy.brier_threshold < 0.0
        or policy.minimum_persistence <= 0
    ):
        raise ValueError("The drift policy is invalid.")
    _validate_history(
        history,
        current_start=start,
        model_hash=model_hash,
        policy_hash=policy.policy_hash,
    )
    mmd = rbf_mmd2(reference, live, policy.bandwidth)
    reference_std = float(np.std(reference, ddof=1))
    live_std = float(np.std(live, ddof=1))
    if reference_std == 0.0:
        std_ratio = None
        scale_status: Literal["OK", "DEGENERATE_REFERENCE"] = "DEGENERATE_REFERENCE"
    else:
        std_ratio = float(live_std / reference_std)
        scale_status = "OK"
    brier_change = None
    if (reference_labels is None) != (live_labels is None):
        raise ValueError("Reference and live labels must be supplied together.")
    if reference_labels is not None and live_labels is not None:
        brier_change = (
            calibration_metrics(live, live_labels).brier
            - calibration_metrics(reference, reference_labels).brier
        )
    magnitude = bool(
        scale_status == "DEGENERATE_REFERENCE"
        or mmd > policy.mmd_threshold
        or (brier_change is not None and brier_change > policy.brier_threshold)
    )
    previous_record_id = history[-1].record_id if history else None
    record_payload = {
        "window_id": window_id.strip(),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "model_hash": model_hash.strip(),
        "data_hash": data_hash.strip(),
        "policy_hash": policy.policy_hash,
        "evidence_id": evidence_id.strip(),
        "magnitude_breach": magnitude,
        "previous_record_id": previous_record_id,
    }
    current = DriftWindowRecord(record_id=_digest(record_payload), **record_payload)
    consecutive = 1 if magnitude else 0
    if magnitude:
        for record in reversed(history):
            if not record.magnitude_breach:
                break
            consecutive += 1
    actionable = magnitude and consecutive >= policy.minimum_persistence
    return DriftEvidence(
        reference_observations=int(reference.size),
        live_observations=int(live.size),
        mean_shift=float(np.mean(live) - np.mean(reference)),
        std_ratio=std_ratio,
        scale_status=scale_status,
        mmd2=mmd,
        bandwidth=policy.bandwidth,
        mmd_threshold=policy.mmd_threshold,
        mmd_estimator=policy.estimator,
        brier_change=brier_change,
        consecutive_breaches=consecutive,
        minimum_persistence=policy.minimum_persistence,
        evidence_id=evidence_id.strip(),
        magnitude_breach=magnitude,
        actionable=bool(actionable),
        current_record=current,
    )
