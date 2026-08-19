from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import pandas as pd


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_probability: float
    event_rate: float


@dataclass(frozen=True)
class CalibrationMetrics:
    brier: float
    log_loss: float
    mean_probability: float
    event_rate: float
    observations: int
    epsilon: float
    binned_brier: float
    reliability: float
    resolution: float
    uncertainty: float
    decomposition_residual: float
    reliability_bins: tuple[ReliabilityBin, ...]


@dataclass(frozen=True)
class TemporalCalibrationResult:
    method: str
    calibration_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    raw_evaluation_scores: NDArray[np.float64]
    calibrated_probabilities: NDArray[np.float64]
    raw_probability_metrics: CalibrationMetrics
    calibrated_metrics: CalibrationMetrics
    lineage_digest: str
    calibration_start: str
    calibration_end: str
    evaluation_start: str
    evaluation_end: str


def _binary_inputs(
    probabilities: ArrayLike,
    labels: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    probability = np.asarray(probabilities, dtype=float)
    label = np.asarray(labels)
    if (
        probability.ndim != 1
        or label.ndim != 1
        or probability.size != label.size
        or probability.size == 0
    ):
        raise ValueError("Probabilities and labels must be aligned vectors.")
    if not np.all(np.isfinite(probability)):
        raise ValueError("Probabilities must be finite.")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1].")
    if not np.all(np.isin(label, (0, 1))):
        raise ValueError("Labels must be binary.")
    return probability, np.asarray(label, dtype=np.int64)


def _score_inputs(
    scores: ArrayLike,
    labels: ArrayLike,
    *,
    minimum: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    score = np.asarray(scores, dtype=float)
    label = np.asarray(labels)
    if (
        score.ndim != 1
        or label.ndim != 1
        or score.size != label.size
        or score.size < minimum
    ):
        raise ValueError(f"At least {minimum} aligned score-label pairs are required.")
    if not np.all(np.isfinite(score)):
        raise ValueError("Calibration scores must be finite.")
    if not np.all(np.isin(label, (0, 1))) or np.unique(label).size != 2:
        raise ValueError("Calibration requires both binary outcome classes.")
    return score, np.asarray(label, dtype=np.int64)


def _epsilon(value: float) -> float:
    if not np.isfinite(value) or not 0.0 < value < 0.5:
        raise ValueError("epsilon must lie strictly between zero and one half.")
    return float(value)


def calibration_metrics(
    probabilities: ArrayLike,
    labels: ArrayLike,
    *,
    epsilon: float = 1.0e-6,
    bins: int = 10,
) -> CalibrationMetrics:
    probability, label = _binary_inputs(probabilities, labels)
    governed_epsilon = _epsilon(epsilon)
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ValueError("bins must be an integer of at least two.")
    clipped = np.clip(probability, governed_epsilon, 1.0 - governed_epsilon)
    brier = float(np.mean((label - probability) ** 2))
    log_loss = float(
        -np.mean(
            label * np.log(clipped)
            + (1 - label) * np.log1p(-clipped)
        )
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    membership = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    membership = np.maximum(membership, 0)
    base_rate = float(np.mean(label))
    reliability = 0.0
    resolution = 0.0
    binned_brier = 0.0
    reliability_rows: list[ReliabilityBin] = []
    for bin_number in range(bins):
        mask = membership == bin_number
        count = int(np.sum(mask))
        if count == 0:
            continue
        mean_probability = float(np.mean(probability[mask]))
        event_rate = float(np.mean(label[mask]))
        weight = count / probability.size
        reliability += weight * (mean_probability - event_rate) ** 2
        resolution += weight * (event_rate - base_rate) ** 2
        binned_brier += weight * (
            event_rate * (1.0 - mean_probability) ** 2
            + (1.0 - event_rate) * mean_probability**2
        )
        reliability_rows.append(
            ReliabilityBin(
                lower=float(edges[bin_number]),
                upper=float(edges[bin_number + 1]),
                count=count,
                mean_probability=mean_probability,
                event_rate=event_rate,
            )
        )
    uncertainty = base_rate * (1.0 - base_rate)
    decomposed = reliability - resolution + uncertainty
    return CalibrationMetrics(
        brier=brier,
        log_loss=log_loss,
        mean_probability=float(np.mean(probability)),
        event_rate=float(np.mean(label)),
        observations=int(probability.size),
        epsilon=governed_epsilon,
        binned_brier=float(binned_brier),
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=float(uncertainty),
        decomposition_residual=float(binned_brier - decomposed),
        reliability_bins=tuple(reliability_rows),
    )


class PlattCalibrator:
    def __init__(self, *, regularization_c: float = 1.0, epsilon: float = 1.0e-6) -> None:
        if not np.isfinite(regularization_c) or regularization_c <= 0.0:
            raise ValueError("regularization_c must be finite and positive.")
        self._epsilon = _epsilon(epsilon)
        self._model = LogisticRegression(C=regularization_c, solver="lbfgs")
        self._fitted = False

    def fit(self, scores: ArrayLike, labels: ArrayLike) -> "PlattCalibrator":
        score, label = _score_inputs(scores, labels, minimum=20)
        self._model.fit(score.reshape(-1, 1), label)
        self._fitted = True
        return self

    def predict(self, scores: ArrayLike) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("The calibrator must be fitted before prediction.")
        score = np.asarray(scores, dtype=float)
        if score.ndim != 1 or score.size == 0 or not np.all(np.isfinite(score)):
            raise ValueError("Prediction scores must be a finite nonempty vector.")
        probability = self._model.predict_proba(score.reshape(-1, 1))[:, 1]
        return np.clip(probability, self._epsilon, 1.0 - self._epsilon)


class IsotonicCalibrator:
    def __init__(self, *, epsilon: float = 1.0e-6) -> None:
        self._epsilon = _epsilon(epsilon)
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, scores: ArrayLike, labels: ArrayLike) -> "IsotonicCalibrator":
        score, label = _score_inputs(scores, labels, minimum=40)
        if np.unique(score).size < 4:
            raise ValueError("Isotonic calibration needs at least four score levels.")
        self._model.fit(score, label)
        self._fitted = True
        return self

    def predict(self, scores: ArrayLike) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("The calibrator must be fitted before prediction.")
        score = np.asarray(scores, dtype=float)
        if score.ndim != 1 or score.size == 0 or not np.all(np.isfinite(score)):
            raise ValueError("Prediction scores must be a finite nonempty vector.")
        probability = np.asarray(self._model.predict(score), dtype=float)
        return np.clip(probability, self._epsilon, 1.0 - self._epsilon)


def temporal_calibration(
    raw_scores: ArrayLike,
    labels: ArrayLike,
    *,
    score_timestamps: ArrayLike,
    label_known_timestamps: ArrayLike,
    model_fit_ids: ArrayLike,
    calibration_indices: ArrayLike,
    evaluation_indices: ArrayLike,
    method: Literal["platt", "isotonic"] = "platt",
    epsilon: float = 1.0e-6,
    bins: int = 10,
) -> TemporalCalibrationResult:
    """Fit on one ordered period and evaluate once on a later untouched period."""

    scores = np.asarray(raw_scores, dtype=float)
    outcome = np.asarray(labels)
    if (
        scores.ndim != 1
        or outcome.ndim != 1
        or scores.shape != outcome.shape
        or not np.all(np.isfinite(scores))
        or not np.all(np.isin(outcome, (0, 1)))
    ):
        raise ValueError("raw_scores and binary labels must be aligned finite vectors.")
    score_times = pd.DatetimeIndex(pd.to_datetime(score_timestamps, utc=True, errors="raise"))
    known_times = pd.DatetimeIndex(
        pd.to_datetime(label_known_timestamps, utc=True, errors="raise")
    )
    fit_ids = np.asarray(model_fit_ids, dtype=object)
    if len(score_times) != scores.size or len(known_times) != scores.size or fit_ids.shape != scores.shape:
        raise ValueError("Temporal lineage fields must align with scores.")
    if score_times.has_duplicates or not score_times.is_monotonic_increasing:
        raise ValueError("score_timestamps must be unique and increasing.")
    if any(not isinstance(value, str) or not value.strip() for value in fit_ids):
        raise ValueError("Every score requires a nonempty model_fit_id.")

    def governed_indices(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
        index = np.asarray(values)
        if (
            index.ndim != 1
            or index.size == 0
            or not np.issubdtype(index.dtype, np.integer)
            or np.unique(index).size != index.size
            or np.any(index < 0)
            or np.any(index >= scores.size)
        ):
            raise ValueError(f"{name} must be unique in-range integer indices.")
        return np.asarray(index, dtype=np.int64)

    calibration = governed_indices(calibration_indices, name="calibration_indices")
    evaluation = governed_indices(evaluation_indices, name="evaluation_indices")
    if np.intersect1d(calibration, evaluation).size:
        raise ValueError("Calibration and evaluation periods must be disjoint.")
    if score_times[calibration].max() >= score_times[evaluation].min():
        raise ValueError("Calibration scores must strictly precede evaluation scores.")
    if known_times[calibration].max() > score_times[evaluation].min():
        raise ValueError("Calibration labels were not known before evaluation began.")
    calibrator: PlattCalibrator | IsotonicCalibrator
    if method == "platt":
        calibrator = PlattCalibrator(epsilon=epsilon)
    elif method == "isotonic":
        calibrator = IsotonicCalibrator(epsilon=epsilon)
    else:
        raise ValueError("method must be platt or isotonic.")
    calibrator.fit(scores[calibration], outcome[calibration])
    probability = calibrator.predict(scores[evaluation])
    raw_probability = 1.0 / (1.0 + np.exp(-np.clip(scores[evaluation], -700.0, 700.0)))
    payload = {
        "method": method,
        "calibration_indices": calibration.tolist(),
        "evaluation_indices": evaluation.tolist(),
        "model_fit_ids": [str(fit_ids[index]) for index in np.r_[calibration, evaluation]],
        "score_timestamps": [score_times[index].isoformat() for index in np.r_[calibration, evaluation]],
        "label_known_timestamps": [known_times[index].isoformat() for index in calibration],
        "epsilon": _epsilon(epsilon),
    }
    lineage_digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TemporalCalibrationResult(
        method=method,
        calibration_indices=tuple(int(value) for value in calibration),
        evaluation_indices=tuple(int(value) for value in evaluation),
        raw_evaluation_scores=scores[evaluation].copy(),
        calibrated_probabilities=probability,
        raw_probability_metrics=calibration_metrics(
            raw_probability,
            outcome[evaluation],
            epsilon=epsilon,
            bins=bins,
        ),
        calibrated_metrics=calibration_metrics(
            probability,
            outcome[evaluation],
            epsilon=epsilon,
            bins=bins,
        ),
        lineage_digest=lineage_digest,
        calibration_start=score_times[calibration].min().isoformat(),
        calibration_end=score_times[calibration].max().isoformat(),
        evaluation_start=score_times[evaluation].min().isoformat(),
        evaluation_end=score_times[evaluation].max().isoformat(),
    )
