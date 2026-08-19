from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class StabilitySelectionResult:
    selection_probability: NDArray[np.float64]
    sign_probability: NDArray[np.float64]
    selected: NDArray[np.bool_]
    convergence_failures: int
    resamples: int
    threshold: float
    successful_resamples: int
    denominator_policy: str


def blocked_stability_selection(
    x: ArrayLike,
    y: ArrayLike,
    *,
    block_size: int,
    n_resamples: int,
    alpha: float,
    l1_ratio: float = 0.8,
    threshold: float = 0.7,
    seed: int = 0,
    convergence_policy: str = "reject_any",
) -> StabilitySelectionResult:
    """Select features with scaling fitted independently inside each resample."""

    features = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    if (
        features.ndim != 2
        or target.ndim != 1
        or features.shape[0] != target.size
        or target.size < 20
        or features.shape[1] == 0
    ):
        raise ValueError("Invalid feature and target dimensions.")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(target)):
        raise ValueError("Features and target must be finite.")
    if (
        block_size <= 1
        or block_size >= target.size
        or n_resamples <= 0
        or not np.isfinite(alpha)
        or alpha <= 0.0
        or not 0.0 <= l1_ratio <= 1.0
        or not 0.0 <= threshold <= 1.0
        or convergence_policy != "reject_any"
    ):
        raise ValueError("Invalid stability-selection configuration.")

    generator = np.random.default_rng(seed)
    observations, feature_count = features.shape
    selected_count = np.zeros(feature_count)
    positive_count = np.zeros(feature_count)
    blocks_per_sample = int(np.ceil(observations / block_size))
    starts = np.arange(0, observations - block_size + 1)
    convergence_failures = 0
    for _ in range(n_resamples):
        chosen = generator.choice(starts, size=blocks_per_sample, replace=True)
        index = np.concatenate(
            [
                np.arange(start, min(start + block_size, observations))
                for start in chosen
            ]
        )[:observations]
        model = make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=20_000,
                tol=1.0e-8,
            ),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features[index], target[index])
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            convergence_failures += 1
            continue
        coefficients = model.named_steps["elasticnet"].coef_
        selected = np.abs(coefficients) > 1.0e-10
        selected_count += selected
        positive_count += coefficients > 1.0e-10
    successful = n_resamples - convergence_failures
    if convergence_failures:
        raise RuntimeError(
            "Stability selection rejected because at least one governed fit "
            f"failed to converge ({convergence_failures}/{n_resamples})."
        )
    probability = selected_count / n_resamples
    sign_probability = np.divide(
        positive_count,
        selected_count,
        out=np.zeros_like(positive_count),
        where=selected_count > 0,
    )
    return StabilitySelectionResult(
        selection_probability=probability,
        sign_probability=sign_probability,
        selected=probability >= threshold,
        convergence_failures=convergence_failures,
        resamples=n_resamples,
        threshold=threshold,
        successful_resamples=successful,
        denominator_policy="all_attempted_resamples",
    )


def blocked_stability_selection_by_outer_fold(
    x: ArrayLike,
    y: ArrayLike,
    outer_train_indices: list[ArrayLike],
    *,
    block_size: int,
    n_resamples: int,
    alpha: float,
    l1_ratio: float = 0.8,
    threshold: float = 0.7,
    seed: int = 0,
) -> tuple[StabilitySelectionResult, ...]:
    """Run selection independently inside each immutable outer training fold."""

    features = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    results: list[StabilitySelectionResult] = []
    for fold_number, indices in enumerate(outer_train_indices):
        index = np.asarray(indices)
        if (
            index.ndim != 1
            or index.size == 0
            or not np.issubdtype(index.dtype, np.integer)
            or np.unique(index).size != index.size
            or np.any(index < 0)
            or np.any(index >= target.size)
        ):
            raise ValueError(f"Invalid outer training indices for fold {fold_number}.")
        results.append(
            blocked_stability_selection(
                features[index],
                target[index],
                block_size=block_size,
                n_resamples=n_resamples,
                alpha=alpha,
                l1_ratio=l1_ratio,
                threshold=threshold,
                seed=seed + fold_number,
            )
        )
    if not results:
        raise ValueError("At least one outer training fold is required.")
    return tuple(results)
