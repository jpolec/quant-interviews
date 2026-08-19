from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import inf
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


@dataclass(frozen=True)
class HMMFilterResult:
    predicted: NDArray[np.float64]
    filtered: NDArray[np.float64]
    log_likelihood: float
    state_order: NDArray[np.int64]
    expected_durations: NDArray[np.float64]
    initial_probability_timing: str


@dataclass(frozen=True)
class StateLabelMatch:
    permutation: NDArray[np.int64]
    assigned_costs: NDArray[np.float64]
    total_cost: float
    second_best_margin: float
    status: Literal["PASS", "REJECT_UNIDENTIFIED"]
    policy_version: str


def canonical_state_order(
    means: ArrayLike,
    stds: ArrayLike,
) -> NDArray[np.int64]:
    mean_array = np.asarray(means, dtype=float)
    std_array = np.asarray(stds, dtype=float)
    if (
        mean_array.ndim != 1
        or std_array.ndim != 1
        or mean_array.shape != std_array.shape
        or mean_array.size == 0
        or not np.all(np.isfinite(mean_array))
        or not np.all(np.isfinite(std_array))
        or np.any(std_array <= 0.0)
    ):
        raise ValueError("State means and standard deviations must be valid vectors.")
    return np.asarray(np.lexsort((mean_array, std_array)), dtype=np.int64)


def match_state_labels(
    previous_means: ArrayLike,
    previous_stds: ArrayLike,
    current_means: ArrayLike,
    current_stds: ArrayLike,
    *,
    ambiguity_margin: float = 1.0e-6,
    policy_version: str = "hmm-label-match-v1",
) -> StateLabelMatch:
    """Match refitted states to the prior release by a scaled Hungarian cost."""

    previous_mean = np.asarray(previous_means, dtype=float)
    previous_std = np.asarray(previous_stds, dtype=float)
    current_mean = np.asarray(current_means, dtype=float)
    current_std = np.asarray(current_stds, dtype=float)
    if (
        previous_mean.ndim != 1
        or previous_std.shape != previous_mean.shape
        or current_mean.shape != previous_mean.shape
        or current_std.shape != previous_mean.shape
        or not all(
            np.all(np.isfinite(array))
            for array in (
                previous_mean,
                previous_std,
                current_mean,
                current_std,
            )
        )
        or np.any(previous_std <= 0.0)
        or np.any(current_std <= 0.0)
    ):
        raise ValueError("Previous and current HMM parameters must align.")
    scale = np.maximum(previous_std[:, None], 1.0e-12)
    cost = (
        ((previous_mean[:, None] - current_mean[None, :]) / scale) ** 2
        + (np.log(previous_std[:, None] / current_std[None, :])) ** 2
    )
    if not np.isfinite(ambiguity_margin) or ambiguity_margin < 0.0:
        raise ValueError("ambiguity_margin must be finite and nonnegative.")
    if not policy_version.strip():
        raise ValueError("policy_version must be nonempty.")
    previous_index, current_index = linear_sum_assignment(cost)
    order = np.empty(previous_mean.size, dtype=np.int64)
    order[previous_index] = current_index
    assigned_costs = cost[np.arange(previous_mean.size), order]
    best = float(np.sum(assigned_costs))
    # Releases normally have few states.  Enumerating assignments makes the
    # ambiguity margin exact and auditable instead of solver-dependent.
    alternative_costs = [
        float(sum(cost[row, column] for row, column in enumerate(candidate)))
        for candidate in permutations(range(previous_mean.size))
        if tuple(candidate) != tuple(int(value) for value in order)
    ]
    second_best = min(alternative_costs, default=inf)
    margin = second_best - best
    status: Literal["PASS", "REJECT_UNIDENTIFIED"] = (
        "PASS" if margin > ambiguity_margin else "REJECT_UNIDENTIFIED"
    )
    return StateLabelMatch(
        permutation=order,
        assigned_costs=np.asarray(assigned_costs, dtype=float),
        total_cost=best,
        second_best_margin=float(margin),
        status=status,
        policy_version=policy_version,
    )


def expected_state_durations(transition: ArrayLike) -> NDArray[np.float64]:
    """Return geometric durations, with absorbing states reported as infinity."""

    matrix = np.asarray(transition, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0.0)
        or not np.allclose(matrix.sum(axis=1), 1.0, atol=1.0e-12, rtol=1.0e-12)
    ):
        raise ValueError("transition must be a finite row-stochastic square matrix.")
    diagonal = np.diag(matrix)
    return np.asarray(
        [inf if value == 1.0 else 1.0 / (1.0 - value) for value in diagonal],
        dtype=float,
    )


def gaussian_hmm_filter(
    observations: ArrayLike,
    transition: ArrayLike,
    means: ArrayLike,
    stds: ArrayLike,
    initial_predicted_probabilities: ArrayLike,
) -> HMMFilterResult:
    observations_array = np.asarray(observations, dtype=float)
    transition_array = np.asarray(transition, dtype=float)
    mean_array = np.asarray(means, dtype=float)
    std_array = np.asarray(stds, dtype=float)
    initial_array = np.asarray(initial_predicted_probabilities, dtype=float)
    if mean_array.ndim != 1 or std_array.ndim != 1 or initial_array.ndim != 1:
        raise ValueError("HMM parameter vectors must be one-dimensional.")
    states = mean_array.size
    if (
        observations_array.ndim != 1
        or observations_array.size == 0
        or transition_array.shape != (states, states)
        or std_array.shape != (states,)
        or initial_array.shape != (states,)
    ):
        raise ValueError("Incompatible HMM dimensions.")
    if not all(
        np.all(np.isfinite(array))
        for array in (
            observations_array,
            transition_array,
            mean_array,
            std_array,
            initial_array,
        )
    ):
        raise ValueError("All HMM inputs must be finite.")
    if (
        np.any(std_array <= 0.0)
        or np.any(transition_array < 0.0)
        or np.any(initial_array < 0.0)
    ):
        raise ValueError("Probabilities must be nonnegative and scales positive.")
    if not np.allclose(
        transition_array.sum(axis=1), 1.0, atol=1.0e-12, rtol=1.0e-12
    ) or not np.isclose(
        initial_array.sum(), 1.0, atol=1.0e-12, rtol=1.0e-12
    ):
        raise ValueError("Transition rows and initial weights must sum to one.")

    predicted = np.empty((observations_array.size, states))
    filtered = np.empty((observations_array.size, states))
    log_likelihood = 0.0
    current = initial_array
    for index, value in enumerate(observations_array):
        # The input is explicitly the state distribution predicted for the
        # first emission.  Transitions begin before the second observation.
        prior = current @ transition_array if index > 0 else current
        log_density = (
            -0.5 * ((value - mean_array) / std_array) ** 2
            - np.log(std_array)
            - 0.5 * np.log(2.0 * np.pi)
        )
        log_prior = np.where(prior > 0.0, np.log(prior), -np.inf)
        log_weights = log_prior + log_density
        log_scale = float(logsumexp(log_weights))
        if not np.isfinite(log_scale):
            raise FloatingPointError("HMM likelihood is non-finite.")
        current = np.exp(log_weights - log_scale)
        predicted[index] = prior
        filtered[index] = current
        log_likelihood += log_scale
    return HMMFilterResult(
        predicted=predicted,
        filtered=filtered,
        log_likelihood=float(log_likelihood),
        state_order=canonical_state_order(mean_array, std_array),
        expected_durations=expected_state_durations(transition_array),
        initial_probability_timing="predicted_before_first_emission",
    )
