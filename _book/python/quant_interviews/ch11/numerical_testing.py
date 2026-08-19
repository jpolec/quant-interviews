from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MonteCarloCheck:
    estimate: float
    standard_error: float
    lower_bound: float
    upper_bound: float
    absolute_error: float
    passed: bool
    sample_size: int
    seed: int
    confidence_level: float
    critical_value: float
    absolute_tolerance: float
    interval_method: str
    assumptions: str


def check_monte_carlo_mean(
    sampler: Callable[[np.random.Generator, int], np.ndarray],
    expected_mean: float,
    sample_size: int,
    seed: int,
    *,
    confidence_level: float = 0.999,
    absolute_tolerance: float = 1e-12,
    iid_finite_variance_verified: bool,
) -> MonteCarloCheck:
    if sample_size < 30:
        raise ValueError("sample_size must meet the configured normal-approximation minimum")
    if not iid_finite_variance_verified:
        raise ValueError(
            "the normal interval requires a separately verified IID finite-variance contract"
        )
    if not math.isfinite(expected_mean):
        raise ValueError("expected_mean must be finite")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be finite and non-negative")
    rng = np.random.default_rng(seed)
    values = np.asarray(sampler(rng, sample_size), dtype=np.float64)
    if values.shape != (sample_size,):
        raise ValueError("sampler returned an unexpected shape")
    if np.any(~np.isfinite(values)):
        raise ValueError("sampler returned non-finite values")
    estimate = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(sample_size))
    if not math.isfinite(estimate) or not math.isfinite(standard_error):
        raise FloatingPointError("Monte Carlo summary is non-finite")
    critical = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    half_width = critical * standard_error + absolute_tolerance
    lower = estimate - half_width
    upper = estimate + half_width
    return MonteCarloCheck(
        estimate=estimate,
        standard_error=standard_error,
        lower_bound=lower,
        upper_bound=upper,
        absolute_error=abs(estimate - expected_mean),
        passed=lower <= expected_mean <= upper,
        sample_size=sample_size,
        seed=seed,
        confidence_level=confidence_level,
        critical_value=critical,
        absolute_tolerance=absolute_tolerance,
        interval_method="normal interval plus additive absolute acceptance tolerance",
        assumptions="IID_FINITE_VARIANCE_VERIFIED",
    )


def assert_metamorphic_relation(
    transform_input: Callable[[np.ndarray], np.ndarray],
    transform_output: Callable[[float], float],
    function: Callable[[np.ndarray], float],
    values: np.ndarray,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be finite and non-negative")
    original = np.array(values, dtype=np.float64, copy=True)
    transform_argument = original.copy()
    transformed = np.asarray(transform_input(transform_argument), dtype=np.float64)
    if not np.array_equal(transform_argument, original, equal_nan=True):
        raise AssertionError("input transform mutated its argument")
    if np.any(~np.isfinite(original)) or np.any(~np.isfinite(transformed)):
        raise ValueError("metamorphic inputs must be finite")
    base_argument = original.copy()
    expected = float(transform_output(float(function(base_argument))))
    if not np.array_equal(base_argument, original, equal_nan=True):
        raise AssertionError("function mutated the base fixture")
    transformed_argument = transformed.copy()
    observed = float(function(transformed_argument))
    if not np.array_equal(transformed_argument, transformed, equal_nan=True):
        raise AssertionError("function mutated the transformed fixture")
    if not math.isclose(observed, expected, rel_tol=rtol, abs_tol=atol):
        raise AssertionError("metamorphic relation failed")


def assert_translation_invariance(
    pricer: Callable[[np.ndarray], float],
    values: np.ndarray,
    shift: float,
) -> None:
    assert_metamorphic_relation(
        lambda x: x + shift,
        lambda result: result + shift,
        pricer,
        values,
    )


def differential_test(
    reference: Callable[[np.ndarray], np.ndarray],
    candidate: Callable[[np.ndarray], np.ndarray],
    values: np.ndarray,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be finite and non-negative")
    original = np.array(values, copy=True)
    reference_argument = original.copy()
    candidate_argument = original.copy()
    expected = np.asarray(reference(reference_argument), dtype=np.float64)
    if not np.array_equal(reference_argument, original, equal_nan=True):
        raise AssertionError("reference mutated its input fixture")
    observed = np.asarray(candidate(candidate_argument), dtype=np.float64)
    if not np.array_equal(candidate_argument, original, equal_nan=True):
        raise AssertionError("candidate mutated its input fixture")
    if expected.shape != observed.shape:
        raise AssertionError("candidate returned a different shape")
    if np.any(~np.isfinite(expected)) or np.any(~np.isfinite(observed)):
        raise AssertionError("non-finite output requires an explicit mask contract")
    if not np.allclose(expected, observed, rtol=rtol, atol=atol):
        raise AssertionError("differential test failed")
