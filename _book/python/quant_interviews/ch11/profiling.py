from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import platform
import re
import sys
from time import perf_counter
from typing import Callable

import numpy as np

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ProfileStats:
    label: str
    samples_seconds: tuple[float, ...]
    median_seconds: float
    p95_seconds: float
    p99_seconds: float
    result: float
    result_hash: str
    measurement_hash: str
    input_hash: str
    warmup_results: tuple[float, ...]
    warmups: int
    repeats: int
    percentile_method: str
    environment: tuple[tuple[str, str], ...]
    clock: str


def profile(
    label: str,
    fn: Callable[[], float],
    *,
    repeats: int = 11,
    warmups: int = 2,
    input_hash: str,
) -> ProfileStats:
    if not label:
        raise ValueError("label is required")
    if repeats < 3:
        raise ValueError("repeats must be at least three")
    if warmups < 1:
        raise ValueError("warmups must be at least one")
    if SHA256_PATTERN.fullmatch(input_hash) is None:
        raise ValueError("input_hash must be a lowercase SHA-256 digest")
    warmup_results: list[float] = []
    for _ in range(warmups):
        warmup_result = float(fn())
        if not math.isfinite(warmup_result):
            raise FloatingPointError("warm-up produced a non-finite result")
        warmup_results.append(warmup_result)
    samples: list[float] = []
    results: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        result = float(fn())
        samples.append(perf_counter() - start)
        results.append(result)
    if not np.all(np.isfinite(results)):
        raise FloatingPointError("profiled function produced a non-finite result")
    if any(result.hex() != results[0].hex() for result in results[1:]):
        raise AssertionError("profiled result checksum changed across repetitions")
    if any(result.hex() != results[0].hex() for result in warmup_results):
        raise AssertionError("profiled result checksum changed after warm-up")
    if np.any(~np.isfinite(samples)) or np.any(np.asarray(samples) <= 0.0):
        raise FloatingPointError("profile timing sample is invalid")
    # The semantic result hash is independent of the chosen repeat count.
    digest = sha256(np.asarray([results[0]], dtype="<f8").tobytes()).hexdigest()
    environment = (
        ("python_implementation", platform.python_implementation()),
        ("python_version", platform.python_version()),
        ("numpy_version", np.__version__),
        ("platform", sys.platform),
        ("machine", platform.machine()),
        ("processor", platform.processor() or "unreported"),
    )
    measurement_hash = sha256(
        json.dumps(
            {
                "clock": "time.perf_counter",
                "environment": environment,
                "input_hash": input_hash,
                "label": label,
                "result_hash": digest,
                "repeats": repeats,
                "samples_hex": [sample.hex() for sample in samples],
                "schema_version": 1,
                "quantile_method": "NumPy linear interpolation (Hyndman-Fan type 7)",
                "warmups": warmups,
                "warmup_results_hex": [result.hex() for result in warmup_results],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return ProfileStats(
        label=label,
        samples_seconds=tuple(samples),
        median_seconds=float(np.median(samples)),
        p95_seconds=float(np.quantile(samples, 0.95, method="linear")),
        p99_seconds=float(np.quantile(samples, 0.99, method="linear")),
        result=results[0],
        result_hash=digest,
        measurement_hash=measurement_hash,
        input_hash=input_hash,
        warmup_results=tuple(warmup_results),
        warmups=warmups,
        repeats=repeats,
        percentile_method="NumPy linear interpolation (Hyndman-Fan type 7)",
        environment=environment,
        clock="time.perf_counter",
    )


def _validated_variance_inputs(
    weights: np.ndarray,
    covariance: np.ndarray,
    *,
    symmetry_tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=np.float64)
    sigma = np.asarray(covariance, dtype=np.float64)
    if w.ndim != 1 or sigma.ndim != 2 or sigma.shape != (w.size, w.size):
        raise ValueError("weights must be 1-D and covariance square with matching dimension")
    if np.any(~np.isfinite(w)) or np.any(~np.isfinite(sigma)):
        raise ValueError("inputs must be finite")
    scale = max(float(np.max(np.abs(sigma), initial=0.0)), 1e-300)
    if float(np.max(np.abs(sigma - sigma.T))) > symmetry_tolerance * scale:
        raise ValueError("covariance must be symmetric")
    minimum = float(np.linalg.eigvalsh((sigma + sigma.T) / 2.0)[0])
    if minimum < -symmetry_tolerance * scale:
        raise ValueError("covariance is materially indefinite")
    return w, sigma


def portfolio_variance_loop(weights: np.ndarray, covariance: np.ndarray) -> float:
    w, sigma = _validated_variance_inputs(weights, covariance)
    total = 0.0
    for i in range(w.size):
        for j in range(w.size):
            total += w[i] * sigma[i, j] * w[j]
    negative_tolerance = 1e-12 * max(
        float(np.max(np.abs(sigma), initial=0.0))
        * float(np.dot(np.abs(w), np.abs(w))),
        1e-300,
    )
    if total < -negative_tolerance:
        raise FloatingPointError("portfolio variance is materially negative")
    if not math.isfinite(float(total)):
        raise FloatingPointError("portfolio variance is non-finite")
    return max(float(total), 0.0)


def portfolio_variance_vectorized(weights: np.ndarray, covariance: np.ndarray) -> float:
    w, sigma = _validated_variance_inputs(weights, covariance)
    total = float(w @ sigma @ w)
    negative_tolerance = 1e-12 * max(
        float(np.max(np.abs(sigma), initial=0.0))
        * float(np.dot(np.abs(w), np.abs(w))),
        1e-300,
    )
    if total < -negative_tolerance:
        raise FloatingPointError("portfolio variance is materially negative")
    if not math.isfinite(total):
        raise FloatingPointError("portfolio variance is non-finite")
    return max(total, 0.0)


def amdahl_speedup_from_serial_fraction(serial_fraction: float, workers: int) -> float:
    if not 0.0 <= serial_fraction <= 1.0:
        raise ValueError("serial_fraction must be in [0, 1]")
    if workers < 1:
        raise ValueError("workers must be positive")
    return 1.0 / (serial_fraction + (1.0 - serial_fraction) / workers)


def amdahl_asymptotic_limit(serial_fraction: float) -> float:
    if not 0.0 <= serial_fraction <= 1.0:
        raise ValueError("serial_fraction must be in [0, 1]")
    return math.inf if serial_fraction == 0.0 else 1.0 / serial_fraction


# Backward-compatible name retained for existing callers; it is a finite-worker speedup.
amdahl_limit = amdahl_speedup_from_serial_fraction
