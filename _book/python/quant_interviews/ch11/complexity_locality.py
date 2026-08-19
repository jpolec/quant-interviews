from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import platform
import re
import sys
from time import perf_counter
from typing import Callable

import numpy as np

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class BenchmarkStats:
    label: str
    samples_seconds: tuple[float, ...]
    median_seconds: float
    p95_seconds: float
    p99_seconds: float
    elements: int
    result: float
    warmup_results: tuple[float, ...]
    warmups: int
    repeats: int
    result_hash: str
    measurement_hash: str
    input_hash: str
    clock: str
    percentile_method: str
    environment: tuple[tuple[str, str], ...]

    @property
    def median_nanoseconds_per_element(self) -> float:
        if self.elements <= 0:
            raise ValueError("elements must be positive")
        return 1e9 * self.median_seconds / self.elements


def timed(
    label: str,
    fn: Callable[[], float],
    elements: int,
    *,
    repeats: int = 9,
    warmups: int = 2,
    input_hash: str,
) -> BenchmarkStats:
    if not label:
        raise ValueError("label is required")
    if repeats < 3:
        raise ValueError("repeats must be at least three")
    if warmups < 1:
        raise ValueError("warmups must be at least one")
    if elements <= 0:
        raise ValueError("elements must be positive")
    if SHA256_PATTERN.fullmatch(input_hash) is None:
        raise ValueError("input_hash must be a lowercase SHA-256 digest")
    warmup_results = [float(fn()) for _ in range(warmups)]
    if warmup_results and not np.all(np.isfinite(warmup_results)):
        raise FloatingPointError("benchmark warm-up produced a non-finite result")
    samples: list[float] = []
    results: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        result = float(fn())
        samples.append(perf_counter() - start)
        results.append(result)
    if not np.all(np.isfinite(results)):
        raise FloatingPointError("benchmark produced a non-finite result")
    if any(result.hex() != results[0].hex() for result in results[1:]):
        raise AssertionError("benchmark result checksum changed across repetitions")
    if any(result.hex() != results[0].hex() for result in warmup_results):
        raise AssertionError("benchmark result checksum changed after warm-up")
    if np.any(~np.isfinite(samples)) or np.any(np.asarray(samples) <= 0.0):
        raise FloatingPointError("benchmark timing sample is invalid")
    result_hash = sha256(
        np.asarray([results[0]], dtype="<f8").tobytes()
    ).hexdigest()
    environment = (
        ("python_implementation", platform.python_implementation()),
        ("python_version", platform.python_version()),
        ("numpy_version", np.__version__),
        ("platform", sys.platform),
        ("machine", platform.machine()),
        ("processor", platform.processor() or "unreported"),
    )
    measurement_payload = {
        "clock": "time.perf_counter",
        "environment": environment,
        "input_hash": input_hash,
        "label": label,
        "result_hash": result_hash,
        "repeats": repeats,
        "samples_hex": [sample.hex() for sample in samples],
        "schema_version": 1,
        "quantile_method": "NumPy linear interpolation (Hyndman-Fan type 7)",
        "warmups": warmups,
        "warmup_results_hex": [result.hex() for result in warmup_results],
    }
    measurement_hash = sha256(
        json.dumps(
            measurement_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return BenchmarkStats(
        label=label,
        samples_seconds=tuple(samples),
        median_seconds=float(np.median(samples)),
        p95_seconds=float(np.quantile(samples, 0.95, method="linear")),
        p99_seconds=float(np.quantile(samples, 0.99, method="linear")),
        elements=elements,
        result=results[0],
        warmup_results=tuple(warmup_results),
        warmups=warmups,
        repeats=repeats,
        result_hash=result_hash,
        measurement_hash=measurement_hash,
        input_hash=input_hash,
        clock="time.perf_counter",
        percentile_method="NumPy linear interpolation (Hyndman-Fan type 7)",
        environment=environment,
    )


def row_major_python_sum(matrix: np.ndarray) -> float:
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a NumPy array; implicit conversion is forbidden")
    a = matrix
    if a.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if a.dtype != np.float64 or not a.flags.c_contiguous:
        raise ValueError("matrix must be a C-contiguous float64 array")
    if not np.all(np.isfinite(a)):
        raise ValueError("matrix must be finite")
    total = 0.0
    rows, cols = a.shape
    for i in range(rows):
        for j in range(cols):
            total += float(a[i, j])
    return total


def column_major_python_sum(matrix: np.ndarray) -> float:
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a NumPy array; implicit conversion is forbidden")
    a = matrix
    if a.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if a.dtype != np.float64 or not a.flags.c_contiguous:
        raise ValueError("matrix must be a C-contiguous float64 array")
    if not np.all(np.isfinite(a)):
        raise ValueError("matrix must be finite")
    total = 0.0
    rows, cols = a.shape
    for j in range(cols):
        for i in range(rows):
            total += float(a[i, j])
    return total


def stride_sum(values: np.ndarray, stride: int) -> float:
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a NumPy array; implicit conversion is forbidden")
    x = values
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if x.dtype != np.float64 or not x.flags.c_contiguous:
        raise ValueError("values must be a C-contiguous native float64 array")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite")
    return float(np.sum(x[::stride], dtype=np.float64))


def amdahl_speedup(parallel_fraction: float, workers: int) -> float:
    if not 0.0 <= parallel_fraction <= 1.0:
        raise ValueError("parallel_fraction must be in [0, 1]")
    if workers < 1:
        raise ValueError("workers must be positive")
    denominator = (1.0 - parallel_fraction) + parallel_fraction / workers
    return 1.0 / denominator
