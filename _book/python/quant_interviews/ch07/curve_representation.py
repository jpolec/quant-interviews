from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .evidence import (
    ConventionSchema,
    EvidenceCertificate,
    InvariantResult,
    TolerancePolicy,
    Violation,
    decision_from_invariants,
    stable_manifest_hash,
)


@dataclass(frozen=True)
class DiscountCurve:
    times: NDArray[np.float64]
    discount_factors: NDArray[np.float64]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        dfs = np.asarray(self.discount_factors, dtype=float)
        if (
            times.ndim != 1
            or dfs.ndim != 1
            or times.size != dfs.size
            or times.size == 0
        ):
            raise ValueError("times and discount_factors must be one-dimensional")
        if (
            not np.all(np.isfinite(times))
            or not np.all(np.isfinite(dfs))
            or np.any(times <= 0.0)
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("times must be positive and strictly increasing")
        if np.any(dfs <= 0.0):
            raise ValueError("discount factors must be positive")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "discount_factors", dfs)

    def zero_rate_cc(self) -> NDArray[np.float64]:
        return -np.log(self.discount_factors) / self.times

    def simple_forward(self, start_index: int, end_index: int) -> float:
        if not (0 <= start_index < end_index < self.times.size):
            raise IndexError("invalid curve interval")
        delta = self.times[end_index] - self.times[start_index]
        return (
            self.discount_factors[start_index]
            / self.discount_factors[end_index]
            - 1.0
        ) / delta

    def interval_forward_cc(self) -> NDArray[np.float64]:
        log_df = np.log(self.discount_factors)
        return -np.diff(log_df) / np.diff(self.times)


@dataclass(frozen=True)
class CurveRepresentationResult:
    zero_rates_cc: NDArray[np.float64]
    interval_forwards_cc: NDArray[np.float64]
    reconstructed_discount_factors: NDArray[np.float64]
    maximum_reconstruction_error: float


def curve_representation_certificate(
    curve: DiscountCurve,
    *,
    tolerance: TolerancePolicy = TolerancePolicy(1.0e-12, 1.0e-10),
) -> EvidenceCertificate[CurveRepresentationResult]:
    zero_rates = curve.zero_rate_cc()
    forwards = curve.interval_forward_cc()
    reconstructed = np.exp(-zero_rates * curve.times)
    error = float(np.max(np.abs(reconstructed - curve.discount_factors)))
    limit = tolerance.limit(float(np.max(curve.discount_factors)))
    invariants = (
        InvariantResult(
            "discount_factor_reconstruction",
            error,
            0.0,
            limit,
            error <= limit,
        ),
        InvariantResult(
            "interval_forward_count",
            int(forwards.size),
            int(curve.times.size - 1),
            None,
            forwards.size == curve.times.size - 1,
        ),
    )
    result = CurveRepresentationResult(
        zero_rates_cc=zero_rates,
        interval_forwards_cc=forwards,
        reconstructed_discount_factors=reconstructed,
        maximum_reconstruction_error=error,
    )
    violations = tuple(
        Violation(
            code=f"CURVE_REPRESENTATION_{invariant.name.upper()}",
            message=f"curve-representation invariant failed: {invariant.name}",
            observed=str(invariant.observed),
            limit=str(invariant.tolerance),
        )
        for invariant in invariants
        if not invariant.passed
    )
    return EvidenceCertificate(
        status=decision_from_invariants(invariants),
        result=result,
        invariants=invariants,
        violations=violations,
        conventions=ConventionSchema(
            currency="unit",
            rate_compounding="continuous zeros and interval forwards",
            day_count="curve year fractions",
            discounting_curve="supplied discount factors",
        ),
        tolerances=tolerance,
        input_manifest_hash=stable_manifest_hash(
            {
                "times": curve.times,
                "discount_factors": curve.discount_factors,
                "tolerance": tolerance,
            }
        ),
        implementation_version="ch07.curve_representation.v0.10",
    )


if __name__ == "__main__":
    curve = DiscountCurve(
        times=np.array([0.5, 1.0, 2.0, 3.0]),
        discount_factors=np.array([0.9780, 0.9551, 0.9069, 0.8585]),
    )
    certificate = curve_representation_certificate(curve)
    print("status=", certificate.status)
    print("zero_cc=", certificate.result.zero_rates_cc)
    print("interval_forward_cc=", certificate.result.interval_forwards_cc)
