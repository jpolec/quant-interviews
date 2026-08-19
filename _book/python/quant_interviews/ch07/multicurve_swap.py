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

BASIS_POINT = 1.0e-4


@dataclass(frozen=True)
class NodeCurve:
    times: NDArray[np.float64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if times.ndim != 1 or values.ndim != 1 or times.size != values.size:
            raise ValueError("curve arrays must be one-dimensional and equal length")
        if (
            times.size == 0
            or not np.all(np.isfinite(times))
            or not np.all(np.isfinite(values))
            or np.any(times <= 0.0)
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("curve times must be strictly increasing")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    def linear(self, maturity: float) -> float:
        if not np.isfinite(maturity):
            raise ValueError("maturity must be finite")
        if maturity < self.times[0] or maturity > self.times[-1]:
            raise ValueError("maturity is outside the curve")
        return float(np.interp(maturity, self.times, self.values))


@dataclass(frozen=True)
class MultiCurveSwap:
    payment_times: NDArray[np.float64]
    accruals: NDArray[np.float64]
    notional: float = 1.0

    def __post_init__(self) -> None:
        times = np.asarray(self.payment_times, dtype=float)
        accruals = np.asarray(self.accruals, dtype=float)
        if (
            times.ndim != 1
            or accruals.ndim != 1
            or times.size == 0
            or times.shape != accruals.shape
        ):
            raise ValueError("payment_times and accruals must be equal non-empty arrays")
        if (
            not np.all(np.isfinite(times))
            or not np.all(np.isfinite(accruals))
            or not np.isfinite(self.notional)
            or np.any(times <= 0.0)
            or np.any(accruals <= 0.0)
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("invalid payment schedule")
        object.__setattr__(self, "payment_times", times)
        object.__setattr__(self, "accruals", accruals)

    def fixed_annuity(self, discount_curve: NodeCurve) -> float:
        return float(
            self.notional
            * np.sum(
                [alpha * np.exp(discount_curve.linear(t)) for t, alpha in zip(self.payment_times, self.accruals, strict=True)]
            )
        )

    def floating_pv(self, discount_curve: NodeCurve, forward_curve: NodeCurve) -> float:
        dfs = np.array([np.exp(discount_curve.linear(t)) for t in self.payment_times])
        forwards = np.array([forward_curve.linear(t) for t in self.payment_times])
        return float(self.notional * np.dot(dfs * self.accruals, forwards))

    def par_rate(self, discount_curve: NodeCurve, forward_curve: NodeCurve) -> float:
        return self.floating_pv(discount_curve, forward_curve) / self.fixed_annuity(discount_curve)

    def payer_pv(self, fixed_rate: float, discount_curve: NodeCurve, forward_curve: NodeCurve) -> float:
        return self.floating_pv(discount_curve, forward_curve) - fixed_rate * self.fixed_annuity(discount_curve)


@dataclass(frozen=True)
class MultiCurveRiskResult:
    par_rate: float
    contractual_rate: float
    payer_pv: float
    par_repricing_residual: float
    discount_zero_node_dv01: NDArray[np.float64]
    projection_forward_node_dv01: NDArray[np.float64]
    node_coordinate_transform: NDArray[np.float64]
    coordinate_transform_condition_number: float
    node_coordinate_ids: tuple[str, ...]
    node_coordinate_units: tuple[str, ...]
    risk_reconciliation_error: float


def _bump_discount_zero_node(
    curve: NodeCurve,
    index: int,
    bump: float,
) -> NodeCurve:
    values = curve.values.copy()
    values[index] -= bump * curve.times[index]
    return NodeCurve(curve.times, values)


def _bump_forward_node(
    curve: NodeCurve,
    index: int,
    bump: float,
) -> NodeCurve:
    values = curve.values.copy()
    values[index] += bump
    return NodeCurve(curve.times, values)


def multicurve_risk_certificate(
    swap: MultiCurveSwap,
    discount_curve: NodeCurve,
    projection_curve: NodeCurve,
    *,
    fixed_rate: float | None = None,
    bump_bp: float = 1.0,
    tolerance: TolerancePolicy = TolerancePolicy(1.0e-7, 1.0e-6),
) -> EvidenceCertificate[MultiCurveRiskResult]:
    if not np.isfinite(bump_bp) or bump_bp <= 0.0:
        raise ValueError("bump_bp must be finite and positive")
    if not np.array_equal(discount_curve.times, projection_curve.times):
        raise ValueError("fixture requires aligned discount and projection nodes")
    par_rate = swap.par_rate(discount_curve, projection_curve)
    contractual_rate = par_rate if fixed_rate is None else fixed_rate
    if not np.isfinite(contractual_rate):
        raise ValueError("fixed_rate must be finite")
    base_pv = swap.payer_pv(
        contractual_rate,
        discount_curve,
        projection_curve,
    )
    par_pv = swap.payer_pv(par_rate, discount_curve, projection_curve)
    bump = bump_bp * BASIS_POINT
    discount_dv01: list[float] = []
    projection_dv01: list[float] = []
    for index in range(discount_curve.times.size):
        discount_down = _bump_discount_zero_node(
            discount_curve,
            index,
            -bump,
        )
        discount_up = _bump_discount_zero_node(
            discount_curve,
            index,
            bump,
        )
        discount_dv01.append(
            (
                swap.payer_pv(
                    contractual_rate,
                    discount_down,
                    projection_curve,
                )
                - swap.payer_pv(
                    contractual_rate,
                    discount_up,
                    projection_curve,
                )
            )
            / (2.0 * bump_bp)
        )
        projection_down = _bump_forward_node(
            projection_curve,
            index,
            -bump,
        )
        projection_up = _bump_forward_node(
            projection_curve,
            index,
            bump,
        )
        projection_dv01.append(
            (
                swap.payer_pv(
                    contractual_rate,
                    discount_curve,
                    projection_down,
                )
                - swap.payer_pv(
                    contractual_rate,
                    discount_curve,
                    projection_up,
                )
            )
            / (2.0 * bump_bp)
        )

    discount_risk = np.asarray(discount_dv01)
    projection_risk = np.asarray(projection_dv01)
    coordinate_transform = np.block(
        [
            [-np.diag(discount_curve.times), np.zeros(
                (discount_curve.times.size, projection_curve.times.size)
            )],
            [np.zeros(
                (projection_curve.times.size, discount_curve.times.size)
            ), np.eye(projection_curve.times.size)],
        ]
    )
    direct_parallel_down = swap.payer_pv(
        contractual_rate,
        NodeCurve(
            discount_curve.times,
            discount_curve.values + bump * discount_curve.times,
        ),
        NodeCurve(
            projection_curve.times,
            projection_curve.values - bump,
        ),
    )
    direct_parallel_up = swap.payer_pv(
        contractual_rate,
        NodeCurve(
            discount_curve.times,
            discount_curve.values - bump * discount_curve.times,
        ),
        NodeCurve(
            projection_curve.times,
            projection_curve.values + bump,
        ),
    )
    direct_parallel_dv01 = (
        direct_parallel_down - direct_parallel_up
    ) / (2.0 * bump_bp)
    bucket_sum = float(np.sum(discount_risk) + np.sum(projection_risk))
    reconciliation_error = abs(bucket_sum - direct_parallel_dv01)
    pv_limit = tolerance.limit(max(1.0, abs(swap.notional)))
    risk_limit = tolerance.limit(max(1.0, abs(bucket_sum)))
    invariants = (
        InvariantResult(
            "par_instrument_repricing",
            abs(par_pv),
            0.0,
            pv_limit,
            abs(par_pv) <= pv_limit,
        ),
        InvariantResult(
            "parallel_risk_reconciliation",
            reconciliation_error,
            0.0,
            risk_limit,
            reconciliation_error <= risk_limit,
        ),
    )
    result = MultiCurveRiskResult(
        par_rate=par_rate,
        contractual_rate=contractual_rate,
        payer_pv=base_pv,
        par_repricing_residual=par_pv,
        discount_zero_node_dv01=discount_risk,
        projection_forward_node_dv01=projection_risk,
        node_coordinate_transform=coordinate_transform,
        coordinate_transform_condition_number=float(
            np.linalg.cond(coordinate_transform)
        ),
        node_coordinate_ids=tuple(
            [f"discount_zero:{time:g}Y" for time in discount_curve.times]
            + [f"projection_forward:{time:g}Y" for time in projection_curve.times]
        ),
        node_coordinate_units=tuple(
            ["log_discount_factor_per_continuous_zero"]
            * discount_curve.times.size
            + ["forward_rate_per_forward_node"]
            * projection_curve.times.size
        ),
        risk_reconciliation_error=reconciliation_error,
    )
    violations = tuple(
        Violation(
            code=f"MULTICURVE_{invariant.name.upper()}",
            message=f"multicurve invariant failed: {invariant.name}",
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
            currency="PV currency per basis point",
            rate_compounding="continuous discount zeros; simple forward rates",
            day_count="schedule accrual fractions",
            discounting_curve="OIS-style discount log-DF nodes",
            projection_curve="tenor forward-rate nodes",
        ),
        tolerances=tolerance,
        input_manifest_hash=stable_manifest_hash(
            {
                "swap": swap,
                "discount_curve": discount_curve,
                "projection_curve": projection_curve,
                "fixed_rate": contractual_rate,
                "bump_bp": bump_bp,
                "tolerance": tolerance,
            }
        ),
        implementation_version="ch07.multicurve_swap.v0.10",
    )


def example_swap() -> tuple[MultiCurveSwap, NodeCurve, NodeCurve]:
    times = np.arange(0.5, 5.0 + 0.5, 0.5)
    discount_zero = np.linspace(0.035, 0.040, times.size)
    discount_log_df = -discount_zero * times
    forwards = np.linspace(0.041, 0.048, times.size)
    return (
        MultiCurveSwap(payment_times=times, accruals=np.full(times.size, 0.5), notional=100_000_000.0),
        NodeCurve(times=times, values=discount_log_df),
        NodeCurve(times=times, values=forwards),
    )


if __name__ == "__main__":
    swap, discount, forward = example_swap()
    certificate = multicurve_risk_certificate(swap, discount, forward)
    print("status=", certificate.status)
    print(f"par_rate={certificate.result.par_rate:.8%}")
    print("discount_dv01=", certificate.result.discount_zero_node_dv01)
    print("projection_dv01=", certificate.result.projection_forward_node_dv01)
