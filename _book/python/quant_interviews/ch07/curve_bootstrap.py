from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

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
        if times.ndim != 1 or dfs.ndim != 1 or times.size != dfs.size:
            raise ValueError("times and discount_factors must be one-dimensional")
        if times.size == 0 or not np.all(np.isfinite(times)):
            raise ValueError("curve times must be finite and non-empty")
        if not np.all(np.isfinite(dfs)):
            raise ValueError("discount factors must be finite")
        if np.any(np.diff(times) <= 0.0) or np.any(times <= 0.0):
            raise ValueError("times must be positive and strictly increasing")
        if np.any(dfs <= 0.0):
            raise ValueError("discount factors must be positive")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "discount_factors", dfs)

    def df(self, maturity: float) -> float:
        if not isfinite(maturity):
            raise ValueError("maturity must be finite")
        if maturity < 0.0 or maturity > self.times[-1]:
            raise ValueError("maturity lies outside the calibrated curve")
        if maturity == 0.0:
            return 1.0
        grid = np.r_[0.0, self.times]
        log_dfs = np.r_[0.0, np.log(self.discount_factors)]
        return float(np.exp(np.interp(maturity, grid, log_dfs)))

    def zero_rate_cc(self, maturity: float) -> float:
        if maturity <= 0.0:
            raise ValueError("maturity must be positive")
        return -np.log(self.df(maturity)) / maturity

    def interval_forwards_cc(self) -> NDArray[np.float64]:
        grid = np.r_[0.0, self.times]
        log_dfs = np.r_[0.0, np.log(self.discount_factors)]
        return -np.diff(log_dfs) / np.diff(grid)


@dataclass(frozen=True)
class CalibrationNode:
    instrument: str
    maturity: float
    quote: float
    discount_factor: float
    residual: float
    solve_method: str
    iterations: int


@dataclass(frozen=True)
class CurveBootstrapResult:
    curve: DiscountCurve
    calibration_nodes: tuple[CalibrationNode, ...]
    maximum_repricing_residual: float
    interval_forwards_cc: NDArray[np.float64]
    minimum_interval_forward: float
    monotone_discount_factors: bool
    nonnegative_forward_policy: bool


def reprice_annual_par_swap(
    curve: DiscountCurve,
    maturity: int,
    fixed_rate: float,
) -> float:
    annuity = sum(curve.df(float(year)) for year in range(1, maturity + 1))
    return fixed_rate * annuity - (1.0 - curve.df(float(maturity)))


def solve_curve_nodes(
    deposit_quotes: dict[float, float],
    swap_quotes: dict[int, float],
    *,
    overlap_tolerance: float = 1.0e-12,
) -> tuple[dict[float, float], tuple[CalibrationNode, ...]]:
    """Solve ordered deposit and annual-swap nodes and reprice the final curve.

    A deposit and swap may share a maturity only when both equations reconcile
    on the same node inside ``overlap_tolerance``.  A conflicting quote is an
    over-determined one-node calibration and is rejected instead of silently
    overwriting the deposit node.
    """

    if not isfinite(overlap_tolerance) or overlap_tolerance < 0.0:
        raise ValueError("overlap_tolerance must be finite and non-negative")

    nodes: dict[float, float] = {}
    calibration: list[CalibrationNode] = []
    for maturity, rate in sorted(deposit_quotes.items()):
        if not isfinite(maturity) or not isfinite(rate) or maturity <= 0.0:
            raise ValueError("deposit quotes must be finite with positive maturity")
        denominator = 1.0 + rate * maturity
        if denominator <= 0.0:
            raise ValueError("deposit quote implies a non-positive discount factor")
        discount_factor = 1.0 / denominator
        nodes[float(maturity)] = discount_factor
        calibration.append(
            CalibrationNode(
                "deposit",
                float(maturity),
                float(rate),
                float(discount_factor),
                float(discount_factor * denominator - 1.0),
                "closed_form",
                1,
            )
        )
    for maturity, par_rate in sorted(swap_quotes.items()):
        if maturity < 1 or not isfinite(par_rate) or 1.0 + par_rate <= 0.0:
            raise ValueError("invalid annual swap quote")
        prior = [nodes.get(float(year)) for year in range(1, maturity)]
        if any(discount_factor is None for discount_factor in prior):
            raise ValueError("all earlier annual discount nodes are required")
        implied_final_df = (
            1.0 - par_rate * float(np.sum(prior))
        ) / (1.0 + par_rate)
        if implied_final_df <= 0.0 or not isfinite(implied_final_df):
            raise ValueError("swap quote implies a non-positive discount factor")
        shared_node = nodes.get(float(maturity))
        if shared_node is not None:
            shared_residual = (1.0 + par_rate) * shared_node - (
                1.0 - par_rate * float(np.sum(prior))
            )
            if abs(shared_residual) > overlap_tolerance:
                raise ValueError(
                    "deposit and annual swap at the same maturity are inconsistent"
                )
            final_df = shared_node
            solve_method = "shared_node_reprice"
            iterations = 0
        else:
            final_df = implied_final_df
            nodes[float(maturity)] = final_df
            solve_method = "closed_form"
            iterations = 1
        trial_curve = DiscountCurve(
            np.array(sorted(nodes), dtype=float),
            np.array([nodes[node] for node in sorted(nodes)], dtype=float),
        )
        calibration.append(
            CalibrationNode(
                "annual_swap",
                float(maturity),
                float(par_rate),
                float(final_df),
                reprice_annual_par_swap(trial_curve, maturity, par_rate),
                solve_method,
                iterations,
            )
        )
    final_curve = DiscountCurve(
        np.array(sorted(nodes), dtype=float),
        np.array([nodes[node] for node in sorted(nodes)], dtype=float),
    )
    final_calibration = tuple(
        replace(
            node,
            discount_factor=final_curve.df(node.maturity),
            residual=(
                final_curve.df(node.maturity)
                * (1.0 + node.quote * node.maturity)
                - 1.0
                if node.instrument == "deposit"
                else reprice_annual_par_swap(
                    final_curve,
                    int(node.maturity),
                    node.quote,
                )
            ),
        )
        for node in calibration
    )
    return nodes, final_calibration


def bootstrap_curve_certificate(
    deposit_quotes: dict[float, float],
    swap_quotes: dict[int, float],
    *,
    require_nonnegative_forwards: bool = False,
    tolerance: TolerancePolicy = TolerancePolicy(1.0e-12, 1.0e-10),
) -> EvidenceCertificate[CurveBootstrapResult]:
    """Bootstrap simple deposits and annual swaps with governed diagnostics.

    Negative quotes are valid when their cash-flow equations imply positive
    discount factors. Monotone discount factors are enforced only when the
    signed non-negative-forward policy requests them.
    """

    if not deposit_quotes and not swap_quotes:
        raise ValueError("at least one calibration quote is required")
    residual_limit = tolerance.limit(1.0)
    nodes, calibration = solve_curve_nodes(
        deposit_quotes,
        swap_quotes,
        overlap_tolerance=residual_limit,
    )

    curve = DiscountCurve(
        times=np.array(sorted(nodes), dtype=float),
        discount_factors=np.array([nodes[node] for node in sorted(nodes)]),
    )
    forwards = curve.interval_forwards_cc()
    maximum_residual = max(abs(node.residual) for node in calibration)
    monotone = bool(np.all(np.diff(np.r_[1.0, curve.discount_factors]) <= 0.0))
    invariants = (
        InvariantResult(
            "positive_discount_factors",
            float(np.min(curve.discount_factors)),
            "> 0",
            None,
            bool(np.min(curve.discount_factors) > 0.0),
        ),
        InvariantResult(
            "maximum_calibration_residual",
            maximum_residual,
            0.0,
            residual_limit,
            maximum_residual <= residual_limit,
        ),
        InvariantResult(
            "signed_nonnegative_forward_policy",
            float(np.min(forwards)),
            ">= 0 when policy is enabled",
            residual_limit,
            not require_nonnegative_forwards
            or bool(np.min(forwards) >= -residual_limit),
        ),
    )
    violations = tuple(
        Violation(
            code=f"CURVE_{invariant.name.upper()}",
            message=f"curve invariant failed: {invariant.name}",
            observed=str(invariant.observed),
            limit=str(invariant.expected),
        )
        for invariant in invariants
        if not invariant.passed
    )
    result = CurveBootstrapResult(
        curve=curve,
        calibration_nodes=calibration,
        maximum_repricing_residual=maximum_residual,
        interval_forwards_cc=forwards,
        minimum_interval_forward=float(np.min(forwards)),
        monotone_discount_factors=monotone,
        nonnegative_forward_policy=require_nonnegative_forwards,
    )
    manifest = {
        "deposit_quotes": deposit_quotes,
        "swap_quotes": swap_quotes,
        "require_nonnegative_forwards": require_nonnegative_forwards,
        "tolerance": tolerance,
    }
    return EvidenceCertificate(
        status=decision_from_invariants(invariants),
        result=result,
        invariants=invariants,
        violations=violations,
        conventions=ConventionSchema(
            currency="unit",
            rate_compounding="simple deposits; annual-pay swaps; log-DF interpolation",
            day_count="year fractions supplied by fixture",
            discounting_curve="single discount curve",
        ),
        tolerances=tolerance,
        input_manifest_hash=stable_manifest_hash(manifest),
        implementation_version="ch07.curve_bootstrap.v0.10",
    )


def bootstrap_simple_deposits_and_annual_swaps(
    deposit_quotes: dict[float, float],
    swap_quotes: dict[int, float],
) -> DiscountCurve:
    return bootstrap_curve_certificate(
        deposit_quotes,
        swap_quotes,
    ).result.curve


def example_curve() -> DiscountCurve:
    return bootstrap_simple_deposits_and_annual_swaps(
        deposit_quotes={0.5: 0.045, 1.0: 0.047},
        swap_quotes={2: 0.050, 3: 0.052},
    )


if __name__ == "__main__":
    certificate = bootstrap_curve_certificate(
        deposit_quotes={0.5: -0.002, 1.0: -0.001},
        swap_quotes={2: 0.001, 3: 0.003},
    )
    print("status=", certificate.status)
    print("maximum_residual=", certificate.result.maximum_repricing_residual)
    print("minimum_forward=", certificate.result.minimum_interval_forward)
