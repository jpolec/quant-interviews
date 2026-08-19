from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
Compounding = Literal["annual", "continuous"]


@dataclass(frozen=True)
class BondCashflows:
    """Annual-compounding bond cash flows at strictly positive year fractions."""

    times: NDArray[np.float64]
    amounts: NDArray[np.float64]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        amounts = np.asarray(self.amounts, dtype=float)
        if times.ndim != 1 or amounts.ndim != 1 or times.size != amounts.size:
            raise ValueError("times and amounts must be one-dimensional and equal length")
        if (
            times.size == 0
            or not np.all(np.isfinite(times))
            or not np.all(np.isfinite(amounts))
            or np.any(times <= 0.0)
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("times must be positive and strictly increasing")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "amounts", amounts)

    def price(self, annual_yield: float) -> float:
        if not np.isfinite(annual_yield) or annual_yield <= -1.0:
            raise ValueError("annual_yield must be finite and exceed -100%")
        discount = (1.0 + annual_yield) ** (-self.times)
        return float(np.dot(self.amounts, discount))

    def price_continuous(self, continuous_yield: float) -> float:
        if not np.isfinite(continuous_yield):
            raise ValueError("continuous_yield must be finite")
        return float(
            np.dot(
                self.amounts,
                np.exp(-continuous_yield * self.times),
            )
        )

    def analytic_risk(
        self,
        yield_value: float,
        *,
        compounding: Compounding = "annual",
    ) -> dict[str, float]:
        if compounding == "annual":
            price = self.price(yield_value)
            discount = (1.0 + yield_value) ** (-self.times)
            macaulay = float(np.dot(self.times, self.amounts * discount) / price)
            modified = macaulay / (1.0 + yield_value)
            convexity = float(
                np.dot(
                    self.times * (self.times + 1.0),
                    self.amounts
                    * (1.0 + yield_value) ** (-self.times - 2.0),
                )
                / price
            )
        elif compounding == "continuous":
            price = self.price_continuous(yield_value)
            discount = np.exp(-yield_value * self.times)
            macaulay = float(np.dot(self.times, self.amounts * discount) / price)
            modified = macaulay
            convexity = float(
                np.dot(
                    self.times**2,
                    self.amounts * discount,
                )
                / price
            )
        else:
            raise ValueError("compounding must be annual or continuous")
        if not np.isfinite(price) or abs(price) <= np.finfo(float).tiny:
            raise ArithmeticError("bond price must be finite and non-zero for risk")
        pv = self.amounts * discount
        if not np.isclose(float(np.sum(pv)), price):
            raise ArithmeticError("cash-flow PVs do not reconcile to price")
        return {
            "price": price,
            "macaulay_duration": macaulay,
            "modified_duration": modified,
            "convexity": convexity,
            "dv01": price * modified * BASIS_POINT,
        }

    def bump_and_revalue_dv01(
        self,
        yield_value: float,
        bump_bp: float = 1.0,
        *,
        compounding: Compounding = "annual",
    ) -> float:
        bump = bump_bp * BASIS_POINT
        if compounding not in {"annual", "continuous"}:
            raise ValueError("compounding must be annual or continuous")
        if not np.isfinite(bump_bp) or bump <= 0.0:
            raise ValueError("bump_bp must be finite and positive")
        if compounding == "annual" and yield_value - bump <= -1.0:
            raise ValueError("annual-yield bump crosses the compounding domain")
        price_function = (
            self.price if compounding == "annual" else self.price_continuous
        )
        price_down = price_function(yield_value - bump)
        price_up = price_function(yield_value + bump)
        return (price_down - price_up) / (2.0 * bump_bp)


@dataclass(frozen=True)
class BondRiskResult:
    price: float
    analytic_dv01: float
    bump_dv01: float
    reconciliation_error: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    compounding: Compounding


def bond_risk_certificate(
    bond: BondCashflows,
    yield_value: float,
    *,
    compounding: Compounding,
    bump_bp: float = 1.0,
    tolerance: TolerancePolicy = TolerancePolicy(1.0e-10, 1.0e-7),
) -> EvidenceCertificate[BondRiskResult]:
    analytic = bond.analytic_risk(
        yield_value,
        compounding=compounding,
    )
    bump_dv01 = bond.bump_and_revalue_dv01(
        yield_value,
        bump_bp,
        compounding=compounding,
    )
    error = abs(analytic["dv01"] - bump_dv01)
    limit = tolerance.limit(analytic["dv01"])
    invariants = (
        InvariantResult(
            "analytic_vs_bump_dv01",
            error,
            0.0,
            limit,
            error <= limit,
        ),
    )
    result = BondRiskResult(
        price=analytic["price"],
        analytic_dv01=analytic["dv01"],
        bump_dv01=bump_dv01,
        reconciliation_error=error,
        macaulay_duration=analytic["macaulay_duration"],
        modified_duration=analytic["modified_duration"],
        convexity=analytic["convexity"],
        compounding=compounding,
    )
    violations = tuple(
        Violation(
            code=f"BOND_RISK_{invariant.name.upper()}",
            message=f"bond-risk invariant failed: {invariant.name}",
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
            currency="price units",
            rate_compounding=compounding,
            day_count="cash-flow year fractions",
            discounting_curve="flat yield-to-maturity coordinate",
        ),
        tolerances=tolerance,
        input_manifest_hash=stable_manifest_hash(
            {
                "times": bond.times,
                "amounts": bond.amounts,
                "yield": yield_value,
                "compounding": compounding,
                "bump_bp": bump_bp,
                "tolerance": tolerance,
            }
        ),
        implementation_version="ch07.bond_risk.v0.10",
    )


def example_bond() -> BondCashflows:
    return BondCashflows(
        times=np.arange(1.0, 6.0),
        amounts=np.array([4.0, 4.0, 4.0, 4.0, 104.0]),
    )


if __name__ == "__main__":
    bond = example_bond()
    certificate = bond_risk_certificate(
        bond,
        0.05,
        compounding="annual",
    )
    print("status=", certificate.status)
    print("result=", certificate.result)
