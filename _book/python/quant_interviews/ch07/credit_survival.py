from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite

from .evidence import (
    ConventionSchema,
    EvidenceCertificate,
    InvariantResult,
    TolerancePolicy,
    decision_from_invariants,
    stable_manifest_hash,
    Violation,
)


@dataclass(frozen=True)
class FlatHazardCredit:
    hazard_rate: float
    recovery: float
    risk_free_rate: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.hazard_rate)
            or not isfinite(self.risk_free_rate)
            or self.hazard_rate < 0.0
        ):
            raise ValueError(
                "hazard must be finite and non-negative; continuous rate must be finite"
            )
        if not isfinite(self.recovery) or not (0.0 <= self.recovery <= 1.0):
            raise ValueError("recovery must lie in [0, 1]")

    def survival(self, maturity: float) -> float:
        if not isfinite(maturity) or maturity < 0.0:
            raise ValueError("maturity must be finite and non-negative")
        return exp(-self.hazard_rate * maturity)

    def default_probability(self, maturity: float) -> float:
        return 1.0 - self.survival(maturity)

    def risky_zero_fractional_recovery_of_par(self, maturity: float) -> float:
        """Continuous-time zero coupon with recovery paid at default."""

        r = self.risk_free_rate
        lam = self.hazard_rate
        recovery = self.recovery
        survival_leg = exp(-(r + lam) * maturity)
        if abs(r + lam) < 1.0e-14:
            recovery_leg = recovery * lam * maturity
        else:
            recovery_leg = (
                recovery
                * lam
                * (1.0 - exp(-(r + lam) * maturity))
                / (r + lam)
            )
        return survival_leg + recovery_leg


@dataclass(frozen=True)
class CreditSurvivalResult:
    maturity: float
    survival_probability: float
    default_probability: float
    risky_zero_price: float
    risk_free_zero_price: float
    survival_leg_pv: float
    recovery_leg_pv: float
    credit_triangle_spread: float


def credit_survival_certificate(
    model: FlatHazardCredit,
    maturity: float,
    *,
    tolerance: TolerancePolicy = TolerancePolicy(1.0e-12, 1.0e-10),
) -> EvidenceCertificate[CreditSurvivalResult]:
    survival = model.survival(maturity)
    default_probability = model.default_probability(maturity)
    risky_zero = model.risky_zero_fractional_recovery_of_par(maturity)
    risk_free_zero = exp(-model.risk_free_rate * maturity)
    survival_leg = exp(
        -(model.risk_free_rate + model.hazard_rate) * maturity
    )
    recovery_leg = risky_zero - survival_leg
    if not all(
        isfinite(value)
        for value in (
            survival,
            default_probability,
            risky_zero,
            risk_free_zero,
            survival_leg,
            recovery_leg,
        )
    ):
        raise ArithmeticError("credit-survival output is non-finite")
    price_limit = tolerance.limit(max(1.0, abs(risky_zero)))
    invariants = (
        InvariantResult(
            "probability_reconciliation",
            survival + default_probability,
            1.0,
            tolerance.limit(1.0),
            abs(survival + default_probability - 1.0)
            <= tolerance.limit(1.0),
        ),
        InvariantResult(
            "probability_domain",
            min(survival, default_probability),
            ">= 0",
            tolerance.limit(1.0),
            survival >= -tolerance.limit(1.0)
            and default_probability >= -tolerance.limit(1.0)
            and survival <= 1.0 + tolerance.limit(1.0)
            and default_probability <= 1.0 + tolerance.limit(1.0),
        ),
        InvariantResult(
            "survival_recovery_leg_reconciliation",
            abs(risky_zero - survival_leg - recovery_leg),
            0.0,
            price_limit,
            abs(risky_zero - survival_leg - recovery_leg) <= price_limit,
        ),
        InvariantResult(
            "positive_risky_zero",
            risky_zero,
            "> 0",
            None,
            risky_zero > 0.0,
        ),
    )
    result = CreditSurvivalResult(
        maturity=maturity,
        survival_probability=survival,
        default_probability=default_probability,
        risky_zero_price=risky_zero,
        risk_free_zero_price=risk_free_zero,
        survival_leg_pv=survival_leg,
        recovery_leg_pv=recovery_leg,
        credit_triangle_spread=(
            model.hazard_rate * (1.0 - model.recovery)
        ),
    )
    violations = tuple(
        Violation(
            code=f"CREDIT_{invariant.name.upper()}",
            message=f"credit-survival invariant failed: {invariant.name}",
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
            currency="fraction of par",
            rate_compounding="continuous",
            day_count="model year fractions",
            discounting_curve="flat risk-free curve",
            credit_convention="fractional recovery of par paid at default",
        ),
        tolerances=tolerance,
        input_manifest_hash=stable_manifest_hash(
            {"model": model, "maturity": maturity, "tolerance": tolerance}
        ),
        implementation_version="ch07.credit_survival.v0.10",
    )


if __name__ == "__main__":
    model = FlatHazardCredit(0.03, 0.40, 0.025)
    for maturity in (1.0, 3.0, 5.0):
        print(credit_survival_certificate(model, maturity))
