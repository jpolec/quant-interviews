from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt

from scipy.stats import norm

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
class MertonResult:
    equity_value: float
    debt_value_from_put: float
    balance_sheet_debt_value: float
    put_value: float
    risk_neutral_default_probability: float
    risk_neutral_distance_to_default: float
    physical_default_probability: float | None
    physical_distance_to_default: float | None
    put_call_parity_residual: float


def merton_pricing_core(
    *,
    asset_value: float,
    debt_face: float,
    asset_volatility: float,
    risk_free_rate: float,
    maturity: float,
) -> tuple[float, float, float, float, float]:
    """Price equity, put and debt from one shared Merton closed form."""

    sigma_sqrt_t = asset_volatility * sqrt(maturity)
    d1 = (
        log(asset_value / debt_face)
        + (risk_free_rate + 0.5 * asset_volatility**2) * maturity
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    discounted_debt = debt_face * exp(-risk_free_rate * maturity)
    equity = (
        asset_value * norm.cdf(d1)
        - discounted_debt * norm.cdf(d2)
    )
    put_value = (
        discounted_debt * norm.cdf(-d2)
        - asset_value * norm.cdf(-d1)
    )
    debt_from_put = discounted_debt - put_value
    return (
        float(equity),
        float(put_value),
        float(debt_from_put),
        float(d1),
        float(d2),
    )


def merton_credit_certificate(
    *,
    asset_value: float,
    debt_face: float,
    asset_volatility: float,
    risk_free_rate: float,
    maturity: float,
    physical_asset_drift: float | None = None,
    tolerance: TolerancePolicy = TolerancePolicy(1.0e-12, 1.0e-10),
) -> EvidenceCertificate[MertonResult]:
    values = (
        asset_value,
        debt_face,
        asset_volatility,
        risk_free_rate,
        maturity,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("Merton inputs must be finite")
    if asset_value <= 0.0 or debt_face <= 0.0 or maturity <= 0.0:
        raise ValueError("asset value, debt and maturity must be positive")
    if asset_volatility <= 0.0:
        raise ValueError("asset volatility must be positive")
    if physical_asset_drift is not None and not isfinite(physical_asset_drift):
        raise ValueError("physical_asset_drift must be finite")

    equity, put_value, debt_from_put, _, d2 = merton_pricing_core(
        asset_value=asset_value,
        debt_face=debt_face,
        asset_volatility=asset_volatility,
        risk_free_rate=risk_free_rate,
        maturity=maturity,
    )
    balance_sheet_debt = asset_value - equity
    residual = abs(debt_from_put - balance_sheet_debt)
    physical_dd: float | None = None
    physical_pd: float | None = None
    if physical_asset_drift is not None:
        sigma_sqrt_t = asset_volatility * sqrt(maturity)
        physical_dd = (
            log(asset_value / debt_face)
            + (
                physical_asset_drift
                - 0.5 * asset_volatility**2
            )
            * maturity
        ) / sigma_sqrt_t
        physical_pd = float(norm.cdf(-physical_dd))
    limit = tolerance.limit(asset_value)
    invariants = (
        InvariantResult(
            "put_call_parity_reconciliation",
            residual,
            0.0,
            limit,
            residual <= limit,
        ),
        InvariantResult(
            "risk_neutral_probability_domain",
            float(norm.cdf(-d2)),
            "[0, 1]",
            None,
            0.0 <= float(norm.cdf(-d2)) <= 1.0,
        ),
    )
    result = MertonResult(
        equity_value=float(equity),
        debt_value_from_put=float(debt_from_put),
        balance_sheet_debt_value=float(balance_sheet_debt),
        put_value=float(put_value),
        risk_neutral_default_probability=float(norm.cdf(-d2)),
        risk_neutral_distance_to_default=float(d2),
        physical_default_probability=physical_pd,
        physical_distance_to_default=physical_dd,
        put_call_parity_residual=residual,
    )
    violations = tuple(
        Violation(
            code=f"MERTON_{invariant.name.upper()}",
            message=f"Merton invariant failed: {invariant.name}",
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
            currency="firm-value currency",
            rate_compounding="continuous",
            day_count="model year fractions",
            discounting_curve="flat risk-free fixture",
            credit_convention=(
                "risk-neutral default probability from d2; physical distance "
                "reported only when a physical asset drift is supplied"
            ),
        ),
        tolerances=tolerance,
        input_manifest_hash=stable_manifest_hash(
            {
                "asset_value": asset_value,
                "debt_face": debt_face,
                "asset_volatility": asset_volatility,
                "risk_free_rate": risk_free_rate,
                "maturity": maturity,
                "physical_asset_drift": physical_asset_drift,
                "tolerance": tolerance,
            }
        ),
        implementation_version="ch07.merton_structural.v0.10",
    )


def merton_equity_and_default_probability(
    asset_value: float,
    debt_face: float,
    asset_volatility: float,
    risk_free_rate: float,
    maturity: float,
) -> dict[str, float]:
    result = merton_credit_certificate(
        asset_value=asset_value,
        debt_face=debt_face,
        asset_volatility=asset_volatility,
        risk_free_rate=risk_free_rate,
        maturity=maturity,
    ).result
    return {
        "equity_value": result.equity_value,
        "debt_value": result.debt_value_from_put,
        "risk_neutral_default_probability": (
            result.risk_neutral_default_probability
        ),
        "risk_neutral_distance_to_default": (
            result.risk_neutral_distance_to_default
        ),
    }


if __name__ == "__main__":
    certificate = merton_credit_certificate(
        asset_value=150.0,
        debt_face=100.0,
        asset_volatility=0.30,
        risk_free_rate=0.03,
        maturity=1.0,
        physical_asset_drift=0.08,
    )
    print("status=", certificate.status)
    print("result=", certificate.result)
