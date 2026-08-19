from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite

import numpy as np
from scipy.optimize import brentq

from .curve_bootstrap import DiscountCurve

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
class CDSQuote:
    maturity_years: int
    running_coupon: float
    upfront: float = 0.0
    payments_per_year: int = 4

    def __post_init__(self) -> None:
        if self.maturity_years <= 0 or self.payments_per_year <= 0:
            raise ValueError("CDS maturity and payment frequency must be positive")
        if (
            not isfinite(self.running_coupon)
            or self.running_coupon < 0.0
            or not isfinite(self.upfront)
        ):
            raise ValueError("CDS coupon/upfront must be finite")


@dataclass(frozen=True)
class PiecewiseHazardCurve:
    maturities: tuple[int, ...]
    hazards: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.maturities) != len(self.hazards) or not self.maturities:
            raise ValueError("maturities and hazards must have equal non-zero length")
        if (
            tuple(sorted(self.maturities)) != self.maturities
            or len(set(self.maturities)) != len(self.maturities)
            or any(not isfinite(hazard) or hazard < 0.0 for hazard in self.hazards)
        ):
            raise ValueError("maturities must increase and hazards be non-negative")

    def hazard_at(self, time: float) -> float:
        if time <= 0.0:
            return self.hazards[0]
        previous = 0
        for maturity, hazard in zip(
            self.maturities,
            self.hazards,
            strict=True,
        ):
            if previous < time <= maturity:
                return hazard
            previous = maturity
        raise ValueError("time exceeds curve horizon")

    def integrated_hazard(self, time: float) -> float:
        if time < 0.0 or time > self.maturities[-1]:
            raise ValueError("time is outside the hazard curve")
        total = 0.0
        previous = 0.0
        for maturity, hazard in zip(
            self.maturities,
            self.hazards,
            strict=True,
        ):
            segment_end = min(time, float(maturity))
            total += hazard * max(segment_end - previous, 0.0)
            if time <= maturity:
                break
            previous = float(maturity)
        return total

    def survival(self, time: float) -> float:
        return exp(-self.integrated_hazard(time))


@dataclass(frozen=True)
class CDSLegs:
    scheduled_annuity: float
    accrued_on_default: float
    risky_pv01: float
    protection_leg: float
    upfront: float
    net_pv: float


@dataclass(frozen=True)
class CDSCalibrationNode:
    maturity_years: int
    running_coupon: float
    upfront: float
    hazard: float
    residual: float
    bracket_lower: float
    bracket_upper: float
    iterations: int
    survival_probability: float
    default_probability: float


@dataclass(frozen=True)
class CDSBootstrapResult:
    curve: PiecewiseHazardCurve | None
    calibration_nodes: tuple[CDSCalibrationNode, ...]
    maximum_repricing_residual: float
    recovery: float
    no_bracket_maturity: int | None
    discount_curve_run_id: str


def flat_discount_curve(
    *,
    continuous_rate: float,
    horizon: float,
) -> DiscountCurve:
    """Explicit educational adapter from a flat rate to a typed curve."""

    if (
        not isfinite(continuous_rate)
        or not isfinite(horizon)
        or horizon <= 0.0
    ):
        raise ValueError("flat discount-curve inputs must be finite")
    return DiscountCurve(
        times=np.array([horizon], dtype=float),
        discount_factors=np.array(
            [exp(-continuous_rate * horizon)],
            dtype=float,
        ),
    )


def cds_legs(
    curve: PiecewiseHazardCurve,
    quote: CDSQuote,
    recovery: float,
    discount_curve: DiscountCurve,
) -> CDSLegs:
    if not 0.0 <= recovery < 1.0:
        raise ValueError("recovery must lie in [0, 1)")
    dt = 1.0 / quote.payments_per_year
    payment_times = np.arange(
        dt,
        quote.maturity_years + 0.5 * dt,
        dt,
    )
    scheduled_annuity = 0.0
    accrued_on_default = 0.0
    protection = 0.0
    previous_time = 0.0
    previous_survival = 1.0
    for payment_time in payment_times:
        survival = curve.survival(float(payment_time))
        discount = discount_curve.df(float(payment_time))
        scheduled_annuity += dt * discount * survival
        default_probability = previous_survival - survival
        midpoint = 0.5 * (previous_time + float(payment_time))
        default_discount = discount_curve.df(midpoint)
        accrued_on_default += 0.5 * dt * default_discount * default_probability
        protection += (
            (1.0 - recovery) * default_discount * default_probability
        )
        previous_survival = survival
        previous_time = float(payment_time)
    risky_pv01 = scheduled_annuity + accrued_on_default
    net_pv = (
        quote.running_coupon * risky_pv01
        + quote.upfront
        - protection
    )
    return CDSLegs(
        scheduled_annuity=scheduled_annuity,
        accrued_on_default=accrued_on_default,
        risky_pv01=risky_pv01,
        protection_leg=protection,
        upfront=quote.upfront,
        net_pv=net_pv,
    )


def _trial_curve(
    maturities: list[int],
    hazards: list[float],
    maturity: int,
    candidate: float,
) -> PiecewiseHazardCurve:
    return PiecewiseHazardCurve(
        tuple(maturities + [maturity]),
        tuple(hazards + [candidate]),
    )


def bootstrap_cds_certificate(
    quotes: tuple[CDSQuote, ...],
    *,
    discount_curve: DiscountCurve,
    discount_curve_run_id: str,
    recovery: float = 0.40,
    initial_hazard_upper: float = 0.25,
    maximum_hazard: float = 100.0,
    tolerance: TolerancePolicy = TolerancePolicy(1.0e-11, 1.0e-10),
) -> EvidenceCertificate[CDSBootstrapResult]:
    if not quotes:
        raise ValueError("at least one CDS quote is required")
    if not discount_curve_run_id.strip():
        raise ValueError("discount_curve_run_id must be non-empty")
    if discount_curve.times[-1] < max(quote.maturity_years for quote in quotes):
        raise ValueError("discount curve does not cover the CDS horizon")
    if tuple(sorted(quote.maturity_years for quote in quotes)) != tuple(
        quote.maturity_years for quote in quotes
    ):
        raise ValueError("CDS quotes must be ordered by maturity")
    if len({quote.maturity_years for quote in quotes}) != len(quotes):
        raise ValueError("CDS maturities must be unique")
    if initial_hazard_upper <= 0.0 or maximum_hazard <= initial_hazard_upper:
        raise ValueError("invalid hazard bracket policy")

    maturities: list[int] = []
    hazards: list[float] = []
    calibration: list[CDSCalibrationNode] = []
    no_bracket: int | None = None
    for quote in quotes:
        def residual(candidate: float) -> float:
            curve = _trial_curve(
                maturities,
                hazards,
                quote.maturity_years,
                candidate,
            )
            return cds_legs(
                curve,
                quote,
                recovery,
                discount_curve,
            ).net_pv

        lower = 0.0
        upper = initial_hazard_upper
        lower_value = residual(lower)
        upper_value = residual(upper)
        while (
            lower_value * upper_value > 0.0
            and upper < maximum_hazard
        ):
            upper = min(2.0 * upper, maximum_hazard)
            upper_value = residual(upper)
        if lower_value * upper_value > 0.0:
            no_bracket = quote.maturity_years
            break
        root, solver = brentq(
            residual,
            lower,
            upper,
            xtol=tolerance.absolute,
            rtol=max(tolerance.relative, 4.0 * np.finfo(float).eps),
            full_output=True,
        )
        maturities.append(quote.maturity_years)
        hazards.append(float(root))
        curve = PiecewiseHazardCurve(tuple(maturities), tuple(hazards))
        node_residual = cds_legs(
            curve,
            quote,
            recovery,
            discount_curve,
        ).net_pv
        survival = curve.survival(float(quote.maturity_years))
        calibration.append(
            CDSCalibrationNode(
                maturity_years=quote.maturity_years,
                running_coupon=quote.running_coupon,
                upfront=quote.upfront,
                hazard=float(root),
                residual=float(node_residual),
                bracket_lower=lower,
                bracket_upper=upper,
                iterations=int(solver.iterations),
                survival_probability=survival,
                default_probability=1.0 - survival,
            )
        )

    curve = (
        PiecewiseHazardCurve(tuple(maturities), tuple(hazards))
        if no_bracket is None
        else None
    )
    maximum_residual = (
        max(abs(node.residual) for node in calibration)
        if calibration
        else float("inf")
    )
    residual_limit = tolerance.limit(1.0)
    survival_monotone = (
        curve is not None
        and all(
            curve.survival(float(right))
            <= curve.survival(float(left)) + residual_limit
            for left, right in zip(
                (0, *curve.maturities[:-1]),
                curve.maturities,
                strict=True,
            )
        )
    )
    invariants = (
        InvariantResult(
            "dynamic_bracket_found",
            no_bracket is None,
            True,
            None,
            no_bracket is None,
        ),
        InvariantResult(
            "maximum_quote_repricing_residual",
            maximum_residual,
            0.0,
            residual_limit,
            maximum_residual <= residual_limit,
        ),
        InvariantResult(
            "survival_is_nonincreasing",
            survival_monotone,
            True,
            None,
            survival_monotone,
        ),
    )
    violations = tuple(
        Violation(
            code=(
                "CDS_NO_BRACKET"
                if invariant.name == "dynamic_bracket_found"
                else f"CDS_{invariant.name.upper()}"
            ),
            message=f"CDS invariant failed: {invariant.name}",
            observed=str(invariant.observed),
            limit=str(invariant.expected),
        )
        for invariant in invariants
        if not invariant.passed
    )
    result = CDSBootstrapResult(
        curve=curve,
        calibration_nodes=tuple(calibration),
        maximum_repricing_residual=maximum_residual,
        recovery=recovery,
        no_bracket_maturity=no_bracket,
        discount_curve_run_id=discount_curve_run_id,
    )
    return EvidenceCertificate(
        status=decision_from_invariants(invariants),
        result=result,
        invariants=invariants,
        violations=violations,
        conventions=ConventionSchema(
            currency="fraction of CDS notional",
            rate_compounding="continuous discounting",
            day_count=f"{quotes[0].payments_per_year} premium periods per year",
            discounting_curve=(
                f"typed discount curve run {discount_curve_run_id}; "
                "log-DF interpolation"
            ),
            credit_convention=(
                "piecewise hazard; scheduled premium plus midpoint "
                "accrued-on-default; coupon plus upfront"
            ),
        ),
        tolerances=tolerance,
        input_manifest_hash=stable_manifest_hash(
            {
                "quotes": quotes,
                "recovery": recovery,
                "discount_curve": discount_curve,
                "discount_curve_run_id": discount_curve_run_id,
                "initial_hazard_upper": initial_hazard_upper,
                "maximum_hazard": maximum_hazard,
                "tolerance": tolerance,
            }
        ),
        implementation_version="ch07.cds_bootstrap.v0.10",
    )


def bootstrap_hazard_curve(
    quotes: dict[int, float],
    recovery: float = 0.40,
    discount_rate: float = 0.03,
) -> PiecewiseHazardCurve:
    discount_curve = flat_discount_curve(
        continuous_rate=discount_rate,
        horizon=float(max(quotes)),
    )
    certificate = bootstrap_cds_certificate(
        tuple(
            CDSQuote(maturity, spread)
            for maturity, spread in sorted(quotes.items())
        ),
        recovery=recovery,
        discount_curve=discount_curve,
        discount_curve_run_id="educational-flat-rate-adapter",
    )
    if certificate.result.curve is None:
        raise ValueError("NO_BRACKET")
    return certificate.result.curve


def cds_pv(
    curve: PiecewiseHazardCurve,
    maturity: int,
    spread: float,
    recovery: float,
    discount_rate: float,
) -> float:
    discount_curve = flat_discount_curve(
        continuous_rate=discount_rate,
        horizon=float(maturity),
    )
    return cds_legs(
        curve,
        CDSQuote(maturity, spread),
        recovery,
        discount_curve,
    ).net_pv


if __name__ == "__main__":
    example_quotes = (
        CDSQuote(1, 0.0100),
        CDSQuote(3, 0.0150),
        CDSQuote(5, 0.0100, upfront=0.035),
    )
    certificate = bootstrap_cds_certificate(
        example_quotes,
        discount_curve=flat_discount_curve(
            continuous_rate=0.03,
            horizon=5.0,
        ),
        discount_curve_run_id="example-flat-curve",
    )
    print("status=", certificate.status)
    print("hazards=", [node.hazard for node in certificate.result.calibration_nodes])
    print("maximum residual=", certificate.result.maximum_repricing_residual)
