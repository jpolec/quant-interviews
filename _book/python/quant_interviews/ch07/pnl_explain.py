from __future__ import annotations

from dataclasses import dataclass, field
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
    certificate_manifest,
    stable_manifest_hash,
)

COMPONENT_NAMES = (
    "carry",
    "roll_down",
    "curve_delta",
    "curve_cross_gamma",
    "basis",
    "credit_delta",
    "credit_convexity",
    "volatility",
    "cash_and_fixings",
    "exercise",
    "model_change",
    "data_correction",
)


@dataclass(frozen=True)
class StateManifest:
    artifact_type: Literal["position", "risk", "market"]
    run_id: str
    state_hash: str
    dependency_hashes: tuple[str, ...]
    record_count: int

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("state-manifest run_id must be non-empty")
        hashes = (self.state_hash, *self.dependency_hashes)
        if any(
            len(value) != 64
            or value.lower() != value
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError("state manifests require lowercase SHA-256 hashes")
        if not isinstance(self.record_count, int) or self.record_count <= 0:
            raise ValueError("state-manifest record_count must be positive")


@dataclass(frozen=True)
class PnLInputs:
    carry: float
    roll_down: float
    curve_dv01: NDArray[np.float64]
    curve_move_bp: NDArray[np.float64]
    curve_cross_gamma: NDArray[np.float64]
    basis_pnl: float
    cs01: float
    credit_spread_move_bp: float
    credit_convexity: float
    volatility_pnl: float
    cash_and_fixings_pnl: float
    exercise_pnl: float
    model_change_pnl: float
    data_correction_pnl: float
    actual_pnl: float
    position_manifest: StateManifest
    risk_manifest: StateManifest
    market_manifest: StateManifest
    component_evidence_ids: dict[str, tuple[str, ...]]
    not_applicable_reasons: dict[str, str] = field(default_factory=dict)
    stale_input_count: int = 0

    def __post_init__(self) -> None:
        dv01 = np.asarray(self.curve_dv01, dtype=float)
        move = np.asarray(self.curve_move_bp, dtype=float)
        gamma = np.asarray(self.curve_cross_gamma, dtype=float)
        if (
            dv01.ndim != 1
            or move.shape != dv01.shape
            or gamma.shape != (dv01.size, dv01.size)
            or not np.all(np.isfinite(dv01))
            or not np.all(np.isfinite(move))
            or not np.all(np.isfinite(gamma))
        ):
            raise ValueError("curve delta, move and cross-gamma shapes are invalid")
        scalar_values = (
            self.carry,
            self.roll_down,
            self.basis_pnl,
            self.cs01,
            self.credit_spread_move_bp,
            self.credit_convexity,
            self.volatility_pnl,
            self.cash_and_fixings_pnl,
            self.exercise_pnl,
            self.model_change_pnl,
            self.data_correction_pnl,
            self.actual_pnl,
        )
        if not all(np.isfinite(value) for value in scalar_values):
            raise ValueError("P&L inputs must be finite")
        if (
            not isinstance(self.stale_input_count, int)
            or self.stale_input_count < 0
        ):
            raise ValueError("stale_input_count must be non-negative")
        if self.position_manifest.artifact_type != "position":
            raise ValueError("position_manifest has the wrong artifact type")
        if self.risk_manifest.artifact_type != "risk":
            raise ValueError("risk_manifest has the wrong artifact type")
        if self.market_manifest.artifact_type != "market":
            raise ValueError("market_manifest has the wrong artifact type")
        unknown_evidence = set(self.component_evidence_ids) - set(COMPONENT_NAMES)
        unknown_reasons = set(self.not_applicable_reasons) - set(COMPONENT_NAMES)
        if unknown_evidence or unknown_reasons:
            raise ValueError("unknown P&L component evidence key")
        for name in COMPONENT_NAMES:
            evidence = self.component_evidence_ids.get(name, ())
            reason = self.not_applicable_reasons.get(name)
            if evidence and reason is not None:
                raise ValueError(
                    f"component {name} cannot have evidence and not-applicable reason"
                )
            if not evidence and (reason is None or not reason.strip()):
                raise ValueError(
                    f"component {name} requires evidence or a not-applicable reason"
                )
            if any(not identifier.strip() for identifier in evidence):
                raise ValueError("component evidence IDs must be non-empty")
        object.__setattr__(self, "curve_dv01", dv01)
        object.__setattr__(self, "curve_move_bp", move)
        object.__setattr__(self, "curve_cross_gamma", gamma)


@dataclass(frozen=True)
class PnLExplainResult:
    components: dict[str, float]
    explained_pnl: float
    actual_pnl: float
    residual: float
    residual_fraction_of_actual: float
    snapshot_alignment: bool
    missing_component_count: int
    stale_input_count: int


def pnl_component_ledger(inputs: PnLInputs) -> dict[str, float]:
    """Return the complete governed P&L component ledger."""

    curve_delta = -float(
        np.dot(inputs.curve_dv01, inputs.curve_move_bp)
    )
    curve_cross_gamma = 0.5 * float(
        inputs.curve_move_bp
        @ inputs.curve_cross_gamma
        @ inputs.curve_move_bp
    )
    credit_delta = -inputs.cs01 * inputs.credit_spread_move_bp
    credit_gamma = (
        0.5
        * inputs.credit_convexity
        * inputs.credit_spread_move_bp**2
    )
    return {
        "carry": inputs.carry,
        "roll_down": inputs.roll_down,
        "curve_delta": curve_delta,
        "curve_cross_gamma": curve_cross_gamma,
        "basis": inputs.basis_pnl,
        "credit_delta": credit_delta,
        "credit_convexity": credit_gamma,
        "volatility": inputs.volatility_pnl,
        "cash_and_fixings": inputs.cash_and_fixings_pnl,
        "exercise": inputs.exercise_pnl,
        "model_change": inputs.model_change_pnl,
        "data_correction": inputs.data_correction_pnl,
    }


def explain_certificate(
    inputs: PnLInputs,
    *,
    absolute_residual_limit: float,
    relative_residual_limit: float,
    tolerance: TolerancePolicy = TolerancePolicy(1.0e-8, 1.0e-10),
) -> EvidenceCertificate[PnLExplainResult]:
    if (
        not np.isfinite(absolute_residual_limit)
        or not np.isfinite(relative_residual_limit)
        or absolute_residual_limit < 0.0
        or not 0.0 <= relative_residual_limit <= 1.0
    ):
        raise ValueError("invalid residual limits")
    components = pnl_component_ledger(inputs)
    explained = float(sum(components.values()))
    residual = inputs.actual_pnl - explained
    denominator = max(abs(inputs.actual_pnl), tolerance.absolute)
    residual_fraction = abs(residual) / denominator
    risk_dependencies = inputs.risk_manifest.dependency_hashes
    aligned = (
        inputs.position_manifest.state_hash in risk_dependencies
        and inputs.market_manifest.state_hash in risk_dependencies
    )
    missing_components = tuple(
        name
        for name in COMPONENT_NAMES
        if not inputs.component_evidence_ids.get(name)
        and not inputs.not_applicable_reasons.get(name)
    )
    invariants = (
        InvariantResult(
            "absolute_residual",
            abs(residual),
            0.0,
            absolute_residual_limit,
            abs(residual) <= absolute_residual_limit,
        ),
        InvariantResult(
            "relative_residual",
            residual_fraction,
            0.0,
            relative_residual_limit,
            residual_fraction <= relative_residual_limit,
        ),
        InvariantResult(
            "snapshot_alignment",
            aligned,
            True,
            None,
            aligned,
        ),
        InvariantResult(
            "component_completeness",
            len(missing_components),
            0,
            None,
            len(missing_components) == 0,
        ),
        InvariantResult(
            "stale_input_count",
            inputs.stale_input_count,
            0,
            None,
            inputs.stale_input_count == 0,
        ),
    )
    violations = tuple(
        Violation(
            code=f"PNL_{invariant.name.upper()}",
            message=f"P&L explain invariant failed: {invariant.name}",
            observed=str(invariant.observed),
            limit=str(invariant.tolerance or invariant.expected),
        )
        for invariant in invariants
        if not invariant.passed
    )
    result = PnLExplainResult(
        components=components,
        explained_pnl=explained,
        actual_pnl=inputs.actual_pnl,
        residual=residual,
        residual_fraction_of_actual=residual_fraction,
        snapshot_alignment=aligned,
        missing_component_count=len(missing_components),
        stale_input_count=inputs.stale_input_count,
    )
    return EvidenceCertificate(
        status=decision_from_invariants(invariants),
        result=result,
        invariants=invariants,
        violations=violations,
        conventions=ConventionSchema(
            currency="book reporting currency",
            rate_compounding="DV01 per basis point; cross-gamma per bp squared",
            day_count="valuation interval",
            discounting_curve="risk-run curve bundle",
            projection_curve="risk-run projection and basis bundle",
            credit_convention="CS01 per basis point",
            exercise_convention="cash and lifecycle events explicitly booked",
        ),
        tolerances=TolerancePolicy(
            absolute=absolute_residual_limit,
            relative=relative_residual_limit,
        ),
        input_manifest_hash=stable_manifest_hash(
            certificate_manifest(
                inputs=inputs,
                algorithm_config={"component_names": COMPONENT_NAMES},
                policy={
                    "absolute_residual_limit": absolute_residual_limit,
                    "relative_residual_limit": relative_residual_limit,
                    "tolerance": tolerance,
                    "policy_version": "ch07.pnl-policy.v0.10",
                },
                conventions={
                    "curve_delta_sign": "-DV01_times_move_bp",
                    "curve_gamma_units": "currency_per_bp_squared",
                    "credit_delta_sign": "-CS01_times_move_bp",
                },
                implementation_version="ch07.pnl_explain.v0.10",
            )
        ),
        implementation_version="ch07.pnl_explain.v0.10",
    )


def explain(inputs: PnLInputs) -> dict[str, float | str]:
    """Return an explicitly ungoverned arithmetic preview without a gate."""

    components = pnl_component_ledger(inputs)
    explained = float(sum(components.values()))
    residual = inputs.actual_pnl - explained
    return {
        **components,
        "explained_pnl": explained,
        "residual": residual,
        "actual_pnl": inputs.actual_pnl,
        "status": "UNGOVERNED",
    }


if __name__ == "__main__":
    position_hash = stable_manifest_hash("sample-position")
    market_hash = stable_manifest_hash("sample-market")
    risk_hash = stable_manifest_hash("sample-risk")
    sample = PnLInputs(
        carry=12_000.0,
        roll_down=5_000.0,
        curve_dv01=np.array([800.0, 1_200.0, 500.0]),
        curve_move_bp=np.array([2.0, -1.0, 3.0]),
        curve_cross_gamma=np.diag([4.0, 6.0, 2.0]),
        basis_pnl=-250.0,
        cs01=900.0,
        credit_spread_move_bp=4.0,
        credit_convexity=3.0,
        volatility_pnl=-75.0,
        cash_and_fixings_pnl=400.0,
        exercise_pnl=0.0,
        model_change_pnl=0.0,
        data_correction_pnl=0.0,
        actual_pnl=13_500.0,
        position_manifest=StateManifest(
            "position", "sample-position-run", position_hash, (), 1
        ),
        risk_manifest=StateManifest(
            "risk",
            "sample-risk-run",
            risk_hash,
            (position_hash, market_hash),
            2,
        ),
        market_manifest=StateManifest(
            "market", "sample-market-run", market_hash, (), 1
        ),
        component_evidence_ids={
            name: (f"sample:{name}",) for name in COMPONENT_NAMES
        },
    )
    result = explain_certificate(
        sample,
        absolute_residual_limit=2_000.0,
        relative_residual_limit=0.15,
    )
    print("status=", result.status)
    print("residual=", result.result.residual)
    print("residual_fraction=", result.result.residual_fraction_of_actual)
