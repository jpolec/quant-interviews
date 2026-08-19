from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np

from .attribution import reconcile
from .capacity import capacity, capacity_coefficient
from .portfolio import portfolio_certificate
from .robust import robust_mean_variance
from .stress import coverage
from .tail_risk import historical_var_es
from .turnover import realized_cost


@dataclass(frozen=True)
class ChapterInputs:
    expected_returns: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    prior_weights: tuple[float, ...]
    uncertainty_diagonal: tuple[float, ...]
    linear_costs: tuple[float, ...]
    impact_diagonal: tuple[float, ...]
    scenario_losses: tuple[tuple[float, ...], ...]
    capacity_volatility: tuple[float, ...]
    capacity_dollar_adv: tuple[float, ...]
    capacity_impact: tuple[float, ...]
    pnl_total: float
    pnl_components: tuple[tuple[str, float], ...]
    nav: float
    coverage_fractions: tuple[float, ...]
    frozen_position_count: int
    repriced_position_count: int
    position_snapshot_sha256: str
    scenario_set_sha256: str
    code_version: str
    dependency_lock_sha256: str


@dataclass(frozen=True)
class ReleasePolicy:
    version: str = "ch09-release-policy-v0.10"
    scenario_es_limit: float = 0.12
    turnover_limit: float = 0.20
    estimated_cost_limit: float = 0.001
    minimum_capacity_aum: float = 1.0e9
    minimum_scenario_coverage: float = 0.999
    pnl_absolute_tolerance: float = 50.0
    pnl_nav_tolerance: float = 0.00002
    pnl_gross_tolerance: float = 0.04
    optimizer_tolerance: float = 1.0e-7
    condition_number_limit: float = 1.0e8


@dataclass(frozen=True)
class GateRecord:
    name: str
    state: str
    observed: float | int | str | bool
    operator: str
    threshold: float | int | str | bool
    unit: str
    reason_code: str
    evidence_id: str
    hard: bool = True


@dataclass(frozen=True)
class ChapterDecisionPacket:
    snapshot_hash: str
    solver_config_hash: str
    evidence_bundle_hash: str
    policy_version: str
    nominal_weights: tuple[float, ...]
    robust_weights: tuple[float, ...]
    robust_support_residual: float
    robust_stationarity_residual: float
    one_way_turnover: float
    estimated_cost: float
    scenario_var: float
    scenario_es: float
    scenario_coverage: float
    capacity_break_even_aum: float
    pnl_residual: float
    pnl_residual_to_nav: float
    gates: tuple[GateRecord, ...]
    final_decision: str


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def default_inputs() -> ChapterInputs:
    return ChapterInputs(
        expected_returns=(0.075, 0.068, 0.055, 0.047),
        covariance=(
            (0.040, 0.012, 0.008, 0.004),
            (0.012, 0.032, 0.010, 0.006),
            (0.008, 0.010, 0.025, 0.007),
            (0.004, 0.006, 0.007, 0.018),
        ),
        prior_weights=(0.30, 0.25, 0.25, 0.20),
        uncertainty_diagonal=(0.025, 0.020, 0.018, 0.015),
        linear_costs=(0.0010, 0.0012, 0.0008, 0.0007),
        impact_diagonal=(0.020, 0.018, 0.015, 0.012),
        scenario_losses=(
            (0.03, 0.02, 0.01, 0.01),
            (0.08, 0.05, 0.03, 0.02),
            (0.16, 0.09, 0.06, 0.04),
            (0.02, 0.03, 0.04, 0.02),
            (0.12, 0.14, 0.07, 0.05),
            (0.05, 0.04, 0.02, 0.02),
            (0.20, 0.11, 0.09, 0.06),
            (0.04, 0.02, 0.02, 0.01),
            (0.10, 0.07, 0.05, 0.03),
            (0.01, 0.01, 0.01, 0.01),
        ),
        capacity_volatility=(0.22, 0.19, 0.16, 0.12),
        capacity_dollar_adv=(2.0e9, 1.5e9, 1.1e9, 0.9e9),
        capacity_impact=(0.45, 0.50, 0.40, 0.35),
        pnl_total=1250.0,
        pnl_components=(
            ("market", 710.0),
            ("carry", 185.0),
            ("trading", 95.0),
            ("financing", -30.0),
            ("fx_translation", 120.0),
            ("fees", -55.0),
        ),
        nav=2_000_000.0,
        coverage_fractions=(1.0, 1.0, 1.0, 1.0),
        frozen_position_count=4,
        repriced_position_count=4,
        position_snapshot_sha256=(
            "1b9cb87973bf78e6f8ee5eaedfc1809b88f052c3b8966dcb3c738c783f6715d1"
        ),
        scenario_set_sha256=(
            "81612ee60fef349103fc3e35fb27d02c72e38591ae5817bd405d47f5b9c51627"
        ),
        code_version="ch09-v0.10",
        dependency_lock_sha256=(
            "d6a7531b3a7ebfa384a9aa482668aca9089257d293a947a81fd91cc36c6a91fa"
        ),
    )


def _gate(
    name: str,
    passed: bool,
    observed: float | int | str | bool,
    operator: str,
    threshold: float | int | str | bool,
    unit: str,
    evidence_id: str,
    *,
    hard: bool = True,
    review: bool = False,
) -> GateRecord:
    state = "REVIEW" if review and passed else "PASS" if passed else "REJECT"
    reason = f"{name}_{state}"
    return GateRecord(
        name=name,
        state=state,
        observed=observed,
        operator=operator,
        threshold=threshold,
        unit=unit,
        reason_code=reason,
        evidence_id=evidence_id,
        hard=hard,
    )


def evaluate_release_gates(gates: tuple[GateRecord, ...]) -> str:
    if not gates or any(not gate.evidence_id for gate in gates):
        return "REJECT"
    if any(gate.hard and gate.state != "PASS" for gate in gates):
        return "REJECT"
    if any(gate.state == "REVIEW" for gate in gates):
        return "REVIEW"
    return "PASS"


def chapter_decision_packet(
    inputs: ChapterInputs | None = None,
    policy: ReleasePolicy | None = None,
) -> ChapterDecisionPacket:
    source = default_inputs() if inputs is None else inputs
    release_policy = ReleasePolicy() if policy is None else policy
    snapshot_hash = _canonical_digest(
        {
            "inputs": asdict(source),
            "policy": asdict(release_policy),
        }
    )
    solver_config_hash = _canonical_digest(
        {
            "solver": "scipy.optimize.SLSQP",
            "optimality_tolerance": release_policy.optimizer_tolerance,
            "condition_number_limit": release_policy.condition_number_limit,
            "code_version": source.code_version,
            "dependency_lock_sha256": source.dependency_lock_sha256,
        }
    )

    expected_returns = np.asarray(source.expected_returns, dtype=float)
    covariance = np.asarray(source.covariance, dtype=float)
    prior_weights = np.asarray(source.prior_weights, dtype=float)
    nominal = portfolio_certificate(
        expected_returns,
        covariance,
        "GMV",
        condition_number_limit=release_policy.condition_number_limit,
        tolerance=release_policy.optimizer_tolerance,
    )
    robust = robust_mean_variance(
        expected_returns,
        covariance,
        np.diag(source.uncertainty_diagonal),
        radius=0.8,
        risk_aversion=3.0,
        uncertainty_set_version="ch09-ellipsoid-v0.10",
        optimality_tolerance=release_policy.optimizer_tolerance,
    )
    trade = robust.weights - prior_weights
    one_way_turnover = 0.5 * float(np.abs(trade).sum())
    estimated_cost = realized_cost(
        trade,
        np.asarray(source.linear_costs, dtype=float),
        np.diag(source.impact_diagonal),
    )
    scenario_var, scenario_es = historical_var_es(
        np.asarray(source.scenario_losses, dtype=float) @ robust.weights,
        alpha=0.90,
    )
    scenario_coverage = coverage(
        np.ones(len(source.coverage_fractions)),
        np.asarray(source.coverage_fractions, dtype=float),
    )
    impact_coefficient = capacity_coefficient(
        trade,
        np.asarray(source.capacity_volatility, dtype=float),
        np.asarray(source.capacity_dollar_adv, dtype=float),
        np.asarray(source.capacity_impact, dtype=float),
    )
    capacity_result = capacity(0.060, 0.008, impact_coefficient)
    pnl = reconcile(
        source.pnl_total,
        dict(source.pnl_components),
        nav=source.nav,
        absolute_tolerance=release_policy.pnl_absolute_tolerance,
        nav_tolerance=release_policy.pnl_nav_tolerance,
        gross_tolerance=release_policy.pnl_gross_tolerance,
    )
    evidence = f"snapshot:{snapshot_hash}"
    gates = (
        _gate(
            "INPUT_SNAPSHOT",
            len(snapshot_hash) == 64 and len(solver_config_hash) == 64,
            snapshot_hash,
            "SHA256",
            True,
            "digest",
            evidence,
        ),
        _gate(
            "NOMINAL_SOLVER",
            nominal.status == "PASS",
            nominal.optimizer_weight_residual_l2,
            "<=",
            10.0 * release_policy.optimizer_tolerance,
            "weight_l2",
            f"portfolio:{nominal.input_sha256}",
        ),
        _gate(
            "ROBUST_SOLVER",
            robust.status == "PASS",
            robust.stationarity_residual,
            "<=",
            release_policy.optimizer_tolerance,
            "gradient",
            f"solver:{solver_config_hash}",
        ),
        _gate(
            "POSITION_COVERAGE",
            source.repriced_position_count == source.frozen_position_count,
            source.repriced_position_count,
            "==",
            source.frozen_position_count,
            "positions",
            evidence,
        ),
        _gate(
            "SCENARIO_COVERAGE",
            scenario_coverage >= release_policy.minimum_scenario_coverage,
            scenario_coverage,
            ">=",
            release_policy.minimum_scenario_coverage,
            "fraction",
            f"scenario:{source.scenario_set_sha256}",
        ),
        _gate(
            "ESTIMATED_COST",
            estimated_cost <= release_policy.estimated_cost_limit,
            estimated_cost,
            "<=",
            release_policy.estimated_cost_limit,
            "return_fraction",
            evidence,
        ),
        _gate(
            "CAPACITY",
            capacity_result.break_even_aum >= release_policy.minimum_capacity_aum,
            capacity_result.break_even_aum,
            ">=",
            release_policy.minimum_capacity_aum,
            "USD_AUM",
            evidence,
        ),
        _gate(
            "SCENARIO_ES",
            scenario_es <= release_policy.scenario_es_limit,
            scenario_es,
            "<=",
            release_policy.scenario_es_limit,
            "loss_fraction",
            f"scenario:{source.scenario_set_sha256}",
        ),
        _gate(
            "TURNOVER",
            one_way_turnover <= release_policy.turnover_limit,
            one_way_turnover,
            "<=",
            release_policy.turnover_limit,
            "one_way_fraction",
            evidence,
        ),
        _gate(
            "PNL_RECONCILIATION",
            pnl.status == "PASS",
            pnl.residual,
            "all_tolerances",
            release_policy.pnl_absolute_tolerance,
            "base_currency",
            evidence,
        ),
        _gate(
            "VAR_THRESHOLD_REVIEW",
            scenario_var <= scenario_es,
            scenario_var,
            "<=",
            scenario_es,
            "loss_fraction",
            f"scenario:{source.scenario_set_sha256}",
            hard=False,
            review=False,
        ),
    )
    final_decision = evaluate_release_gates(gates)
    metrics = {
        "snapshot_hash": snapshot_hash,
        "solver_config_hash": solver_config_hash,
        "nominal_weights": nominal.weights.tolist(),
        "robust_weights": robust.weights.tolist(),
        "one_way_turnover": one_way_turnover,
        "estimated_cost": estimated_cost,
        "scenario_var": scenario_var,
        "scenario_es": scenario_es,
        "scenario_coverage": scenario_coverage,
        "capacity_break_even_aum": capacity_result.break_even_aum,
        "pnl_residual": pnl.residual,
        "gates": [asdict(gate) for gate in gates],
        "decision": final_decision,
    }
    evidence_bundle_hash = _canonical_digest(metrics)
    return ChapterDecisionPacket(
        snapshot_hash=snapshot_hash,
        solver_config_hash=solver_config_hash,
        evidence_bundle_hash=evidence_bundle_hash,
        policy_version=release_policy.version,
        nominal_weights=tuple(float(value) for value in nominal.weights),
        robust_weights=tuple(float(value) for value in robust.weights),
        robust_support_residual=robust.support_function_residual,
        robust_stationarity_residual=robust.stationarity_residual,
        one_way_turnover=one_way_turnover,
        estimated_cost=estimated_cost,
        scenario_var=scenario_var,
        scenario_es=scenario_es,
        scenario_coverage=scenario_coverage,
        capacity_break_even_aum=capacity_result.break_even_aum,
        pnl_residual=pnl.residual,
        pnl_residual_to_nav=pnl.residual_to_nav,
        gates=gates,
        final_decision=final_decision,
    )


def main() -> None:
    print(asdict(chapter_decision_packet()))


if __name__ == "__main__":
    main()
