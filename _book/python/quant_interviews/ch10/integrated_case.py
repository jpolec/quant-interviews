from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .benchmarks import execution_benchmarks
from .impact_models import square_root_capacity
from .order_choice import OrderChoiceInputs, choose_order_type
from .pretrade import OrderIntent, RiskLimits, pretrade_check
from .queue_fill import QueueModel
from .smart_router import VenueQuote, route_order
from .tca import tca_decomposition


FLOAT_SIGNIFICANT_DIGITS = 12


@dataclass(frozen=True)
class ExecutionGateEvidence:
    gate_id: str
    required: bool
    observed: float
    operator: str
    signed_limit: float
    unit: str
    owner: str
    status: str


@dataclass(frozen=True)
class ExecutionDecisionPacket:
    input_snapshot_hash: str
    result_bundle_hash: str
    policy_hash: str
    release_evidence_hash: str
    digest_algorithm: str
    signature_status: str
    source_hashes: tuple[tuple[str, str], ...]
    route_hash: str
    queue_fill_probability: float
    order_type: str
    capacity_optimal_quantity: float
    capacity_break_even_quantity: float
    capacity_optimal_status: str
    capacity_break_even_status: str
    route_total_submitted: int
    route_expected_fill: float
    route_completion_fraction: float
    route_overfill_exposure: int
    implementation_shortfall: float
    tca_all_in_cost_bps: float
    tca_residual_bps: float
    child_pretrade_breaches: tuple[tuple[str, tuple[str, ...]], ...]
    aggregate_pretrade_breaches: tuple[str, ...]
    gate_evidence: tuple[ExecutionGateEvidence, ...]
    decision: str
    decision_reason: str

    @property
    def fixture_hash(self) -> str:
        """Compatibility alias for the full input-snapshot digest."""

        return self.input_snapshot_hash

    @property
    def pretrade_breaches(self) -> tuple[str, ...]:
        return self.aggregate_pretrade_breaches


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, np.ndarray):
        return [_canonical_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("evidence bundles cannot contain non-finite values")
        text = (
            "0"
            if value == 0.0
            else format(value, f".{FLOAT_SIGNIFICANT_DIGITS}g")
        )
        return {"__float__": text}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported evidence type: {type(value).__name__}")


def _digest(value: Any) -> str:
    payload = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _source_hashes() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[4]
    paths = [
        *sorted((root / "_book/python/quant_interviews/ch10").glob("*.py")),
        *sorted((root / "_book/sql/ch10").glob("*.sql")),
        *sorted((root / "_book/cpp/ch10").glob("*.cpp")),
        root / "_book/json/ch10/tool_contract_library.json",
    ]
    return tuple(
        (
            str(path.relative_to(root)),
            sha256(path.read_bytes()).hexdigest(),
        )
        for path in paths
    )


def _reduce_gates(
    gates: tuple[ExecutionGateEvidence, ...],
) -> tuple[str, str]:
    required_failures = [
        gate.gate_id
        for gate in gates
        if gate.required and gate.status == "REJECT"
    ]
    optional_failures = [
        gate.gate_id
        for gate in gates
        if not gate.required and gate.status == "REJECT"
    ]
    if required_failures:
        return "REJECT", "required gates failed: " + ", ".join(required_failures)
    if optional_failures:
        return (
            "CONDITIONAL_PASS",
            "optional gates require review: " + ", ".join(optional_failures),
        )
    return "PASS", "all required and optional gates passed"


def governed_execution_fixture(
    *,
    queue_ahead_lots: int = 3,
    route_completion_required: bool = False,
) -> ExecutionDecisionPacket:
    source_hashes = _source_hashes()
    venue_quotes = (
        VenueQuote("A", 800, 0.85, 1.0, 0.1, 0.2, 0.6, 0.1, 2.0, 0.1),
        VenueQuote("B", 700, 0.65, 1.1, 0.0, 0.0, 1.2, 0.2, 2.3, 0.2),
    )
    fixture = {
        "fixture_version": "ch10-governed-v0.10",
        "symbol": "XYZ",
        "side": "BUY",
        "decision_time": 0.0,
        "arrival_time": 1.0,
        "end_time": 3.0,
        "knowledge_cutoff": "2026-07-30T08:00:00Z",
        "target_quantity": 1000,
        "decision_price": 100.0,
        "arrival_price": 100.1,
        "terminal_price": 100.4,
        "market_volume": 100_000.0,
        "queue": {
            "queue_ahead_lots": queue_ahead_lots,
            "marketable_lots_per_second": 2.0,
            "cancellation_ahead_lots_per_second": 0.5,
            "horizon_seconds": 2.0,
            "model_version": "ordered-ctmc-v0.10",
        },
        "order_choice": {
            "half_spread_bps": 2.0,
            "market_fee_bps": 0.2,
            "passive_fee_bps": 0.1,
            "passive_rebate_bps": 0.2,
            "market_impact_bps": 0.5,
            "adverse_selection_bps_if_filled": 0.8,
            "nonfill_opportunity_cost_bps": 3.5,
            "replacement_cost_bps": 0.8,
        },
        "capacity": {
            "alpha_per_unit": 0.012,
            "fixed_cost_per_unit": 0.002,
            "impact_coefficient": 0.8,
            "volatility": 0.02,
            "daily_volume": 100_000.0,
            "maximum_calibration_participation": 0.25,
        },
        "routing": {
            "venues": venue_quotes,
            "non_completion_penalty_bps": 10.0,
            "minimum_expected_completion_fraction": 0.75,
            "release_mode": "SIMULTANEOUS_HARD_CAP",
        },
        "benchmark": {
            "child_prices": [100.15, 100.20],
            "child_quantities": [400.0, 400.0],
            "market_prices": [100.00, 100.20, 100.40],
            "market_volumes": [30_000.0, 40_000.0, 30_000.0],
            "market_timestamps": [1.0, 2.0, 3.0],
        },
        "tca": {
            "fill_times": [1.2, 1.8],
            "fill_quantities": [400.0, 400.0],
            "fill_prices": [100.15, 100.20],
            "venues": ["A", "B"],
            "fee_bps": [0.12, 0.12],
            "midpoint_at_fills": [100.12, 100.14],
            "counterfactual_midpoint_at_fills": [100.11, 100.13],
            "post_trade_midpoints": [100.08, 100.08],
            "markout_horizon_seconds": 60.0,
            "reconciliation_tolerance_bps": 1e-10,
        },
        "risk": {
            "limit_snapshot_version": "limits-v4",
            "max_order_quantity": 10_000,
            "max_notional": 2_000_000.0,
            "max_participation": 0.02,
            "price_collar_bps": 30.0,
            "max_market_volume_age_ms": 500,
            "max_position_after_order": 50_000.0,
            "max_credit_notional": 2_000_000.0,
            "starting_position": 9_000.0,
            "available_credit_notional": 1_500_000.0,
            "existing_working_quantity": 0,
        },
        "policy": {
            "policy_version": "execution-release-v0.10",
            "route_completion_required": route_completion_required,
        },
        "source_hashes": source_hashes,
    }
    input_snapshot_hash = _digest(fixture)

    queue_probability = QueueModel(
        queue_ahead_lots,
        marketable_lots_per_second=2.0,
        cancellation_ahead_lots_per_second=0.5,
    ).fill_probability(2.0)
    order_decision = choose_order_type(
        OrderChoiceInputs(
            fill_probability=queue_probability,
            **fixture["order_choice"],
        )
    )
    capacity = square_root_capacity(**fixture["capacity"])
    routing = fixture["routing"]
    route = route_order(
        fixture["target_quantity"],
        list(venue_quotes),
        non_completion_penalty_bps=routing["non_completion_penalty_bps"],
        minimum_expected_completion_fraction=routing[
            "minimum_expected_completion_fraction"
        ],
        release_mode=routing["release_mode"],
    )
    route_hash = _digest(route)

    benchmark = fixture["benchmark"]
    benchmarks = execution_benchmarks(
        "buy",
        fixture["decision_price"],
        fixture["arrival_price"],
        np.asarray(benchmark["child_prices"]),
        np.asarray(benchmark["child_quantities"]),
        np.asarray(benchmark["market_prices"]),
        np.asarray(benchmark["market_volumes"]),
        fixture["target_quantity"],
        market_timestamps=np.asarray(benchmark["market_timestamps"]),
        decision_time=fixture["decision_time"],
        arrival_time=fixture["arrival_time"],
        end_time=fixture["end_time"],
    )

    tca_inputs = fixture["tca"]
    executed = float(sum(tca_inputs["fill_quantities"]))
    accounting_price_cost_bps = (
        sum(
            quantity * (price - fixture["arrival_price"])
            for quantity, price in zip(
                tca_inputs["fill_quantities"],
                tca_inputs["fill_prices"],
                strict=True,
            )
        )
        / (executed * fixture["arrival_price"])
        * 10_000.0
    )
    ledger_all_in_cost_bps = accounting_price_cost_bps + 0.12
    tca = tca_decomposition(
        side="buy",
        arrival_price=fixture["arrival_price"],
        order_quantity=fixture["target_quantity"],
        market_volume=fixture["market_volume"],
        fill_times=np.asarray(tca_inputs["fill_times"]),
        fill_quantities=np.asarray(tca_inputs["fill_quantities"]),
        fill_prices=np.asarray(tca_inputs["fill_prices"]),
        venues=tuple(tca_inputs["venues"]),
        fee_bps=np.asarray(tca_inputs["fee_bps"]),
        midpoint_at_fills=np.asarray(tca_inputs["midpoint_at_fills"]),
        counterfactual_midpoint_at_fills=np.asarray(
            tca_inputs["counterfactual_midpoint_at_fills"]
        ),
        ledger_all_in_cost_bps=ledger_all_in_cost_bps,
        reconciliation_tolerance_bps=tca_inputs[
            "reconciliation_tolerance_bps"
        ],
        post_trade_midpoints=np.asarray(tca_inputs["post_trade_midpoints"]),
        markout_horizon_seconds=tca_inputs["markout_horizon_seconds"],
    )

    risk = fixture["risk"]
    limits = RiskLimits(
        max_order_quantity=risk["max_order_quantity"],
        max_notional=risk["max_notional"],
        max_participation=risk["max_participation"],
        price_collar_bps=risk["price_collar_bps"],
        max_market_volume_age_ms=risk["max_market_volume_age_ms"],
        limit_snapshot_version=risk["limit_snapshot_version"],
        max_position_after_order=risk["max_position_after_order"],
        max_credit_notional=risk["max_credit_notional"],
    )
    child_breaches: list[tuple[str, tuple[str, ...]]] = []
    cumulative_quantity = risk["existing_working_quantity"]
    venue_prices = {"A": 100.11, "B": 100.12}
    for allocation in route.allocations:
        cumulative_quantity += allocation.submitted_quantity
        child = OrderIntent(
            quantity=allocation.submitted_quantity,
            price=venue_prices[allocation.venue],
            reference_price=fixture["arrival_price"],
            forecast_market_volume=fixture["market_volume"],
            market_volume_age_ms=120,
            limit_snapshot_version=risk["limit_snapshot_version"],
            projected_position=risk["starting_position"] + cumulative_quantity,
            available_credit_notional=risk["available_credit_notional"],
            self_trade_risk=False,
        )
        child_breaches.append(
            (allocation.venue, tuple(pretrade_check(child, limits)))
        )
    aggregate = OrderIntent(
        quantity=route.total_submitted_quantity + risk["existing_working_quantity"],
        price=max(venue_prices[allocation.venue] for allocation in route.allocations),
        reference_price=fixture["arrival_price"],
        forecast_market_volume=fixture["market_volume"],
        market_volume_age_ms=120,
        limit_snapshot_version=risk["limit_snapshot_version"],
        projected_position=risk["starting_position"]
        + route.total_submitted_quantity
        + risk["existing_working_quantity"],
        available_credit_notional=risk["available_credit_notional"],
        self_trade_risk=False,
    )
    aggregate_breaches = tuple(pretrade_check(aggregate, limits))

    result_payload = {
        "queue_probability": queue_probability,
        "order_decision": order_decision,
        "capacity": capacity,
        "route": route,
        "route_hash": route_hash,
        "benchmarks": benchmarks,
        "tca": tca,
        "child_pretrade_breaches": child_breaches,
        "aggregate_pretrade_breaches": aggregate_breaches,
    }
    result_bundle_hash = _digest(result_payload)
    policy = {
        "policy_version": "execution-release-v0.10",
        "route_completion_required": route_completion_required,
        "gates": [
            "queue.probability",
            "order.choice",
            "capacity.optimal",
            "capacity.break_even",
            "route.hard_cap",
            "route.expected_completion",
            "tca.reconciliation",
            "pretrade.child",
            "pretrade.aggregate",
        ],
    }
    policy_hash = _digest(policy)
    gates = (
        ExecutionGateEvidence(
            "queue.probability", True, queue_probability, ">=", 0.0, "probability", "model-risk", "PASS"
        ),
        ExecutionGateEvidence(
            "order.choice", True, float(order_decision.status != "REJECT"), ">=", 1.0, "flag", "execution", "PASS" if order_decision.status != "REJECT" else "REJECT"
        ),
        ExecutionGateEvidence(
            "capacity.optimal", True, float(capacity.optimal_status == "PASS"), ">=", 1.0, "flag", "research", "PASS" if capacity.optimal_status == "PASS" else "REJECT"
        ),
        ExecutionGateEvidence(
            "capacity.break_even", False, float(capacity.break_even_status == "PASS"), ">=", 1.0, "flag", "research", "PASS" if capacity.break_even_status == "PASS" else "REJECT"
        ),
        ExecutionGateEvidence(
            "route.hard_cap", True, float(route.maximum_overfill_exposure), "<=", 0.0, "shares", "execution-platform", "PASS" if route.maximum_overfill_exposure == 0 else "REJECT"
        ),
        ExecutionGateEvidence(
            "route.expected_completion", route_completion_required, route.expected_completion_fraction, ">=", 1.0, "fraction", "execution", "PASS" if route.expected_completion_fraction >= 1.0 - 1e-12 else "REJECT"
        ),
        ExecutionGateEvidence(
            "tca.reconciliation", True, abs(tca.residual_bps), "<=", tca.reconciliation_tolerance_bps, "bps", "TCA", "PASS" if tca.status == "PASS" else "REJECT"
        ),
        ExecutionGateEvidence(
            "pretrade.child", True, float(sum(bool(item[1]) for item in child_breaches)), "<=", 0.0, "breach count", "risk", "PASS" if not any(item[1] for item in child_breaches) else "REJECT"
        ),
        ExecutionGateEvidence(
            "pretrade.aggregate", True, float(len(aggregate_breaches)), "<=", 0.0, "breach count", "risk", "PASS" if not aggregate_breaches else "REJECT"
        ),
    )
    decision, reason = _reduce_gates(gates)
    release_evidence_hash = _digest(
        {
            "input_snapshot_hash": input_snapshot_hash,
            "result_bundle_hash": result_bundle_hash,
            "policy_hash": policy_hash,
            "route_hash": route_hash,
            "gate_evidence": gates,
            "decision": decision,
        }
    )
    return ExecutionDecisionPacket(
        input_snapshot_hash=input_snapshot_hash,
        result_bundle_hash=result_bundle_hash,
        policy_hash=policy_hash,
        release_evidence_hash=release_evidence_hash,
        digest_algorithm="SHA-256",
        signature_status="UNSIGNED_LOCAL_EVIDENCE",
        source_hashes=source_hashes,
        route_hash=route_hash,
        queue_fill_probability=queue_probability,
        order_type=order_decision.decision,
        capacity_optimal_quantity=capacity.optimal_quantity,
        capacity_break_even_quantity=capacity.break_even_quantity,
        capacity_optimal_status=capacity.optimal_status,
        capacity_break_even_status=capacity.break_even_status,
        route_total_submitted=route.total_submitted_quantity,
        route_expected_fill=route.expected_filled_quantity,
        route_completion_fraction=route.expected_completion_fraction,
        route_overfill_exposure=route.maximum_overfill_exposure,
        implementation_shortfall=benchmarks.implementation_shortfall,
        tca_all_in_cost_bps=tca.all_in_cost_bps,
        tca_residual_bps=tca.residual_bps,
        child_pretrade_breaches=tuple(child_breaches),
        aggregate_pretrade_breaches=aggregate_breaches,
        gate_evidence=gates,
        decision=decision,
        decision_reason=reason,
    )


if __name__ == "__main__":
    print(json.dumps(asdict(governed_execution_fixture()), indent=2))
