from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Mapping

from .agent_governance import (
    IdempotencyStore,
    ToolCall,
    ToolPolicy,
    execute_tool,
)
from .bitemporal import BitemporalRow, SelectionStatus, as_of_value
from .corporate_actions import (
    ActionStatus,
    ActionType,
    ApplicationStatus,
    CorporateAction,
    Position,
    apply_action,
)
from .data_quality import QualityStatus, UniverseMember, coverage
from .event_backtester import (
    Event,
    EventLoop,
    EventPriority,
    HandlerEvidence,
)
from .event_sourcing import DomainEvent, EventStore, build_snapshot, verify_snapshot
from .instrument_master import IdentityRecord, ResolutionStatus, resolve_identifier
from .order_lifecycle import Order, OrderState, OrderType, Side, TimeInForce, acknowledge, apply_fill
from .pipeline_orchestration import (
    PublicationRegistry,
    TaskSpec,
    create_run_manifest,
    mark_succeeded,
    publish_atomically,
)
from .portfolio_accounting import LedgerState, NavStatus, nav, open_cash
from .pretrade_risk import DecisionState, Limits, RiskSnapshot, pretrade_decision
from .reconciliation import BreakState, reconcile, service_level
from .snapshot_lineage import canonical_hash
from .walk_forward import WindowMode, walk_forward_folds


UTC = timezone.utc
CASE_TIME = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


@dataclass(frozen=True)
class GateEvidence:
    gate: str
    status: str
    evidence_hash: str
    detail: str


@dataclass(frozen=True)
class PromotionReport:
    fixture_id: str
    gates: tuple[GateEvidence, ...]
    decision: str
    report_hash: str

    def gate(self, name: str) -> GateEvidence:
        return next(item for item in self.gates if item.gate == name)


def _gate(name: str, status: str, detail: str, evidence: object) -> GateEvidence:
    if status not in {"PASS", "CONDITIONAL_PASS", "REJECT"}:
        raise ValueError("invalid integrated gate status")
    return GateEvidence(name, status, canonical_hash(evidence), detail)


def _final_status(gates: tuple[GateEvidence, ...]) -> str:
    if any(item.status == "REJECT" for item in gates):
        return "REJECT"
    if any(item.status == "CONDITIONAL_PASS" for item in gates):
        return "CONDITIONAL_PASS"
    return "PASS"


def run_governed_promotion_case(
    *,
    proposed_gross: float = 2.5,
    data_quality_issue_count: int = 1,
) -> PromotionReport:
    """Execute the synthetic CH12 release chain; no gate is inferred from prose."""
    gates: list[GateEvidence] = []

    identity_row = IdentityRecord(
        "I1", "ticker", "ABC", date(2020, 1, 1), None,
        CASE_TIME - timedelta(days=1), None, "m1", "master", 10,
    )
    identity = resolve_identifier(
        (identity_row,), "ticker", "ABC", CASE_TIME.date(), CASE_TIME
    )
    gates.append(
        _gate(
            "instrument_identity",
            "PASS" if identity.status is ResolutionStatus.RESOLVED else "REJECT",
            f"{identity.status.value}; mapping={identity.mapping_version}",
            {"status": identity.status.value, "lineage": identity_row.lineage_hash},
        )
    )

    feature_row = BitemporalRow(
        "eps", 1.2, CASE_TIME - timedelta(days=30), None,
        CASE_TIME - timedelta(hours=1), None, "f2", "vendor",
        CASE_TIME - timedelta(hours=3), CASE_TIME - timedelta(hours=2),
        CASE_TIME - timedelta(hours=1), source_priority=10,
    )
    selection = as_of_value((feature_row,), "eps", CASE_TIME, CASE_TIME)
    gates.append(
        _gate(
            "bitemporal_selection",
            "PASS" if selection.status is SelectionStatus.SELECTED else "REJECT",
            f"{selection.status.value}; version={feature_row.version_id}",
            {"status": selection.status.value, "row": selection.selected_row_hash},
        )
    )

    position = Position("I1", Decimal("100"), "USD", Decimal("8000"), "L1")
    corporate_action = CorporateAction(
        "A1", "v2", "exchange", "I1", ActionType.CASH_DIVIDEND,
        CASE_TIME - timedelta(days=5), CASE_TIME - timedelta(days=5), None,
        CASE_TIME.date(), CASE_TIME.date(), CASE_TIME.date(), "USD",
        ActionStatus.CONFIRMED, cash_amount=Decimal("3"),
    )
    application = apply_action(
        position,
        corporate_action,
        portfolio_id="P1",
        effective_date=CASE_TIME.date(),
        knowledge_cutoff=CASE_TIME,
        application_ledger=set(),
    )
    gates.append(
        _gate(
            "corporate_actions",
            "PASS" if application.status is ApplicationStatus.APPLIED else "REJECT",
            f"receivable={application.cash_postings[0].amount} USD; raw price unchanged",
            {"status": application.status.value, "application_key": application.application_key},
        )
    )

    loop = EventLoop(master_seed=7)
    loop.publish(
        Event.from_payload(
            event_id="market-1",
            event_time_ns=10,
            available_at_ns=20,
            priority=EventPriority.MARKET_DATA,
            sequence=1,
            kind="market",
            payload={"instrument_id": "I1", "price": 10.0},
        )
    )
    replay = loop.run(
        {"market": lambda _event, _loop, _seed: HandlerEvidence(output_hashes=("c" * 64,))},
        handler_versions={"market": "market-handler-v1"},
        input_snapshot_hash="a" * 64,
        config_hash="b" * 64,
    )
    gates.append(
        _gate(
            "event_replay",
            "PASS" if replay.status == "PASS" else "REJECT",
            f"events={len(replay.processed)}; trace={replay.trace_hash[:12]}",
            {"trace": replay.trace_hash, "terminal": replay.terminal_state_hash},
        )
    )

    order = Order(
        "O1", Side.BUY, OrderType.LIMIT, TimeInForce.DAY, 100, "XNAS",
        CASE_TIME, limit_price=Decimal("11"),
    )
    order = acknowledge(order, CASE_TIME + timedelta(seconds=1))
    order = apply_fill(order, 100, Decimal("10.5"), CASE_TIME + timedelta(seconds=2))
    gates.append(
        _gate(
            "order_lifecycle",
            "PASS" if order.state is OrderState.FILLED and order.quantity_conserved else "REJECT",
            f"state={order.state.value}; average={order.average_price}; remaining={order.remaining_lots}",
            {"state": order.state.value, "filled": order.filled_lots, "average": order.average_price},
        )
    )

    ledger = open_cash(
        LedgerState(), event_id="cash-1", currency="USD",
        amount=Decimal("1000"), booked_at=CASE_TIME,
    )
    nav_result = nav(
        ledger, marks=(), fx_rates=(), base_currency="USD", as_of=CASE_TIME,
        stale_after=timedelta(days=1),
    )
    gates.append(
        _gate(
            "ledger_nav",
            "PASS" if nav_result.status is NavStatus.PASS else "REJECT",
            f"NAV={nav_result.nav} USD from cash-by-currency ledger",
            {"status": nav_result.status.value, "nav": nav_result.nav},
        )
    )

    quality = coverage(
        (UniverseMember("I1", 99.0), UniverseMember("I2", 1.0)),
        ("I1",),
        issue_count=data_quality_issue_count,
        minimum_count_coverage=0.5,
        minimum_value_coverage=0.95,
    )
    gates.append(
        _gate(
            "data_quality",
            quality.status.value,
            f"count={quality.count_coverage:.2%}; value={quality.market_value_coverage:.2%}; issues={quality.issue_count}",
            asdict(quality),
        )
    )

    sessions = tuple(date(2024, 1, day) for day in range(1, 16))
    folds = walk_forward_folds(
        sessions,
        train_sessions=5,
        validation_sessions=2,
        test_sessions=2,
        embargo_sessions=1,
        step_sessions=2,
        mode=WindowMode.ROLLING,
    )
    gates.append(
        _gate(
            "walk_forward",
            "PASS" if folds else "REJECT",
            f"folds={len(folds)}; validation-to-test embargo=1 session",
            [asdict(fold) for fold in folds],
        )
    )

    events = (
        DomainEvent.create(
            event_id="E1", aggregate_id="P1", sequence=1,
            event_type="add", payload={"delta": 3},
            schema_version="event-v1", reducer_version="sum-v1",
            producer_id="case", event_time=CASE_TIME, known_at=CASE_TIME,
            recorded_at=CASE_TIME,
        ),
        DomainEvent.create(
            event_id="E2", aggregate_id="P1", sequence=2,
            event_type="add", payload={"delta": 4},
            schema_version="event-v1", reducer_version="sum-v1",
            producer_id="case", event_time=CASE_TIME, known_at=CASE_TIME,
            recorded_at=CASE_TIME,
        ),
    )
    store = EventStore()
    for event in events:
        store.append(event)
    reducers = {"add": lambda state, payload: state + int(payload["delta"])}
    snapshot = build_snapshot("P1", 0, events, reducers, reducer_version="sum-v1")
    snapshot_valid = verify_snapshot(snapshot, 0, tuple(reversed(events)), reducers)
    gates.append(
        _gate(
            "event_state",
            "PASS" if snapshot_valid and snapshot.state == 7 else "REJECT",
            f"sequence={snapshot.sequence}; state={snapshot.state}; prefix verified",
            asdict(snapshot),
        )
    )

    tasks = (TaskSpec("raw", ()), TaskSpec("publish", ("raw",)))
    run = create_run_manifest(
        run_id="run-1", input_manifest={"snapshot": "a" * 64},
        code_version="ch12-v0.10", config={"seed": 7}, tasks=tasks,
        environment={"python": "reference"},
    )
    output_hash = "d" * 64
    run = mark_succeeded(
        run,
        output_hash=output_hash,
        partition_certificates=(("publish", "all", "PASS"),),
    )
    publication = publish_atomically(
        run, output_hash=output_hash, registry=PublicationRegistry()
    )
    gates.append(
        _gate(
            "atomic_publication",
            "PASS" if publication.status == "PUBLISHED" else "REJECT",
            f"{publication.status}; semantic_run={publication.semantic_run_hash[:12]}",
            asdict(publication),
        )
    )

    limits = Limits("limits-v1", 2.0, 0.5, 0.2, 0.1, 5.0)
    before = RiskSnapshot("before", CASE_TIME, "risk-v1", 1.0, 0.1, 0.05, 0.02, 1.0)
    after = RiskSnapshot("after", CASE_TIME, "risk-v1", proposed_gross, 0.1, 0.05, 0.02, 1.0)
    risk = pretrade_decision(
        before, after, limits, decision_time=CASE_TIME,
        maximum_age=timedelta(minutes=1), portfolio_id="P1", order_id="O1",
        risk_data_hash="e" * 64,
    )
    risk_gate = {
        DecisionState.PASS: "PASS",
        DecisionState.REQUIRES_APPROVAL: "CONDITIONAL_PASS",
        DecisionState.BLOCK: "REJECT",
        DecisionState.INVALID_OR_STALE_INPUT: "REJECT",
    }[risk.status]
    gates.append(
        _gate(
            "pretrade_risk",
            risk_gate,
            f"decision={risk.status.value}; proposed gross={proposed_gross}; hard limit=2.0",
            asdict(risk),
        )
    )

    breaks = reconcile(
        {"cash:USD": Decimal("100.00")},
        {"cash:USD": Decimal("100.00")},
        {"cash:USD": Decimal("0.01")},
        unit="USD",
        policy_version="recon-v1",
    )
    slo = service_level(
        Decimal("0.01"), Decimal("2"), Decimal("1000"),
        slo_id="availability", policy_version="slo-v1", input_hash="f" * 64,
    )
    operational_pass = all(item.state is BreakState.MATCH for item in breaks) and slo.within_budget
    gates.append(
        _gate(
            "reconciliation_slo",
            "PASS" if operational_pass else "REJECT",
            f"breaks={sum(item.material for item in breaks)}; burn_rate={slo.burn_rate}",
            {"breaks": [asdict(item) for item in breaks], "slo": asdict(slo)},
        )
    )

    call = ToolCall.create(
        tool="publisher", action="publish", actor_id="agent-1",
        actor_role="operator", tenant_id="T1", portfolio_id="P1",
        request_id="R1", idempotency_key="K1", expected_state_hash="1" * 64,
        arguments={"run_id": "run-1"},
    )
    policy = ToolPolicy(
        "publisher", "agent-policy-v1", frozenset({"operator"}),
        frozenset({"T1"}), frozenset({"P1"}), frozenset({"publish"}),
        True, True,
        {"type": "object", "properties": {"run_id": {"type": "string"}}, "required": ["run_id"], "additionalProperties": False},
        {"type": "object", "properties": {"published": {"const": True}}, "required": ["published"], "additionalProperties": False},
    )
    blocked_without_approval = False
    try:
        execute_tool(
            call,
            policy,
            lambda run_id: {"published": bool(run_id)},
            tool_version="publisher-v1",
            decision_time=CASE_TIME,
            authoritative_state_hash="1" * 64,
            precondition=lambda _call: True,
            staged_invariant=lambda output: output["published"] is True,
            local_outbox_commit=lambda _output: None,
            idempotency_store=IdempotencyStore(),
        )
    except PermissionError:
        blocked_without_approval = True
    gates.append(
        _gate(
            "agent_governance",
            "PASS" if blocked_without_approval else "REJECT",
            "unapproved mutation blocked before implementation and commit",
            {"blocked_without_approval": blocked_without_approval, "envelope": call.envelope_hash},
        )
    )

    frozen_gates = tuple(gates)
    decision = _final_status(frozen_gates)
    report_body: Mapping[str, object] = {
        "fixture_id": "ch12_governed_promotion_v1",
        "gates": [asdict(item) for item in frozen_gates],
        "decision": decision,
    }
    return PromotionReport(
        "ch12_governed_promotion_v1",
        frozen_gates,
        decision,
        canonical_hash(report_body),
    )
