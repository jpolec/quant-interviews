from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import importlib
from pathlib import Path

import pytest

from quant_interviews.ch12.agent_governance import (
    AuthorizationStatus,
    HumanApproval,
    IdempotencyStore,
    ToolCall,
    ToolPolicy,
    dispatch_outbox,
    execute_read_only,
    execute_tool,
)
from quant_interviews.ch12.bitemporal import (
    BitemporalRow,
    RowStatus,
    SelectionStatus,
    TemporalRequest,
    as_of_value,
    temporal_join,
)
from quant_interviews.ch12.corporate_actions import (
    ActionStatus,
    ActionType,
    ApplicationStatus,
    CorporateAction,
    Position,
    apply_action,
    theoretical_split_reference_price,
    total_return,
)
from quant_interviews.ch12.data_quality import (
    Observation,
    ObservationStatus,
    QualityStatus,
    UniverseMember,
    coverage,
    validate_observations,
)
from quant_interviews.ch12.event_backtester import (
    Event,
    EventLoop,
    EventPriority,
    HandlerEvidence,
    assert_no_lookahead,
    event_seed,
)
from quant_interviews.ch12.event_sourcing import (
    DomainEvent,
    EventStore,
    build_snapshot,
    fold_events,
    verify_snapshot,
)
from quant_interviews.ch12.instrument_master import (
    IdentityRecord,
    ResolutionStatus,
    resolve_identifier,
    validate_no_overlap,
)
from quant_interviews.ch12.order_lifecycle import (
    Order,
    OrderState,
    OrderType,
    Side,
    TimeInForce,
    acknowledge,
    activate_stop,
    apply_fill,
    cancel,
    cancel_oco_sibling,
    expire,
    reject,
)
from quant_interviews.ch12.pipeline_orchestration import (
    PublicationRegistry,
    RunState,
    TaskSpec,
    create_run_manifest,
    mark_succeeded,
    publish_atomically,
    retry_delay,
    topological_order,
)
from quant_interviews.ch12.portfolio_accounting import (
    Account,
    FxRate,
    LedgerState,
    Mark,
    NavStatus,
    account_balances,
    buy,
    nav,
    open_cash,
    sell_fifo,
)
from quant_interviews.ch12.pretrade_risk import (
    DecisionState,
    Limits,
    RiskSnapshot,
    pretrade_decision,
)
from quant_interviews.ch12.reconciliation import (
    BreakState,
    reconcile,
    service_level,
)
from quant_interviews.ch12.snapshot_lineage import (
    build_manifest,
    canonical_bytes,
    canonical_hash,
    verify_artifact_bytes,
    verify_manifest_integrity,
    verify_output_equivalence,
)
from quant_interviews.ch12.walk_forward import (
    WindowMode,
    purge_overlapping_labels,
    purge_overlapping_supports,
    walk_forward_folds,
)


UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def identity(
    instrument: str,
    version: str,
    *,
    valid_from: date = date(2020, 1, 1),
    valid_to: date | None = None,
    known_from: datetime = T0,
    known_to: datetime | None = None,
    source: str = "master",
    priority: int = 10,
) -> IdentityRecord:
    return IdentityRecord(
        instrument,
        "ticker",
        "ABC",
        valid_from,
        valid_to,
        known_from,
        known_to,
        version,
        source,
        priority,
    )


def feature(
    value: float | None,
    version: str,
    *,
    known_from: datetime,
    known_to: datetime | None = None,
    status: RowStatus = RowStatus.ACTIVE,
    source: str = "vendor",
) -> BitemporalRow:
    return BitemporalRow(
        "eps",
        value,
        datetime(2023, 1, 1, tzinfo=UTC),
        None,
        known_from,
        known_to,
        version,
        source,
        known_from - timedelta(hours=2),
        known_from - timedelta(hours=1),
        known_from,
        status,
    )


def action(
    action_type: ActionType,
    **overrides: object,
) -> CorporateAction:
    base: dict[str, object] = {
        "action_id": "action-1",
        "version_id": "v1",
        "source_id": "exchange",
        "parent_instrument_id": "PARENT",
        "action_type": action_type,
        "announced_at": T0,
        "known_from": T0,
        "known_to": None,
        "ex_date": date(2024, 2, 1),
        "record_date": date(2024, 2, 2),
        "payable_date": date(2024, 2, 10),
        "currency": "USD",
        "status": ActionStatus.CONFIRMED,
    }
    base.update(overrides)
    return CorporateAction(**base)


def test_all_ch12_python_modules_import() -> None:
    root = Path("_book/python/quant_interviews/ch12")
    for path in sorted(root.glob("*.py")):
        if path.name != "__init__.py":
            importlib.import_module(f"quant_interviews.ch12.{path.stem}")


def test_instrument_resolution_is_bitemporal_and_typed() -> None:
    rows = (
        identity(
            "OLD",
            "v1",
            valid_to=date(2022, 1, 1),
            known_to=T0 + timedelta(days=10),
        ),
        identity(
            "NEW",
            "v2",
            valid_from=date(2022, 1, 1),
            known_from=T0 + timedelta(days=10),
        ),
    )
    old = resolve_identifier(
        rows, "ticker", "ABC", date(2021, 6, 1), T0 + timedelta(days=5)
    )
    new = resolve_identifier(
        rows, "ticker", "ABC", date(2023, 6, 1), T0 + timedelta(days=15)
    )
    assert (old.status, old.instrument_id, old.mapping_version) == (
        ResolutionStatus.RESOLVED,
        "OLD",
        "v1",
    )
    assert new.instrument_id == "NEW"


def test_instrument_equal_precedence_is_ambiguous() -> None:
    result = resolve_identifier(
        (identity("A", "v1"), identity("B", "v2")),
        "ticker",
        "ABC",
        date(2024, 1, 1),
        T0,
    )
    assert result.status is ResolutionStatus.AMBIGUOUS
    assert result.instrument_id is None


def test_overlap_check_detects_nested_intervals() -> None:
    outer = identity(
        "A",
        "outer",
        valid_from=date(2020, 1, 1),
        valid_to=date(2030, 1, 1),
        known_to=T0 + timedelta(days=100),
    )
    inner = identity(
        "B",
        "inner",
        valid_from=date(2022, 1, 1),
        valid_to=date(2023, 1, 1),
        known_from=T0 + timedelta(days=1),
        known_to=T0 + timedelta(days=2),
    )
    assert validate_no_overlap((outer, inner))[0].code == "ambiguous_mapping"


def test_bitemporal_revision_and_tombstone_semantics() -> None:
    first = feature(1.0, "v1", known_from=T0, known_to=T0 + timedelta(days=5))
    second = feature(1.2, "v2", known_from=T0 + timedelta(days=5))
    tombstone = feature(
        None,
        "v3",
        known_from=T0 + timedelta(days=10),
        status=RowStatus.TOMBSTONE,
    )
    before = as_of_value(
        (first, second, tombstone),
        "eps",
        T0,
        T0 + timedelta(days=2),
    )
    after = as_of_value(
        (first, second, tombstone),
        "eps",
        T0,
        T0 + timedelta(days=12),
    )
    assert before.row is not None and before.row.value == 1.0
    assert after.status is SelectionStatus.WITHDRAWN


def test_bitemporal_ties_and_duplicate_requests_fail_closed() -> None:
    rows = (
        feature(1.0, "v1", known_from=T0, source="a"),
        feature(2.0, "v2", known_from=T0, source="b"),
    )
    assert as_of_value(rows, "eps", T0, T0).status is SelectionStatus.AMBIGUOUS
    request = TemporalRequest("d1", "eps", T0, T0)
    with pytest.raises(ValueError, match="duplicate temporal request"):
        temporal_join((request, request), rows)


def test_corporate_actions_do_not_double_count_value() -> None:
    parent = Position(
        "PARENT", Decimal("100"), "USD", Decimal("8000"), "lot-1"
    )
    dividend = apply_action(
        parent,
        action(ActionType.CASH_DIVIDEND, cash_amount=Decimal("3")),
        portfolio_id="portfolio-1",
        effective_date=date(2024, 2, 1),
        knowledge_cutoff=T0,
        application_ledger=set(),
    )
    spinoff = apply_action(
        parent,
        action(
            ActionType.SPINOFF,
            ratio=Decimal("0.5"),
            child_instrument_id="CHILD",
            parent_cost_basis_fraction=Decimal("0.75"),
        ),
        portfolio_id="portfolio-1",
        effective_date=date(2024, 2, 1),
        knowledge_cutoff=T0,
        application_ledger=set(),
    )
    assert dividend.parent_position == parent
    assert dividend.cash_postings[0].amount == Decimal("300")
    assert not dividend.raw_market_price_adjusted
    assert spinoff.child_positions[0].quantity == Decimal("50")
    assert spinoff.parent_position is not None
    assert (
        spinoff.parent_position.cost_basis
        + spinoff.child_positions[0].cost_basis
        == parent.cost_basis
    )
    assert spinoff.cash_postings == ()


def test_total_return_includes_cash_and_child_security() -> None:
    result = total_return(
        Decimal("100"),
        Decimal("95"),
        (Decimal("2"),),
        (Decimal("4"),),
    )
    assert result == Decimal("0.01")


def test_event_loop_orders_by_availability_before_event_time() -> None:
    loop = EventLoop(master_seed=17)
    loop.publish(
        Event.from_payload(
            event_id="late",
            event_time_ns=1,
            available_at_ns=20,
            priority=EventPriority.MARKET_DATA,
            sequence=2,
            kind="late",
        )
    )
    loop.publish(
        Event.from_payload(
            event_id="now",
            event_time_ns=10,
            available_at_ns=10,
            priority=EventPriority.MARKET_DATA,
            sequence=1,
            kind="now",
        )
    )
    handlers = {
        kind: (lambda _event, _loop, _seed: HandlerEvidence())
        for kind in ("now", "late")
    }
    result = loop.run(
        handlers,
        handler_versions={"now": "v1", "late": "v1"},
        input_snapshot_hash="a" * 64,
        config_hash="b" * 64,
    )
    assert [event.kind for event in result.processed] == ["now", "late"]
    assert len(result.trace_hash) == 64
    assert event_seed(17, "now") == event_seed(17, "now")


def test_event_loop_rejects_duplicate_and_lookahead() -> None:
    event = Event.from_payload(
        event_id="e1",
        event_time_ns=1,
        available_at_ns=1,
        priority=EventPriority.FEATURE,
        sequence=1,
        kind="feature",
    )
    loop = EventLoop()
    loop.publish(event)
    with pytest.raises(ValueError, match="duplicate event_id"):
        loop.publish(event)
    with pytest.raises(ValueError, match="look-ahead"):
        assert_no_lookahead(
            (10, int(EventPriority.SIGNAL), 10, 2, "decision"),
            (10, int(EventPriority.FILL), 9, 1, "later-priority-source"),
        )
    assert_no_lookahead(
        (10, int(EventPriority.SIGNAL), 10, 2, "decision"),
        (10, int(EventPriority.FEATURE), 10, 1, "eligible-source"),
    )


def base_order(order_type: OrderType = OrderType.LIMIT) -> Order:
    return Order(
        order_id="o1",
        side=Side.BUY,
        order_type=order_type,
        time_in_force=TimeInForce.DAY,
        quantity_lots=100,
        venue="XNAS",
        submitted_at=T0,
        limit_price=Decimal("11") if order_type is OrderType.LIMIT else None,
        stop_price=Decimal("12") if order_type is OrderType.STOP else None,
    )


def test_order_lifecycle_conserves_exact_quantity() -> None:
    order = acknowledge(base_order(), T0 + timedelta(seconds=1))
    order = apply_fill(
        order, 40, Decimal("10"), T0 + timedelta(seconds=2)
    )
    assert order.state is OrderState.PARTIALLY_FILLED
    order = apply_fill(
        order, 60, Decimal("11"), T0 + timedelta(seconds=3)
    )
    assert order.state is OrderState.FILLED
    assert order.average_price == Decimal("10.6")
    assert order.quantity_conserved and order.remaining_lots == 0
    with pytest.raises(ValueError):
        cancel(order, T0 + timedelta(seconds=4), "too late")


def test_sell_stop_triggers_on_down_move() -> None:
    order = replace(
        base_order(OrderType.STOP),
        side=Side.SELL,
        stop_price=Decimal("9"),
    )
    stopped = acknowledge(order, T0)
    assert activate_stop(stopped, Decimal("10")).state is OrderState.STOPPED
    assert (
        activate_stop(stopped, Decimal("8.5")).state
        is OrderState.ACKNOWLEDGED
    )


def test_order_rejects_nonfinite_decimal() -> None:
    with pytest.raises(ValueError):
        replace(base_order(), limit_price=Decimal("NaN"))


def test_reject_expire_and_oco_terminal_paths() -> None:
    rejected = reject(base_order(), T0 + timedelta(seconds=1), "venue_reject")
    assert rejected.state is OrderState.REJECTED
    assert rejected.closed_unfilled_lots == rejected.quantity_lots
    acknowledged = acknowledge(base_order(), T0 + timedelta(seconds=1))
    expired = expire(acknowledged, T0 + timedelta(seconds=2))
    assert expired.state is OrderState.EXPIRED and expired.remaining_lots == 0

    leader = replace(base_order(), order_id="leader", oco_group_id="oco-1")
    leader = acknowledge(leader, T0 + timedelta(seconds=1))
    leader = apply_fill(
        leader,
        leader.quantity_lots,
        Decimal("10"),
        T0 + timedelta(seconds=2),
    )
    sibling = acknowledge(
        replace(base_order(), order_id="sibling", oco_group_id="oco-1"),
        T0 + timedelta(seconds=1),
    )
    sibling = cancel_oco_sibling(leader, sibling, T0 + timedelta(seconds=3))
    assert sibling.state is OrderState.CANCELLED
    assert sibling.close_reason == "oco:leader"


def test_double_entry_fifo_and_multicurrency_nav() -> None:
    state = open_cash(
        LedgerState(),
        event_id="cash-1",
        currency="USD",
        amount=Decimal("1000"),
        booked_at=T0,
    )
    state = buy(
        state,
        event_id="trade-1",
        lot_id="lot-1",
        instrument_id="A",
        currency="USD",
        quantity=Decimal("10"),
        price=Decimal("20"),
        fee=Decimal("2"),
        trade_at=T0,
        settlement_at=T0,
    )
    state = sell_fifo(
        state,
        event_id="trade-2",
        instrument_id="A",
        currency="USD",
        quantity=Decimal("4"),
        price=Decimal("25"),
        fee=Decimal("1"),
        trade_at=T0,
        settlement_at=T0,
    )
    assert account_balances(state.entries, Account.CASH)["USD"] == Decimal("897")
    certificate = nav(
        state,
        marks=(Mark("A", "USD", Decimal("22"), T0),),
        fx_rates=(),
        base_currency="USD",
        as_of=T0,
        stale_after=timedelta(days=1),
    )
    assert certificate.status is NavStatus.PASS
    assert certificate.nav == Decimal("1029")


def test_nav_fails_closed_on_missing_fx() -> None:
    state = open_cash(
        LedgerState(),
        event_id="cash-eur",
        currency="EUR",
        amount=Decimal("100"),
        booked_at=T0,
    )
    certificate = nav(
        state,
        marks=(),
        fx_rates=(),
        base_currency="USD",
        as_of=T0,
        stale_after=timedelta(days=1),
    )
    assert certificate.status is NavStatus.INVALID_OR_STALE_INPUT
    assert certificate.nav is None


def test_data_quality_models_missing_and_future_observations() -> None:
    rows = (
        Observation(
            "obs-1",
            "A",
            "price",
            T0 - timedelta(days=10),
            T0 + timedelta(days=1),
            1,
            "vendor",
            None,
        ),
    )
    codes = {
        issue.code
        for issue in validate_observations(
            rows, as_of=T0, stale_after=timedelta(days=5)
        )
    }
    assert codes == {"future_knowledge"}


def test_count_and_market_value_coverage_are_separate() -> None:
    certificate = coverage(
        (
            UniverseMember("A", 99.0),
            UniverseMember("B", 1.0),
        ),
        ("A",),
        minimum_count_coverage=0.4,
        minimum_value_coverage=0.95,
    )
    assert certificate.count_coverage == 0.5
    assert certificate.market_value_coverage == 0.99
    assert certificate.status is QualityStatus.PASS
    with pytest.raises(ValueError, match="must not be empty"):
        coverage((), ())


def test_manifest_verifies_integrity_and_artifact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "fixture.txt"
    path.write_text("abc", encoding="utf-8")
    manifest = build_manifest(
        {"fixture://x": path},
        config={"model": "x"},
        seed=7,
        implementation_version="ch12-v0.9",
        environment={"python": "3.12"},
    )
    assert verify_manifest_integrity(manifest)
    assert verify_artifact_bytes(manifest).status == "PASS"
    path.write_text("changed", encoding="utf-8")
    assert verify_artifact_bytes(manifest).changed_artifacts == ("fixture://x",)


def test_split_reference_canonical_bytes_and_output_hash() -> None:
    assert theoretical_split_reference_price(
        Decimal("80"), Decimal("4")
    ) == Decimal("20")
    payload = {"amount": Decimal("1.25"), "known_at": T0}
    encoded = canonical_bytes(payload)
    assert encoded.startswith(b'{"canonical_format":"QJ-CANONICAL-JSON-1"')
    assert verify_output_equivalence(canonical_hash(payload), payload)
    assert not verify_output_equivalence("a" * 64, payload)
    with pytest.raises(ValueError, match="collide"):
        canonical_bytes({"e\u0301": 1, "é": 2})


def test_canonical_hash_rejects_unsupported_or_nonfinite_values() -> None:
    assert canonical_hash({"x": Decimal("1.25")}) == canonical_hash(
        {"x": Decimal("1.25")}
    )
    with pytest.raises(ValueError):
        canonical_hash({"x": float("nan")})
    with pytest.raises(TypeError):
        canonical_hash({"x": {1, 2}})


def test_walk_forward_uses_sessions_and_rolling_windows() -> None:
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
    assert len(folds) == 3
    assert folds[0].train[0] == date(2024, 1, 1)
    assert folds[1].train[0] == date(2024, 1, 3)
    assert folds[0].test[0] == date(2024, 1, 9)


def test_purge_uses_half_open_overlap_and_requires_label_end() -> None:
    times = (date(2024, 1, 1), date(2024, 1, 2))
    retained = purge_overlapping_labels(
        times,
        {
            times[0]: date(2024, 1, 5),
            times[1]: date(2024, 1, 6),
        },
        test_start=date(2024, 1, 5),
        test_end=date(2024, 1, 7),
    )
    assert retained == (times[0],)
    with pytest.raises(ValueError, match="missing support endpoint"):
        purge_overlapping_labels(
            times,
            {times[0]: date(2024, 1, 5)},
            test_start=date(2024, 1, 5),
            test_end=date(2024, 1, 7),
        )


def test_purge_protects_train_validation_and_validation_test_boundaries() -> None:
    train = (date(2024, 1, 1), date(2024, 1, 2))
    validation = (date(2024, 1, 5), date(2024, 1, 6))
    support_start = {
        train[0]: date(2023, 12, 29),
        train[1]: date(2024, 1, 2),
        validation[0]: date(2024, 1, 4),
        validation[1]: date(2024, 1, 6),
    }
    support_end = {
        train[0]: date(2024, 1, 5),
        train[1]: date(2024, 1, 7),  # leaks into validation
        validation[0]: date(2024, 1, 9),  # leaks into test
        validation[1]: date(2024, 1, 8),
    }
    retained_train = purge_overlapping_supports(
        train,
        support_start,
        support_end,
        protected_intervals=(
            (date(2024, 1, 5), date(2024, 1, 7)),
            (date(2024, 1, 8), date(2024, 1, 10)),
        ),
    )
    retained_validation = purge_overlapping_supports(
        validation,
        support_start,
        support_end,
        protected_intervals=((date(2024, 1, 8), date(2024, 1, 10)),),
    )
    assert retained_train == (train[0],)
    assert retained_validation == (validation[1],)


def domain_event(
    event_id: str,
    sequence: int,
    delta: int,
) -> DomainEvent:
    return DomainEvent.create(
        event_id=event_id,
        aggregate_id="portfolio-1",
        sequence=sequence,
        event_type="add",
        payload={"delta": delta},
        schema_version="domain-event-v1",
        reducer_version="sum-v1",
        producer_id="test-suite",
        event_time=T0,
        known_at=T0,
        recorded_at=T0,
    )


def test_event_store_distinguishes_replay_from_conflict() -> None:
    store = EventStore()
    event = domain_event("e1", 1, 3)
    assert store.append(event).status == "APPENDED"
    assert store.append(event).status == "DUPLICATE_SAME_CONTENT"
    conflict = domain_event("e1", 1, 4)
    with pytest.raises(ValueError, match="event_id conflict"):
        store.append(conflict)
    with pytest.raises(ValueError, match="sequence gap"):
        store.append(domain_event("e3", 3, 1))


def test_event_snapshot_has_prefix_hash_and_replays() -> None:
    events = (domain_event("e1", 1, 3), domain_event("e2", 2, 4))
    reducers = {"add": lambda state, payload: state + int(payload["delta"])}
    assert fold_events(0, events, reducers) == 7
    snapshot = build_snapshot(
        "portfolio-1", 0, events, reducers, reducer_version="sum-v1"
    )
    assert snapshot.sequence == 2 and len(snapshot.prefix_hash) == 64
    assert verify_snapshot(snapshot, 0, events, reducers)


def test_pipeline_dag_retry_and_atomic_publication() -> None:
    tasks = (
        TaskSpec("raw", ()),
        TaskSpec("features", ("raw",)),
        TaskSpec("report", ("features",)),
    )
    assert topological_order(tasks) == ("raw", "features", "report")
    assert retry_delay(20) == 60.0
    run = create_run_manifest(
        run_id="run-1",
        input_manifest={"snapshot": "abc"},
        code_version="v1",
        config={"seed": 7},
        tasks=tasks,
    )
    output_hash = "a" * 64
    succeeded = mark_succeeded(
        run,
        output_hash=output_hash,
        partition_certificates=(("report", "2024-01-01", "PASS"),),
    )
    registry = PublicationRegistry()
    assert (
        publish_atomically(
            succeeded,
            output_hash=output_hash,
            registry=registry,
        ).status
        == "PUBLISHED"
    )
    assert (
        publish_atomically(
            succeeded,
            output_hash=output_hash,
            registry=registry,
        ).status
        == "DUPLICATE_SAME_OUTPUT"
    )


def test_pipeline_rejects_duplicate_tasks_and_cycles() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        topological_order((TaskSpec("a", ()), TaskSpec("a", ())))
    with pytest.raises(ValueError, match="cycle"):
        topological_order(
            (TaskSpec("a", ("b",)), TaskSpec("b", ("a",)))
        )


def risk(snapshot_id: str, known_at: datetime, gross: float) -> RiskSnapshot:
    return RiskSnapshot(snapshot_id, known_at, "risk-v1", gross, 0.1, 0.05, 0.02, 1.0)


def test_pretrade_distinguishes_existing_from_new_breaches() -> None:
    limits = Limits("policy-v1", 2.0, 0.5, 0.2, 0.1, 5.0)
    unchanged = pretrade_decision(
        risk("before", T0, 2.5),
        risk("after", T0, 2.4),
        limits,
        decision_time=T0,
        maximum_age=timedelta(minutes=1),
        portfolio_id="portfolio-1",
        order_id="order-1",
        risk_data_hash="a" * 64,
    )
    new = pretrade_decision(
        risk("before", T0, 1.0),
        risk("after", T0, 2.5),
        limits,
        decision_time=T0,
        maximum_age=timedelta(minutes=1),
        portfolio_id="portfolio-1",
        order_id="order-2",
        risk_data_hash="a" * 64,
    )
    assert unchanged.status is DecisionState.PASS
    assert new.status is DecisionState.BLOCK
    assert new.new_breaches[0].metric == "gross"


def test_pretrade_rejects_nan_and_stale_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        risk("bad", T0, float("nan"))
    limits = Limits("p", 2.0, 0.5, 0.2, 0.1, 5.0)
    result = pretrade_decision(
        risk("before", T0, 1.0),
        risk("after", T0, 1.0),
        limits,
        decision_time=T0 + timedelta(days=1),
        maximum_age=timedelta(minutes=5),
        portfolio_id="portfolio-1",
        order_id="order-1",
        risk_data_hash="a" * 64,
    )
    assert result.status is DecisionState.BLOCK


def test_reconciliation_preserves_missing_states_and_decimal_tolerance() -> None:
    breaks = reconcile(
        {"cash": Decimal("100.00")},
        {"cash": Decimal("99.99"), "fee": Decimal("1")},
        {"cash": Decimal("0.01"), "fee": Decimal("0")},
    )
    assert [item.state for item in breaks] == [
        BreakState.MATCH,
        BreakState.MISSING_INTERNAL,
    ]
    slo = service_level(
        Decimal("0.01"),
        Decimal("2"),
        Decimal("1000"),
        slo_id="availability",
        policy_version="slo-v1",
        input_hash="a" * 64,
    )
    assert slo.within_budget
    with pytest.raises(ValueError):
        service_level(
            Decimal("0.01"),
            Decimal("1001"),
            Decimal("1000"),
            slo_id="availability",
            policy_version="slo-v1",
            input_hash="a" * 64,
        )


def read_policy(*, mutating: bool = False) -> ToolPolicy:
    return ToolPolicy(
        name="nav",
        policy_version="policy-v1",
        allowed_roles=frozenset({"reader", "operator"}),
        allowed_tenants=frozenset({"tenant-1"}),
        allowed_portfolios=frozenset({"portfolio-1"}),
        allowed_actions=frozenset({"read", "publish"}),
        mutating=mutating,
        requires_human_approval=mutating,
        input_schema={
            "type": "object",
            "required": ["x"],
            "properties": {"x": {"type": "integer"}},
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "integer"}},
            "additionalProperties": False,
        },
    )


def tool_call(action: str = "read") -> ToolCall:
    return ToolCall.create(
        tool="nav",
        action=action,
        actor_id="actor-1",
        actor_role="operator" if action == "publish" else "reader",
        tenant_id="tenant-1",
        portfolio_id="portfolio-1",
        request_id=f"request-{action}",
        idempotency_key=f"idempotency-{action}",
        expected_state_hash="a" * 64 if action == "publish" else None,
        arguments={"x": 2},
    )


def test_ai_read_only_call_has_schema_and_evidence() -> None:
    result = execute_read_only(
        tool_call(),
        read_policy(),
        lambda x: {"value": x * 3},
        tool_version="nav-v1",
    )
    assert result["result"] == {"value": 6}
    assert (
        result["evidence"].authorization_status
        is AuthorizationStatus.AUTHORIZED_READ_ONLY
    )
    assert len(result["evidence"].evidence_hash) == 64


def test_ai_mutation_requires_bound_approval_and_is_idempotent() -> None:
    policy = read_policy(mutating=True)
    call = tool_call("publish")
    store = IdempotencyStore()
    local_outbox_rows: list[dict[str, object]] = []
    controls = {
        "decision_time": T0 + timedelta(minutes=1),
        "authoritative_state_hash": "a" * 64,
        "precondition": lambda candidate: candidate.expected_state_hash == "a" * 64,
        "staged_invariant": lambda output: output["value"] == 2,
        "local_outbox_commit": lambda output: local_outbox_rows.append(dict(output)),
        "idempotency_store": store,
    }
    with pytest.raises(PermissionError, match="human approval"):
        execute_tool(
            call,
            policy,
            lambda x: {"value": x},
            tool_version="1.0",
            **controls,
        )
    approval = HumanApproval.create(
        approval_id="approval-1",
        approver_id="risk-owner",
        call=call,
        policy_version=policy.policy_version,
        approved_at=T0,
        not_before=T0,
        expires_at=T0 + timedelta(hours=1),
    )
    result, evidence = execute_tool(
        call,
        policy,
        lambda x: {"value": x},
        tool_version="1.0",
        approval=approval,
        **controls,
    )
    assert result == {"value": 2}
    assert evidence.approval_id == "approval-1"
    replay, same_evidence = execute_tool(
        call,
        policy,
        lambda x: {"value": x + 100},
        tool_version="1.0",
        approval=approval,
        **controls,
    )
    assert replay == {"value": 2}
    assert same_evidence == evidence
    assert local_outbox_rows == [{"value": 2}]
    external_effects: list[dict[str, object]] = []

    def external_executor(
        output: dict[str, object] | object,
        _scoped_key: tuple[str, str, str, str, str, str],
    ) -> dict[str, object]:
        value = dict(output)  # type: ignore[arg-type]
        external_effects.append(value)
        return {"published": value["value"] == 2}

    receipt = dispatch_outbox(
        store,
        call.scoped_idempotency_key,
        external_executor,
        lambda value: value["published"] is True,
    )
    assert receipt == {"published": True}
    assert external_effects == [{"value": 2}]
    assert store.outbox[call.scoped_idempotency_key].dispatched


def test_mutating_policy_cannot_disable_human_approval() -> None:
    with pytest.raises(ValueError, match="must require"):
        replace(read_policy(mutating=True), requires_human_approval=False)
