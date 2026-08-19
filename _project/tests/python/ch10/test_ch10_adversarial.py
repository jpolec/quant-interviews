from __future__ import annotations

from math import exp

import numpy as np
import pytest

from quant_interviews.ch10.almgren_chriss import optimal_schedule
from quant_interviews.ch10.benchmarks import execution_benchmarks
from quant_interviews.ch10.execution_state import (
    EventType,
    OrderEvent,
    reconstruct_order,
)
from quant_interviews.ch10.hawkes import (
    HawkesParameters,
    hawkes_diagnostics,
    log_likelihood,
)
from quant_interviews.ch10.iceberg import infer_iceberg
from quant_interviews.ch10.impact_models import (
    fit_kyle_lambda,
    fit_square_root_coefficient,
    square_root_capacity,
)
from quant_interviews.ch10.integrated_case import _digest, governed_execution_fixture
from quant_interviews.ch10.market_maker import avellaneda_stoikov_quotes
from quant_interviews.ch10.order_book import BookSnapshot
from quant_interviews.ch10.order_choice import (
    OrderChoiceInputs,
    break_even_fill_probability,
)
from quant_interviews.ch10.pretrade import OrderIntent, RiskLimits, pretrade_check
from quant_interviews.ch10.queue_fill import QueueModel
from quant_interviews.ch10.smart_router import VenueQuote, route_order
from quant_interviews.ch10.tca import tca_decomposition


ABS = 1e-12


def event(
    event_id: str,
    sequence: int,
    event_type: EventType,
    quantity: int = 0,
    price: float | None = None,
) -> OrderEvent:
    return OrderEvent(event_id, sequence, event_type, quantity, price)


def test_ordered_queue_counterexample_and_integer_domain() -> None:
    result = QueueModel(1, 1.0, 1.0).fill_probability(1.0)
    assert result == pytest.approx((1.0 - exp(-1.0)) ** 2, rel=0.0, abs=ABS)
    with pytest.raises(ValueError):
        QueueModel(0.5, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        QueueModel(True, 1.0)


def test_execution_state_rejects_nan_unknown_type_and_preserves_cancel_race() -> None:
    with pytest.raises(ValueError):
        event("bad", 1, EventType.FILL, 1, float("nan"))
    with pytest.raises(ValueError):
        OrderEvent("bad", 1, "BOGUS", 0, None)  # type: ignore[arg-type]
    state = reconstruct_order(
        [
            event("e1", 1, EventType.SUBMIT, 100),
            event("e2", 2, EventType.ACK),
            event("e3", 3, EventType.CANCEL_REQUEST),
            event("e4", 4, EventType.PARTIAL_FILL, 40, 100.0),
            event("e5", 5, EventType.CANCEL_ACK),
        ]
    )
    assert state.status == "PARTIALLY_FILLED_CANCELLED"
    assert state.filled_quantity == 40


def test_cancel_request_preserves_already_partial_state() -> None:
    state = reconstruct_order(
        [
            event("e1", 1, EventType.SUBMIT, 100),
            event("e2", 2, EventType.ACK),
            event("e3", 3, EventType.PARTIAL_FILL, 20, 100.0),
            event("e4", 4, EventType.CANCEL_REQUEST),
        ]
    )
    assert state.status == "PARTIALLY_FILLED_CANCEL_PENDING"
    assert state.cancel_requested


def test_capacity_reports_each_operating_point_support() -> None:
    result = square_root_capacity(
        alpha_per_unit=0.012,
        fixed_cost_per_unit=0.002,
        impact_coefficient=0.8,
        volatility=0.02,
        daily_volume=100_000.0,
        maximum_calibration_participation=0.25,
    )
    assert result.optimal_extrapolation_ratio == pytest.approx(
        0.6944444444444444, rel=0.0, abs=ABS
    )
    assert result.break_even_extrapolation_ratio == pytest.approx(
        1.5625, rel=0.0, abs=ABS
    )
    assert result.optimal_status == "PASS"
    assert result.break_even_status == "OUTSIDE_SUPPORT"
    assert result.status == "CONDITIONAL_PASS"


def test_impact_fit_fails_closed_on_bad_or_unidentified_data() -> None:
    x = np.arange(5.0)
    y = 0.5 * x
    with pytest.raises(ValueError):
        fit_kyle_lambda(np.array([0.0, 1.0, 2.0, 3.0, np.nan]), y)
    with pytest.raises(ValueError):
        fit_kyle_lambda(np.ones(10), np.linspace(0.0, 1.0, 10))
    with pytest.raises(ValueError):
        fit_square_root_coefficient(
            np.array([-0.1, 0.0, 0.1, 0.2, 0.3]), y
        )


def test_almgren_chriss_never_returns_nonfinite_pass() -> None:
    with pytest.raises(ValueError):
        optimal_schedule(1e200, 1.0, 4, 1e200, 0.0, 0.02)


def test_zero_fill_shortfall_is_valid_and_timestamp_weighted() -> None:
    result = execution_benchmarks(
        "buy",
        100.0,
        100.1,
        np.array([], dtype=float),
        np.array([], dtype=float),
        np.array([100.1, 100.4]),
        np.array([1000.0, 1000.0]),
        100.0,
        market_timestamps=np.array([1.0, 3.0]),
        decision_time=0.0,
        arrival_time=1.0,
        end_time=3.0,
    )
    assert result.realized_average_price is None
    assert result.executed_quantity == pytest.approx(0.0, rel=0.0, abs=ABS)
    assert result.trading_cost == pytest.approx(0.0, rel=0.0, abs=ABS)
    assert result.market_twap == pytest.approx(100.25, rel=0.0, abs=ABS)
    with pytest.raises(ValueError):
        execution_benchmarks(
            "buy", 100.0, 100.1, np.array([]), np.array([]),
            np.array([100.1, 100.2]), np.array([1.0, 1.0]), 100.0,
            market_timestamps=np.array([2.0, 1.0]),
            decision_time=0.0, arrival_time=1.0, end_time=1.0,
        )


def _venue(name: str, size: int, probability: float, cost_shift: float = 0.0) -> VenueQuote:
    return VenueQuote(
        name,
        size,
        probability,
        1.0,
        0.1,
        0.0,
        0.5 + cost_shift,
        0.1,
        2.0,
        0.1,
    )


def test_router_never_oversubmits_and_names_noncompletion_risk() -> None:
    plan = route_order(1000, [_venue("A", 800, 0.85), _venue("B", 700, 0.65)])
    assert plan.total_submitted_quantity == 1000
    assert sum(item.submitted_quantity for item in plan.allocations) == 1000
    assert plan.maximum_overfill_exposure == 0
    assert plan.status == "CONDITIONAL_PASS"
    assert plan.decision_reason == "HARD_CAP_PRESERVED_WITH_NONCOMPLETION_RISK"


def test_market_maker_zero_gamma_and_boundary_suppression() -> None:
    zero_gamma = avellaneda_stoikov_quotes(
        100.0, 0.0, 0.0, 0.02, 1.0, 2.0,
        tick_size=0.01, max_absolute_inventory=500.0,
    )
    assert zero_gamma.half_spread == pytest.approx(0.5, rel=0.0, abs=ABS)
    boundary = avellaneda_stoikov_quotes(
        100.0, 500.0, 0.1, 0.02, 1.0, 1.5,
        tick_size=0.01, max_absolute_inventory=500.0, order_size=1.0,
    )
    assert boundary.bid is None
    assert boundary.ask is not None
    assert boundary.bid_suppression_reason == "PROJECTED_LONG_INVENTORY_LIMIT"
    assert boundary.status == "CONDITIONAL_PASS"


def test_python_book_enforces_price_tick_and_lot_domains() -> None:
    with pytest.raises(ValueError):
        BookSnapshot(
            np.array([10.0, -5.0]), np.array([1.0, 1.0]),
            np.array([11.0]), np.array([1.0]),
        )
    with pytest.raises(ValueError):
        BookSnapshot(
            np.array([99.995]), np.array([1.0]),
            np.array([100.01]), np.array([1.0]), tick_size=0.01,
        )
    with pytest.raises(ValueError):
        BookSnapshot(
            np.array([99.99]), np.array([1.5]),
            np.array([100.01]), np.array([1.0]), lot_size=1.0,
        )


def _choice_for_endpoint(*, nonfill: float, adverse: float, probability: float) -> OrderChoiceInputs:
    return OrderChoiceInputs(
        half_spread_bps=2.0,
        market_fee_bps=0.2,
        passive_fee_bps=0.1,
        passive_rebate_bps=0.2,
        market_impact_bps=0.5,
        fill_probability=probability,
        adverse_selection_bps_if_filled=adverse,
        nonfill_opportunity_cost_bps=nonfill,
        replacement_cost_bps=0.5,
    )


def test_order_choice_break_even_endpoints_name_ties() -> None:
    zero, zero_direction = break_even_fill_probability(
        _choice_for_endpoint(nonfill=2.2, adverse=0.3, probability=0.0)
    )
    one, one_direction = break_even_fill_probability(
        _choice_for_endpoint(nonfill=2.5, adverse=4.8, probability=1.0)
    )
    assert zero == pytest.approx(0.0, rel=0.0, abs=ABS)
    assert zero_direction == "LIMIT_IF_P_GT_0_TIE_AT_0"
    assert one == pytest.approx(1.0, rel=0.0, abs=ABS)
    assert one_direction == "MARKET_IF_P_LT_1_TIE_AT_1"


def test_iceberg_tolerance_and_zero_peak_domain() -> None:
    with pytest.raises(ValueError):
        infer_iceberg(np.array([1.0]), np.array([1.0]), float("nan"))
    with pytest.raises(ValueError):
        infer_iceberg(np.array([1.0]), np.array([1.0]), -1.0)
    result = infer_iceberg(np.zeros(2), np.zeros(2))
    assert result.execution_to_peak_display_ratio is None
    assert result.status == "NO_EVIDENCE"


def test_hawkes_small_sample_cannot_receive_full_pass_and_history_matters() -> None:
    parameters = HawkesParameters(1.0, 0.4, 1.0)
    events = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    diagnostics = hawkes_diagnostics(events, parameters, 6.0)
    assert diagnostics.evidence_status == "INSUFFICIENT_EVIDENCE"
    assert diagnostics.status == "CONDITIONAL_PASS"
    without_history = log_likelihood(events, parameters, 6.0)
    with_history = log_likelihood(
        events, parameters, 6.0, history_before_zero=np.array([-0.5])
    )
    assert with_history != pytest.approx(without_history, rel=0.0, abs=ABS)


def _tca(ledger: float, order_quantity: float = 100.0) -> object:
    return tca_decomposition(
        side="buy",
        arrival_price=100.0,
        order_quantity=order_quantity,
        market_volume=1000.0,
        fill_times=np.array([1.0]),
        fill_quantities=np.array([100.0]),
        fill_prices=np.array([100.05]),
        venues=("A",),
        fee_bps=np.array([0.1]),
        midpoint_at_fills=np.array([100.02]),
        counterfactual_midpoint_at_fills=np.array([100.01]),
        ledger_all_in_cost_bps=ledger,
        reconciliation_tolerance_bps=1e-9,
    )


def test_tca_uses_independent_ledger_and_blocks_impossible_participation() -> None:
    rejected = _tca(ledger=999.0)
    assert rejected.status == "REJECT"
    assert abs(rejected.residual_bps) > rejected.reconciliation_tolerance_bps
    with pytest.raises(ValueError):
        _tca(ledger=5.1, order_quantity=2000.0)


def test_pretrade_unknown_state_and_invalid_limits_fail_closed() -> None:
    with pytest.raises(ValueError):
        RiskLimits(100, float("nan"), 0.1, 50.0, 1000, "v1", 1000.0, 1000.0)
    limits = RiskLimits(100, 100000.0, 0.1, 50.0, 1000, "v1", 1000.0, 100000.0)
    order = OrderIntent(
        10, 100.0, 100.0, 1000.0, 10, "v1",
        float("nan"), float("nan"), False,
    )
    assert pretrade_check(order, limits) == [
        "UNKNOWN_POSITION_STATE", "UNKNOWN_CREDIT_STATE"
    ]


@pytest.mark.skip(reason="requires non-public publication provenance assets")
def test_integrated_evidence_hashes_inputs_results_policy_and_actual_route() -> None:
    baseline = governed_execution_fixture()
    mutated = governed_execution_fixture(queue_ahead_lots=4)
    required_completion = governed_execution_fixture(route_completion_required=True)
    for digest in (
        baseline.input_snapshot_hash,
        baseline.result_bundle_hash,
        baseline.policy_hash,
        baseline.release_evidence_hash,
        baseline.route_hash,
    ):
        assert len(digest) == 64
        int(digest, 16)
    assert baseline.input_snapshot_hash != mutated.input_snapshot_hash
    assert baseline.result_bundle_hash != mutated.result_bundle_hash
    assert baseline.route_total_submitted == 1000
    assert not baseline.aggregate_pretrade_breaches
    assert all(not breaches for _, breaches in baseline.child_pretrade_breaches)
    assert baseline.signature_status == "UNSIGNED_LOCAL_EVIDENCE"
    assert required_completion.decision == "REJECT"


def test_integrated_evidence_hash_quantizes_sub_tolerance_float_noise() -> None:
    baseline = {"probability": 0.6915015566393556, "zero": 0.0}
    perturbed = {"probability": 0.6915015566393548, "zero": -0.0}
    material_change = {"probability": 0.6915015566, "zero": 0.0}

    assert _digest(baseline) == _digest(perturbed)
    assert _digest(baseline) != _digest(material_change)
