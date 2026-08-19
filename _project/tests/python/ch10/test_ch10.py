from __future__ import annotations

import numpy as np
import pytest

from quant_interviews.ch10.almgren_chriss import optimal_schedule
from quant_interviews.ch10.benchmarks import execution_benchmarks
from quant_interviews.ch10.execution_state import EventType, OrderEvent, reconstruct_order
from quant_interviews.ch10.hawkes import (
    HawkesParameters,
    hawkes_diagnostics,
    log_likelihood,
    simulate_hawkes,
)
from quant_interviews.ch10.iceberg import infer_iceberg
from quant_interviews.ch10.impact_models import (
    fit_kyle_lambda,
    fit_square_root_coefficient,
    square_root_capacity,
)
from quant_interviews.ch10.integrated_case import governed_execution_fixture
from quant_interviews.ch10.market_maker import avellaneda_stoikov_quotes
from quant_interviews.ch10.order_book import BookSnapshot
from quant_interviews.ch10.order_choice import OrderChoiceInputs, choose_order_type
from quant_interviews.ch10.pretrade import OrderIntent, RiskLimits, pretrade_check
from quant_interviews.ch10.queue_fill import QueueModel, competing_risks
from quant_interviews.ch10.smart_router import VenueQuote, route_order
from quant_interviews.ch10.tca import moving_block_bootstrap_mean, tca_decomposition


PRICE_ABS = 1e-10
PROBABILITY_ABS = 1e-10
QUANTITY_ABS = 1e-8
BPS_ABS = 1e-9


def test_order_book_metrics_and_sweep() -> None:
    book = BookSnapshot(
        bid_prices=np.array([99.99, 99.98, 99.97]),
        bid_sizes=np.array([700.0, 800.0, 900.0]),
        ask_prices=np.array([100.01, 100.02, 100.03]),
        ask_sizes=np.array([300.0, 700.0, 900.0]),
    )
    metrics = book.metrics()
    assert metrics.spread == pytest.approx(0.02, rel=0.0, abs=PRICE_ABS)
    assert metrics.top_imbalance == pytest.approx(0.4, rel=0.0, abs=PROBABILITY_ABS)
    assert metrics.microprice > metrics.midpoint
    assert book.market_order_vwap("buy", 500.0) == pytest.approx(
        (300 * 100.01 + 200 * 100.02) / 500,
        rel=0.0,
        abs=PRICE_ABS,
    )


@pytest.mark.parametrize(
    "bid,ask",
    [(100.0, 99.9), (100.0, 100.0), (float("nan"), 100.1)],
)
def test_invalid_book_rejected(bid: float, ask: float) -> None:
    with pytest.raises(ValueError):
        BookSnapshot(
            np.array([bid]), np.array([1.0]), np.array([ask]), np.array([1.0])
        )


def test_queue_fill_analytic_matches_ordered_event_simulation() -> None:
    model = QueueModel(
        queue_ahead_lots=3,
        marketable_lots_per_second=2.0,
        cancellation_ahead_lots_per_second=0.5,
    )
    analytic = model.fill_probability(2.0)
    simulated = model.simulate_fill_probability(2.0, simulations=200_000, seed=3)
    assert simulated == pytest.approx(analytic, abs=0.01)


def test_queue_fill_analytic_matches_marked_monte_carlo() -> None:
    """Compatibility entry point for the preserved v0.9 printed listing."""

    test_queue_fill_analytic_matches_ordered_event_simulation()


test_queue_fill_analytic_matches_marked_monte_carlo.__test__ = False


def test_cancellation_alone_cannot_fill_own_lot() -> None:
    model = QueueModel(0, marketable_lots_per_second=0.0, cancellation_ahead_lots_per_second=9.0)
    assert model.fill_probability(10.0) == pytest.approx(
        0.0, rel=0.0, abs=PROBABILITY_ABS
    )
    with pytest.raises(ValueError):
        model.simulate_fill_probability(float("nan"))


def test_competing_risk_probabilities_reconcile() -> None:
    result = competing_risks(0.4, 0.2, 0.1, 5.0)
    assert (
        result.fill_first_probability
        + result.cancel_first_probability
        + result.adverse_move_first_probability
        + result.survival_probability
    ) == pytest.approx(1.0, rel=0.0, abs=PROBABILITY_ABS)


def _order_choice(**overrides: float) -> OrderChoiceInputs:
    values: dict[str, float] = {
        "half_spread_bps": 2.0,
        "market_fee_bps": 0.2,
        "passive_fee_bps": 0.1,
        "passive_rebate_bps": 0.2,
        "market_impact_bps": 0.5,
        "fill_probability": 0.8,
        "adverse_selection_bps_if_filled": 0.3,
        "nonfill_opportunity_cost_bps": 2.5,
        "replacement_cost_bps": 0.5,
    }
    values.update(overrides)
    return OrderChoiceInputs(**values)


def test_order_choice_changes_with_adverse_selection() -> None:
    assert choose_order_type(_order_choice()).decision == "LIMIT"
    assert (
        choose_order_type(_order_choice(adverse_selection_bps_if_filled=5.0)).decision
        == "MARKET"
    )


def test_order_choice_reports_break_even_direction_and_fee() -> None:
    decision = choose_order_type(_order_choice(fill_probability=0.5))
    assert decision.break_even_fill_probability is not None
    assert "FILL_PROBABILITY" in decision.threshold_direction
    with pytest.raises(ValueError):
        _order_choice(passive_fee_bps=float("nan"))


def test_kyle_and_square_root_fits_recover_coefficients() -> None:
    rng = np.random.default_rng(5)
    volume = rng.normal(0.0, 2.0, 400)
    price_change = 0.001 + 0.004 * volume + rng.normal(0.0, 0.001, 400)
    kyle = fit_kyle_lambda(volume, price_change)
    assert kyle.coefficient == pytest.approx(0.004, abs=3e-4)
    participation = np.linspace(0.001, 0.25, 200)
    impact = 0.05 + 0.8 * np.sqrt(participation)
    square_root = fit_square_root_coefficient(participation, impact)
    assert square_root.coefficient == pytest.approx(0.8, abs=1e-10)
    assert square_root.residual_standard_error < 1e-10


def test_capacity_distinguishes_profit_optimum_from_break_even() -> None:
    result = square_root_capacity(
        alpha_per_unit=0.012,
        fixed_cost_per_unit=0.002,
        impact_coefficient=0.8,
        volatility=0.02,
        daily_volume=1_000_000,
        maximum_calibration_participation=0.5,
    )
    assert result.optimal_quantity == pytest.approx(
        4.0 * result.break_even_quantity / 9.0,
        rel=1e-12,
        abs=QUANTITY_ABS,
    )
    assert result.expected_profit_at_optimum > 0.0


def test_almgren_chriss_schedule_is_monotone_and_finishes() -> None:
    schedule = optimal_schedule(1_000_000, 1.0, 20, 1e-6, 1e-6, 0.02)
    assert schedule.holdings[0] == pytest.approx(
        1_000_000, rel=0.0, abs=QUANTITY_ABS
    )
    assert schedule.holdings[-1] == pytest.approx(0.0, abs=1e-8)
    assert np.all(np.diff(schedule.holdings) <= 1e-9)
    assert schedule.risk_penalty == pytest.approx(
        1e-6 * schedule.inventory_variance,
        rel=1e-12,
        abs=1e-12,
    )


def test_almgren_chriss_linear_and_large_kappa_stable() -> None:
    linear = optimal_schedule(100.0, 1.0, 4, 1.0, 0.0, 0.02)
    assert linear.holdings.tolist() == pytest.approx(
        [100.0, 75.0, 50.0, 25.0, 0.0],
        rel=0.0,
        abs=QUANTITY_ABS,
    )
    steep = optimal_schedule(100.0, 100.0, 20, 1e-8, 10.0, 10.0)
    assert np.all(np.isfinite(steep.holdings))
    assert np.all(np.diff(steep.holdings) <= 1e-12)


def test_execution_shortfall_exact_counterexample_is_220() -> None:
    result = execution_benchmarks(
        "buy",
        100.0,
        100.1,
        np.array([100.15, 100.2]),
        np.array([400.0, 400.0]),
        np.array([100.0, 100.2, 100.4]),
        np.array([1000.0, 1200.0, 900.0]),
        1000.0,
        market_timestamps=np.array([1.0, 2.0, 3.0]),
        decision_time=0.0,
        arrival_time=1.0,
        end_time=3.0,
    )
    assert result.implementation_shortfall == pytest.approx(
        220.0, rel=0.0, abs=PRICE_ABS
    )
    assert result.opportunity_cost == pytest.approx(
        60.0, rel=0.0, abs=PRICE_ABS
    )
    assert result.implementation_shortfall == pytest.approx(
        result.delay_cost + result.trading_cost + result.opportunity_cost,
        rel=0.0,
        abs=PRICE_ABS,
    )


def test_execution_benchmark_rejects_overfill() -> None:
    with pytest.raises(ValueError):
        execution_benchmarks(
            "buy",
            100.0,
            100.1,
            np.array([100.2]),
            np.array([101.0]),
            np.array([100.3]),
            np.array([1000.0]),
            100.0,
            market_timestamps=np.array([1.0, 2.0]),
            decision_time=0.0,
            arrival_time=1.0,
            end_time=2.0,
        )


def test_router_preserves_simultaneous_parent_cap_and_reports_completion_risk() -> None:
    venues = [
        VenueQuote("A", 500, 0.9, 1.0, 0.1, 0.2, 0.5, 0.1, 2.0, 0.1),
        VenueQuote("B", 500, 0.7, 1.0, 0.1, 0.0, 1.5, 0.4, 2.5, 0.2),
    ]
    plan = route_order(700, venues)
    assert plan.allocations[0].venue == "A"
    assert plan.total_submitted_quantity == 700
    assert plan.expected_filled_quantity == pytest.approx(
        590.0, rel=0.0, abs=QUANTITY_ABS
    )
    assert plan.maximum_overfill_exposure == 0
    assert plan.status == "CONDITIONAL_PASS"


def test_router_meets_expected_completion_and_reports_overfill_exposure() -> None:
    """Compatibility entry point for the preserved v0.9 printed listing."""

    test_router_preserves_simultaneous_parent_cap_and_reports_completion_risk()


test_router_meets_expected_completion_and_reports_overfill_exposure.__test__ = False


def test_iceberg_reports_replenishment_not_false_hidden_lower_bound() -> None:
    evidence = infer_iceberg(
        np.array([100.0, 80.0, 100.0, 70.0, 100.0]),
        np.array([20.0, 30.0, 40.0, 50.0, 60.0]),
    )
    assert evidence.replenishment_count == 2
    assert evidence.excess_execution_over_peak_display == pytest.approx(
        100.0, rel=0.0, abs=QUANTITY_ABS
    )
    assert evidence.hidden_quantity_lower_bound == pytest.approx(
        0.0, rel=0.0, abs=QUANTITY_ABS
    )
    assert evidence.execution_to_peak_display_ratio == pytest.approx(
        2.0, rel=0.0, abs=PROBABILITY_ABS
    )
    assert evidence.status == "EVIDENCE_OF_REPLENISHMENT"


def test_iceberg_empty_episode_rejected() -> None:
    with pytest.raises(ValueError):
        infer_iceberg(np.array([]), np.array([]))


def test_tca_all_in_decomposition_reconciles_and_signs_toxicity() -> None:
    result = tca_decomposition(
        side="buy",
        arrival_price=100.0,
        order_quantity=50_000.0,
        market_volume=1_000_000.0,
        fill_times=np.array([1.0, 2.0]),
        fill_quantities=np.array([25_000.0, 25_000.0]),
        fill_prices=np.array([100.07, 100.09]),
        venues=("A", "B"),
        fee_bps=np.array([0.25, 0.25]),
        midpoint_at_fills=np.array([100.02, 100.04]),
        counterfactual_midpoint_at_fills=np.array([100.01, 100.03]),
        ledger_all_in_cost_bps=8.25,
        reconciliation_tolerance_bps=1e-10,
        post_trade_midpoints=np.array([100.00, 100.02]),
        markout_horizon_seconds=60.0,
    )
    assert result.all_in_cost_bps == pytest.approx(
        result.price_slippage_bps + result.fee_component_bps,
        rel=0.0,
        abs=BPS_ABS,
    )
    assert result.all_in_cost_bps == pytest.approx(
        result.model_attribution_total_bps
        + result.residual_bps,
        rel=0.0,
        abs=BPS_ABS,
    )
    assert result.model_attribution_total_bps == pytest.approx(
        result.spread_component_bps
        + result.impact_component_bps
        + result.timing_component_bps
        + result.fee_component_bps,
        rel=0.0,
        abs=BPS_ABS,
    )
    assert result.post_trade_drift_bps == pytest.approx(
        -2.0, rel=0.0, abs=BPS_ABS
    )
    assert result.adverse_selection_cost_bps == pytest.approx(
        2.0, rel=0.0, abs=BPS_ABS
    )
    assert result.status == "PASS"


def test_block_bootstrap_interval_contains_mean_and_validates_simulations() -> None:
    observations = np.sin(np.linspace(0, 8, 200)) + 0.1
    mean, lower, upper = moving_block_bootstrap_mean(
        observations,
        block_length=10,
        simulations=1000,
        seed=4,
    )
    assert lower < mean < upper
    with pytest.raises(ValueError):
        moving_block_bootstrap_mean(observations, 10, simulations=0)


def test_hawkes_simulation_and_rescaling_diagnostics_are_finite() -> None:
    params = HawkesParameters(0.5, 0.4, 1.0)
    first = simulate_hawkes(params, 40.0, seed=8)
    second = simulate_hawkes(params, 40.0, seed=8)
    assert np.array_equal(first.events, second.events)
    assert np.array_equal(first.history_before_zero, second.history_before_zero)
    assert np.isfinite(
        log_likelihood(
            first.events,
            params,
            40.0,
            history_before_zero=first.history_before_zero,
        )
    )
    diagnostics = hawkes_diagnostics(
        first.events,
        params,
        40.0,
        history_before_zero=first.history_before_zero,
    )
    assert diagnostics.event_count == len(first.events)
    assert diagnostics.status in {"PASS", "REJECT", "CONDITIONAL_PASS"}


def test_hawkes_rejects_nonfinite_or_explosive_parameters() -> None:
    with pytest.raises(ValueError):
        HawkesParameters(0.5, 1.1, 1.0)
    with pytest.raises(ValueError):
        HawkesParameters(float("nan"), 0.1, 1.0)


def test_market_maker_rounds_ticks_and_blocks_inventory_breach() -> None:
    flat = avellaneda_stoikov_quotes(
        100.0, 0.0, 0.1, 0.02, 1.0, 1.5, tick_size=0.01, max_absolute_inventory=500
    )
    long = avellaneda_stoikov_quotes(
        100.0, 100.0, 0.1, 0.02, 1.0, 1.5, tick_size=0.01, max_absolute_inventory=500
    )
    blocked = avellaneda_stoikov_quotes(
        100.0, 501.0, 0.1, 0.02, 1.0, 1.5, tick_size=0.01, max_absolute_inventory=500
    )
    assert long.reservation_price < flat.reservation_price
    assert flat.bid is not None and flat.ask is not None
    assert flat.bid * 100 == pytest.approx(
        round(flat.bid * 100), rel=0.0, abs=PRICE_ABS
    )
    assert flat.ask > flat.bid
    assert blocked.status == "REJECT"


def _event(
    event_id: str,
    sequence: int,
    event_type: EventType,
    quantity: int = 0,
    price: float | None = None,
) -> OrderEvent:
    return OrderEvent(event_id, sequence, event_type, quantity, price)


def test_event_sourcing_is_idempotent_and_preserves_partial_cancel_state() -> None:
    partial = _event("e3", 3, EventType.PARTIAL_FILL, 300, 100.1)
    state = reconstruct_order(
        [
            _event("e1", 1, EventType.SUBMIT, 1000),
            _event("e2", 2, EventType.ACK),
            partial,
            partial,
            _event("e4", 4, EventType.CANCEL_REQUEST),
            _event("e5", 5, EventType.CANCEL_ACK),
        ]
    )
    assert state.filled_quantity == 300
    assert state.status == "PARTIALLY_FILLED_CANCELLED"
    assert state.applied_event_ids == ("e1", "e2", "e3", "e4", "e5")


@pytest.mark.parametrize(
    "events",
    [
        [_event("e1", 1, EventType.CANCEL_ACK)],
        [
            _event("e1", 1, EventType.SUBMIT, 100),
            _event("e2", 2, EventType.ACK),
            _event("e3", 3, EventType.FILL, 100, 10.0),
            _event("e4", 4, EventType.ACK),
        ],
        [
            _event("e1", 1, EventType.SUBMIT, 100),
            _event("e3", 3, EventType.ACK),
        ],
        [
            _event("e1", 1, EventType.SUBMIT, 100),
            _event("e2", 2, EventType.ACK),
            _event("e3", 3, EventType.FILL, 99, 10.0),
        ],
    ],
)
def test_event_sourcing_rejects_invalid_transitions(events: list[OrderEvent]) -> None:
    with pytest.raises(ValueError):
        reconstruct_order(events)


def _limits(**overrides: object) -> RiskLimits:
    values: dict[str, object] = {
        "max_order_quantity": 100_000,
        "max_notional": 5_000_000.0,
        "max_participation": 0.10,
        "price_collar_bps": 50.0,
        "max_market_volume_age_ms": 1_000,
        "limit_snapshot_version": "limits-v3",
        "max_position_after_order": 200_000.0,
        "max_credit_notional": 5_000_000.0,
    }
    values.update(overrides)
    return RiskLimits(**values)


def test_pretrade_checks_multiple_limits() -> None:
    order = OrderIntent(
        75_000,
        101.0,
        100.0,
        500_000.0,
        100,
        "limits-v3",
        projected_position=250_000.0,
        available_credit_notional=4_000_000.0,
        self_trade_risk=False,
    )
    breaches = pretrade_check(order, _limits())
    assert set(breaches) == {
        "MAX_NOTIONAL",
        "MAX_PARTICIPATION",
        "PRICE_COLLAR",
        "POSITION_LIMIT",
        "CREDIT_LIMIT",
    }


@pytest.mark.parametrize("volume", [float("nan"), 0.0, -1.0])
def test_pretrade_fails_closed_for_invalid_volume(volume: float) -> None:
    order = OrderIntent(
        100,
        100.0,
        100.0,
        volume,
        100,
        "limits-v3",
        0.0,
        1_000_000.0,
        False,
    )
    assert "INVALID_OR_STALE_MARKET_VOLUME" in pretrade_check(order, _limits())


def test_pretrade_blocks_stale_state_kill_switch_and_self_trade() -> None:
    order = OrderIntent(
        100,
        100.0,
        100.0,
        10_000.0,
        2_000,
        "limits-v2",
        projected_position=0.0,
        available_credit_notional=1_000_000.0,
        self_trade_risk=True,
    )
    breaches = pretrade_check(order, _limits(kill_switch_active=True))
    assert {
        "KILL_SWITCH_ACTIVE",
        "STALE_LIMIT_SNAPSHOT",
        "SELF_TRADE_RISK",
        "INVALID_OR_STALE_MARKET_VOLUME",
    }.issubset(breaches)


@pytest.mark.skip(reason="requires non-public publication provenance assets")
def test_integrated_case_is_deterministic_and_governed() -> None:
    first = governed_execution_fixture()
    second = governed_execution_fixture()
    assert first == second
    assert len(first.fixture_hash) == 64
    assert first.fixture_hash == second.fixture_hash
    assert first.implementation_shortfall == pytest.approx(
        220.0, rel=0.0, abs=PRICE_ABS
    )
    assert first.tca_residual_bps == pytest.approx(
        0.0, rel=0.0, abs=BPS_ABS
    )
    assert first.route_total_submitted == 1000
    assert first.route_overfill_exposure == 0
    assert first.signature_status == "UNSIGNED_LOCAL_EVIDENCE"
    assert first.decision == "CONDITIONAL_PASS"
