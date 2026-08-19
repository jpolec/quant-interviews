from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from math import log2, sqrt
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from quant_interviews.ch14.adaptive_interview import (
    Evidence,
    Question,
    choose_next_question,
    competency_need,
)
from quant_interviews.ch14.calibration import (
    InterviewScore,
    interviewer_diagnostics,
    pairwise_agreement,
    standard_error,
)
from quant_interviews.ch14.developer_case import (
    OnlineStats,
    UpdateableTopK,
    batch_updateable_top_k,
    deterministic_top_k,
    relative_error,
)
from quant_interviews.ch14.desk_quant_case import (
    black_scholes_call,
    black_scholes_pde_residual,
    delta_hedged_pnl,
    normal_cdf,
    normal_pdf,
    weighted_rmse,
)
from quant_interviews.ch14.ml_case import (
    brier_score,
    log_loss,
    net_value,
    nonoverlapping_net_value,
    population_stability_index,
    purged_train_indices,
)
from quant_interviews.ch14.platform_case import (
    Event,
    JournalState,
    approval_commitment,
    approval_gate,
    canonical_event_payload,
    fold_events,
)
from quant_interviews.ch14.pm_case import (
    allocate_sleeves,
    canonical_four_sleeve_fixture,
    covariance_certificate,
    optimal_capital,
    risk_contributions,
)
from quant_interviews.ch14.preparation_plan import Topic, allocate_hours, weekly_plan
from quant_interviews.ch14.research_case import (
    annualized_sharpe,
    benjamini_hochberg_qvalues,
    bounded_fold_stability,
    expected_max_standard_normal,
    net_information_ratio,
    research_launch_gate,
    selection_adjusted_score,
)
from quant_interviews.ch14.risk_case import (
    historical_var_es,
    liquidity_adjusted_capital,
    limit_utilization,
    reverse_stress_scale,
    scenario_pnl,
    tail_risk_gate,
)
from quant_interviews.ch14.scoring import (
    EvidenceItem,
    ScoreCard,
    evidence_completeness,
    hiring_decision,
    mean_and_se,
)
from quant_interviews.ch14.takehome_case import (
    SubmissionManifest,
    canonical_bytes,
    canonical_hash,
    score_takehome,
    validate_manifest,
)
from quant_interviews.ch14.trader_case import (
    choose_order,
    fill_probability_threshold,
    limit_order_expected_cost,
    market_order_cost,
)


ROOT = Path(__file__).resolve().parents[4]


def digest(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return sha256(payload).hexdigest()


def decision_hashes() -> dict[str, str]:
    return {
        "market_snapshot_hash": digest("market"),
        "fill_model_hash": digest("fill-model"),
        "portfolio_snapshot_hash": digest("portfolio"),
        "limit_snapshot_hash": digest("limits"),
    }


def test_gaussian_maximum_small_m_exact_monotone_and_large_fixture() -> None:
    assert expected_max_standard_normal(1.0) == 0.0
    assert expected_max_standard_normal(2.0) == pytest.approx(
        1.0 / sqrt(np.pi), abs=1.0e-12
    )
    assert expected_max_standard_normal(3.0) == pytest.approx(
        0.8462843753216345, abs=1.0e-12
    )
    values = [expected_max_standard_normal(float(value)) for value in range(1, 31)]
    assert np.all(np.diff(values) > 0.0)
    assert expected_max_standard_normal(500.0) == pytest.approx(
        3.0366993459289464, abs=1.0e-12
    )


def test_selection_certificate_uses_explicit_se_and_launch_aggregator() -> None:
    certificate = selection_adjusted_score(
        1.9,
        500.0,
        12.0,
        sharpe_standard_error=1.0 / sqrt(12.0),
        standard_error_method="iid_demo_only",
        dependence_method="effective_trials_bootstrap_v1",
    )
    assert certificate.adjusted_sharpe == pytest.approx(1.0233804075899808)
    assert certificate.adjusted_lower_bound == pytest.approx(0.5485520654409983)
    assert certificate.adjusted_lower_bound > 0.0
    economics = net_information_ratio(
        0.12,
        0.08,
        1.8,
        0.02,
        0.009,
        turnover_horizon="annual",
        turnover_convention="one_way",
    )
    assert economics.net_alpha == pytest.approx(0.075)
    decision = research_launch_gate(
        certificate,
        economics,
        point_in_time_pass=True,
        holdout_untouched=True,
        capacity_pass=True,
        operational_readiness_pass=True,
    )
    assert decision.decision == "PASS"
    assert research_launch_gate(
        certificate,
        economics,
        point_in_time_pass=True,
        holdout_untouched=False,
        capacity_pass=True,
        operational_readiness_pass=True,
    ).decision == "REJECT"
    with pytest.raises(ValueError, match="annual"):
        net_information_ratio(
            0.12,
            0.08,
            1.8,
            0.02,
            turnover_horizon="monthly",  # type: ignore[arg-type]
        )


def test_research_statistics_bh_by_and_direct_helpers() -> None:
    returns = np.array([0.01, -0.005, 0.007, 0.003, -0.002, 0.008])
    assert np.isfinite(annualized_sharpe(returns))
    pvalues = np.array([0.001, 0.02, 0.5])
    bh, rejected = benjamini_hochberg_qvalues(pvalues)
    by, _ = benjamini_hochberg_qvalues(
        pvalues, method="BY_ARBITRARY_DEPENDENCE"
    )
    assert rejected.tolist() == [True, True, False]
    assert np.all(by >= bh)
    assert 0.0 <= bounded_fold_stability(np.array([0.8, 1.0, 0.6])) <= 1.0
    with pytest.raises(ValueError):
        benjamini_hochberg_qvalues(np.array([0.01, np.nan]))


@pytest.mark.parametrize(
    ("market", "filled", "miss", "classification", "threshold"),
    [
        (2.6, 0.6, 4.0, "INTERIOR", 0.4117647058823529),
        (5.0, 1.0, 4.0, "LIMIT_DOMINATES", None),
        (0.0, 1.0, 4.0, "MARKET_DOMINATES", None),
        (2.0, 2.0, 2.0, "EQUAL", None),
        (-1.0, 1.0, 4.0, "INVALID_MODEL", None),
    ],
)
def test_fill_threshold_is_typed(
    market: float,
    filled: float,
    miss: float,
    classification: str,
    threshold: float | None,
) -> None:
    result = fill_probability_threshold(market, filled, miss)
    assert result.classification == classification
    assert result.probability == pytest.approx(threshold) if threshold is not None else result.probability is None


def test_order_gate_recomputes_freshness_risk_and_hash() -> None:
    market = market_order_cost(1.0, 1.5, 0.1)
    limit = limit_order_expected_cost(0.8, 0.2, 0.8, 4.0)
    kwargs = {
        "market_cost_bps": market,
        "limit_expected_cost_bps": limit,
        "no_trade_cost_bps": 7.0,
        "current_inventory": 0.0,
        "order_quantity": 1.0,
        "fill_probability": 0.8,
        "volatility": 0.2,
        "horizon_years": 1.0 / 252.0,
        "risk_aversion_bps_per_variance": 10.0,
        "proposed_risk": 5.0,
        "risk_limit": 10.0,
        "decision_time": "2026-01-01T12:00:01Z",
        "market_observed_at": "2026-01-01T12:00:00Z",
        "maximum_market_age_seconds": 2.0,
        "policy_version": "order-gate-v1",
        **decision_hashes(),
    }
    decision = choose_order(**kwargs)
    assert decision.selected_action == "LIMIT"
    assert decision.status == "PASS"
    assert len(decision.decision_hash) == 64
    changed = dict(kwargs)
    changed["market_snapshot_hash"] = digest("changed-market")
    assert choose_order(**changed).decision_hash != decision.decision_hash
    stale = dict(kwargs)
    stale["decision_time"] = "2026-01-01T12:00:03Z"
    assert choose_order(**stale).selected_action == "BLOCK_STALE_MARKET"
    over_limit = dict(kwargs)
    over_limit["proposed_risk"] = 11.0
    assert choose_order(**over_limit).selected_action == "BLOCK_RISK"


def test_black_scholes_pde_dividend_boundary_and_finite_postconditions() -> None:
    result = black_scholes_call(100.0, 100.0, 1.0, 0.04, 0.25, 0.01)
    assert result.price == pytest.approx(11.23555759403962)
    assert abs(
        black_scholes_pde_residual(
            result, spot=100.0, rate=0.04, volatility=0.25
        )
    ) < 1.0e-12
    expiry = black_scholes_call(100.0, 100.0, 0.0, 0.04, 0.25, 0.01)
    assert expiry.delta == 0.5
    assert expiry.status == "BOUNDARY_CONVENTION"
    assert expiry.boundary_convention == "symmetric_subgradient_at_expiry_kink"
    with pytest.raises(FloatingPointError):
        black_scholes_call(1.0e308, 1.0e308, 1.0e-10, 0.04, 1.0, -1.0e308)


def test_self_financing_hedge_ledger_and_pricing_helpers() -> None:
    before = black_scholes_call(100.0, 100.0, 1.0, 0.04, 0.25, 0.01)
    after = black_scholes_call(
        102.0, 100.0, 1.0 - 1.0 / 252.0, 0.04, 0.25, 0.01
    )
    ledger = delta_hedged_pnl(
        price_before=before.price,
        price_after=after.price,
        spot_before=100.0,
        spot_after=102.0,
        delta_before=before.delta,
        gamma_before=before.gamma,
        raw_vega_before=before.raw_vega_per_unit_volatility,
        theta_per_year=before.theta_per_year,
        volatility_move=0.0,
        elapsed_years=1.0 / 252.0,
        rate=0.04,
        dividend_yield=0.01,
        tolerance=0.02,
    )
    assert ledger.initial_cash == pytest.approx(
        before.delta * 100.0 - before.price
    )
    assert ledger.dividend_cash_flow < 0.0
    assert ledger.status == "PASS"
    assert weighted_rmse([1.0, -1.0], [1.0, 1.0]) == 1.0
    assert normal_cdf(0.0) == 0.5
    assert normal_pdf(0.0) == pytest.approx(1.0 / sqrt(2.0 * np.pi))


def test_online_stats_rejects_corruption_and_updates_transactionally() -> None:
    with pytest.raises(ValueError):
        OnlineStats(count=-1)
    with pytest.raises(ValueError):
        OnlineStats(count=2, mean=0.0, m2=-1.0)
    stats = OnlineStats()
    with pytest.raises(ValueError, match="at least two"):
        _ = stats.variance
    for value in [1.0, 2.0, 3.0, 4.0]:
        stats.update(value)
    assert stats.mean == 2.5
    assert stats.variance == pytest.approx(5.0 / 3.0)
    extreme = OnlineStats()
    extreme.update(1.0e308)
    with pytest.raises(FloatingPointError):
        extreme.update(-1.0e308)
    assert extreme.count == 1 and extreme.mean == 1.0e308 and extreme.m2 == 0.0


def test_updateable_top_k_expected_height_reference_and_relative_error() -> None:
    book = UpdateableTopK()
    for index in range(500):
        book.update(f"id-{index:04d}", float(index))
    assert book.top_k(2) == [(499.0, "id-0499"), (498.0, "id-0498")]
    assert book.tree_height < 6.0 * log2(book.active_count + 1)
    book.update("id-0499", -1.0)
    assert book.top_k(1) == [(498.0, "id-0498")]
    assert book.delete("id-0498")
    assert not book.delete("missing")
    assert book.top_k(1) == [(497.0, "id-0497")]
    updates = [(5.0, "A"), (4.0, "B"), (1.0, "A")]
    assert deterministic_top_k(updates, 2) == batch_updateable_top_k(updates, 2)
    assert relative_error(1.01, 1.0) == pytest.approx(0.01)


def test_correlated_four_sleeve_fixture_and_singular_covariance_reject() -> None:
    fixture = canonical_four_sleeve_fixture()
    assert fixture.covariance.shape == (4, 4)
    assert np.count_nonzero(fixture.covariance - np.diag(np.diag(fixture.covariance)))
    certificate = covariance_certificate(
        fixture.current_weights, fixture.covariance
    )
    assert certificate.status == "PASS"
    assert abs(certificate.contribution_residual) < 1.0e-12
    assert np.allclose(
        risk_contributions(fixture.current_weights, fixture.covariance),
        certificate.risk_contributions,
    )
    singular = covariance_certificate(np.array([0.0, 1.0]), np.diag([0.0, 1.0]))
    assert singular.status == "REJECT_SINGULAR"
    assert singular.condition_number is None


def test_full_four_sleeve_allocation_costs_residuals_and_shrinkage() -> None:
    fixture = canonical_four_sleeve_fixture()

    def solve(shrinkage: float):
        return allocate_sleeves(
            fixture.expected_returns,
            fixture.covariance,
            fixture.current_weights,
            fixture.factor_loadings,
            fixture.factor_target,
            risk_aversion=4.0,
            borrow_cost_rates=fixture.borrow_cost_rates,
            turnover_cost_rates=fixture.turnover_cost_rates,
            impact_coefficients=fixture.impact_coefficients,
            capacity_units=fixture.capacity_units,
            lower_bounds=fixture.lower_bounds,
            upper_bounds=fixture.upper_bounds,
            return_shrinkage=shrinkage,
            uncertainty_margin=0.005,
        )

    allocation = solve(0.75)
    conservative = solve(0.50)
    assert allocation.decision == conservative.decision == "PASS"
    assert np.sum(allocation.weights) == pytest.approx(1.0, abs=1.0e-7)
    assert abs(allocation.factor_residual) < 1.0e-7
    assert allocation.maximum_bound_violation == 0.0
    assert allocation.linear_turnover_cost > 0.0
    assert allocation.nonlinear_impact_cost > 0.0
    assert not np.allclose(allocation.weights, conservative.weights)


def test_short_enabled_allocation_exercises_borrow_cost() -> None:
    allocation = allocate_sleeves(
        expected_returns=np.array([0.00, 0.10]),
        covariance=np.diag([0.04, 0.04]),
        current_weights=np.array([0.5, 0.5]),
        factor_loadings=np.array([1.0, 0.0]),
        factor_target=-0.2,
        risk_aversion=2.0,
        borrow_cost_rates=np.array([0.02, 0.0]),
        lower_bounds=np.array([-0.3, 0.0]),
        upper_bounds=np.array([0.8, 1.3]),
    )
    assert allocation.decision == "PASS"
    assert allocation.weights == pytest.approx([-0.2, 1.2], abs=1.0e-7)
    assert allocation.borrow_cost == pytest.approx(0.004, abs=1.0e-9)


def test_capacity_has_interior_optimum_and_boundary_status() -> None:
    interior = optimal_capital(0.8, 0.12, 0.025, maximum_capital=50.0)
    assert interior.status == "PASS" and not interior.boundary
    assert abs(interior.derivative_residual) < 1.0e-10
    constrained = optimal_capital(0.8, 0.12, 0.025, maximum_capital=1.0)
    assert constrained.status == "CONDITIONAL_PASS" and constrained.boundary


def test_empirical_var_es_convention_and_separate_risk_gate() -> None:
    certificate = historical_var_es(np.array([1.0, 2.0, 3.0, 10.0]), 0.75)
    assert certificate.value_at_risk == 3.0
    assert certificate.expected_shortfall == 10.0
    assert certificate.exception_count == 1
    decile = historical_var_es(np.arange(1.0, 11.0), 0.9)
    assert decile.value_at_risk == 9.0 and decile.expected_shortfall == 10.0
    assert tail_risk_gate(
        decile,
        limit=12.0,
        coverage=0.99,
        minimum_coverage=0.98,
        tail_model_adequate=True,
    ).decision == "PASS"
    assert tail_risk_gate(
        decile,
        limit=9.0,
        coverage=0.99,
        minimum_coverage=0.98,
        tail_model_adequate=True,
    ).decision == "REJECT"
    signed = historical_var_es(np.array([-2.0, -1.0, 1.0, 2.0]), 0.75)
    assert signed.value_at_risk == 1.0 and signed.expected_shortfall == 2.0


def test_reverse_stress_requires_root_residual_and_monotonicity() -> None:
    smooth = reverse_stress_scale(lambda scale: 2.0 * scale + scale * scale, 15.0)
    assert smooth.status == "PASS"
    assert smooth.scale == pytest.approx(3.0)
    already = reverse_stress_scale(lambda scale: 20.0 + scale, 15.0)
    assert already.scale == 0.0 and already.status == "ALREADY_EXCEEDS"
    discontinuous = reverse_stress_scale(
        lambda scale: 20.0
        if 0.2 <= scale <= 0.3 or scale >= 0.8
        else 0.0,
        15.0,
    )
    assert discontinuous.status == "NON_MONOTONE"
    step = reverse_stress_scale(lambda scale: 0.0 if scale < 0.5 else 20.0, 15.0)
    assert step.status == "BOUND_ONLY"
    assert abs(step.residual) > 1.0


def test_scenario_pnl_full_gamma_and_overflow_reject() -> None:
    gamma = np.array([[1.0, 0.5], [0.5, 2.0]])
    exposures = np.array([2.0, -1.0])
    shocks = np.array([0.1, 0.2])
    expected = float(exposures @ shocks) + 0.5 * float(shocks @ gamma @ shocks)
    assert scenario_pnl(exposures, shocks, gamma) == pytest.approx(expected)
    with pytest.raises(FloatingPointError):
        scenario_pnl(np.array([1.0e308]), np.array([1.0e308]))
    assert limit_utilization(5.0, 10.0) == 0.5
    assert liquidity_adjusted_capital(12.0, 2.0) == 14.0


def test_purging_uses_half_open_exact_union_and_each_embargo() -> None:
    starts = np.array(
        ["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-11", "2026-01-12"],
        dtype="datetime64[D]",
    )
    ends = starts + np.timedelta64(1, "D")
    train = purged_train_indices(
        starts,
        ends,
        np.array([0, 3]),
        candidate_train_indices=np.array([1, 2, 4]),
        embargo=np.timedelta64(2, "D"),
        walk_forward=False,
    )
    assert train.tolist() == [2]
    touching_starts = np.array(
        ["2026-02-01", "2026-02-03"], dtype="datetime64[D]"
    )
    touching_ends = np.array(
        ["2026-02-03", "2026-02-04"], dtype="datetime64[D]"
    )
    touching = purged_train_indices(
        touching_starts,
        touching_ends,
        np.array([1]),
        candidate_train_indices=np.array([0]),
        walk_forward=False,
    )
    assert touching.tolist() == [0]


def test_ml_boundary_conventions_and_psi_certificate() -> None:
    probabilities = np.array([0.0, 1.0])
    outcomes = np.array([1, 0])
    assert np.isfinite(log_loss(probabilities, outcomes, epsilon=1.0e-6))
    assert brier_score(np.array([0.2, 0.8]), np.array([0, 1])) == pytest.approx(
        0.04
    )
    rng = np.random.default_rng(1)
    psi = population_stability_index(
        rng.normal(size=1000), rng.normal(0.5, 1.0, 1000), floor=1.0e-6
    )
    assert psi.value > 0.0
    assert psi.smoothing_policy == "floor_then_renormalize"
    assert psi.floor == 1.0e-6


def test_per_label_value_is_not_deployment_pnl_and_nonoverlap_is_enforced() -> None:
    probabilities = np.array([0.7, 0.3, 0.8])
    returns = np.array([0.02, -0.01, 0.03])
    descriptive = net_value(
        probabilities,
        returns,
        threshold=0.5,
        cost_per_unit_turnover=0.001,
    )
    assert descriptive.status == "PER_LABEL_ONLY"
    starts = np.array(
        ["2026-01-01", "2026-01-06", "2026-01-11"], dtype="datetime64[D]"
    )
    ends = starts + np.timedelta64(5, "D")
    tradable = nonoverlapping_net_value(
        probabilities,
        returns,
        starts,
        ends,
        threshold=0.5,
        cost_per_unit_turnover=0.001,
    )
    assert tradable.status == "PASS"
    overlapping = starts.copy()
    overlapping[1] = np.datetime64("2026-01-03")
    with pytest.raises(ValueError, match="non-overlapping"):
        nonoverlapping_net_value(
            probabilities,
            returns,
            overlapping,
            ends,
            threshold=0.5,
            cost_per_unit_turnover=0.001,
        )


def test_event_fold_persists_sequence_state_and_hash_chain_across_restart() -> None:
    first = fold_events(
        [
            Event("a", 1, "fill", "AAPL", "USD", quantity=2.0, price=10.0),
            Event("b", 2, "fee", "", "USD", amount=1.0),
        ]
    )
    assert first.positions[("AAPL", "USD")] == 2.0
    assert first.cash["USD"] == -21.0
    assert first.state.event_count == 2
    second = fold_events(
        [Event("c", 3, "fill", "AAPL", "USD", quantity=-1.0, price=11.0)],
        prior_state=first.state,
    )
    assert second.positions[("AAPL", "USD")] == 1.0
    assert second.cash["USD"] == -10.0
    assert second.replay_hash != first.replay_hash
    with pytest.raises(ValueError, match="durable sequence"):
        fold_events(
            [Event("later", 2, "fee", "", "USD", amount=1.0)],
            prior_state=first.state,
        )
    with pytest.raises(ValueError, match="contiguous"):
        fold_events(
            [Event("gap", 4, "fee", "", "USD", amount=1.0)],
            prior_state=first.state,
        )
    duplicate = fold_events(
        [Event("a", 1, "fill", "AAPL", "USD", quantity=2.0, price=10.0)],
        prior_state=first.state,
    )
    assert duplicate.applied_event_ids == ()
    assert canonical_event_payload(
        Event("a", 1, "fill", "AAPL", "USD", quantity=2.0, price=10.0)
    ).startswith("{")


def test_event_schema_and_overflow_fail_before_commit() -> None:
    with pytest.raises(ValueError, match="schema"):
        Event("bad", 1, "fee", "", "USD", amount=1.0, schema_version=2)
    with pytest.raises(ValueError, match="fee"):
        Event("bad", 1, "fee", "AAPL", "USD", quantity=1.0, amount=1.0)
    prior = JournalState()
    with pytest.raises(FloatingPointError):
        fold_events(
            [
                Event(
                    "overflow",
                    1,
                    "fill",
                    "AAPL",
                    "USD",
                    quantity=1.0e308,
                    price=1.0e308,
                )
            ],
            prior_state=prior,
        )
    assert prior.event_count == 0 and prior.cash == {}


def approval_fixture():
    return approval_commitment(
        proposal_hash=digest("proposal"),
        portfolio_snapshot_hash=digest("portfolio"),
        risk_result_hash=digest("risk"),
        policy_version="approval-policy-v1",
        approver_ids=("risk-owner", "trading-owner"),
        issued_at="2026-01-01T00:00:00Z",
        expires_at="2026-01-02T00:00:00Z",
        tool_version="approval-tool-v1",
        authorization_receipt_hash=digest("authorization-receipt"),
    )


def test_approval_gate_enforces_expiry_dual_control_hash_and_coverage() -> None:
    approval = approval_fixture()
    common = {
        "proposed_risk": 5.0,
        "limit": 10.0,
        "approval": approval,
        "current_commitment": approval.commitment,
        "coverage": 0.99,
        "minimum_coverage": 0.98,
        "evidence_complete": True,
    }
    assert approval_gate(
        **common, decision_time="2026-01-01T12:00:00Z"
    ) == "PASS"
    assert approval_gate(
        **common, decision_time="2026-01-02T00:00:00Z"
    ) == "BLOCK_EXPIRED_APPROVAL"
    stale = dict(common)
    stale["current_commitment"] = digest("other")
    assert approval_gate(
        **stale, decision_time="2026-01-01T12:00:00Z"
    ) == "BLOCK_STALE_APPROVAL"
    invalid = dict(common)
    invalid["coverage"] = -1.0
    assert approval_gate(
        **invalid, decision_time="2026-01-01T12:00:00Z"
    ) == "INVALID_INPUT"


def manifest_fixture() -> tuple[SubmissionManifest, dict[str, bytes]]:
    artifacts = {
        "data": b"data",
        "config": b"config",
        "environment": b"environment",
        "result": b"result",
    }
    manifest = SubmissionManifest(
        candidate_id="candidate-1",
        schema_version="1.0",
        code_commit="a" * 40,
        data_hash=digest(artifacts["data"]),
        config_hash=digest(artifacts["config"]),
        environment_hash=digest(artifacts["environment"]),
        result_hash=digest(artifacts["result"]),
    )
    return manifest, artifacts


def test_jcs_golden_numbers_manifest_artifact_bytes_and_four_scores() -> None:
    assert canonical_bytes(
        {"a": 1.0, "b": -0.0, "c": 1.0e-7, "d": 1.0e-6, "e": 1.0e21}
    ) == b'{"a":1,"b":0,"c":1e-7,"d":0.000001,"e":1e+21}'
    manifest, artifacts = manifest_fixture()
    expected = {
        name: str(value)
        for name, value in vars(manifest).items()
    }
    assert validate_manifest(manifest, expected, artifact_bytes=artifacts) == ()
    mutated = dict(artifacts)
    mutated["result"] = b"different"
    assert any(
        "result_hash" in issue
        for issue in validate_manifest(manifest, expected, artifact_bytes=mutated)
    )
    with pytest.raises(ValueError, match="schema"):
        replace(manifest, schema_version="999")
    decision = score_takehome(
        reproducibility=5.0,
        research_quality=5.0,
        implementation=5.0,
        communication=5.0,
        leakage=False,
        fabricated_result=False,
        core_claim_reproduced=True,
        decision_supported=True,
    )
    assert decision.component_count == 4 and decision.status == "ACCEPT"
    assert len(canonical_hash({"manifest": vars(manifest)})) == 64


def evidence_items(card: ScoreCard) -> tuple[EvidenceItem, ...]:
    return tuple(
        EvidenceItem(
            evidence_id=f"e-{index}",
            competency=name,
            observation=f"observed {name}",
            quality=1.0,
            source_hash=digest(name),
            observed_at=f"2026-01-01T0{index}:00:00Z",
            scorer_id="scorer-a",
        )
        for index, name in enumerate(card.as_dict())
    )


def test_weighted_completeness_intersection_and_extra_evidence() -> None:
    evidence = (
        EvidenceItem(
            "e1",
            "critical",
            "observed",
            1.0,
            digest("critical"),
            "2026-01-01T00:00:00Z",
            "s1",
        ),
        EvidenceItem(
            "e2",
            "extra",
            "extra",
            1.0,
            digest("extra"),
            "2026-01-01T00:00:00Z",
            "s1",
        ),
    )
    certificate = evidence_completeness(
        {"critical": 0.8, "minor": 0.8},
        {"critical": 0.9, "minor": 0.1},
        evidence,
    )
    assert certificate.weighted_completeness == pytest.approx(0.9)
    assert certificate.missing_evidence == ("minor",)
    assert certificate.extra_evidence_categories == ("extra",)


def test_hiring_decision_integrity_dissent_taxonomy_and_complete_hash() -> None:
    card = ScoreCard(4.5, 4.4, 4.3, 4.2, 4.1)
    weights = {name: 1.0 for name in card.as_dict()}
    required = {name: 0.8 for name in card.as_dict()}
    evidence = evidence_items(card)
    common = {
        "card": card,
        "weights": weights,
        "hard_floors": {"correctness": 2.0, "judgment": 2.0},
        "required_evidence_quality": required,
        "required_evidence_weights": weights,
        "evidence": evidence,
        "scorer_ids": ("scorer-a", "scorer-b"),
        "decision_time": "2026-01-02T00:00:00Z",
        "policy_version": "hiring-policy-v1",
    }
    accepted = hiring_decision(**common)
    assert accepted.state == "ACCEPT"
    dissent = hiring_decision(**common, dissent=("material systems concern",))
    assert dissent.state == "PANEL_REVIEW"
    integrity = hiring_decision(
        **common, integrity_failures=("candidate identity mismatch",)
    )
    assert integrity.state == "REJECT"
    changed = dict(common)
    changed["weights"] = {
        "correctness": 5.0,
        "reasoning": 1.0,
        "implementation": 1.0,
        "judgment": 1.0,
        "communication": 1.0,
    }
    assert hiring_decision(**changed).decision_hash != accepted.decision_hash
    with pytest.raises(ValueError, match="panel_lower"):
        hiring_decision(
            **common,
            pass_score=3.5,
            panel_lower=3.5,
            panel_upper=3.75,
            strong_hire_score=4.0,
        )


def test_score_statistics_helpers_are_directly_exercised() -> None:
    mean, se = mean_and_se([3.0, 4.0, 5.0])
    assert mean == 4.0 and se > 0.0
    assert standard_error([3.0, 4.0, 5.0]) == pytest.approx(se)


def test_interviewer_calibration_rejects_duplicates_conflicting_and_no_reference() -> None:
    records = [
        InterviewScore("i1", "c1", 4.0, 3.5, "anchor-v1"),
        InterviewScore("i2", "c1", 3.0, 3.5, "anchor-v1"),
        InterviewScore("i1", "c2", 3.0),
        InterviewScore("i2", "c2", 4.0),
    ]
    diagnostics = interviewer_diagnostics(records)
    assert {item.interviewer_id for item in diagnostics} == {"i1", "i2"}
    assert {item.reference for item in diagnostics} == {
        "independent_anchor@anchor-v1",
        "leave_one_out_panel",
    }
    with pytest.raises(ValueError, match="unique"):
        interviewer_diagnostics(records + [records[0]])
    with pytest.raises(ValueError, match="contradictory"):
        interviewer_diagnostics(
            [
                InterviewScore("i1", "c1", 4.0, 3.5, "v1"),
                InterviewScore("i2", "c1", 4.0, 4.0, "v2"),
            ]
        )
    with pytest.raises(ValueError, match="neither anchor nor peer"):
        interviewer_diagnostics([InterviewScore("i1", "solo", 4.0)])
    assert pairwise_agreement([3.0, 4.0], [3.2, 4.4], 0.5) == 1.0


def test_adaptive_interview_contract_unique_targets_and_information_rate() -> None:
    questions = [
        Question("Q1", "math", 3, 10),
        Question("Q2", "code", 4, 10),
        Question("Q3", "code", 2, 5),
    ]
    chosen = choose_next_question(
        questions,
        set(),
        {"math": 3.5, "code": 3.5},
        [Evidence("math", 4.0, 1.0)],
        15,
    )
    assert chosen is not None and chosen.question_id == "Q3"
    assert competency_need(
        {"math": 3.0}, [Evidence("math", 2.0, 1.0)]
    ) == {"math": 1.0}
    with pytest.raises(ValueError, match="unique"):
        choose_next_question(
            [questions[0], questions[0]],
            set(),
            {"math": 3.5},
            [],
            15,
        )
    assert choose_next_question(questions, set(), {"math": 3.5, "code": 3.5}, [], 0) is None


def test_preparation_hours_are_capped_by_actual_need() -> None:
    topics = [
        Topic("math", 2.0, 4.0, 10.0, 2.0),
        Topic("code", 3.0, 4.0, 1.0, 5.0),
    ]
    allocation = allocate_hours(topics, 100.0)
    assert allocation["math"] == pytest.approx(4.0)
    assert allocation["code"] == pytest.approx(5.0)
    partial = allocate_hours(topics, 3.0)
    assert sum(partial.values()) == pytest.approx(3.0)
    assert partial["math"] <= 4.0 and partial["code"] <= 5.0
    plan = weekly_plan(allocation, 4)
    assert sum(sum(week.values()) for week in plan) == pytest.approx(9.0)
    with pytest.raises(ValueError, match="positive"):
        Topic("bad", 1.0, 2.0, 1.0, 0.0)


def test_cpp20_kernels_require_compiler_execute_negative_self_tests_and_parity(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++") or shutil.which("clang++")
    if compiler is None:
        pytest.fail("C++ compiler is required for the CH14 release suite")
    outputs: dict[str, str] = {}
    for source in sorted((ROOT / "_book/cpp/ch14").glob("*.cpp")):
        binary = tmp_path / source.stem
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                str(source),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        outputs[source.name] = subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True
        ).stdout.strip()
    assert outputs["online_stats.cpp"] == "4 2.500000 1.666667"
    assert outputs["risk_gate.cpp"] == "PASS"
    assert outputs["top_k.cpp"].splitlines() == ["B:4", "C:3"]

    replay_binary = tmp_path / "replay_hash"
    canonical = '{"schema_version":"1.0","value":17}'
    cpp_digest = subprocess.run(
        [str(replay_binary), canonical],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert cpp_digest == digest(canonical)
    assert subprocess.run(
        [str(replay_binary), ""],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == digest("")
