from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import minimize

from quant_interviews.ch09.attribution import (
    brinson,
    execution_cost,
    geometric_link,
    reconcile,
)
from quant_interviews.ch09.black_litterman import (
    ViewLineage,
    implied_equilibrium_returns,
    posterior,
)
from quant_interviews.ch09.capacity import (
    capacity,
    capacity_coefficient,
    capacity_diagnostics,
    net_alpha,
)
from quant_interviews.ch09.covariance import (
    participation_ratio_effective_rank,
    perturbation_diagnostics,
    precision_portfolio,
    shrink_covariance,
)
from quant_interviews.ch09.drawdown import (
    conditional_drawdown_at_risk,
    drawdown_stats,
    moving_block_paths,
    wealth_from_returns,
)
from quant_interviews.ch09.factor_neutral import (
    covariance_aware_neutral,
    constrained_neutral_portfolio,
    euclidean_residualize,
    exposure_residual,
)
from quant_interviews.ch09.kelly import (
    multivariate_kelly,
    risk_adjusted_certainty_equivalent,
    scalar_kelly,
    simulate_wealth,
)
from quant_interviews.ch09.integrated_case import (
    ReleasePolicy,
    chapter_decision_packet,
    default_inputs,
    evaluate_release_gates,
)
from quant_interviews.ch09.performance import (
    directional_effective_breadth,
    effective_breadth,
    information_ratio,
    sharpe_ratio,
    spectral_effective_rank,
    transfer_coefficient,
)
from quant_interviews.ch09.portfolio import (
    gmv_weights,
    portfolio_certificate,
    tangency_direction,
    tangency_weights,
    target_return_weights,
)
from quant_interviews.ch09.rebalancing import (
    captured_alpha,
    event_trigger,
    event_trigger_decision,
    policy_value,
)
from quant_interviews.ch09.risk_parity import (
    risk_contributions,
    solve_risk_budget,
)
from quant_interviews.ch09.robust import (
    robust_mean_variance,
    worst_case_mean,
)
from quant_interviews.ch09.stress import (
    coverage,
    liquidation_cost,
    risk_coverage_profile,
    reverse_stress,
    stress_pnl,
)
from quant_interviews.ch09.tail_risk import (
    cvar_objective,
    es_contributions,
    historical_var_es,
    normal_var_es,
    optimize_cvar,
)
from quant_interviews.ch09.turnover import (
    hysteresis_action,
    hysteresis_decision,
    realized_cost,
    soft_threshold,
)
from quant_interviews.ch09.var_backtest import (
    VaRForecastRecord,
    bernoulli_var_counterexample,
    exceedance_summary,
    horizon_variance,
    iid_normal_horizon_var,
    point_in_time_exceedance_summary,
)

COVARIANCE = np.array([[0.04, 0.006], [0.006, 0.09]])
MEAN = np.array([0.07, 0.11])
ABS_TOL = 1e-10
REL_TOL = 1e-10


def _approx(
    expected: object,
    *,
    abs: float = ABS_TOL,
    rel: float = REL_TOL,
) -> object:
    """Chapter-wide explicit numerical-comparison policy."""
    return pytest.approx(expected, abs=abs, rel=rel)


def _view_lineage(view_id: str) -> ViewLineage:
    return ViewLineage(
        view_id=view_id,
        author="researcher@example.test",
        valid_time="2026-07-29T00:00:00Z",
        knowledge_time="2026-07-30T08:00:00Z",
        horizon="1M",
        units="annual_return_fraction",
        source_sha256="a" * 64,
        approver="risk@example.test",
    )


def test_gmv_and_target_match_numerical_optimizer() -> None:
    gmv = gmv_weights(COVARIANCE)
    target = target_return_weights(MEAN, COVARIANCE, 0.09)
    result = minimize(
        lambda weights: float(weights @ COVARIANCE @ weights),
        np.array([0.5, 0.5]),
        constraints=[
            {"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            {"type": "eq", "fun": lambda weights: weights @ MEAN - 0.09},
        ],
        method="SLSQP",
        options={"ftol": 1e-14},
    )
    assert result.success
    assert gmv.sum() == _approx(1.0)
    assert target.weights.sum() == _approx(1.0)
    assert target.expected_return == _approx(0.09)
    assert target.weights == _approx(result.x, abs=1e-8)


def test_portfolio_gates_and_tangency_direction() -> None:
    assert tangency_weights(MEAN, COVARIANCE, 0.02).sum() == _approx(1.0)
    assert tangency_direction(MEAN, COVARIANCE, 0.02).shape == (2,)
    with pytest.raises(ValueError):
        gmv_weights(np.array([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        gmv_weights(np.array([[1.0, 1.0], [1.0, 1.0]]))
    with pytest.raises(ValueError):
        target_return_weights(
            np.array([0.1, 0.1]),
            np.eye(2),
            0.1,
        )


def test_covariance_normalization_and_shrinkage_contract() -> None:
    sample = np.array([[1.0, 0.99], [0.99, 1.0]])
    shrunk = shrink_covariance(sample, 0.5)
    assert participation_ratio_effective_rank(shrunk) > (
        participation_ratio_effective_rank(sample)
    )
    assert np.linalg.cond(shrunk) < np.linalg.cond(sample)
    net = precision_portfolio(np.array([1.0, 0.5]), shrunk, "net")
    gross = precision_portfolio(np.array([1.0, -0.5]), shrunk, "gross")
    assert net.sum() == _approx(1.0)
    assert np.abs(gross).sum() == _approx(1.0)
    with pytest.raises(ValueError):
        shrink_covariance(sample, 0.5, np.array([[1.0, 2.0], [0.0, 1.0]]))


def test_low_eigenvalue_perturbation_is_material() -> None:
    covariance = np.diag([0.01, 1.0])
    signal = np.array([1.0, 1.0])
    low_changed = np.diag([0.02, 1.0])
    high_changed = np.diag([0.01, 1.01])
    low = perturbation_diagnostics(signal, covariance, low_changed)
    high = perturbation_diagnostics(signal, covariance, high_changed)
    assert low.weight_change_l2 > high.weight_change_l2


def test_black_litterman_limits_and_correlated_views() -> None:
    prior = implied_equilibrium_returns(
        COVARIANCE,
        np.array([0.6, 0.4]),
        2.5,
    )
    views = np.array([[1.0, -1.0], [1.0, 0.0]])
    targets = np.array([0.05, 0.08])
    low_confidence = posterior(
        COVARIANCE,
        prior,
        0.05,
        views,
        targets,
        np.eye(2) * 1000.0,
        (_view_lineage("relative"), _view_lineage("asset-one")),
        decision_time="2026-07-30T10:00:00Z",
    )
    high_confidence = posterior(
        COVARIANCE,
        prior,
        0.05,
        views,
        targets,
        np.array([[1e-8, 2e-9], [2e-9, 1e-8]]),
        (_view_lineage("relative"), _view_lineage("asset-one")),
        decision_time="2026-07-30T10:00:00Z",
    )
    assert np.linalg.norm(low_confidence.posterior_mean - prior) < 1e-4
    assert np.linalg.norm(views @ high_confidence.posterior_mean - targets) < 1e-5
    assert np.linalg.eigvalsh(
        high_confidence.posterior_mean_covariance
    ).min() >= -1e-12
    assert high_confidence.prior_view_surprise == _approx(
        targets - views @ prior,
        abs=ABS_TOL,
        rel=REL_TOL,
    )
    assert high_confidence.posterior_view_residual == _approx(
        targets - views @ high_confidence.posterior_mean,
        abs=ABS_TOL,
        rel=REL_TOL,
    )
    assert high_confidence.surprise_degrees_of_freedom == 2


def test_risk_budget_arbitrary_budgets_reconcile() -> None:
    result = solve_risk_budget(COVARIANCE, np.array([0.7, 0.3]))
    contributions = risk_contributions(result.weights, COVARIANCE)
    assert result.status == "PASS"
    assert contributions.sum() == _approx(
        np.sqrt(result.weights @ COVARIANCE @ result.weights)
    )
    assert result.realized_budgets == _approx([0.7, 0.3], abs=1e-7)
    with pytest.raises(ValueError):
        solve_risk_budget(COVARIANCE, np.array([1.0, 0.0]))


def test_factor_neutral_budget_and_rank_certificate() -> None:
    exposures = np.array([[1.0], [0.5], [-0.2]])
    constraints = np.column_stack((exposures, np.ones(3)))
    result = constrained_neutral_portfolio(
        np.array([0.4, 0.2, 0.1]),
        np.array(
            [
                [1.0, 0.2, 0.0],
                [0.2, 2.0, 0.1],
                [0.0, 0.1, 1.5],
            ]
        ),
        constraints,
        np.array([0.0, 1.0]),
    )
    assert result.constraint_residual == _approx([0.0, 0.0], abs=1e-10)
    assert result.effective_rank == 2
    residual = euclidean_residualize(
        np.array([0.4, 0.2, 0.1]),
        exposures,
    )
    assert exposures.T @ residual == _approx([0.0], abs=1e-10)


def test_turnover_cost_and_hysteresis_reversal() -> None:
    trade = soft_threshold(
        np.array([0.2, 1.0]),
        np.array([0.3, 0.3]),
        np.ones(2),
    )
    assert trade[0] == 0.0 and trade[1] > 0.0
    assert hysteresis_action(-0.8, 1, 0.5, 0.2) == -1
    assert hysteresis_action(0.8, -1, 0.5, 0.2) == 1
    decision = hysteresis_decision(
        -0.8,
        1,
        0.5,
        0.2,
        "2026-07-30T10:00:00Z",
    )
    assert decision.reason_code == "REVERSE"
    with pytest.raises(ValueError):
        realized_cost(
            np.ones(2),
            np.ones(2),
            np.array([[1.0, 2.0], [2.0, 1.0]]),
        )


def test_empirical_es_handles_atoms_at_var() -> None:
    losses = np.concatenate((np.zeros(96), np.full(4, 10.0)))
    value_at_risk, expected_shortfall = historical_var_es(losses, 0.95)
    assert value_at_risk == 0.0
    assert expected_shortfall == _approx(8.0)


def test_es_contributions_reconcile_and_cvar_optimizes() -> None:
    scenarios = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 4.0],
            [2.0, 2.0],
        ]
    )
    weights = np.array([0.5, 0.5])
    contributions = es_contributions(scenarios, weights, 0.8)
    _, total_es = historical_var_es(scenarios @ weights, 0.8)
    assert contributions.sum() == _approx(total_es)
    optimized = optimize_cvar(scenarios, 0.8)
    assert optimized.status == "PASS"
    assert optimized.weights.sum() == _approx(1.0)


def test_normal_es_validation() -> None:
    var, expected_shortfall = normal_var_es(0.0, 1.0, 0.95)
    assert expected_shortfall > var
    with pytest.raises(ValueError):
        normal_var_es(0.0, -1.0, 0.95)


def test_var_counterexample_and_horizon_mean() -> None:
    counterexample = bernoulli_var_counterexample(p_loss=0.05)
    assert counterexample["combined_var"] == 1.0
    with pytest.raises(ValueError):
        bernoulli_var_counterexample(p_loss=0.001)
    assert horizon_variance(np.array([1.0, 0.2, 0.1]), 3) > 3.0
    horizon_var = iid_normal_horizon_var(0.01, 0.02, 0.95, 4)
    assert horizon_var == _approx(
        4 * 0.01 + 2 * 0.02 * 1.6448536269514722
    )


def test_exceedance_backtest_returns_coverage_and_independence() -> None:
    result = exceedance_summary(
        np.array([0.0, 2.0, 0.0, 3.0, 0.0, 0.0]),
        np.ones(6),
        0.95,
    )
    assert result["count"] == 2.0
    assert 0.0 <= result["kupiec_pvalue"] <= 1.0
    assert result["n01"] + result["n11"] == 2.0


def test_stress_metric_coverage_and_liquidation() -> None:
    result = stress_pnl(
        np.array([1.0, 2.0]),
        np.eye(2),
        np.array([-0.1, -0.2]),
        lambda shock: float(np.exp(shock[0]) - 1.0 + 2.0 * shock[1]),
    )
    assert result.full_pnl is not None
    shock = reverse_stress(np.array([1.0, 0.0]), np.eye(2), 0.5)
    assert shock.shock[0] <= -0.49
    assert shock.status == "PASS"
    assert coverage(np.array([100.0, 50.0]), np.array([True, False])) == _approx(2 / 3)
    assert liquidation_cost(
        np.array([-100.0, 100.0]),
        np.array([0.1, 0.1]),
        np.array([1000.0, 1000.0]),
    ) > 0.0


def test_derivative_risk_coverage_cannot_pass_on_market_value_alone() -> None:
    profile = risk_coverage_profile(
        market_values=np.array([1000.0, 1.0]),
        deltas=np.array([1.0, 100.0]),
        vegas=np.array([0.0, 500.0]),
        stress_losses=np.array([1.0, 1000.0]),
        notionals=np.array([1000.0, 1000.0]),
        covered=np.array([1.0, 0.0]),
        minimum_coverage=0.95,
    )
    assert profile.market_value > 0.99
    assert profile.vega == 0.0
    assert profile.stress_loss < 0.01
    assert profile.status == "REJECT"


def test_drawdown_includes_initial_wealth_and_last_peak() -> None:
    first_loss = drawdown_stats(np.array([-0.1, 0.05]))
    assert first_loss.maximum_drawdown == _approx(0.1)
    assert first_loss.peak_index == 0
    repeated_peak = drawdown_stats(np.array([0.1, 0.0, -0.1, 1 / 9]))
    assert repeated_peak.peak_index == 2
    assert repeated_peak.recovery_index == 4
    assert repeated_peak.recovery_time == 2
    assert repeated_peak.maximum_drawdown_duration == 1


def test_drawdown_bootstrap_validation() -> None:
    returns = np.array([0.1, -0.2, 0.05, 0.2])
    assert conditional_drawdown_at_risk(returns, 0.5) >= 0.0
    assert moving_block_paths(returns, 2, 3, 1).shape == (3, 4)
    with pytest.raises(ValueError):
        moving_block_paths(returns, 0, 3)
    with pytest.raises(ValueError):
        drawdown_stats(np.array([-1.0]))


def test_cdar_uses_exact_fractional_tail_mass_at_quantile_atom() -> None:
    # Drawdowns after the initial wealth point are [0.0, 0.2, 0.1, 0.1].
    returns = np.array([0.0, -0.2, 0.125, 0.0])
    assert conditional_drawdown_at_risk(returns, 0.5) == _approx(0.15)


def test_kelly_uncertainty_reduces_certainty_equivalent() -> None:
    assert scalar_kelly(0.1, 0.2, 0.1) < scalar_kelly(0.1, 0.2)
    base = multivariate_kelly(MEAN, COVARIANCE)
    half = multivariate_kelly(MEAN, COVARIANCE, fraction=0.5)
    assert half == _approx(0.5 * base)
    ce_base = risk_adjusted_certainty_equivalent(MEAN, COVARIANCE)
    ce_uncertain = risk_adjusted_certainty_equivalent(
        MEAN,
        COVARIANCE + np.eye(2) * 0.02,
    )
    assert ce_uncertain <= ce_base
    with pytest.raises(ValueError):
        multivariate_kelly(MEAN, COVARIANCE, fraction=1.5)


def test_wealth_guard() -> None:
    with pytest.raises(ValueError):
        simulate_wealth(np.array([2.0]), np.array([[-0.6]]))


def test_capacity_handles_signed_trades_and_negative_aum() -> None:
    long = capacity_coefficient(
        np.array([0.1, 0.2]),
        np.array([0.2, 0.3]),
        np.array([1e6, 2e6]),
        np.array([0.5, 0.6]),
    )
    short = capacity_coefficient(
        np.array([-0.1, -0.2]),
        np.array([0.2, 0.3]),
        np.array([1e6, 2e6]),
        np.array([0.5, 0.6]),
    )
    assert short == _approx(long)
    result = capacity(0.12, 0.02, long)
    assert result.break_even_aum > result.profit_optimal_aum
    assert net_alpha(result.break_even_aum, 0.12, 0.02, long) == _approx(0.0)
    with pytest.raises(ValueError):
        net_alpha(-1.0, 0.12, 0.02, long)


def test_attribution_execution_and_fail_closed_reconciliation() -> None:
    result = brinson(
        np.array([0.6, 0.4]),
        np.array([0.5, 0.5]),
        np.array([0.1, 0.02]),
        np.array([0.06, 0.03]),
    )
    active = (
        np.array([0.6, 0.4]) @ np.array([0.1, 0.02])
        - np.array([0.5, 0.5]) @ np.array([0.06, 0.03])
    )
    assert (
        result.allocation + result.selection + result.interaction
    ).sum() == _approx(active)
    assert execution_cost(
        np.array([10.0, -10.0]),
        np.array([101.0, 99.0]),
        np.array([100.0, 100.0]),
    ) == _approx(20.0)
    reconciliation = reconcile(
        0.0,
        {"gain": 100.0, "loss": -99.0},
        nav=1000.0,
        absolute_tolerance=0.1,
        nav_tolerance=0.0001,
        gross_tolerance=0.0001,
    )
    assert reconciliation.status == "REJECT"
    assert reconciliation.residual_to_gross_attribution == _approx(1 / 199)


def test_geometric_link() -> None:
    returns = np.array([0.1, 0.05])
    contributions = np.array([[0.06, 0.04], [0.02, 0.03]])
    result = geometric_link(returns, contributions)
    assert result.status == "PASS"
    assert result.linked_contributions.sum() == _approx(
        np.prod(1.0 + returns) - 1.0
    )


def test_performance_breadth_and_negative_transfer_coefficient() -> None:
    returns = np.array([0.01, 0.02, -0.01, 0.03, 0.0, 0.02])
    assert np.isfinite(sharpe_ratio(returns, hac_lags=1))
    assert information_ratio(returns, hac_lags=1) == _approx(
        sharpe_ratio(returns, hac_lags=1)
    )
    rho = 0.5
    correlation = np.full((3, 3), rho)
    np.fill_diagonal(correlation, 1.0)
    equal = directional_effective_breadth(np.ones(3), correlation)
    spectral = spectral_effective_rank(correlation)
    assert equal == _approx(3 / (1 + 2 * rho))
    assert spectral == _approx(3 / (1 + 2 * rho**2))
    assert equal != _approx(spectral)
    assert transfer_coefficient(
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.eye(2),
    ) == _approx(-1.0)


def test_rebalancing_policy_domain_and_state() -> None:
    intervals = np.array([1.0, 5.0, 10.0])
    assert np.all(np.diff(captured_alpha(intervals, 0.1, 0.2)) < 0.0)
    assert np.isfinite(
        policy_value(intervals, 0.1, 0.2, 0.01, 0.001, 0.0001)
    ).all()
    with pytest.raises(ValueError):
        policy_value(0.0, 0.1, 0.2, 0.01, 0.001, 0.0001)
    decision = event_trigger_decision(
        0.5,
        0.2,
        0.1,
        0.0,
        (1.0, 1.0, 1.0, 1.0),
        0.4,
        decision_time="2026-07-30T10:00:00Z",
    )
    assert decision.triggered and decision.reason_code == "SCORE_BREACH"
    stale = event_trigger_decision(
        0.0,
        0.0,
        0.0,
        1.0,
        (1.0, 1.0, 1.0, 1.0),
        1.0,
        decision_time="2026-07-30T10:00:00Z",
        maximum_staleness_breached=True,
    )
    assert stale.triggered and stale.reason_code == "MAX_STALENESS"


def test_robust_support_function_certificate() -> None:
    uncertainty = np.eye(2) * 0.02
    result = robust_mean_variance(
        MEAN,
        COVARIANCE,
        uncertainty,
        0.5,
        3.0,
    )
    assert result.weights.sum() == _approx(1.0)
    assert result.status == "PASS"
    assert result.support_function_residual <= 1e-10
    assert np.linalg.norm(result.worst_case_latent_direction) == _approx(0.5)
    assert result.worst_case_mean_shock == _approx(
        uncertainty @ result.worst_case_latent_direction
    )
    assert worst_case_mean(
        MEAN,
        uncertainty,
        0.5,
        result.weights,
    ) == _approx(result.worst_case_expected_return)


def test_integrated_case_rejects_incomplete_pnl_evidence() -> None:
    packet = chapter_decision_packet()
    assert sum(packet.nominal_weights) == _approx(1.0)
    assert sum(packet.robust_weights) == _approx(1.0)
    assert packet.robust_support_residual <= 1e-10
    assert packet.one_way_turnover <= 0.20
    assert packet.pnl_residual == _approx(225.0)
    assert packet.final_decision == "REJECT"


def test_kelly_uses_predictive_second_moment_and_exact_ce_scale() -> None:
    expected_weight = 0.1 / (0.2 + 0.1**2)
    assert scalar_kelly(0.1, 0.2) == _approx(
        expected_weight,
        abs=ABS_TOL,
        rel=REL_TOL,
    )
    expected_ce = 0.5 * 0.1**2 / (0.2 + 0.1**2)
    assert risk_adjusted_certainty_equivalent(
        np.array([0.1]),
        np.array([[0.2]]),
    ) == _approx(expected_ce, abs=ABS_TOL, rel=REL_TOL)
    with pytest.raises(ValueError, match="symmetric"):
        risk_adjusted_certainty_equivalent(
            np.array([0.1, 0.2]),
            np.array([[1.0, 2.0], [0.0, 1.0]]),
        )


def test_black_litterman_lineage_is_fail_closed() -> None:
    prior = np.array([0.02, 0.03])
    with pytest.raises(ValueError, match="lineage"):
        posterior(
            np.eye(2),
            prior,
            0.1,
            np.array([[1.0, -1.0]]),
            np.array([0.01]),
            np.array([[0.02]]),
            (),
            decision_time="2026-07-30T10:00:00Z",
        )
    late = replace(
        _view_lineage("late"),
        knowledge_time="2026-07-30T11:00:00Z",
    )
    with pytest.raises(ValueError, match="exceeds"):
        posterior(
            np.eye(2),
            prior,
            0.1,
            np.array([[1.0, -1.0]]),
            np.array([0.01]),
            np.array([[0.02]]),
            (late,),
            decision_time="2026-07-30T10:00:00Z",
        )


def test_tail_policy_is_quantile_consistent_and_permutation_invariant() -> None:
    value_at_risk, _ = historical_var_es(np.arange(1.0, 101.0), 0.95)
    assert value_at_risk == _approx(95.0, abs=0.0, rel=0.0)
    scenarios = np.array([[10.0, 0.0], [0.0, 10.0]])
    weights = np.array([0.5, 0.5])
    first = es_contributions(scenarios, weights, 0.5)
    second = es_contributions(scenarios[::-1], weights, 0.5)
    assert first == _approx(second, abs=ABS_TOL, rel=REL_TOL)
    assert first == _approx([2.5, 2.5], abs=ABS_TOL, rel=REL_TOL)
    assert cvar_objective(np.array([0.0, 2.0]), 1.0, 0.5) == _approx(
        2.0,
        abs=ABS_TOL,
        rel=REL_TOL,
    )


def test_stress_coverage_nonfinite_and_reverse_certificate() -> None:
    assert coverage(
        np.array([100.0, 50.0]),
        np.array([0.1, 0.0]),
    ) == _approx(1.0 / 15.0, abs=ABS_TOL, rel=REL_TOL)
    with pytest.raises(ValueError):
        coverage(np.array([100.0]), np.array([np.nan]))
    rejected = stress_pnl(
        np.array([1.0]),
        np.array([[0.0]]),
        np.array([0.1]),
        lambda _: float("nan"),
    )
    assert rejected.status == "REJECT"
    assert rejected.reason_code == "FULL_REPRICE_NONFINITE"
    with pytest.raises(ValueError, match="threshold"):
        reverse_stress(np.array([1.0]), np.eye(1), float("nan"))


def test_horizon_covariance_and_pit_backtest_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="PSD"):
        horizon_variance(np.array([1.0, -2.0]), 2)
    records = tuple(
        VaRForecastRecord(
            forecast_id=f"f-{index}",
            forecast_origin=f"2026-07-0{index + 1}T08:00:00Z",
            target_start=f"2026-07-0{index + 1}T09:00:00Z",
            target_end=f"2026-07-0{index + 1}T16:00:00Z",
            knowledge_time=f"2026-07-0{index + 1}T07:00:00Z",
            model_version="var-v1",
            position_snapshot_sha256="b" * 64,
            forecast_loss=1.0,
            realized_loss=2.0 if index == 1 else 0.0,
            realized_loss_knowledge_time=(
                "2026-08-02T00:00:00Z"
                if index == 3
                else f"2026-07-0{index + 1}T17:00:00Z"
            ),
        )
        for index in range(4)
    )
    certificate = point_in_time_exceedance_summary(
        records,
        evaluation_time="2026-08-01T00:00:00Z",
    )
    assert certificate.used_count == 3
    assert certificate.excluded_count == 1
    assert certificate.alignment_status == "PASS"


def test_drawdown_observation_convention_and_empty_rejection() -> None:
    returns = np.array([-0.1, 0.05])
    stats = drawdown_stats(returns)
    expected_second_drawdown = 1.0 - 0.9 * 1.05
    expected_ulcer = np.sqrt((0.1**2 + expected_second_drawdown**2) / 2.0)
    assert stats.ulcer_index == _approx(
        expected_ulcer,
        abs=ABS_TOL,
        rel=REL_TOL,
    )
    assert wealth_from_returns(returns).shape == (3,)
    with pytest.raises(ValueError):
        drawdown_stats(np.array([]))


def test_hard_risk_override_and_timestamped_compatibility_alias() -> None:
    decision = event_trigger_decision(
        0.0,
        0.0,
        1.0,
        1.0,
        (1.0, 1.0, 1.0, 1.0),
        0.5,
        decision_time="2026-07-30T10:00:00Z",
        hard_risk_breached=True,
    )
    assert decision.triggered
    assert decision.reason_code == "HARD_RISK_OVERRIDE"
    assert event_trigger(
        0.0,
        0.0,
        1.0,
        1.0,
        (1.0, 1.0, 1.0, 1.0),
        0.5,
        decision_time="2026-07-30T10:00:00Z",
        hard_risk_breached=True,
    )
    with pytest.raises(ValueError):
        event_trigger(
            0.0,
            0.0,
            0.0,
            0.0,
            (1.0, 1.0, 1.0, 1.0),
            0.5,
            decision_time="",
        )


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_public_scalar_contracts_reject_nonfinite(nonfinite: float) -> None:
    with pytest.raises(ValueError):
        net_alpha(1.0, nonfinite, 0.01, 0.02)
    with pytest.raises(ValueError):
        policy_value(1.0, 0.1, 0.2, 0.01, nonfinite, 0.01)
    with pytest.raises(ValueError):
        worst_case_mean(np.array([0.1]), np.eye(1), nonfinite, np.ones(1))


def test_attribution_and_breadth_domains_are_explicit() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        brinson(
            np.array([0.3, 0.3]),
            np.array([0.3, 0.2]),
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
        )
    with pytest.raises(ValueError, match="reconcile"):
        geometric_link(
            np.array([0.1]),
            np.array([[0.01, 0.02]]),
        )
    with pytest.raises(ValueError):
        directional_effective_breadth(np.array([1.0, -1.0]), np.eye(2))


def test_portfolio_robust_and_capacity_certificates() -> None:
    portfolio = portfolio_certificate(MEAN, COVARIANCE, "GMV")
    assert portfolio.status == "PASS"
    assert portfolio.budget_residual <= portfolio.tolerance
    robust = robust_mean_variance(
        MEAN,
        COVARIANCE,
        np.eye(2) * 0.02,
        0.5,
        3.0,
    )
    assert robust.solver_success
    assert robust.stationarity_residual <= 1e-7
    nonunique = robust_mean_variance(
        MEAN,
        COVARIANCE,
        np.zeros((2, 1)),
        0.5,
        3.0,
    )
    assert not nonunique.worst_case_direction_unique
    diagnostics = capacity_diagnostics(
        np.array([0.01, 0.02]),
        1.0e8,
        np.array([2.0e8, 1.0e8]),
        2.0,
        0.1,
        0.5,
    )
    assert diagnostics.maximum_stressed_participation == _approx(
        0.02,
        abs=ABS_TOL,
        rel=REL_TOL,
    )
    assert diagnostics.status == "PASS"


def test_compatibility_symbols_are_directly_covered() -> None:
    alpha = np.array([0.2, -0.1, 0.0])
    exposures = np.array([[1.0], [0.5], [-0.5]])
    covariance = np.eye(3)
    neutral = covariance_aware_neutral(alpha, covariance, exposures)
    assert exposure_residual(neutral, exposures) == _approx(
        [0.0],
        abs=ABS_TOL,
        rel=REL_TOL,
    )
    assert effective_breadth(np.eye(3)) == _approx(
        3.0,
        abs=ABS_TOL,
        rel=REL_TOL,
    )


def test_integrated_hashes_and_every_hard_gate_are_mutation_sensitive() -> None:
    base_inputs = default_inputs()
    base_policy = ReleasePolicy()
    base = chapter_decision_packet(base_inputs, base_policy)
    changed_inputs = replace(
        base_inputs,
        expected_returns=(0.076, *base_inputs.expected_returns[1:]),
    )
    changed_policy = replace(base_policy, turnover_limit=0.19)
    assert chapter_decision_packet(changed_inputs, base_policy).snapshot_hash != (
        base.snapshot_hash
    )
    assert chapter_decision_packet(base_inputs, changed_policy).snapshot_hash != (
        base.snapshot_hash
    )
    assert len(base.evidence_bundle_hash) == 64
    for index, gate in enumerate(base.gates):
        if not gate.hard:
            continue
        mutated = list(base.gates)
        mutated[index] = replace(gate, state="REJECT")
        assert evaluate_release_gates(tuple(mutated)) == "REJECT"
