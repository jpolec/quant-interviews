from __future__ import annotations

from math import exp

import numpy as np
import pytest

from quant_interviews.ch06.black_scholes_greeks import (
    GreekConventions,
    call_greeks,
    call_price,
)
from quant_interviews.ch06.gamma_vega import (
    long_gamma_short_vega_calendar,
)
from quant_interviews.ch06.hedging_pnl import (
    self_financing_hedge_reconciliation,
    variance_pnl_approx,
)
from quant_interviews.ch06.heston import heston_diagnostics
from quant_interviews.ch06.implied_vol import (
    call_price_bounds,
    implied_vol_bid_ask,
    implied_vol_call,
)
from quant_interviews.ch06.integrated_case import chapter_release_case
from quant_interviews.ch06.jump_hedging import (
    hedge_frequency_study,
    jump_gap_exposure,
)
from quant_interviews.ch06.roughness import (
    exponential_forecast_weights,
    roughness_recovery_study,
)
from quant_interviews.ch06.sabr import (
    hagan_black_vol,
    sabr_smile_audit,
)
from quant_interviews.ch06.skew import skew_fit
from quant_interviews.ch06.sticky import sticky_delta_scenario
from quant_interviews.ch06.surface import surface_arbitrage_audit
from quant_interviews.ch06.vanna_volga import vanna_volga_audit
from quant_interviews.ch06.variance_products import (
    corridor_realized_variance_audit,
    variance_swap_audit,
)


def test_greek_schema_encodes_units_and_general_delta_bound() -> None:
    conventions = GreekConventions(currency="GBP")
    greeks = call_greeks(
        100.0,
        100.0,
        0.03,
        -0.02,
        0.20,
        1.0,
        conventions=conventions,
    )
    assert 0.0 < greeks.delta < greeks.delta_upper_bound
    assert greeks.delta_upper_bound == pytest.approx(exp(0.02))
    assert greeks.vega_per_vol_point == pytest.approx(
        0.01 * greeks.raw_vega_per_unit_vol
    )
    assert greeks.rho_per_basis_point == pytest.approx(
        0.0001 * greeks.raw_rho_per_unit_rate
    )
    assert greeks.theta_per_calendar_day == pytest.approx(
        greeks.theta_per_year / 365.0
    )
    assert greeks.conventions.currency == "GBP"


def test_analytic_greeks_reconcile_with_independent_bumps() -> None:
    spot = 103.0
    strike = 97.0
    rate = 0.025
    dividend = 0.012
    vol = 0.24
    horizon = 0.8
    greeks = call_greeks(spot, strike, rate, dividend, vol, horizon)
    spot_bump = 1e-2
    vol_bump = 1e-4
    price = call_price(spot, strike, rate, dividend, vol, horizon)
    up = call_price(spot + spot_bump, strike, rate, dividend, vol, horizon)
    down = call_price(spot - spot_bump, strike, rate, dividend, vol, horizon)
    bump_delta = (up - down) / (2.0 * spot_bump)
    bump_gamma = (up - 2.0 * price + down) / spot_bump**2
    bump_vega = (
        call_price(spot, strike, rate, dividend, vol + vol_bump, horizon)
        - call_price(spot, strike, rate, dividend, vol - vol_bump, horizon)
    ) / (2.0 * vol_bump)
    assert bump_delta == pytest.approx(greeks.delta, rel=1e-7)
    assert bump_gamma == pytest.approx(greeks.gamma, rel=2e-5)
    assert bump_vega == pytest.approx(
        greeks.raw_vega_per_unit_vol, rel=1e-7
    )


def test_variance_pnl_sign_is_gamma_weighted() -> None:
    conventions = {
        "variance_convention": "annualized_continuously_compounded_return",
        "horizon_unit": "years",
    }
    assert variance_pnl_approx(
        0.02, 100.0, 0.09, 0.04, 1.0, **conventions
    ) > 0.0
    assert variance_pnl_approx(
        0.02, 100.0, 0.01, 0.04, 1.0, **conventions
    ) < 0.0


def test_self_financing_reconciliation_has_explicit_ledger() -> None:
    result = self_financing_hedge_reconciliation(
        option_pnl=np.array([1.0, -0.4]),
        delta_hedge_pnl=np.array([-0.6, 0.3]),
        financing_pnl=np.array([0.01, 0.01]),
        dividend_pnl=np.array([-0.02, -0.01]),
        execution_cost=np.array([0.01, 0.02]),
        accounting_pnl=np.array([0.38, -0.12]),
        tolerance=1e-12,
    )
    assert result.attributed_pnl == pytest.approx(0.26)
    assert result.residual == pytest.approx(0.0)
    assert result.status == "PASS"


def test_calendar_constructs_long_gamma_and_strictly_short_vega() -> None:
    result = long_gamma_short_vega_calendar(
        spot=100.0,
        rate=0.03,
        dividend=0.01,
        vol=0.20,
        short_horizon=0.10,
        long_horizon=1.0,
        log_forward_moneyness=0.0,
    )
    assert (
        result.minimum_short_long_weight
        < result.selected_short_long_weight
        < result.maximum_short_long_weight
    )
    assert result.net_gamma > 0.0
    assert result.net_vega_per_vol_point < 0.0
    assert result.greek_feasibility_status == "PASS"
    assert result.economic_scenario_status == "REJECT"
    assert result.status == "REJECT"


def test_volga_sign_is_fixture_specific_not_an_invariant() -> None:
    def pricer(spot: float, volatility: float) -> float:
        return call_price(spot, 100.0, 0.03, 0.01, volatility, 1.0)

    atm = vanna_volga_audit(
        price_fn=pricer,
        spot=100.0,
        strike=100.0,
        rate=0.03,
        dividend=0.01,
        vol=0.20,
        horizon=1.0,
    )
    assert atm.status == "PASS"
    assert atm.volga_sign == "NEGATIVE"

    def wing_pricer(spot: float, volatility: float) -> float:
        return call_price(spot, 140.0, 0.03, 0.01, volatility, 1.0)

    wing = vanna_volga_audit(
        price_fn=wing_pricer,
        spot=100.0,
        strike=140.0,
        rate=0.03,
        dividend=0.01,
        vol=0.20,
        horizon=1.0,
    )
    assert wing.status == "PASS"
    assert wing.volga_sign == "POSITIVE"


def test_sticky_delta_round_trip_reprices_fixed_strike() -> None:
    smile = lambda delta: 0.20 + 0.08 * (0.50 - delta)
    result = sticky_delta_scenario(
        spot=100.0,
        spot_move=-5.0,
        target_delta=0.40,
        rate=0.02,
        dividend=0.0,
        horizon=0.75,
        smile_by_delta=smile,
        delta_convention="forward_call",
    )
    assert result.initial_strike > 0.0
    assert result.moved_log_forward_moneyness != pytest.approx(
        result.initial_log_forward_moneyness
    )
    assert result.sticky_delta_vol == pytest.approx(
        smile(result.moved_delta_at_fixed_strike)
    )
    assert result.status == "PASS"


def test_skew_fit_reports_uncertainty_in_forward_coordinate() -> None:
    result = skew_fit(
        strikes=np.array([80.0, 90.0, 100.0, 110.0, 120.0]),
        forward=100.0,
        vols=np.array([0.31, 0.255, 0.22, 0.198, 0.185]),
    )
    assert result.slope < 0.0
    assert result.slope_confidence_interval[1] < 0.0
    assert result.coordinate == "log-forward-moneyness"


def test_implied_vol_returns_governed_certificate_and_bound_states() -> None:
    price = call_price(100.0, 105.0, 0.03, 0.01, 0.27, 1.2)
    result = implied_vol_call(price, 100.0, 105.0, 0.03, 0.01, 1.2)
    assert result.status == "CONVERGED"
    assert result.implied_vol == pytest.approx(0.27, abs=1e-10)
    assert abs(result.repricing_residual) < 1e-10
    assert result.bracket_width <= 1e-10
    lower, upper = call_price_bounds(100.0, 105.0, 0.03, 0.01, 1.2)
    assert implied_vol_call(
        lower, 100.0, 105.0, 0.03, 0.01, 1.2
    ).status == "LOWER_BOUND"
    assert implied_vol_call(
        upper, 100.0, 105.0, 0.03, 0.01, 1.2
    ).status == "UPPER_BOUND_LIMIT"


def test_implied_vol_bid_ask_preserves_economic_uncertainty() -> None:
    bid = call_price(100.0, 100.0, 0.02, 0.0, 0.19, 0.5)
    ask = call_price(100.0, 100.0, 0.02, 0.0, 0.23, 0.5)
    interval = implied_vol_bid_ask(
        bid, ask, 100.0, 100.0, 0.02, 0.0, 0.5
    )
    assert interval.status == "PASS"
    assert interval.bid.implied_vol == pytest.approx(0.19, abs=1e-10)
    assert interval.ask.implied_vol == pytest.approx(0.23, abs=1e-10)


def _clean_surface_fixture() -> tuple[np.ndarray, ...]:
    log_moneyness = np.array([-0.20, -0.10, 0.0, 0.10, 0.20])
    maturities = np.array([0.5, 1.0])
    forwards = np.array([100.0, 100.0])
    discounts = np.exp(-0.02 * maturities)
    implied_vols = np.full((2, 5), 0.20)
    call_prices = np.empty_like(implied_vols)
    for row, maturity in enumerate(maturities):
        spot = forwards[row] * exp(-0.02 * maturity)
        for column, coordinate in enumerate(log_moneyness):
            strike = forwards[row] * exp(coordinate)
            call_prices[row, column] = call_price(
                spot, strike, 0.02, 0.0, 0.20, maturity
            )
    return (
        log_moneyness,
        maturities,
        forwards,
        discounts,
        call_prices,
        implied_vols,
    )


def test_surface_gate_audits_bounds_slopes_convexity_and_calendar() -> None:
    k, t, f, d, calls, vols = _clean_surface_fixture()
    result = surface_arbitrage_audit(
        log_forward_moneyness=k,
        maturities=t,
        forwards=f,
        discounts=d,
        call_prices=calls,
        implied_vols=vols,
    )
    assert result.violations == ()
    assert result.minimum_call_slope_residual >= 0.0
    assert result.minimum_convexity_residual >= 0.0
    assert result.minimum_calendar_total_variance_increment > 0.0
    assert result.status == "PASS"


def test_surface_mid_failure_can_remain_bid_ask_feasible() -> None:
    k, t, f, d, calls, vols = _clean_surface_fixture()
    bad_mid = calls.copy()
    bad_mid[0, 2] += 5.0
    result = surface_arbitrage_audit(
        log_forward_moneyness=k,
        maturities=t,
        forwards=f,
        discounts=d,
        call_prices=bad_mid,
        implied_vols=vols,
        bid_prices=calls - 0.02,
        ask_prices=calls + 0.02,
    )
    assert result.violations
    assert result.bid_ask_feasible
    assert result.status == "REJECT"


def test_variance_strip_recovers_black_scholes_variance() -> None:
    forward = 100.0
    maturity = 1.0
    rate = 0.02
    discount = exp(-rate * maturity)
    spot = forward * discount
    volatility = 0.20
    strikes = np.linspace(10.0, 400.0, 3_901)
    calls = np.asarray(
        [
            call_price(spot, strike, rate, 0.0, volatility, maturity)
            for strike in strikes
        ]
    )
    puts = calls - discount * (forward - strikes)
    result = variance_swap_audit(
        forward=forward,
        strikes=strikes,
        puts=puts,
        calls=calls,
        discount=discount,
        horizon=maturity,
    )
    assert result.fair_variance == pytest.approx(volatility**2, abs=3e-4)
    assert result.k0 == pytest.approx(forward)
    assert result.status == "PASS"


def test_corridor_variance_uses_total_contract_time_denominator() -> None:
    result = corridor_realized_variance_audit(
        spots=np.array([90.0, 100.0, 110.0, 120.0, 105.0]),
        lower=95.0,
        upper=115.0,
        observation_convention="start",
        boundary_convention="strict",
    )
    assert result.corridor_variance <= result.full_variance
    assert result.occupancy_fraction == pytest.approx(0.5)
    assert result.status == "PASS"


def test_heston_reports_moment_martingale_and_refinement_diagnostics() -> None:
    result = heston_diagnostics(
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.30,
        rho=-0.70,
        horizon=1.0,
        step_counts=(32, 64),
        paths=20_000,
        seed=11,
        variance_z_limit=5.0,
        martingale_limit=0.02,
    )
    assert result.feller_condition
    assert result.rows[-1].variance_mean_z_score < 5.0
    assert result.rows[-1].forward_martingale_error < 0.02
    assert result.rows[-1].negative_pre_truncation_fraction >= 0.0
    assert result.status == "PASS"


def test_sabr_audit_exposes_skew_curvature_and_shifted_domain() -> None:
    strikes = np.linspace(70.0, 130.0, 13)
    base = sabr_smile_audit(
        forward=100.0,
        strikes=strikes,
        alpha=0.20,
        beta=0.50,
        rho=-0.20,
        nu=0.60,
        horizon=1.0,
    )
    more_negative_rho = sabr_smile_audit(
        forward=100.0,
        strikes=strikes,
        alpha=0.20,
        beta=0.50,
        rho=-0.60,
        nu=0.60,
        horizon=1.0,
        parameter_shock="rho -0.2 -> -0.6",
    )
    assert more_negative_rho.downside_skew < base.downside_skew
    assert np.isfinite(base.curvature)
    assert hagan_black_vol(
        forward=-0.01,
        strike=0.0,
        alpha=0.02,
        beta=0.50,
        rho=-0.20,
        nu=0.50,
        horizon=1.0,
        shift=0.05,
    ) > 0.0


def test_roughness_recovers_synthetic_hurst_and_separates_forecast_weights() -> None:
    study = roughness_recovery_study(
        true_hurst=0.20,
        points=256,
        paths=12,
        max_lag=12,
        seed=7,
        tolerance=0.12,
    )
    assert study.status == "PASS"
    assert abs(study.estimated_hurst_mean - 0.20) < 0.12
    weights = exponential_forecast_weights(5.0, np.array([1.0, 5.0, 20.0]))
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > weights[-1]


def test_jump_gap_is_delta_hedged_residual_not_option_move() -> None:
    spot = 100.0
    strike = 100.0
    volatility = 0.20
    maturity = 0.5
    price_before = call_price(
        spot, strike, 0.02, 0.0, volatility, maturity
    )
    greeks = call_greeks(
        spot, strike, 0.02, 0.0, volatility, maturity
    )
    result = jump_gap_exposure(
        price_fn=lambda moved_spot: call_price(
            moved_spot, strike, 0.02, 0.0, volatility, maturity
        ),
        price_before_jump=price_before,
        delta_before=greeks.delta,
        gamma_before=greeks.gamma,
        spot=spot,
        jump_return=-0.02,
    )
    assert result.exact_delta_hedged_gap > 0.0
    assert result.gamma_approximation == pytest.approx(
        result.exact_delta_hedged_gap, rel=0.10
    )


def test_hedge_frequency_study_exposes_cost_risk_frontier_and_jump_floor() -> None:
    result = hedge_frequency_study(
        rebalance_counts=(4, 8, 16),
        paths=1_000,
        transaction_cost_bps=2.0,
        jump_intensity=2.0,
        jump_return=-0.12,
        seed=17,
    )
    assert result.selected_rebalances in {4, 8, 16}
    assert result.diffusion_rows[-1].mean_transaction_cost > (
        result.diffusion_rows[0].mean_transaction_cost
    )
    assert result.jump_loss_floor > 0.0
    assert result.status == "PASS"


def test_integrated_case_produces_governed_rejection_record() -> None:
    result = chapter_release_case()
    assert result["fixture_type"] == "synthetic governed release fixture"
    assert len(result["snapshot_hash"]) == 64
    assert result["calendar_status"] == "PASS"
    assert result["surface_violation_count"] > 0
    assert result["surface_bid_ask_feasible"] is False
    assert result["surface_status"] == "REJECT"
    assert result["final_decision"] == "REJECT"
