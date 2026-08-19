from __future__ import annotations

import math

import numpy as np
import pytest

from quant_interviews.ch05.american_binomial import (
    american_lattice_convergence,
    american_put_binomial,
    american_put_lattice,
)
from quant_interviews.ch05.barrier_mc import (
    barrier_monitoring_experiment,
    down_and_out_call_mc,
)
from quant_interviews.ch05.binomial_state_prices import (
    two_state_replication_certificate,
)
from quant_interviews.ch05.black_scholes import (
    black_scholes_call,
    black_scholes_delta,
    normal_pdf,
)
from quant_interviews.ch05.black_scholes_pde import (
    black_scholes_pde_call,
    black_scholes_pde_convergence_certificate,
)
from quant_interviews.ch05.digital_replication import (
    analytic_cash_digital_call,
    digital_replication_convergence,
)
from quant_interviews.ch05.dupire import (
    dupire_local_variance,
    dupire_local_variance_diagnostic,
    dupire_surface_sha256,
)
from quant_interviews.ch05.forward_start import (
    forward_start_call_price,
    forward_start_call_ratio,
)
from quant_interviews.ch05.integrated_case import (
    GateRecord,
    ReleasePolicy,
    chapter_release_case,
    evaluate_release_gates,
)
from quant_interviews.ch05.model_free_bounds import (
    discrete_model_free_bounds,
    discrete_superhedge,
)
from quant_interviews.ch05.parity_audit import (
    executable_put_call_parity_band,
    put_call_parity_residual,
)
from quant_interviews.ch05.payoff_bounds import (
    bound_violations,
    european_bounds,
    executable_call_bound_certificate,
)
from quant_interviews.ch05.quanto import (
    quanto_adjustment_certificate,
    quanto_drift_adjustment,
)
from quant_interviews.ch05.risk_neutral import (
    discounted_expectation,
    physical_probability_invariance_certificate,
)
from quant_interviews.ch05.static_arbitrage import (
    SurfaceQuote,
    bid_ask_static_arbitrage_lp,
    butterfly_audit,
    butterfly_violations,
    calendar_violations,
    calendar_surface_audit,
    vertical_spread_violations,
    vertical_spread_audit,
)


def test_executable_call_bound_returns_trade_and_cash_flow() -> None:
    certificate = executable_call_bound_certificate(
        call_bid=7.8,
        call_ask=8.0,
        spot_bid=100.0,
        spot_ask=100.2,
        strike=90.0,
        discount_factor=0.95,
        execution_cost=0.1,
    )
    assert certificate["status"] == "REJECT"
    assert certificate["violation"] == "below_executable_lower_bound"
    assert certificate["initial_cash_flow"] == pytest.approx(6.4)
    assert certificate["terminal_payoff_floor"] == 0.0
    assert len(certificate["legs"]) == 3
    assert certificate["terminal_payoff_at_breakpoints"]["terminal_spot_strike"] == 0.0


def test_two_state_replication_certificate_closes_every_identity() -> None:
    certificate = two_state_replication_certificate(
        spot=100.0,
        up=120.0,
        down=80.0,
        payoff_up=20.0,
        payoff_down=0.0,
        gross_rate=1.02,
    )
    assert certificate["status"] == "PASS"
    assert certificate["replication_price"] == pytest.approx(
        certificate["state_price_value"]
    )
    assert certificate["up_state_residual"] == pytest.approx(0.0, abs=1e-12)
    assert certificate["down_state_residual"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_executable_parity_band_uses_both_quote_sides() -> None:
    certificate = executable_put_call_parity_band(
        call_bid=8.9,
        call_ask=9.1,
        put_bid=6.1,
        put_ask=6.3,
        spot_bid=99.9,
        spot_ask=100.1,
        strike=100.0,
        maturity=1.0,
        long_funding_rate=0.03,
        short_funding_rate=0.031,
        borrow_rate=0.002,
        dividend_yield=0.0,
    )
    assert certificate["status"] == "PASS"
    assert certificate["observed_call_minus_put_lower"] == pytest.approx(2.6)
    assert certificate["observed_call_minus_put_upper"] == pytest.approx(3.0)
    assert certificate["conversion_lower"] < certificate[
        "reverse_conversion_upper"
    ]


def test_executable_parity_band_rejects_nonintersecting_quotes() -> None:
    certificate = executable_put_call_parity_band(
        call_bid=12.0,
        call_ask=12.1,
        put_bid=5.0,
        put_ask=5.1,
        spot_bid=99.9,
        spot_ask=100.1,
        strike=100.0,
        maturity=1.0,
        long_funding_rate=0.03,
        short_funding_rate=0.031,
        borrow_rate=0.002,
    )
    assert certificate["status"] == "REJECT"
    assert certificate["upper_residual"] < 0.0


def test_vertical_and_butterfly_audits_return_economic_coordinates() -> None:
    strikes = np.array([110.0, 90.0, 100.0])
    calls = np.array([5.0, 15.0, 16.0])
    quote_ids = ("q110", "q90", "q100")
    vertical = vertical_spread_audit(
        strikes,
        calls,
        1.0,
        quote_ids=quote_ids,
    )
    assert vertical[0]["left_strike"] == 90.0
    assert vertical[0]["right_strike"] == 100.0
    assert vertical[0]["left_quote_id"] == "q90"

    butterfly = butterfly_audit(
        np.array([90.0, 100.0, 110.0]),
        np.array([15.0, 8.0, 0.0]),
        quote_ids=("q90", "q100", "q110"),
    )
    assert butterfly[0]["middle_strike"] == 100.0
    assert butterfly[0]["quote_ids"] == ("q90", "q100", "q110")


def test_calendar_surface_audit_requires_common_forward_moneyness() -> None:
    report = calendar_surface_audit(
        log_moneyness=np.array([-0.1, 0.0, 0.1]),
        maturities=np.array([0.5, 1.0]),
        total_variance=np.array(
            [
                [0.04, 0.04, 0.04],
                [0.05, 0.03, 0.05],
            ]
        ),
        forwards=np.array([101.0, 102.0]),
        discount_factors=np.array([0.99, 0.97]),
    )
    assert report["status"] == "REJECT"
    assert report["violation_count"] == 1
    assert report["violations"][0]["log_moneyness"] == 0.0
    assert "common forward" in report["violations"][0]["assumption"]


def _quotes(
    middle_bid: float,
    middle_ask: float,
) -> tuple[SurfaceQuote, ...]:
    return (
        SurfaceQuote("q90", "XYZ", 1.0, 90.0, 11.9, 12.1, 100.0, 0.97),
        SurfaceQuote(
            "q100",
            "XYZ",
            1.0,
            100.0,
            middle_bid,
            middle_ask,
            100.0,
            0.97,
        ),
        SurfaceQuote("q110", "XYZ", 1.0, 110.0, 2.9, 3.1, 100.0, 0.97),
    )


def test_bid_ask_lp_can_pass_when_mids_violate() -> None:
    report = bid_ask_static_arbitrage_lp(_quotes(7.4, 8.1))
    assert report["status"] == "PASS"
    assert report["mid_violations"]
    assert report["minimum_quote_repair"] == 0.0
    selected = report["selected_prices"]
    assert 7.4 <= selected["q100"] <= 8.1


def test_bid_ask_lp_reports_minimum_repair_when_box_is_infeasible() -> None:
    report = bid_ask_static_arbitrage_lp(_quotes(8.0, 8.0))
    assert report["status"] == "REJECT"
    assert report["minimum_quote_repair"] > 0.0
    assert report["moved_quotes"]
    assert report["dual_prices"]


def test_digital_replication_exposes_accuracy_cost_tradeoff() -> None:
    parameters = {
        "spot": 105.0,
        "rate": 0.03,
        "dividend_yield": 0.01,
        "volatility": 0.2,
        "maturity": 1.0,
    }
    call_price = lambda strike: black_scholes_call(
        strike=strike,
        **parameters,
    )
    analytic = analytic_cash_digital_call(strike=100.0, **parameters)
    study = digital_replication_convergence(
        call_price,
        strike=100.0,
        widths=(10.0, 5.0, 2.5, 1.25),
        analytic_digital=analytic,
        call_half_spread=0.02,
    )
    rows = study.rows
    errors = np.array([row["absolute_error"] for row in rows])
    costs = np.array([row["estimated_crossing_cost"] for row in rows])
    assert np.all(np.diff(errors) < 0.0)
    assert np.all(np.diff(costs) > 0.0)
    assert rows[-1]["observed_order"] == pytest.approx(2.0, rel=0.2)
    assert study.status == "PASS"


def test_american_lattice_returns_exercise_policy() -> None:
    result = american_put_lattice(
        spot=90.0,
        strike=100.0,
        rate=0.05,
        dividend_yield=0.0,
        volatility=0.25,
        maturity=1.0,
        steps=200,
    )
    assert result.status == "RAW_GRID_ONLY"
    assert result.price >= 10.0
    assert 0.0 < result.risk_neutral_probability < 1.0
    assert np.count_nonzero(np.isfinite(result.exercise_boundary)) > 20
    assert len(result.exercise_mask) == 201


def test_risk_neutral_price_does_not_depend_on_physical_probability() -> None:
    report = physical_probability_invariance_certificate(
        payoffs=np.array([0.0, 10.0]),
        state_prices=np.array([0.45, 0.50]),
        physical_probabilities_a=np.array([0.8, 0.2]),
        physical_probabilities_b=np.array([0.2, 0.8]),
        cash_price=0.95,
        cash_payoff=1.0,
        traded_payoffs=np.array([[80.0], [120.0]]),
        traded_prices=np.array([96.0]),
    )
    assert report["no_arbitrage_price_a"] == report["no_arbitrage_price_b"]
    assert report["physical_expected_payoff_a"] != report[
        "physical_expected_payoff_b"
    ]


def test_black_scholes_contract_limits_and_validation() -> None:
    assert black_scholes_call(
        spot=110.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.01,
        volatility=0.0,
        maturity=1.0,
    ) > 0.0
    assert black_scholes_delta(
        spot=110.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.01,
        volatility=0.0,
        maturity=1.0,
    ) == pytest.approx(math.exp(-0.01))
    with pytest.raises(ValueError):
        black_scholes_call(
            spot=100.0,
            strike=100.0,
            rate=0.03,
            dividend_yield=0.01,
            volatility=-0.2,
            maturity=1.0,
        )


def test_black_scholes_pde_converges_to_independent_closed_form() -> None:
    analytic = black_scholes_call(
        spot=100.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.01,
        volatility=0.2,
        maturity=1.0,
    )
    coarse = black_scholes_pde_call(
        spot=100.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.01,
        volatility=0.2,
        maturity=1.0,
        maximum_spot=400.0,
        spot_steps=100,
        time_steps=200,
    )
    fine = black_scholes_pde_call(
        spot=100.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.01,
        volatility=0.2,
        maturity=1.0,
        maximum_spot=400.0,
        spot_steps=200,
        time_steps=400,
    )
    assert abs(fine.price - analytic) < abs(coarse.price - analytic)
    assert abs(fine.price - analytic) < 0.01
    assert fine.final_time == pytest.approx(1.0)
    assert 0.0 < fine.delta < 1.0
    assert fine.gamma > 0.0


def test_forward_start_full_price_is_homogeneous_in_spot() -> None:
    arguments = {
        "alpha": 1.0,
        "rate": 0.03,
        "dividend_yield": 0.01,
        "volatility": 0.2,
        "start": 0.5,
        "maturity": 1.5,
    }
    ratio = forward_start_call_ratio(**arguments)
    price_100 = forward_start_call_price(spot=100.0, **arguments)
    price_150 = forward_start_call_price(spot=150.0, **arguments)
    assert price_150 == pytest.approx(1.5 * price_100)
    assert price_100 == pytest.approx(
        100.0 * math.exp(-0.01 * 0.5) * ratio
    )


def test_quanto_quote_inversion_preserves_economic_adjustment() -> None:
    common = {
        "asset_currency": "JPY",
        "payoff_currency": "USD",
        "collateral_currency": "USD",
        "pricing_numeraire": "USD",
        "asset_volatility": 0.2,
        "fx_volatility": 0.1,
        "correlation": 0.5,
    }
    domestic = quanto_adjustment_certificate(
        fx_quote_direction="DOMESTIC_PER_FOREIGN",
        **common,
    )
    inverse = quanto_adjustment_certificate(
        fx_quote_direction="FOREIGN_PER_DOMESTIC",
        **common,
    )
    assert domestic["drift_adjustment"] == pytest.approx(-0.01)
    assert inverse["drift_adjustment"] == pytest.approx(-0.01)
    assert domestic["covariance_term"] == inverse["covariance_term"]
    assert domestic["quoted_fx_return_correlation"] == pytest.approx(0.5)
    assert inverse["quoted_fx_return_correlation"] == pytest.approx(-0.5)


def test_barrier_experiment_is_coupled_and_pathwise_reconciled() -> None:
    report = barrier_monitoring_experiment(
        spot=100.0,
        strike=100.0,
        barrier=80.0,
        rate=0.03,
        dividend_yield=0.0,
        volatility=0.2,
        maturity=1.0,
        fine_steps=252,
        paths=12000,
        monitoring_indices={
            "monthly": tuple(range(21, 253, 21)),
            "weekly": tuple(range(5, 251, 5)) + (252,),
            "daily": tuple(range(1, 253)),
        },
        seed=17,
        batch_size=2000,
    )
    monthly = report["discrete"]["monthly"]
    weekly = report["discrete"]["weekly"]
    daily = report["discrete"]["daily"]
    assert monthly["knock_out_value"] >= weekly["knock_out_value"]
    assert weekly["knock_out_value"] >= daily["knock_out_value"]
    assert monthly["maximum_pathwise_parity_error"] == 0.0
    assert weekly["maximum_pathwise_parity_error"] == 0.0
    assert daily["maximum_pathwise_parity_error"] == 0.0
    assert abs(report["bridge_reference_error"]) <= (
        5.0 * report["bridge_standard_error"] + 0.03
    )
    assert report["status"] == "PASS"


def _flat_black_scholes_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strikes = np.linspace(65.0, 135.0, 141)
    maturities = np.linspace(0.2, 2.0, 19)
    surface = np.array(
        [
            [
                black_scholes_call(
                    spot=100.0,
                    strike=float(strike),
                    rate=0.02,
                    dividend_yield=0.0,
                    volatility=0.2,
                    maturity=float(maturity),
                )
                for strike in strikes
            ]
            for maturity in maturities
        ]
    )
    return strikes, maturities, surface


def test_dupire_recovers_flat_black_scholes_variance() -> None:
    strikes, maturities, surface = _flat_black_scholes_surface()
    forwards = 100.0 * np.exp(0.02 * maturities)
    discounts = np.exp(-0.02 * maturities)
    surface_hash = dupire_surface_sha256(
        strike=strikes,
        maturity=maturities,
        call_surface=surface,
        forwards=forwards,
        discount_factors=discounts,
        rate=0.02,
        dividend_yield=0.0,
    )
    report = dupire_local_variance_diagnostic(
        strike=strikes,
        maturity=maturities,
        call_surface=surface,
        rate=0.02,
        dividend_yield=0.0,
        forwards=forwards,
        discount_factors=discounts,
        upstream_surface_sha256=surface_hash,
        upstream_static_arbitrage_status="PASS",
        expected_local_variance=0.04,
        density_absolute_tolerance=1e-12,
        density_relative_tolerance=1e-9,
    )
    center = report.local_variance[2:-2, 20:-20]
    assert np.nanmedian(center) == pytest.approx(0.04, abs=0.002)
    assert np.nanmax(np.abs(center - 0.04)) < 0.008
    assert report.negative_fraction == 0.0
    assert report.nonfinite_fraction == 0.0
    assert report.status == "PASS"


def test_dupire_rejects_negative_density_nodes() -> None:
    strikes, maturities, surface = _flat_black_scholes_surface()
    surface[:, 70] += 2.0
    forwards = 100.0 * np.exp(0.02 * maturities)
    discounts = np.exp(-0.02 * maturities)
    surface_hash = dupire_surface_sha256(
        strike=strikes,
        maturity=maturities,
        call_surface=surface,
        forwards=forwards,
        discount_factors=discounts,
        rate=0.02,
        dividend_yield=0.0,
    )
    report = dupire_local_variance_diagnostic(
        strike=strikes,
        maturity=maturities,
        call_surface=surface,
        rate=0.02,
        dividend_yield=0.0,
        forwards=forwards,
        discount_factors=discounts,
        upstream_surface_sha256=surface_hash,
        upstream_static_arbitrage_status="PASS",
        expected_local_variance=0.04,
    )
    assert report.status == "REJECT"
    assert report.denominator_floor_hits > 0


def test_model_free_bounds_return_primal_dual_certificates() -> None:
    states = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    hedge_payoffs = np.column_stack(
        (
            np.ones(states.size),
            states,
            np.maximum(states - 90.0, 0.0),
            np.maximum(states - 110.0, 0.0),
        )
    )
    reference_state_prices = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
    hedge_prices = reference_state_prices @ hedge_payoffs
    exotic = (states >= 105.0).astype(float)
    result = discrete_model_free_bounds(
        states,
        exotic,
        hedge_payoffs,
        hedge_prices,
    )
    assert result.solver_status == "PASS"
    assert result.lower_bound < result.upper_bound
    assert np.min(result.subhedge_residual) >= -1e-9
    assert np.min(result.superhedge_residual) >= -1e-9
    assert abs(result.lower_duality_gap) < 1e-9
    assert abs(result.upper_duality_gap) < 1e-9
    assert result.lower_binding_states.size > 0
    assert result.upper_binding_states.size > 0


def test_integrated_release_case_is_reproducible_and_rejects_failed_gates() -> None:
    first = chapter_release_case()
    second = chapter_release_case()
    assert first == second
    gates = {gate["name"]: gate for gate in first["gates"]}
    assert gates["executable_parity"]["status"] == "PASS"
    assert gates["surface_feasibility"]["status"] == "PASS"
    assert gates["calendar_total_variance"]["status"] == "REJECT"
    assert gates["dupire_density"]["status"] == "REJECT"
    assert first["snapshot"]["rows_received"] == 4
    assert first["snapshot"]["rows_excluded"] == 2
    assert len(first["snapshot"]["snapshot_sha256"]) == 64
    assert len(first["evidence_bundle_sha256"]) == 64
    assert first["decision"] == "REJECT"


def test_surface_lp_rejects_prices_outside_call_bounds() -> None:
    overpriced = tuple(
        SurfaceQuote(f"q{strike:g}", "XYZ", 1.0, strike, 200.0, 200.0, 100.0, 0.97)
        for strike in (90.0, 100.0, 110.0)
    )
    report = bid_ask_static_arbitrage_lp(overpriced)
    assert report["status"] == "REJECT"
    assert report["minimum_quote_repair"] > 0.0
    assert any(
        item["type"] == "call_upper_bound"
        for item in report["executable_violations"]
    )

    below_intrinsic = (
        SurfaceQuote("q80", "XYZ", 1.0, 80.0, 1.0, 1.0, 100.0, 0.97),
        SurfaceQuote("q90", "XYZ", 1.0, 90.0, 0.5, 0.5, 100.0, 0.97),
        SurfaceQuote("q100", "XYZ", 1.0, 100.0, 0.0, 0.0, 100.0, 0.97),
    )
    assert bid_ask_static_arbitrage_lp(below_intrinsic)["status"] == "REJECT"


def test_surface_lp_separates_underlyings_and_rejects_curve_mismatch() -> None:
    xyz = _quotes(7.4, 8.1)
    abc = tuple(
        SurfaceQuote(
            f"abc-{quote.quote_id}", "ABC", quote.maturity, quote.strike,
            quote.bid, quote.ask, quote.forward, quote.discount_factor,
        )
        for quote in xyz
    )
    assert bid_ask_static_arbitrage_lp(xyz + abc)["status"] == "PASS"
    inconsistent = list(xyz)
    inconsistent[1] = SurfaceQuote(
        "q100", "XYZ", 1.0, 100.0, 7.4, 8.1, 101.0, 0.97
    )
    with pytest.raises(ValueError, match="forward differs"):
        bid_ask_static_arbitrage_lp(tuple(inconsistent))
    duplicate_ids = xyz + tuple(
        SurfaceQuote(
            quote.quote_id, "ABC", quote.maturity, quote.strike,
            quote.bid, quote.ask, quote.forward, quote.discount_factor,
        )
        for quote in xyz
    )
    with pytest.raises(ValueError, match="globally unique"):
        bid_ask_static_arbitrage_lp(duplicate_ids)


def test_calendar_rejects_nonfinite_tolerance_and_compatibility_wrappers() -> None:
    with pytest.raises(ValueError):
        calendar_surface_audit(
            log_moneyness=np.array([0.0]),
            maturities=np.array([0.5, 1.0]),
            total_variance=np.array([[0.04], [0.01]]),
            forwards=np.array([100.0, 100.0]),
            discount_factors=np.array([1.0, 1.0]),
            tolerance=float("nan"),
        )
    with pytest.raises(ValueError):
        calendar_violations(
            np.array([0.5, 1.0]), np.array([0.04, 0.01]), float("nan")
        )
    assert vertical_spread_violations(
        np.array([90.0, 100.0]), np.array([10.0, 11.0]), 1.0
    ) == [0]
    assert butterfly_violations(
        np.array([90.0, 100.0, 110.0]), np.array([15.0, 8.0, 0.0])
    ) == [1]
    assert calendar_violations(
        np.array([0.5, 1.0]), np.array([0.04, 0.01])
    ) == [0]


def test_digital_study_rejects_constant_pricer_and_handles_exact_case() -> None:
    rejected = digital_replication_convergence(
        lambda _strike: 0.0,
        strike=100.0,
        widths=(9.0, 4.0, 1.5),
        analytic_digital=0.5,
        call_half_spread=0.01,
    )
    assert rejected.status == "REJECT"
    exact = digital_replication_convergence(
        lambda strike: max(100.0 - strike, 0.0),
        strike=90.0,
        widths=(8.0, 3.0, 1.0),
        analytic_digital=1.0,
        call_half_spread=0.01,
    )
    assert exact.status == "PASS"
    assert exact.final_observed_order is None


def test_american_certificate_and_raw_grid_semantics() -> None:
    raw = american_put_lattice(
        spot=90.0, strike=100.0, rate=0.05, dividend_yield=0.0,
        volatility=0.25, maturity=1.0, steps=1,
    )
    assert raw.status == "RAW_GRID_ONLY"
    expiry = american_put_lattice(
        spot=90.0, strike=100.0, rate=0.05, dividend_yield=0.0,
        volatility=0.25, maturity=0.0, steps=2,
    )
    assert expiry.status == "EXPIRY"
    assert expiry.risk_neutral_probability is None
    with pytest.raises(ValueError, match="risk-neutral probability"):
        american_put_lattice(
            spot=90.0, strike=100.0, rate=10.0, dividend_yield=0.0,
            volatility=0.01, maturity=1.0, steps=1,
        )
    with pytest.raises(ValueError, match="invalid binomial"):
        american_put_lattice(
            spot=90.0, strike=100.0, rate=0.05, dividend_yield=0.0,
            volatility=0.25, maturity=1.0, steps=2,
            exercise_style="EUROPEAN_PUT",
        )
    certificate = american_lattice_convergence(
        spot=90.0, strike=100.0, rate=0.05, dividend_yield=0.0,
        volatility=0.25, maturity=1.0, base_steps=(100, 200, 400),
        boundary_tolerance=5.0,
    )
    assert certificate.status == "PASS"
    assert certificate.finest_price >= certificate.intrinsic_lower_bound
    assert american_put_binomial(
        spot=90.0, strike=100.0, rate=0.05, dividend_yield=0.0,
        volatility=0.25, maturity=1.0, steps=200,
    ) == pytest.approx(american_put_lattice(
        spot=90.0, strike=100.0, rate=0.05, dividend_yield=0.0,
        volatility=0.25, maturity=1.0, steps=200,
    ).price)


def test_risk_neutral_certificate_rejects_unnormalized_state_prices_and_no_contrast() -> None:
    report = physical_probability_invariance_certificate(
        payoffs=np.array([0.0, 10.0]),
        state_prices=np.array([100.0, 100.0]),
        physical_probabilities_a=np.array([0.5, 0.5]),
        physical_probabilities_b=np.array([0.5, 0.5]),
        cash_price=1.0,
        cash_payoff=1.0,
        traded_payoffs=np.array([[1.0], [1.0]]),
        traded_prices=np.array([1.0]),
    )
    assert report["status"] == "REJECT"
    assert report["maximum_pricing_residual"] > 1.0
    assert discounted_expectation(
        np.array([0.0, 10.0]), np.array([0.4, 0.6]), 0.95
    ) == pytest.approx(5.7)


def test_pde_raw_grid_cannot_claim_pass_and_certificate_checks_boundary() -> None:
    raw = black_scholes_pde_call(
        spot=100.0, strike=100.0, rate=0.03, dividend_yield=0.01,
        volatility=0.2, maturity=1.0, maximum_spot=101.0,
        spot_steps=20, time_steps=10,
    )
    assert raw.status == "RAW_GRID_ONLY"
    certificate = black_scholes_pde_convergence_certificate(
        spot=100.0, strike=100.0, rate=0.03, dividend_yield=0.01,
        volatility=0.2, maturity=1.0, maximum_spot=400.0,
        grids=((80, 160), (160, 320), (320, 640)),
    )
    assert certificate.status == "PASS"
    with pytest.raises(ValueError, match="PDE convergence policy"):
        black_scholes_pde_convergence_certificate(
            spot=100.0, strike=100.0, rate=0.03, dividend_yield=0.01,
            volatility=0.2, maturity=1.0, maximum_spot=101.0,
        )


def test_barrier_negative_fixture_and_calendar_validation() -> None:
    report = barrier_monitoring_experiment(
        spot=100.0, strike=100.0, barrier=80.0, rate=0.03,
        dividend_yield=0.0, volatility=0.2, maturity=1.0,
        fine_steps=2, paths=2,
        monitoring_indices={"terminal": (2,)}, seed=2, batch_size=2,
    )
    assert report["status"] == "REJECT"
    assert report["reference_gate_status"] == "REJECT"
    with pytest.raises(ValueError):
        barrier_monitoring_experiment(
            spot=100.0, strike=100.0, barrier=80.0, rate=0.03,
            dividend_yield=0.0, volatility=0.2, maturity=1.0,
            fine_steps=10, paths=10,
            monitoring_indices={"bad": (0, 10)},
        )
    with pytest.raises(ValueError, match="distinct observation counts"):
        barrier_monitoring_experiment(
            spot=100.0, strike=100.0, barrier=80.0, rate=0.03,
            dividend_yield=0.0, volatility=0.2, maturity=1.0,
            fine_steps=10, paths=10,
            monitoring_indices={"first": (5, 10), "second": (4, 10)},
        )
    with pytest.raises(ValueError, match="unique exact divisors"):
        barrier_monitoring_experiment(
            spot=100.0, strike=100.0, barrier=80.0, rate=0.03,
            dividend_yield=0.0, volatility=0.2, maturity=1.0,
            fine_steps=10, paths=10, monitoring_observations=(2, 2),
        )
    value, standard_error = down_and_out_call_mc(
        spot=100.0, strike=100.0, barrier=80.0, rate=0.03,
        volatility=0.2, maturity=1.0, steps=10, paths=100, seed=3,
    )
    assert value >= 0.0 and standard_error >= 0.0


def test_dupire_rejects_non_call_surface_and_hash_mismatch() -> None:
    strikes = np.linspace(80.0, 120.0, 9)
    maturities = np.linspace(0.25, 1.0, 5)
    surface = maturities[:, None] + 0.01 * strikes[None, :] ** 2
    forwards = np.full(maturities.shape, 100.0)
    discounts = np.ones(maturities.shape)
    surface_hash = dupire_surface_sha256(
        strike=strikes, maturity=maturities, call_surface=surface,
        forwards=forwards, discount_factors=discounts,
        rate=0.0, dividend_yield=0.0,
    )
    report = dupire_local_variance_diagnostic(
        strike=strikes, maturity=maturities, call_surface=surface,
        rate=0.0, dividend_yield=0.0, forwards=forwards,
        discount_factors=discounts, upstream_surface_sha256=surface_hash,
        upstream_static_arbitrage_status="PASS",
    )
    assert report.status == "REJECT"
    assert report.call_bound_violation_count > 0
    assert report.strike_slope_violation_count > 0
    clean_strikes, clean_maturities, clean_surface = _flat_black_scholes_surface()
    clean_forwards = 100.0 * np.exp(0.02 * clean_maturities)
    clean_discounts = np.exp(-0.02 * clean_maturities)
    mismatched = dupire_local_variance_diagnostic(
        strike=clean_strikes, maturity=clean_maturities,
        call_surface=clean_surface, rate=0.02, dividend_yield=0.0,
        forwards=clean_forwards, discount_factors=clean_discounts,
        upstream_surface_sha256="0" * 64,
        upstream_static_arbitrage_status="PASS", expected_local_variance=0.04,
    )
    assert mismatched.status == "REJECT"
    local = dupire_local_variance(
        strike=clean_strikes, maturity=clean_maturities,
        call_surface=clean_surface, rate=0.02, dividend_yield=0.0,
        forwards=clean_forwards, discount_factors=clean_discounts,
        upstream_surface_sha256=dupire_surface_sha256(
            strike=clean_strikes, maturity=clean_maturities,
            call_surface=clean_surface, forwards=clean_forwards,
            discount_factors=clean_discounts, rate=0.02, dividend_yield=0.0,
        ),
        upstream_static_arbitrage_status="PASS",
    )
    assert local.shape == clean_surface.shape


def test_robust_bounds_are_explicitly_discrete_and_wrappers_are_tested() -> None:
    states = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    hedge_payoffs = np.column_stack((np.ones(5), states))
    prices = np.array([1.0, 100.0])
    target = (states >= 105.0).astype(float)
    result = discrete_model_free_bounds(
        states, target, hedge_payoffs, prices,
        target_breakpoints=np.array([105.0]),
    )
    assert result.scope == "DISCRETE_FINITE_STATE_GRID"
    assert not result.continuum_certified
    assert result.unresolved_target_breakpoints.tolist() == [105.0]
    upper, positions = discrete_superhedge(
        states, target, hedge_payoffs, prices
    )
    assert upper == pytest.approx(result.upper_bound)
    assert positions.shape == (2,)


def test_release_reducer_all_pass_conditional_missing_and_each_reject() -> None:
    base = tuple(
        GateRecord(
            name=name, required=True, value=0, operator="<=",
            signed_limit=0, unit="count", owner="owner",
            evidence_sha256=str(index + 1) * 64, status="PASS", reason="ok",
        )
        for index, name in enumerate(("a", "b", "c"))
    )
    policy = ReleasePolicy("policy-v1", ("a", "b", "c"), ())
    assert evaluate_release_gates(base, policy)["decision"] == "PASS"
    conditional = (base[0], base[1], GateRecord(**{**base[2].__dict__, "status": "CONDITIONAL_PASS"}))
    assert evaluate_release_gates(conditional, policy)["decision"] == "CONDITIONAL_PASS"
    assert evaluate_release_gates(base[:-1], policy)["decision"] == "REJECT"
    for index in range(len(base)):
        mutated = tuple(
            GateRecord(**{**gate.__dict__, "status": "REJECT"})
            if item == index else gate
            for item, gate in enumerate(base)
        )
        assert evaluate_release_gates(mutated, policy)["decision"] == "REJECT"


def test_small_public_compatibility_functions_are_covered() -> None:
    bounds = european_bounds(
        spot=100.0, strike=100.0, rate=0.02, maturity=1.0
    )
    assert bound_violations(200.0, *bounds["call"]) == ["above_upper_bound"]
    assert put_call_parity_residual(
        call=10.0, put=10.0 - 100.0 + 100.0 * math.exp(-0.02),
        spot=100.0, strike=100.0, rate=0.02, maturity=1.0,
    ) == pytest.approx(0.0)
    assert normal_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi))
    assert quanto_drift_adjustment(
        asset_volatility=0.2, fx_volatility=0.1, correlation=0.5
    ) == pytest.approx(-0.01)
