"""Executable numerical contracts for Chapter 3."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from quant_interviews.ch03.adjoint import linear_adjoint_audit
from quant_interviews.ch03.autodiff import (
    Dual,
    black_scholes_greek_crosscheck,
    polynomial_value_gradient,
    reverse_polynomial_gradient,
)
from quant_interviews.ch03.barrier_pde import (
    analytic_down_and_out_call,
    analytic_down_and_out_greeks,
    barrier_far_boundary_study,
    barrier_pde_refinement,
    barrier_terminal_grid,
    down_and_out_call_pde,
)
from quant_interviews.ch03.calibration import CalibrationPolicy, calibrate
from quant_interviews.ch03.case_study import governed_numerical_case
from quant_interviews.ch03.finite_difference import (
    crank_nicolson_heat,
    explicit_heat,
    rannacher_comparison,
    rannacher_startup_study,
)
from quant_interviews.ch03.implied_vol import call_price, implied_vol
from quant_interviews.ch03.interpolation import (
    call_price_arbitrage_audit,
    curve_surface_audit,
    interpolate_log_discount,
)
from quant_interviews.ch03.mean_reversion import (
    exact_mean_reversion_step,
    mean_reversion_convergence,
)
from quant_interviews.ch03.qmc import randomized_qmc_study
from quant_interviews.ch03.quadrature import (
    composite_simpson,
    composite_trapezoid,
    quadrature_convergence_table,
    singularity_transform_integral,
)
from quant_interviews.ch03.root_finding import safeguarded_newton


def test_safeguarded_newton_returns_full_convergence_certificate() -> None:
    result = safeguarded_newton(
        lambda value: value * value - 2.0,
        lambda value: 2.0 * value,
        0.0,
        2.0,
        0.1,
        residual_tolerance=1e-12,
        bracket_tolerance=1e-10,
    )

    assert result.converged
    assert result.root == pytest.approx(math.sqrt(2.0), abs=1e-10)
    assert result.residual <= 1e-12
    assert result.bracket_width <= 1e-10
    assert result.newton_steps > 0
    assert result.bisection_steps > 0


def test_safeguarded_newton_handles_endpoint_and_invalid_contracts() -> None:
    endpoint = safeguarded_newton(
        lambda value: value,
        lambda _value: 1.0,
        0.0,
        2.0,
        1.0,
    )

    assert endpoint.status == "CONVERGED_ENDPOINT"
    with pytest.raises(ValueError, match="ordered bracket"):
        safeguarded_newton(lambda x: x, lambda _x: 1.0, 2.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        safeguarded_newton(
            lambda _x: float("nan"),
            lambda _x: 1.0,
            0.0,
            1.0,
            0.5,
        )


def test_implied_vol_roundtrip_and_boundary_statuses() -> None:
    price = call_price(100.0, 100.0, 0.03, 0.01, 0.25, 1.0)
    result = implied_vol(price, 100.0, 100.0, 0.03, 0.01, 1.0)

    assert result.status == "CONVERGED"
    assert result.volatility == pytest.approx(0.25, abs=1e-9)
    assert result.price_residual <= 1e-10
    lower = call_price(100.0, 100.0, 0.03, 0.01, 0.0, 1.0)
    assert implied_vol(
        lower,
        100.0,
        100.0,
        0.03,
        0.01,
        1.0,
    ).status == "LOWER_BOUND"
    upper = 100.0 * math.exp(-0.01)
    assert implied_vol(
        upper,
        100.0,
        100.0,
        0.03,
        0.01,
        1.0,
    ).status == "UPPER_BOUND_LIMIT"
    assert implied_vol(
        upper + 1.0,
        100.0,
        100.0,
        0.03,
        0.01,
        1.0,
    ).status == "INVALID_QUOTE"


def test_implied_vol_expands_bracket_for_high_volatility() -> None:
    price = call_price(100.0, 120.0, 0.0, 0.0, 2.0, 1.0)

    result = implied_vol(price, 100.0, 120.0, 0.0, 0.0, 1.0)

    assert result.status == "CONVERGED"
    assert result.volatility == pytest.approx(2.0, rel=1e-9)


def test_implied_vol_relative_price_tolerance_is_currency_scale_aware() -> None:
    volatility = 0.31
    results = []
    for currency_scale in (1.0e-4, 1.0, 1.0e6):
        spot = 100.0 * currency_scale
        strike = 105.0 * currency_scale
        price = call_price(spot, strike, 0.02, 0.01, volatility, 0.75)
        results.append(
            implied_vol(
                price,
                spot,
                strike,
                0.02,
                0.01,
                0.75,
                price_tolerance=1.0e-14 * currency_scale,
                price_relative_tolerance=1.0e-12,
            ).volatility
        )
    assert results == pytest.approx([volatility] * 3, abs=1.0e-9)
    with pytest.raises(ValueError, match="solver controls"):
        implied_vol(
            1.0,
            100.0,
            100.0,
            0.02,
            0.01,
            0.75,
            volatility_scale=0.0,
        )


def test_implied_vol_publication_contract_is_compact_and_complete() -> None:
    contract = dict(
        spot=100.0,
        strike=100.0,
        rate=0.03,
        dividend=0.01,
        maturity=1.0,
    )
    price = call_price(**contract, volatility=0.25)
    solved = implied_vol(price, **contract)
    lower = call_price(**contract, volatility=0.0)
    upper = contract["spot"] * math.exp(-contract["dividend"])

    assert solved.status == "CONVERGED"
    assert solved.volatility == pytest.approx(0.25, abs=1e-9)
    assert implied_vol(lower, **contract).status == "LOWER_BOUND"
    assert implied_vol(upper, **contract).status == "UPPER_BOUND_LIMIT"


def test_quadrature_convergence_table_recovers_expected_orders() -> None:
    trapezoid = quadrature_convergence_table(
        composite_trapezoid,
        np.exp,
        0.0,
        1.0,
        math.e - 1.0,
        [20, 40, 80],
    )
    simpson = quadrature_convergence_table(
        composite_simpson,
        np.exp,
        0.0,
        1.0,
        math.e - 1.0,
        [20, 40, 80],
    )

    assert trapezoid[-1]["observed_order"] == pytest.approx(2.0, rel=0.01)
    assert simpson[-1]["observed_order"] == pytest.approx(4.0, rel=0.02)


def test_quadrature_order_supports_non_doubling_refinement() -> None:
    rows = quadrature_convergence_table(
        composite_simpson,
        np.exp,
        0.0,
        1.0,
        math.e - 1.0,
        [18, 30, 54],
    )

    assert rows[-1]["observed_order"] == pytest.approx(4.0, rel=0.03)
    with pytest.raises(ValueError, match="strictly increasing"):
        quadrature_convergence_table(
            composite_trapezoid,
            np.exp,
            0.0,
            1.0,
            math.e - 1.0,
            [20, 20],
        )


def test_singularity_transform_separates_two_distinct_examples() -> None:
    square_root = singularity_transform_integral(
        np.sqrt,
        0.0,
        1.0,
        2_000,
        transformed_endpoint_limit=0.0,
    )
    inverse_square_root = singularity_transform_integral(
        lambda values: 1.0 / np.sqrt(values),
        0.0,
        1.0,
        2_000,
        transformed_endpoint_limit=2.0,
    )

    assert square_root == pytest.approx(2.0 / 3.0, abs=2e-7)
    assert inverse_square_root == pytest.approx(2.0, abs=1e-12)


def test_randomized_qmc_study_reports_governed_replicates() -> None:
    integrand = lambda points: np.prod(1.0 + points, axis=1)
    exact = 1.5**4

    rows = randomized_qmc_study(
        integrand,
        4,
        [8, 10, 12],
        scrambles=16,
        seed=7,
        exact=exact,
    )

    repeated = randomized_qmc_study(
        integrand,
        4,
        [8, 10, 12],
        scrambles=16,
        seed=7,
        exact=exact,
    )

    assert [row["samples"] for row in rows] == [2**8, 2**10, 2**12]
    assert all(row["scrambles"] == 16 for row in rows)
    assert np.asarray(rows[-1]["rqmc_replicates"]).shape == (16,)
    assert np.asarray(rows[-1]["rqmc_replicates"]) == pytest.approx(
        repeated[-1]["rqmc_replicates"]
    )


def test_rqmc_outperforms_mc_on_governed_smooth_fixture() -> None:
    integrand = lambda points: np.prod(1.0 + points, axis=1)
    rows = randomized_qmc_study(
        integrand,
        4,
        [8, 10, 12],
        scrambles=16,
        seed=7,
        exact=1.5**4,
    )

    assert rows[-1]["rqmc_replicate_se"] < rows[-1]["mc_replicate_se"]
    assert rows[-1]["rqmc_absolute_error"] < rows[-1]["mc_absolute_error"]


def test_explicit_heat_enforces_cfl_and_discrete_maximum_principle() -> None:
    grid = np.linspace(0.0, 1.0, 101)
    initial = np.sin(np.pi * grid)

    result = explicit_heat(initial, 0.1, 0.01, 0.0004, 50)
    exact = np.exp(-0.1 * np.pi**2 * result.final_time) * initial

    assert result.cfl == pytest.approx(0.4)
    assert result.stability_margin == pytest.approx(0.1)
    assert np.max(result.values) <= np.max(initial)
    assert result.values == pytest.approx(exact, abs=2e-5)
    with pytest.raises(ValueError, match="CFL"):
        explicit_heat(initial, 0.1, 0.01, 0.0006, 5)


def test_rannacher_uses_two_half_steps_per_replaced_full_step() -> None:
    grid = np.linspace(0.0, 1.0, 101)
    initial = (grid >= 0.5).astype(float)
    initial[0], initial[-1] = 0.0, 1.0

    result = crank_nicolson_heat(
        initial,
        0.1,
        0.01,
        0.01,
        5,
        rannacher_full_steps=2,
    )

    assert result.final_time == pytest.approx(0.05)
    assert result.backward_euler_half_steps == 4
    assert result.crank_nicolson_steps == 3


def test_rannacher_reduces_curvature_oscillation_on_discontinuous_data() -> None:
    grid = np.linspace(0.0, 1.0, 101)
    initial = (grid >= 0.5).astype(float)
    initial[0], initial[-1] = 0.0, 1.0

    evidence = rannacher_comparison(initial, 0.1, 0.01, 0.01, 5)

    assert evidence["rannacher_full_steps"] == 1
    assert evidence["gamma_variation_reduction"] > 0.0
    assert evidence["rannacher_gamma_sign_flips"] <= evidence[
        "cn_gamma_sign_flips"
    ]

    startup = rannacher_startup_study(initial, 0.1, 0.01, 0.01, 5)
    assert [row["rannacher_full_steps"] for row in startup] == [0, 1, 2]
    assert startup[1]["gamma_total_variation"] < startup[0][
        "gamma_total_variation"
    ]


def test_forward_and_reverse_ad_match_polynomial_derivative() -> None:
    forward = polynomial_value_gradient(2.0)

    assert isinstance(forward, Dual)
    assert forward.value == pytest.approx(17.0)
    assert forward.derivative == pytest.approx(14.0)
    assert reverse_polynomial_gradient(2.0) == pytest.approx(14.0)


def test_black_scholes_ad_greeks_match_independent_references() -> None:
    evidence = black_scholes_greek_crosscheck(
        100.0,
        105.0,
        0.03,
        0.01,
        0.22,
        1.4,
    )

    assert evidence["ad_delta"] == pytest.approx(
        evidence["analytic_delta"],
        abs=1e-12,
    )
    assert evidence["ad_vega"] == pytest.approx(
        evidence["analytic_vega"],
        abs=1e-11,
    )
    assert evidence["complex_delta"] == pytest.approx(
        evidence["analytic_delta"],
        abs=1e-11,
    )
    assert evidence["complex_vega"] == pytest.approx(
        evidence["analytic_vega"],
        abs=1e-10,
    )
    assert evidence["bump_delta"] == pytest.approx(
        evidence["analytic_delta"],
        abs=1e-8,
    )


def test_calibration_record_exposes_fit_conditioning_and_restarts() -> None:
    coordinates = np.arange(8.0)
    observations = 1.0 + 2.0 * coordinates
    model = lambda parameters: parameters[0] + parameters[1] * coordinates

    record = calibrate(
        [0.0, 0.0],
        model,
        observations,
        np.ones_like(observations),
        parameter_scale=[1.0, 2.0],
        restart_points=([3.0, 1.0], [-2.0, 3.0]),
        heldout_model=lambda parameters: parameters[0]
        + parameters[1] * np.array([8.0, 9.0]),
        heldout_observations=[17.0, 19.0],
    )

    assert record.parameters == pytest.approx([1.0, 2.0])
    assert record.weighted_rmse < 1e-12
    assert record.heldout_rmse is not None and record.heldout_rmse < 1e-12
    assert record.restart_dispersion < 1e-10
    assert record.status == "PASS"


def test_calibration_flags_weak_identification() -> None:
    coordinates = np.ones(5)
    model = lambda parameters: parameters[0] + parameters[1] * coordinates

    record = calibrate(
        [0.0, 0.0],
        model,
        np.full(5, 2.0),
        np.ones(5),
        regularization=1e-10,
        prior=[0.0, 0.0],
    )

    assert record.scaled_jacobian_condition > 1e12
    assert record.status == "NOT_IDENTIFIED"


def test_nonlinear_calibration_uses_bounds_prior_and_restarts() -> None:
    times = np.linspace(0.1, 3.0, 30)
    true_parameters = np.array([0.35, 1.1, 0.04])
    model = lambda parameters: (
        parameters[0] * np.exp(-parameters[1] * times)
        + parameters[2]
    )
    observations = model(true_parameters)

    record = calibrate(
        [0.20, 0.30, 0.02],
        model,
        observations,
        np.full(times.size, 1.0 / 0.002**2),
        bounds=([0.0, 0.05, 0.0], [1.0, 4.0, 0.20]),
        prior=[0.30, 1.0, 0.03],
        regularization=1e-10,
        parameter_scale=[0.30, 1.0, 0.05],
        restart_points=(
            [0.60, 2.0, 0.08],
            [0.15, 0.10, 0.01],
        ),
        heldout_model=lambda parameters: (
            parameters[0] * np.exp(-parameters[1] * np.array([3.2, 3.5]))
            + parameters[2]
        ),
        heldout_observations=(
            true_parameters[0]
            * np.exp(-true_parameters[1] * np.array([3.2, 3.5]))
            + true_parameters[2]
        ),
        policy=CalibrationPolicy(regularization_fraction_limit=1.0),
    )

    assert record.parameters == pytest.approx(true_parameters, abs=2e-6)
    assert record.heldout_rmse is not None and record.heldout_rmse < 2e-7
    assert record.restart_dispersion < 2e-6
    assert record.status == "PASS"


def test_adjoint_matches_multiple_independent_directional_solves() -> None:
    matrix = np.array([[2.0, 0.4], [0.4, 1.5]])
    rhs = np.array([1.0, 0.5])
    matrix_derivatives = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.2], [0.2, 0.0]]),
    ]
    rhs_derivatives = [np.zeros(2), np.array([0.1, -0.2])]

    evidence = linear_adjoint_audit(
        matrix,
        rhs,
        [1.0, -0.5],
        matrix_derivatives,
        rhs_derivatives,
        [[1.0, 0.0], [0.0, 1.0], [0.3, -0.7]],
    )

    assert evidence["primal_residual"] < 1e-14
    assert evidence["adjoint_residual"] < 1e-14
    assert evidence["maximum_absolute_gradient_error"] < 1e-8
    assert evidence["maximum_scaled_gradient_error"] <= 1.0


def test_adjoint_mixed_error_gate_handles_near_zero_direction() -> None:
    evidence = linear_adjoint_audit(
        np.eye(2),
        [1.0, 2.0],
        [1.0, 0.0],
        [np.zeros((2, 2)), np.zeros((2, 2))],
        [np.zeros(2), np.zeros(2)],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    assert evidence["smallest_reference_directional_derivative"] == 0.0
    assert evidence["maximum_absolute_gradient_error"] == 0.0
    assert evidence["maximum_scaled_gradient_error"] == 0.0


def test_log_discount_interpolation_rejects_silent_extrapolation() -> None:
    midpoint = interpolate_log_discount(
        [1.0, 2.0],
        [0.97, 0.93],
        [1.5],
    )

    assert 0.93 < midpoint[0] < 0.97
    with pytest.raises(ValueError, match="in-range"):
        interpolate_log_discount([1.0, 2.0], [0.97, 0.93], [2.5])


def test_dense_curve_and_call_surface_audit_encodes_arbitrage_conditions() -> None:
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    calls = np.array([21.0, 13.0, 7.0, 3.0, 1.0])

    audit = curve_surface_audit(
        [0.5, 1.0, 2.0],
        [0.99, 0.975, 0.94],
        [0.5, 1.0, 2.0],
        [0.02, 0.045, 0.09],
        strikes,
        calls,
        forward=100.0,
        discount=0.98,
    )

    assert audit["status"] == "PASS"
    invalid = call_price_arbitrage_audit(
        strikes,
        [21.0, 13.0, 8.0, 9.0, 1.0],
        forward=100.0,
        discount=0.98,
    )
    assert invalid["status"] == "REJECT"
    assert invalid["monotonicity_violations"] > 0

    negative_rate_discount = interpolate_log_discount(
        [1.0, 2.0],
        [1.01, 1.03],
        [1.5],
    )
    assert negative_rate_discount[0] > 1.0

    vertical_spread = call_price_arbitrage_audit(
        [80.0, 100.0, 120.0],
        [20.0, 0.0, 0.0],
        forward=100.0,
        discount=0.5,
    )
    assert vertical_spread["slope_lower_violations"] == 1

    nonuniform_density = call_price_arbitrage_audit(
        [1.0, 2.0, 4.0],
        [8.0, 4.5, 0.5],
        forward=10.0,
        discount=1.0,
    )
    assert nonuniform_density["minimum_density_proxy"] == pytest.approx(1.0)


def test_exact_mean_reversion_step_includes_piecewise_constant_forcing() -> None:
    result = exact_mean_reversion_step(
        0.0,
        2.0,
        1.0,
        0.4,
        0.5,
    )
    equilibrium = 1.0 + 0.4 / 2.0

    assert result == pytest.approx(
        equilibrium * (1.0 - math.exp(-1.0))
    )
    assert exact_mean_reversion_step(
        1.0,
        0.0,
        2.0,
        0.4,
        0.5,
    ) == pytest.approx(1.2)
    assert exact_mean_reversion_step(
        1.0,
        1e-14,
        2.0,
        0.4,
        0.5,
    ) == pytest.approx(1.2, abs=1e-13)


def test_euler_mean_reversion_evidence_recovers_first_order() -> None:
    rows = mean_reversion_convergence(
        0.0,
        2.0,
        1.0,
        0.4,
        1.0,
        [25, 50, 100],
    )

    assert rows[-1]["observed_order"] == pytest.approx(1.0, rel=0.05)
    assert rows[-1]["stability"] == "STABLE"


def test_barrier_terminal_grid_is_inclusive_and_aligned() -> None:
    spots = np.array([85.0, 90.0, 100.0, 110.0])

    values = barrier_terminal_grid(spots, 100.0, 85.0)

    assert values == pytest.approx([0.0, 0.0, 0.0, 10.0])


def test_barrier_pde_converges_to_independent_continuous_formula() -> None:
    reference, reference_delta, reference_gamma = analytic_down_and_out_greeks(
        100.0,
        100.0,
        85.0,
        0.03,
        0.01,
        0.22,
        1.0,
    )
    rows = barrier_pde_refinement(
        100.0,
        100.0,
        85.0,
        0.03,
        0.01,
        0.22,
        1.0,
        s_max=300.0,
        space_steps=[86, 172, 344],
    )

    assert reference > 0.0
    assert rows[-1]["absolute_reference_error"] < rows[0][
        "absolute_reference_error"
    ]
    assert rows[-1]["absolute_reference_error"] < 0.02
    assert rows[-1]["absolute_delta_error"] < rows[0][
        "absolute_delta_error"
    ]
    assert rows[-1]["absolute_gamma_error"] < rows[0][
        "absolute_gamma_error"
    ]
    assert rows[-1]["price_observed_order"] == pytest.approx(2.0, rel=0.03)
    assert rows[-1]["delta_observed_order"] == pytest.approx(2.0, rel=0.03)
    assert rows[-1]["gamma_observed_order"] == pytest.approx(2.0, rel=0.03)
    assert rows[-1]["price"] == pytest.approx(reference, abs=0.02)
    assert rows[-1]["delta"] == pytest.approx(reference_delta, abs=2e-4)
    assert rows[-1]["gamma"] == pytest.approx(reference_gamma, abs=2e-5)


def test_barrier_pde_reports_contract_and_far_boundary_stability() -> None:
    base = down_and_out_call_pde(
        100.0,
        100.0,
        85.0,
        0.03,
        0.01,
        0.22,
        1.0,
        s_max=250.0,
        space_steps=180,
        time_steps=360,
    )
    far_boundaries = barrier_far_boundary_study(
        100.0,
        100.0,
        85.0,
        0.03,
        0.01,
        0.22,
        1.0,
        s_max_values=[250.0, 320.0, 400.0],
        spatial_spacing=1.0,
        time_steps=500,
    )

    assert base.barrier_aligned
    assert base.monitoring == "continuous"
    assert base.barrier_inclusive
    assert base.rebate_timing == "immediate"
    assert max(
        row["spatial_spacing"] for row in far_boundaries
    ) == pytest.approx(
        min(row["spatial_spacing"] for row in far_boundaries)
    )
    assert len({row["time_steps"] for row in far_boundaries}) == 1
    assert far_boundaries[0]["price"] == pytest.approx(
        far_boundaries[-1]["price"],
        abs=1e-8,
    )
    assert np.isfinite([base.price, base.delta, base.gamma]).all()


def test_pit_sql_contains_both_temporal_axes_and_full_quote_key() -> None:
    sql = (
        Path(__file__).resolve().parents[4]
        / "_book/sql/ch03/calibration_snapshot.sql"
    ).read_text(encoding="utf-8")

    for field in (
        "valid_from",
        "valid_to",
        "known_at",
        "superseded_at",
        "expiry",
        "strike",
        "option_type",
        "quote_convention",
        "snapshot_id",
        "snapshot_hash",
        "row_count",
        "minimum_known_at",
        "maximum_known_at",
        "schema_version",
        "quote_payload_hash",
    ):
        assert field in sql


def test_governed_numerical_case_reconciles_all_release_gates() -> None:
    evidence = governed_numerical_case()

    assert evidence.fixture_type.startswith("Synthetic governed fixture")
    assert evidence.fixture_version == "ch03-case-v0.11"
    assert len(evidence.snapshot_hash) == 64
    assert evidence.quotes_received == 24
    assert evidence.quotes_rejected == 2
    assert evidence.implied_vol_failures == 0
    assert evidence.weighted_calibration_rmse < 1e-7
    assert evidence.finest_pde_reference_error < 0.02
    assert (
        evidence.rannacher_curvature_oscillation
        < evidence.cn_curvature_oscillation
    )
    assert evidence.ad_bump_delta_error < 1e-8
    assert evidence.greek_instrument.startswith("Black-Scholes vanilla")
    assert [row[0] for row in evidence.pde_grid_evidence] == [86, 172, 344]
    assert evidence.decision == "CONDITIONAL PASS"
