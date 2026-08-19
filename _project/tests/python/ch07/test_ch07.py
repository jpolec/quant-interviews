from __future__ import annotations

import sqlite3
from dataclasses import replace
from math import exp
from pathlib import Path

import numpy as np
import pytest

from quant_interviews.ch07.bond_risk import (
    BondCashflows,
    bond_risk_certificate,
    example_bond,
)
from quant_interviews.ch07.callable_bond_tree import (
    CallableBond,
    calibrate_additive_short_rate_tree,
    callable_bond_oas_certificate,
    effective_duration,
    price_on_additive_short_rate_tree,
    price_on_calibrated_tree,
)
from quant_interviews.ch07.cds_bootstrap import (
    CDSQuote,
    bootstrap_hazard_curve,
    bootstrap_cds_certificate,
    cds_pv,
    cds_legs,
    flat_discount_curve,
)
from quant_interviews.ch07.credit_survival import (
    FlatHazardCredit,
    credit_survival_certificate,
)
from quant_interviews.ch07.curve_bootstrap import (
    DiscountCurve as BootstrapDiscountCurve,
    bootstrap_simple_deposits_and_annual_swaps,
    bootstrap_curve_certificate,
    example_curve,
)
from quant_interviews.ch07.curve_representation import (
    DiscountCurve as RepresentationCurve,
)
from quant_interviews.ch07.curve_representation import (
    curve_representation_certificate,
)
from quant_interviews.ch07.hjm_drift import (
    hjm_martingale_audit,
    nested_brownian_increments,
)
from quant_interviews.ch07.hull_white_exact import (
    hull_white_curve_fit_audit,
)
from quant_interviews.ch07.integrated_case import (
    REQUIRED_ARTIFACT_IDS,
    ReleaseGate,
    chapter_release_case,
    reduce_release_gates,
)
from quant_interviews.ch07.key_rate_hedge import (
    key_rate_hedge_certificate,
    solve_key_rate_hedge,
)
from quant_interviews.ch07.merton_structural import (
    merton_credit_certificate,
    merton_equity_and_default_probability,
)
from quant_interviews.ch07.multicurve_swap import (
    MultiCurveSwap,
    example_swap,
    multicurve_risk_certificate,
)
from quant_interviews.ch07.pnl_explain import (
    PnLInputs,
    StateManifest,
    explain,
    explain_certificate,
)
from quant_interviews.ch07.evidence import (
    EvidenceProvenance,
    TolerancePolicy,
    stable_manifest_hash,
)

ROOT = Path(__file__).resolve().parents[4]


def test_curve_representation_certificate_reconciles_coordinate_systems() -> None:
    curve = RepresentationCurve(
        times=np.array([0.5, 1.0, 2.0]),
        discount_factors=np.array([0.98, 0.95, 0.90]),
    )
    certificate = curve_representation_certificate(curve)
    assert certificate.status == "PASS"
    assert certificate.result.maximum_reconstruction_error < 1.0e-14
    expected = (0.98 / 0.95 - 1.0) / 0.5
    assert curve.simple_forward(0, 1) == pytest.approx(expected)


def test_curve_bootstrap_accepts_negative_quotes_when_discount_factors_are_valid() -> None:
    certificate = bootstrap_curve_certificate(
        deposit_quotes={0.5: -0.0020, 1.0: -0.0010},
        swap_quotes={2: 0.0010, 3: 0.0030},
    )
    assert certificate.status == "PASS"
    assert certificate.result.maximum_repricing_residual < 1.0e-12
    assert np.all(certificate.result.curve.discount_factors > 0.0)
    assert len(certificate.result.calibration_nodes) == 4
    assert all(node.iterations == 1 for node in certificate.result.calibration_nodes)


def test_curve_bootstrap_rejects_conflicting_shared_maturity() -> None:
    with pytest.raises(ValueError, match="same maturity are inconsistent"):
        bootstrap_curve_certificate(
            deposit_quotes={1.0: 0.01},
            swap_quotes={1: 0.05},
        )


def test_curve_bootstrap_accepts_redundant_consistent_shared_maturity() -> None:
    certificate = bootstrap_curve_certificate(
        deposit_quotes={1.0: 0.01},
        swap_quotes={1: 0.01},
    )
    assert certificate.status == "PASS"
    assert certificate.result.curve.times.tolist() == [1.0]
    assert len(certificate.result.calibration_nodes) == 2
    assert certificate.result.calibration_nodes[-1].solve_method == "shared_node_reprice"
    assert certificate.result.maximum_repricing_residual <= 1.0e-12


def test_public_curve_bootstrap_wrappers_are_exercised() -> None:
    wrapped = bootstrap_simple_deposits_and_annual_swaps(
        {0.5: 0.02, 1.0: 0.025},
        {2: 0.03},
    )
    assert wrapped.df(2.0) > 0.0
    fixture = example_curve()
    assert fixture.times.tolist() == [0.5, 1.0, 2.0, 3.0]


def test_evidence_provenance_requires_hashes_timezone_and_valid_exception() -> None:
    with pytest.raises(ValueError, match="UTC offset"):
        EvidenceProvenance(
            "run-1",
            "2026-07-30T16:00:00",
            "owner",
            "policy-v1",
            ("a" * 64,),
        )
    with pytest.raises(ValueError, match="supplied together"):
        EvidenceProvenance(
            "run-1",
            "2026-07-30T16:00:00+02:00",
            "owner",
            "policy-v1",
            ("a" * 64,),
            exception_scope="one gate",
        )
    with pytest.raises(ValueError, match="later"):
        EvidenceProvenance(
            "run-1",
            "2026-07-30T16:00:00+02:00",
            "owner",
            "policy-v1",
            ("a" * 64,),
            exception_scope="one gate",
            exception_expires_at="2026-07-30T15:59:00+02:00",
        )


def test_nonnegative_forward_policy_is_separate_from_positive_df_gate() -> None:
    certificate = bootstrap_curve_certificate(
        deposit_quotes={0.5: -0.0100, 1.0: -0.0100},
        swap_quotes={2: -0.0050},
        require_nonnegative_forwards=True,
    )
    assert certificate.status == "REJECT"
    assert certificate.result.minimum_interval_forward < 0.0
    assert np.all(certificate.result.curve.discount_factors > 0.0)


def test_bond_risk_declares_and_reconciles_both_compounding_conventions() -> None:
    bond = example_bond()
    annual = bond_risk_certificate(
        bond,
        0.05,
        compounding="annual",
    )
    continuous = bond_risk_certificate(
        bond,
        np.log1p(0.05),
        compounding="continuous",
    )
    assert annual.status == "PASS"
    assert continuous.status == "PASS"
    assert annual.result.price == pytest.approx(continuous.result.price)
    assert annual.result.analytic_dv01 == pytest.approx(
        annual.result.bump_dv01,
        rel=5.0e-6,
    )


def test_public_fixed_income_value_objects_reject_nonfinite_inputs() -> None:
    with pytest.raises(ValueError):
        RepresentationCurve(
            times=np.array([1.0, 2.0]),
            discount_factors=np.array([0.95, np.nan]),
        )
    with pytest.raises(ValueError):
        BondCashflows(np.array([1.0]), np.array([np.nan]))
    with pytest.raises(ValueError):
        MultiCurveSwap(
            payment_times=np.array([1.0]),
            accruals=np.array([1.0]),
            notional=np.nan,
        )


def _key_rate_fixture() -> tuple[BondCashflows, tuple[BondCashflows, ...]]:
    portfolio = BondCashflows(
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 3.0, 103.0]),
    )
    hedges = (
        BondCashflows(np.array([1.0]), np.array([100.0])),
        BondCashflows(np.array([2.0]), np.array([100.0])),
        BondCashflows(np.array([3.0]), np.array([100.0])),
    )
    return portfolio, hedges


def test_key_rate_certificate_performs_bump_rebuild_reprice_and_svd_solve() -> None:
    portfolio, hedges = _key_rate_fixture()
    certificate = key_rate_hedge_certificate(
        portfolio_bonds=(portfolio,),
        portfolio_notionals=np.array([1.0]),
        hedge_bonds=hedges,
        deposit_quotes={1.0: 0.020},
        swap_quotes={2: 0.025, 3: 0.030},
        notional_rounding=1.0e-8,
    )
    assert certificate.status == "PASS"
    assert certificate.result.quote_coordinates == (
        "deposit:1Y",
        "swap:2Y",
        "swap:3Y",
    )
    assert np.linalg.norm(
        certificate.result.residual_after_rounding
    ) < 1.0e-7
    assert certificate.result.maximum_curve_repricing_residual < 1.0e-12


def test_ridge_hedge_uses_augmented_svd_not_normal_equations() -> None:
    result = solve_key_rate_hedge(
        np.array([1.0, 2.0]),
        np.array([[1.0, 0.99], [0.99, 0.98]]),
        ridge_penalty=1.0e-4,
    )
    assert result.solver == "augmented_svd_ridge"
    assert np.all(np.isfinite(result.notionals))


def test_weighted_hedge_enforces_bounds_and_tradability() -> None:
    result = solve_key_rate_hedge(
        np.array([1.0, 2.0]),
        np.eye(2),
        ridge_penalty=0.0,
        bucket_weights=np.array([100.0, 1.0]),
        lower_bounds=np.array([-0.5, -10.0]),
        upper_bounds=np.array([0.5, 10.0]),
        tradable=np.array([True, False]),
    )
    np.testing.assert_allclose(result.notionals, [-0.5, 0.0], atol=1.0e-10)
    assert "hedge:0:lower" in result.active_constraints
    assert "hedge:1:nontradable" in result.active_constraints
    assert result.solver == "weighted_bounded_lsq"
    assert result.weighted_residual_norm == pytest.approx(
        np.linalg.norm(np.array([5.0, 2.0]))
    )


def test_multicurve_certificate_reports_honest_node_coordinate_risk() -> None:
    swap, discount, forward = example_swap()
    certificate = multicurve_risk_certificate(swap, discount, forward)
    assert certificate.status == "PASS"
    assert abs(certificate.result.par_repricing_residual) < 1.0e-6
    assert certificate.result.payer_pv == pytest.approx(0.0, abs=1.0e-6)
    assert certificate.result.discount_zero_node_dv01.shape == discount.values.shape
    assert certificate.result.projection_forward_node_dv01.shape == forward.values.shape
    assert certificate.result.node_coordinate_transform.shape == (
        2 * discount.values.size,
        2 * discount.values.size,
    )
    assert all(
        coordinate.startswith(("discount_zero:", "projection_forward:"))
        for coordinate in certificate.result.node_coordinate_ids
    )
    risk_scale = abs(
        np.sum(certificate.result.discount_zero_node_dv01)
        + np.sum(certificate.result.projection_forward_node_dv01)
    )
    assert (
        certificate.result.risk_reconciliation_error / risk_scale
        < 1.0e-7
    )


def test_multicurve_dv01_sign_and_nonpar_contract_are_explicit() -> None:
    swap, discount, forward = example_swap()
    certificate = multicurve_risk_certificate(
        swap,
        discount,
        forward,
        fixed_rate=0.20,
    )
    bump = 1.0e-4
    projection_down = type(forward)(
        forward.times,
        forward.values - np.eye(forward.values.size)[0] * bump,
    )
    projection_up = type(forward)(
        forward.times,
        forward.values + np.eye(forward.values.size)[0] * bump,
    )
    expected = (
        swap.payer_pv(0.20, discount, projection_down)
        - swap.payer_pv(0.20, discount, projection_up)
    ) / 2.0
    assert certificate.result.projection_forward_node_dv01[0] == pytest.approx(
        expected
    )
    assert certificate.result.payer_pv < 0.0
    assert abs(certificate.result.par_repricing_residual) < 1.0e-6
    assert certificate.status == "PASS"


def test_hull_white_certificate_recovers_curve_and_exact_moments() -> None:
    times = np.arange(1.0, 6.0)
    certificate = hull_white_curve_fit_audit(
        curve_times=times,
        market_discount_factors=np.exp(-0.04 * times),
        mean_reversion=0.60,
        volatility=0.012,
        paths=30_000,
        seed=711,
    )
    assert certificate.status == "PASS"
    assert certificate.result.maximum_curve_fit_residual < 5.0e-6
    assert certificate.result.mean_z_score < 4.0
    assert certificate.result.variance_z_score < 4.0


@pytest.mark.parametrize("x0", [0.10, -0.07])
def test_hull_white_nonzero_initial_state_is_bound_to_returned_shift(
    x0: float,
) -> None:
    times = np.arange(1.0, 6.0)
    mean_reversion = 0.60
    volatility = 0.012
    market_dfs = np.exp(-0.04 * times)
    certificate = hull_white_curve_fit_audit(
        curve_times=times,
        market_discount_factors=market_dfs,
        mean_reversion=mean_reversion,
        volatility=volatility,
        x0=x0,
        paths=4_000,
        seed=713,
        z_limit=10.0,
    )
    assert certificate.status == "PASS"
    phi_times = certificate.result.phi_times
    phi = certificate.result.phi_values
    cumulative_phi = np.r_[
        0.0,
        np.cumsum(0.5 * (phi[1:] + phi[:-1]) * np.diff(phi_times)),
    ]
    reconstructed = []
    for maturity in times:
        integrated_phi = np.interp(maturity, phi_times, cumulative_phi)
        integrated_variance = volatility**2 / mean_reversion**2 * (
            maturity
            - 2.0 * (1.0 - np.exp(-mean_reversion * maturity))
            / mean_reversion
            + (1.0 - np.exp(-2.0 * mean_reversion * maturity))
            / (2.0 * mean_reversion)
        )
        b_zero_maturity = (
            1.0 - np.exp(-mean_reversion * maturity)
        ) / mean_reversion
        reconstructed.append(
            np.exp(
                -integrated_phi
                - x0 * b_zero_maturity
                + 0.5 * integrated_variance
            )
        )
    np.testing.assert_allclose(reconstructed, market_dfs, atol=5.0e-6, rtol=5.0e-6)


def test_hjm_discounted_bond_values_pass_martingale_and_refinement_gates() -> None:
    certificate = hjm_martingale_audit(
        maturities=np.array([1.0, 2.0, 4.0]),
        initial_forward_rate=0.035,
        sigma0=0.006,
        decay=0.30,
        paths=3_000,
        seed=712,
        z_limit=5.0,
        refinement_limit=1.0e-3,
    )
    assert certificate.status == "PASS"
    assert certificate.result.maximum_absolute_z_score < 5.0
    assert np.max(certificate.result.refinement_differences) < 1.0e-3
    assert certificate.result.nested_brownian_paths


def test_hjm_coarse_brownian_paths_are_exact_sums_of_fine_paths() -> None:
    coarse, fine = nested_brownian_increments(
        time_steps=8,
        paths=11,
        audit_horizon=0.5,
        seed=712,
    )
    np.testing.assert_array_equal(
        coarse,
        fine.reshape(8, 2, 11).sum(axis=1),
    )


def test_hjm_accepts_finite_continuous_forward_below_minus_one() -> None:
    certificate = hjm_martingale_audit(
        maturities=np.array([0.5, 1.0]),
        initial_forward_rate=-2.0,
        sigma0=0.0,
        decay=0.30,
        audit_horizon=0.10,
        time_steps=8,
        maturity_steps=32,
        paths=200,
        seed=714,
        z_limit=10.0,
        refinement_limit=1.0e-8,
    )
    assert certificate.status == "PASS"


def test_credit_survival_certificate_reconciles_probabilities_and_legs() -> None:
    certificate = credit_survival_certificate(
        FlatHazardCredit(0.03, 0.40, 0.025),
        5.0,
    )
    assert certificate.status == "PASS"
    assert certificate.result.survival_probability + certificate.result.default_probability == pytest.approx(1.0)
    assert certificate.result.risky_zero_price == pytest.approx(
        certificate.result.survival_leg_pv
        + certificate.result.recovery_leg_pv
    )


def test_high_recovery_can_exceed_risk_free_zero_without_false_reject() -> None:
    certificate = credit_survival_certificate(
        FlatHazardCredit(0.10, 1.00, 0.05),
        5.0,
    )
    assert certificate.result.risky_zero_price > certificate.result.risk_free_zero_price
    assert certificate.status == "PASS"


def test_continuously_compounded_negative_rate_has_no_minus_one_domain() -> None:
    certificate = credit_survival_certificate(
        FlatHazardCredit(0.03, 0.40, -1.50),
        1.0,
    )
    assert certificate.status == "PASS"
    with pytest.raises(ValueError):
        FlatHazardCredit(float("nan"), 0.40, 0.02)


def test_cds_bootstrap_includes_accrual_on_default_and_standard_upfront() -> None:
    quotes = (
        CDSQuote(1, 0.0100),
        CDSQuote(3, 0.0100, upfront=0.0125),
        CDSQuote(5, 0.0100, upfront=0.0350),
    )
    discount_curve = flat_discount_curve(
        continuous_rate=0.03,
        horizon=5.0,
    )
    certificate = bootstrap_cds_certificate(
        quotes,
        discount_curve=discount_curve,
        discount_curve_run_id="test-flat-curve",
    )
    assert certificate.status == "PASS"
    assert certificate.result.curve is not None
    assert certificate.result.maximum_repricing_residual < 1.0e-10
    legs = cds_legs(
        certificate.result.curve,
        quotes[-1],
        0.40,
        discount_curve,
    )
    assert legs.accrued_on_default > 0.0
    assert abs(legs.net_pv) < 1.0e-10


def test_cds_recovery_sensitivity_changes_calibrated_hazard() -> None:
    quotes = (CDSQuote(1, 0.01), CDSQuote(3, 0.015), CDSQuote(5, 0.018))
    discount_curve = flat_discount_curve(
        continuous_rate=0.03,
        horizon=5.0,
    )
    low_recovery = bootstrap_cds_certificate(
        quotes,
        recovery=0.20,
        discount_curve=discount_curve,
        discount_curve_run_id="test-flat-curve",
    )
    high_recovery = bootstrap_cds_certificate(
        quotes,
        recovery=0.60,
        discount_curve=discount_curve,
        discount_curve_run_id="test-flat-curve",
    )
    assert low_recovery.status == "PASS"
    assert high_recovery.status == "PASS"
    assert (
        high_recovery.result.calibration_nodes[-1].hazard
        > low_recovery.result.calibration_nodes[-1].hazard
    )


def test_cds_bootstrap_uses_full_discount_term_structure_and_lineage() -> None:
    quotes = (CDSQuote(1, 0.01), CDSQuote(3, 0.015), CDSQuote(5, 0.018))
    flat = flat_discount_curve(continuous_rate=0.03, horizon=5.0)
    shaped = BootstrapDiscountCurve(
        times=np.array([1.0, 3.0, 5.0]),
        discount_factors=np.exp(-np.array([0.01, 0.03, 0.06]) * np.array([1.0, 3.0, 5.0])),
    )
    flat_certificate = bootstrap_cds_certificate(
        quotes,
        discount_curve=flat,
        discount_curve_run_id="flat-run",
    )
    shaped_certificate = bootstrap_cds_certificate(
        quotes,
        discount_curve=shaped,
        discount_curve_run_id="shaped-run",
    )
    assert flat_certificate.status == "PASS"
    assert shaped_certificate.status == "PASS"
    assert shaped_certificate.result.discount_curve_run_id == "shaped-run"
    assert not np.allclose(
        [node.hazard for node in flat_certificate.result.calibration_nodes],
        [node.hazard for node in shaped_certificate.result.calibration_nodes],
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_public_flat_curve_cds_adapters_reprice_their_quotes() -> None:
    quotes = {1: 0.0100, 3: 0.0150, 5: 0.0180}
    curve = bootstrap_hazard_curve(quotes, recovery=0.40, discount_rate=0.03)
    for maturity, spread in quotes.items():
        assert abs(cds_pv(curve, maturity, spread, 0.40, 0.03)) < 1.0e-10


def test_merton_debt_reconciles_by_put_call_parity() -> None:
    certificate = merton_credit_certificate(
        asset_value=150.0,
        debt_face=100.0,
        asset_volatility=0.30,
        risk_free_rate=0.03,
        maturity=1.0,
        physical_asset_drift=0.08,
    )
    assert certificate.status == "PASS"
    assert certificate.result.put_call_parity_residual < 1.0e-12
    assert certificate.result.physical_distance_to_default is not None
    assert 0.0 < certificate.result.risk_neutral_default_probability < 1.0
    wrapper = merton_equity_and_default_probability(
        150.0, 100.0, 0.30, 0.03, 1.0
    )
    assert wrapper["equity_value"] == pytest.approx(
        certificate.result.equity_value
    )
    assert 0.0 < wrapper["risk_neutral_default_probability"] < 1.0
    assert certificate.provenance is not None
    assert certificate.provenance.run_id.endswith(
        certificate.input_manifest_hash[:16]
    )
    assert certificate.provenance.artifact_hashes == (
        certificate.input_manifest_hash,
    )


def test_callable_certificate_calibrates_curve_recovers_oas_and_reports_exercise() -> None:
    bond = CallableBond(100.0, 0.06, 5, 100.0, 2)
    market_dfs = np.exp(-0.04 * np.arange(1.0, 6.0))
    tree = calibrate_additive_short_rate_tree(market_dfs, 0.01)
    market_price = price_on_calibrated_tree(
        bond,
        tree,
        callable_=True,
        oas=0.0125,
    ).price
    certificate = callable_bond_oas_certificate(
        bond=bond,
        market_discount_factors=market_dfs,
        rate_step=0.01,
        market_price=market_price,
        market_price_source="self_roundtrip",
    )
    assert certificate.status == "CONDITIONAL_PASS"
    assert certificate.result.market_price_source == "self_roundtrip"
    assert any(
        violation.code == "OAS_NONINDEPENDENT_MARKET_PRICE"
        for violation in certificate.violations
    )
    assert certificate.result.oas == pytest.approx(0.0125, abs=1.0e-10)
    assert certificate.result.oas_repricing_residual == pytest.approx(
        0.0,
        abs=1.0e-9,
    )
    assert certificate.result.callable_price <= certificate.result.straight_price
    assert certificate.result.calibrated_tree.maximum_curve_residual < 1.0e-12
    assert np.sum(certificate.result.exercise_probability_by_year) >= 0.0


def test_callable_certificate_requires_hash_for_independent_benchmark() -> None:
    bond = CallableBond(100.0, 0.06, 5, 100.0, 2)
    market_dfs = np.exp(-0.04 * np.arange(1.0, 6.0))
    tree = calibrate_additive_short_rate_tree(market_dfs, 0.01)
    reference_price = price_on_calibrated_tree(
        bond,
        tree,
        callable_=True,
        oas=0.0125,
    ).price
    with pytest.raises(ValueError, match="SHA-256"):
        callable_bond_oas_certificate(
            bond=bond,
            market_discount_factors=market_dfs,
            rate_step=0.01,
            market_price=reference_price,
            market_price_source="independent",
        )
    # Independent two-period hand check: with a zero-width tree, the issuer
    # calls at year one because 110*exp(-4%) > 100; the time-zero value is the
    # discounted year-one call price plus coupon, 110*exp(-4%).
    hand_bond = CallableBond(100.0, 0.10, 2, 100.0, 1)
    hand_dfs = np.exp(-0.04 * np.arange(1.0, 3.0))
    hand_price = 110.0 * exp(-0.04)
    evidence_hash = stable_manifest_hash(
        {
            "reference": "two-period hand calculation",
            "formula": "(face + coupon) * exp(-r)",
            "face": 100.0,
            "coupon": 10.0,
            "rate": 0.04,
        }
    )
    certificate = callable_bond_oas_certificate(
        bond=hand_bond,
        market_discount_factors=hand_dfs,
        rate_step=0.0,
        market_price=hand_price,
        market_price_source="independent",
        market_price_evidence_hash=evidence_hash,
    )
    assert certificate.status == "PASS"
    assert certificate.result.oas == pytest.approx(0.0, abs=1.0e-10)
    assert certificate.result.callable_price == pytest.approx(hand_price)


def test_callable_boundary_is_highest_exercised_rate_and_reconstructs_mask() -> None:
    bond = CallableBond(100.0, 0.06, 5, 100.0, 2)
    market_dfs = np.exp(-0.04 * np.arange(1.0, 6.0))
    tree = calibrate_additive_short_rate_tree(market_dfs, 0.01)
    result = price_on_calibrated_tree(
        bond,
        tree,
        callable_=True,
        oas=0.0125,
    )
    for step, (mask, boundary, first_nonexercise) in enumerate(
        zip(
            result.exercise_masks,
            result.exercise_boundary_rate_by_year,
            result.first_nonexercise_rate_by_year,
            strict=True,
        )
    ):
        rates = tree.level_shifts[step] + (
            2.0 * np.arange(step + 1) - step
        ) * tree.rate_step
        if np.any(mask):
            assert boundary == pytest.approx(float(np.max(rates[mask])))
            np.testing.assert_array_equal(mask, rates <= boundary + 1.0e-15)
            if np.any(~mask):
                assert first_nonexercise == pytest.approx(
                    float(np.min(rates[~mask]))
                )
                assert boundary < first_nonexercise
        else:
            assert boundary is None
    assert "rate <= boundary" in result.exercise_rule


def test_public_callable_flat_curve_helpers_are_exercised() -> None:
    bond = CallableBond(100.0, 0.06, 5, 100.0, 2)
    price = price_on_additive_short_rate_tree(
        bond,
        initial_rate=0.04,
        rate_step=0.01,
        callable_=True,
    )
    duration = effective_duration(
        bond,
        initial_rate=0.04,
        rate_step=0.01,
        callable_=True,
    )
    assert price > 0.0
    assert np.isfinite(duration)
    assert duration > 0.0


def test_call_date_at_maturity_is_rejected_as_ambiguous_redemption() -> None:
    with pytest.raises(ValueError, match="before maturity"):
        CallableBond(100.0, 0.06, 5, 100.0, 5)


def _pnl_inputs(actual_pnl: float, *, state_id: str = "state-1") -> PnLInputs:
    position_hash = stable_manifest_hash(
        {"state_id": state_id, "artifact_type": "position"}
    )
    market_hash = stable_manifest_hash(
        {"state_id": state_id, "artifact_type": "market"}
    )
    risk_hash = stable_manifest_hash(
        {"state_id": state_id, "artifact_type": "risk"}
    )
    return PnLInputs(
        carry=100.0,
        roll_down=20.0,
        curve_dv01=np.array([10.0, 20.0]),
        curve_move_bp=np.array([1.0, -1.0]),
        curve_cross_gamma=np.diag([0.2, 0.1]),
        basis_pnl=5.0,
        cs01=2.0,
        credit_spread_move_bp=3.0,
        credit_convexity=0.1,
        volatility_pnl=1.0,
        cash_and_fixings_pnl=2.0,
        exercise_pnl=0.0,
        model_change_pnl=0.0,
        data_correction_pnl=0.0,
        actual_pnl=actual_pnl,
        position_manifest=StateManifest(
            "position", "position-run-1", position_hash, (), 1
        ),
        risk_manifest=StateManifest(
            "risk",
            "risk-run-1",
            risk_hash,
            (position_hash, market_hash),
            2,
        ),
        market_manifest=StateManifest(
            "market", "market-run-1", market_hash, (), 2
        ),
        component_evidence_ids={
            name: (f"evidence:{name}",)
            for name in (
                "carry",
                "roll_down",
                "curve_delta",
                "curve_cross_gamma",
                "basis",
                "credit_delta",
                "credit_convexity",
                "volatility",
                "cash_and_fixings",
                "exercise",
                "model_change",
                "data_correction",
            )
        },
    )


def test_pnl_certificate_status_depends_on_real_residual_and_completeness() -> None:
    passing = explain_certificate(
        _pnl_inputs(132.60),
        absolute_residual_limit=0.01,
        relative_residual_limit=0.001,
    )
    assert passing.status == "PASS"
    assert passing.result.residual == pytest.approx(0.0)
    failing = explain_certificate(
        _pnl_inputs(150.0),
        absolute_residual_limit=5.0,
        relative_residual_limit=0.02,
    )
    assert failing.status == "REJECT"
    assert failing.result.residual_fraction_of_actual > 0.02


def test_pnl_requires_dependency_linked_state_manifests() -> None:
    inputs = _pnl_inputs(132.60)
    broken_risk = replace(inputs.risk_manifest, dependency_hashes=())
    certificate = explain_certificate(
        replace(inputs, risk_manifest=broken_risk),
        absolute_residual_limit=0.01,
        relative_residual_limit=0.001,
    )
    assert certificate.status == "REJECT"
    assert any(
        violation.code == "PNL_SNAPSHOT_ALIGNMENT"
        for violation in certificate.violations
    )


def test_pnl_zero_components_still_require_evidence_or_reason() -> None:
    inputs = _pnl_inputs(132.60)
    incomplete = dict(inputs.component_evidence_ids)
    incomplete.pop("exercise")
    with pytest.raises(ValueError, match="exercise requires evidence"):
        replace(inputs, component_evidence_ids=incomplete)


def test_pnl_policy_changes_manifest_hash_and_preview_is_ungoverned() -> None:
    inputs = _pnl_inputs(132.60)
    strict = explain_certificate(
        inputs,
        absolute_residual_limit=0.01,
        relative_residual_limit=0.001,
    )
    loose = explain_certificate(
        inputs,
        absolute_residual_limit=1.0,
        relative_residual_limit=0.01,
    )
    assert strict.input_manifest_hash != loose.input_manifest_hash
    assert explain(inputs)["status"] == "UNGOVERNED"
    with pytest.raises(ValueError, match="finite"):
        TolerancePolicy(float("inf"), 0.0)


def test_all_previously_omitted_controls_change_certificate_hashes() -> None:
    curve = RepresentationCurve(
        np.array([1.0, 2.0]),
        np.exp(-0.03 * np.array([1.0, 2.0])),
    )
    assert (
        curve_representation_certificate(curve).input_manifest_hash
        != curve_representation_certificate(
            curve,
            tolerance=TolerancePolicy(2.0e-12, 1.0e-10),
        ).input_manifest_hash
    )

    swap, discount, projection = example_swap()
    assert (
        multicurve_risk_certificate(swap, discount, projection).input_manifest_hash
        != multicurve_risk_certificate(
            swap,
            discount,
            projection,
            tolerance=TolerancePolicy(2.0e-7, 1.0e-6),
        ).input_manifest_hash
    )

    market_dfs = np.exp(-0.04 * np.arange(1.0, 6.0))
    hw = hull_white_curve_fit_audit(
        curve_times=np.arange(1.0, 6.0),
        market_discount_factors=market_dfs,
        mean_reversion=0.60,
        volatility=0.012,
        paths=800,
        seed=717,
        z_limit=50.0,
        integration_points=501,
    )
    hw_changed = hull_white_curve_fit_audit(
        curve_times=np.arange(1.0, 6.0),
        market_discount_factors=market_dfs,
        mean_reversion=0.60,
        volatility=0.012,
        paths=800,
        seed=717,
        z_limit=60.0,
        integration_points=501,
    )
    assert hw.input_manifest_hash != hw_changed.input_manifest_hash

    hjm_default = hjm_martingale_audit(
        maturities=np.array([1.0, 2.0]),
        initial_forward_rate=0.03,
        sigma0=0.005,
        decay=0.25,
        paths=600,
        seed=719,
        z_limit=50.0,
        refinement_limit=0.1,
    )
    hjm_changed = hjm_martingale_audit(
        maturities=np.array([1.0, 2.0]),
        initial_forward_rate=0.03,
        sigma0=0.005,
        decay=0.25,
        paths=600,
        seed=719,
        z_limit=50.0,
        refinement_limit=0.1,
        tolerance=TolerancePolicy(2.0e-10, 1.0e-8),
    )
    assert hjm_default.input_manifest_hash != hjm_changed.input_manifest_hash

    callable_bond = CallableBond(100.0, 0.06, 5, 100.0, 2)
    tree = calibrate_additive_short_rate_tree(market_dfs, 0.01)
    market_price = price_on_calibrated_tree(
        callable_bond,
        tree,
        callable_=True,
        oas=0.0125,
    ).price
    callable_default = callable_bond_oas_certificate(
        bond=callable_bond,
        market_discount_factors=market_dfs,
        rate_step=0.01,
        market_price=market_price,
        market_price_source="self_roundtrip",
    )
    callable_changed = callable_bond_oas_certificate(
        bond=callable_bond,
        market_discount_factors=market_dfs,
        rate_step=0.01,
        market_price=market_price,
        market_price_source="self_roundtrip",
        initial_oas_bracket=(-0.04, 0.04),
    )
    assert callable_default.input_manifest_hash != callable_changed.input_manifest_hash

    cds_quotes = (CDSQuote(1, 0.01), CDSQuote(3, 0.015))
    discount_curve = flat_discount_curve(continuous_rate=0.03, horizon=3.0)
    cds_default = bootstrap_cds_certificate(
        cds_quotes,
        discount_curve=discount_curve,
        discount_curve_run_id="hash-test-curve",
    )
    cds_changed = bootstrap_cds_certificate(
        cds_quotes,
        discount_curve=discount_curve,
        discount_curve_run_id="hash-test-curve",
        maximum_hazard=120.0,
    )
    assert cds_default.input_manifest_hash != cds_changed.input_manifest_hash

    merton_default = merton_credit_certificate(
        asset_value=150.0,
        debt_face=100.0,
        asset_volatility=0.30,
        risk_free_rate=0.03,
        maturity=1.0,
    )
    merton_changed = merton_credit_certificate(
        asset_value=150.0,
        debt_face=100.0,
        asset_volatility=0.30,
        risk_free_rate=0.03,
        maturity=1.0,
        tolerance=TolerancePolicy(2.0e-12, 1.0e-10),
    )
    assert merton_default.input_manifest_hash != merton_changed.input_manifest_hash


@pytest.mark.skip(reason="requires non-public publication provenance assets")
def test_integrated_case_rejects_release_for_material_pnl_residual() -> None:
    result = chapter_release_case()
    assert result["curve_maximum_repricing_residual"] < 1.0e-10
    assert result["cds_maximum_repricing_residual"] < 1.0e-10
    assert result["oas"] == pytest.approx(0.0125, abs=1.0e-10)
    assert result["pnl_residual_fraction"] > 0.05
    assert result["component_statuses"]["pnl_python"] == "REJECT"
    assert set(result["component_statuses"]) == REQUIRED_ARTIFACT_IDS
    assert result["component_statuses"]["asof_sql"] == "PASS"
    assert result["component_statuses"]["pnl_sql"] == "PASS"
    assert result["component_statuses"]["cpp_risk_kernel"] == "PASS"
    assert result["component_statuses"]["callable_oas"] == "CONDITIONAL_PASS"
    assert result["final_decision"] == "REJECT"


def test_release_reducer_fails_closed_for_every_required_gate() -> None:
    passing = tuple(
        ReleaseGate(
            artifact_id=artifact_id,
            required=True,
            status="PASS",
            evidence_hash=stable_manifest_hash({"artifact_id": artifact_id}),
            owner="test owner",
            observed=True,
            operator="==",
            limit=True,
            unit="boolean",
        )
        for artifact_id in sorted(REQUIRED_ARTIFACT_IDS)
    )
    assert reduce_release_gates(passing) == "PASS"
    for index, gate in enumerate(passing):
        assert reduce_release_gates(passing[:index] + passing[index + 1 :]) == "REJECT"
        rejected = passing[:index] + (
            replace(gate, status="REJECT"),
        ) + passing[index + 1 :]
        assert reduce_release_gates(rejected) == "REJECT"
    assert reduce_release_gates(
        (replace(passing[0], status="CONDITIONAL_PASS"), *passing[1:])
    ) == "CONDITIONAL_PASS"


@pytest.mark.skip(reason="requires non-public publication provenance assets")
def test_integrated_hashes_bind_market_cells_policies_and_results() -> None:
    baseline = chapter_release_case()
    shifted = chapter_release_case(projection_first_node_shift=0.05)
    policy_changed = chapter_release_case(
        policy_overrides={"curve_policy": "FI-CURVE-v4-test"}
    )
    assert baseline["market_snapshot_hash"] != shifted["market_snapshot_hash"]
    assert baseline["release_bundle_hash"] != shifted["release_bundle_hash"]
    assert (
        baseline["release_bundle_hash"]
        != policy_changed["release_bundle_hash"]
    )
    assert (
        baseline["release_bundle"]["source_revision"]["kind"]
        == "local_content_addressed_source_tree"
    )
    assert len(baseline["release_bundle"]["source_revision"]["source_hashes"]) >= 19


def test_asof_sql_selects_curve_quotes_fixings_and_manifest_without_future_leakage() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE market_quote_history (
            instrument_id TEXT, curve_id TEXT, tenor TEXT, quote_type TEXT,
            quote_value REAL, valid_from TEXT, valid_to TEXT,
            known_from TEXT, known_to TEXT, revision_id INTEGER,
            source_priority INTEGER, is_cancelled INTEGER, row_hash TEXT
        );
        CREATE TABLE fixing_history (
            index_id TEXT, fixing_date TEXT, fixing_value REAL,
            valid_from TEXT, valid_to TEXT, known_from TEXT, known_to TEXT,
            revision_id INTEGER, source_priority INTEGER, is_cancelled INTEGER,
            row_hash TEXT
        );
        CREATE TABLE expected_market_data (
            record_type TEXT, economic_key TEXT
        );
        """
    )
    connection.executemany(
        "INSERT INTO market_quote_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                "IRS5Y", "OIS", "5Y", "PAR", 0.0400,
                "2026-07-30 08:00", None,
                "2026-07-30 08:00", "2026-07-30 09:00",
                1, 10, 0, "a" * 64,
            ),
            (
                "IRS5Y", "OIS", "5Y", "PAR", 0.0410,
                "2026-07-30 08:00", None,
                "2026-07-30 09:00", None,
                2, 10, 0, "b" * 64,
            ),
        ],
    )
    connection.execute(
        "INSERT INTO fixing_history VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "SOFR", "2026-07-29", 0.0435,
            "2026-07-29 00:00", None,
            "2026-07-30 07:30", None,
            1, 10, 0, "c" * 64,
        ),
    )
    connection.executemany(
        "INSERT INTO expected_market_data VALUES (?,?)",
        [
            ("QUOTE", "IRS5Y|OIS|5Y|PAR"),
            ("FIXING", "SOFR|2026-07-29"),
        ],
    )
    sql = (
        ROOT / "_book/sql/ch07/fixed_income_asof_market_data.sql"
    ).read_text(encoding="utf-8")
    rows = connection.execute(
        sql,
        {
            "valuation_time": "2026-07-30 08:30",
            "knowledge_cutoff": "2026-07-30 08:30",
        },
    ).fetchall()
    assert len(rows) == 2
    assert {row[0] for row in rows} == {"QUOTE", "FIXING"}
    quote = next(row for row in rows if row[0] == "QUOTE")
    assert quote[2] == pytest.approx(0.0400)
    assert all(row[6] == 2 for row in rows)
    assert rows[0][9] == rows[1][9]
    assert all(row[-1] == "PASS" for row in rows)


def test_pnl_sql_preserves_complete_position_and_returns_pass() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE position_pnl (
            valuation_date TEXT, book_id TEXT, instrument_id TEXT,
            state_id TEXT, position_run_id TEXT, risk_run_id TEXT,
            market_run_id TEXT, actual_pnl REAL, carry_pnl REAL,
            roll_down_pnl REAL, basis_pnl REAL, cs01 REAL,
            credit_spread_move_bp REAL, credit_convexity REAL,
            volatility_pnl REAL, cash_and_fixings_pnl REAL,
            exercise_pnl REAL, model_change_pnl REAL,
            data_correction_pnl REAL, expected_curve_bucket_count INTEGER,
            expected_gamma_pair_count INTEGER, stale_input_count INTEGER
        );
        CREATE TABLE curve_risk_bucket (
            valuation_date TEXT, book_id TEXT, instrument_id TEXT,
            state_id TEXT, risk_run_id TEXT, curve_id TEXT,
            key_rate_tenor TEXT, dv01 REAL
        );
        CREATE TABLE curve_market_move (
            valuation_date TEXT, state_id TEXT, market_run_id TEXT,
            curve_id TEXT,
            key_rate_tenor TEXT, curve_move_bp REAL
        );
        CREATE TABLE curve_cross_gamma (
            valuation_date TEXT, book_id TEXT, instrument_id TEXT,
            state_id TEXT, risk_run_id TEXT,
            curve_id_1 TEXT, key_rate_tenor_1 TEXT,
            curve_id_2 TEXT, key_rate_tenor_2 TEXT, cross_gamma REAL
        );
        """
    )
    connection.execute(
        "INSERT INTO position_pnl VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "2026-07-30", "BOOK1", "BOND1", "state-1",
            "pos-1", "risk-1", "market-1", 132.60, 100.0, 20.0,
            5.0, 2.0, 3.0, 0.1, 1.0, 2.0, 0.0, 0.0, 0.0, 2, 2, 0,
        ),
    )
    connection.executemany(
        "INSERT INTO curve_risk_bucket VALUES (?,?,?,?,?,?,?,?)",
        [
            ("2026-07-30", "BOOK1", "BOND1", "state-1", "risk-1", "OIS", "1Y", 10.0),
            ("2026-07-30", "BOOK1", "BOND1", "state-1", "risk-1", "OIS", "2Y", 20.0),
        ],
    )
    connection.executemany(
        "INSERT INTO curve_market_move VALUES (?,?,?,?,?,?)",
        [
            ("2026-07-30", "state-1", "market-1", "OIS", "1Y", 1.0),
            ("2026-07-30", "state-1", "market-1", "OIS", "2Y", -1.0),
        ],
    )
    connection.executemany(
        "INSERT INTO curve_cross_gamma VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (
                "2026-07-30", "BOOK1", "BOND1", "state-1", "risk-1",
                "OIS", "1Y", "OIS", "1Y", 0.2,
            ),
            (
                "2026-07-30", "BOOK1", "BOND1", "state-1", "risk-1",
                "OIS", "2Y", "OIS", "2Y", 0.1,
            ),
        ],
    )
    sql = (
        ROOT / "_book/sql/ch07/fixed_income_pnl_reconciliation.sql"
    ).read_text(encoding="utf-8")
    row = connection.execute(
        sql,
        {
            "absolute_residual_limit": 0.01,
            "relative_residual_limit": 0.001,
            "pnl_floor": 1.0,
        },
    ).fetchone()
    assert row is not None
    assert row[-1] == "PASS"
    assert row[9] == pytest.approx(0.0)
