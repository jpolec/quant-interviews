"""Executable invariants for Chapter 2."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.linalg import hilbert

from quant_interviews.ch02.case_study import governed_risk_model_case
from quant_interviews.ch02.covariance import (
    condition_diagnostics,
    covariance_psd_certificate,
    equicorrelation_condition_number,
    equicorrelation_eigenvalues,
    governed_pca_components,
    pca_risk_decomposition,
)
from quant_interviews.ch02.factor import simulate_correlated, woodbury_solve
from quant_interviews.ch02.optimization import (
    long_only_markowitz_audit,
    markowitz_kkt,
    turnover_regularized,
)
from quant_interviews.ch02.regularization import (
    linear_shrinkage,
    nearest_correlation,
    oas_shrinkage_stability,
)
from quant_interviews.ch02.risk import risk_contributions, risk_parity
from quant_interviews.ch02.sparse import (
    allocation_stability_metrics,
    gross_exposure_constrained_portfolio,
    long_only_l1_constant_audit,
    long_only_l1_norm,
    norm_constrained_portfolio_comparison,
    project_l2_ball,
    soft_threshold,
)
from quant_interviews.ch02.updates import (
    least_squares_svd_audit,
    rank_one_inverse_core,
    sherman_morrison_inverse,
    solve_audit,
    symmetric_rank_one_update_audit,
)


def test_covariance_psd_certificate_exposes_rank_and_scale() -> None:
    covariance = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
        ]
    )

    certificate = covariance_psd_certificate(covariance)

    assert certificate["is_psd"]
    assert certificate["effective_rank"] == 2
    assert certificate["minimum_eigenvalue"] == pytest.approx(0.0, abs=1e-14)


def test_covariance_psd_certificate_rejects_nonfinite_or_asymmetric_input() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        covariance_psd_certificate([[1.0, 0.2], [0.3, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        covariance_psd_certificate([[1.0, np.nan], [np.nan, 1.0]])


def test_covariance_psd_floor_scales_with_daily_return_covariance() -> None:
    covariance = np.diag([1e-4, 1e-12])

    certificate = covariance_psd_certificate(
        covariance,
        absolute_tolerance=0.0,
        relative_tolerance=1e-8,
    )

    assert certificate["numerical_floor"] == pytest.approx(1e-12)
    assert certificate["effective_rank"] == 1


def test_covariance_symmetry_gate_is_scale_aware() -> None:
    covariance = np.array([[20_000.0, 3_000.0 + 1e-11], [3_000.0, 10_000.0]])

    certificate = covariance_psd_certificate(
        covariance,
        absolute_tolerance=1e-14,
        relative_tolerance=1e-14,
    )

    assert certificate["is_psd"]


@pytest.mark.parametrize("bad_tolerance", [np.nan, np.inf, -np.inf])
def test_covariance_psd_certificate_rejects_nonfinite_tolerances(
    bad_tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="tolerances"):
        covariance_psd_certificate(
            np.eye(2),
            relative_tolerance=bad_tolerance,
        )


def test_equicorrelation_spectrum_and_negative_rho_condition_number() -> None:
    eigenvalues = equicorrelation_eigenvalues(4, 0.2)

    assert np.sort(eigenvalues) == pytest.approx([0.8, 0.8, 0.8, 1.6])
    assert equicorrelation_condition_number(2, -0.8) == pytest.approx(9.0)
    assert math.isinf(equicorrelation_condition_number(2, -1.0))


def test_conditioning_reports_effective_rank_without_inverse() -> None:
    diagnostics = condition_diagnostics([[1.0, 0.999], [0.999, 1.0]])

    assert diagnostics["condition_number"] == pytest.approx(1_999.0)
    assert diagnostics["effective_rank"] == 2


def test_pca_decomposition_reconciles_direct_portfolio_variance() -> None:
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    weights = np.array([0.6, 0.4])

    decomposition = pca_risk_decomposition(covariance, weights)

    assert decomposition["total_variance"] == pytest.approx(
        weights @ covariance @ weights
    )
    assert np.sum(decomposition["variance_contributions"]) == pytest.approx(
        decomposition["total_variance"]
    )
    assert decomposition["reconciliation_error"] < 1e-14


def test_pca_clips_only_numerical_negative_risk_directions() -> None:
    covariance = np.diag([1e-4, -1e-16])

    decomposition = pca_risk_decomposition(covariance, [0.5, 0.5])

    assert np.min(decomposition["raw_eigenvalues"]) < 0.0
    assert np.min(decomposition["used_eigenvalues"]) == 0.0
    assert np.all(decomposition["variance_contributions"] >= 0.0)
    assert decomposition["numerical_repair_norm"] == pytest.approx(1e-16)


def test_governed_pca_core_rejects_material_negative_eigenvalues() -> None:
    allowed = governed_pca_components(
        np.array([-1e-16, 1.0]),
        np.eye(2),
        np.array([0.5, 0.5]),
        numerical_floor=1e-12,
    )

    assert allowed[1] == pytest.approx([1.0, 0.0])
    with pytest.raises(ValueError, match="materially indefinite"):
        governed_pca_components(
            np.array([-0.5, 1.0]),
            np.eye(2),
            np.array([0.5, 0.5]),
            numerical_floor=1e-12,
        )


def test_woodbury_solve_matches_dense_solve_with_psd_factor_covariance() -> None:
    specific_variance = np.array([0.7, 1.1, 0.9, 1.4])
    loadings = np.array(
        [
            [1.0, 0.2],
            [0.5, -0.3],
            [-0.2, 0.8],
            [0.1, 0.4],
        ]
    )
    factor_covariance = np.diag([0.3, 0.0])
    rhs = np.array([1.0, -0.5, 0.25, 0.8])
    full_covariance = (
        np.diag(specific_variance)
        + loadings @ factor_covariance @ loadings.T
    )

    result = woodbury_solve(
        specific_variance,
        loadings,
        factor_covariance,
        rhs,
    )

    assert result["solution"] == pytest.approx(
        np.linalg.solve(full_covariance, rhs),
        abs=1e-13,
    )
    assert result["relative_residual"] < 1e-14
    assert result["factor_rank"] == 1


def test_woodbury_solve_rejects_nonpositive_specific_risk() -> None:
    with pytest.raises(ValueError, match="positive specific"):
        woodbury_solve(
            [1.0, 0.0],
            [[1.0], [0.5]],
            [[0.2]],
            [1.0, 1.0],
        )


def test_woodbury_zero_factor_covariance_reduces_to_specific_risk() -> None:
    specific_variance = np.array([0.5, 2.0])

    result = woodbury_solve(
        specific_variance,
        [[1.0, 0.0], [0.0, 1.0]],
        np.zeros((2, 2)),
        [1.0, -1.0],
    )

    assert result["factor_rank"] == 0
    assert result["factor_system_condition_number"] == 1.0
    assert result["solution"] == pytest.approx([2.0, -0.5])


def test_correlated_simulation_reports_factorization_and_empirical_fit() -> None:
    correlation = np.array(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0, -0.5],
            [0.5, -0.5, 1.0],
        ]
    )

    evidence = simulate_correlated(correlation, draws=30_000, seed=1)
    empirical = np.corrcoef(np.asarray(evidence["samples"]).T)

    assert evidence["factorization_method"] == "eigen-root"
    assert evidence["effective_rank"] == 2
    assert evidence["repair_frobenius_norm"] < 1e-12
    assert empirical == pytest.approx(correlation, abs=0.02)


def test_correlated_simulation_seed_is_reproducible() -> None:
    correlation = np.array([[1.0, 0.4], [0.4, 1.0]])

    first = simulate_correlated(correlation, draws=50, seed=123)
    second = simulate_correlated(correlation, draws=50, seed=123)

    assert np.array_equal(first["samples"], second["samples"])


def test_correlated_simulation_rejects_material_repair_or_bad_diagonal() -> None:
    with pytest.raises(ValueError, match="materially indefinite"):
        simulate_correlated(
            [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]],
            draws=100,
        )
    with pytest.raises(ValueError, match="diagonal"):
        simulate_correlated([[1.0, 0.2], [0.2, 0.9]], draws=100)


def test_correlated_simulation_enforces_signed_repair_materiality() -> None:
    almost_correlation = np.array(
        [[1.0, 1.0 + 5e-11], [1.0 + 5e-11, 1.0]]
    )

    evidence = simulate_correlated(
        almost_correlation,
        draws=100,
        maximum_repair_norm=1e-12,
    )

    assert evidence["status"] == "REJECT"
    assert np.asarray(evidence["samples"]).shape == (0, 2)


def _short_panel() -> np.ndarray:
    generator = np.random.default_rng(7)
    observations = 45
    assets = 25
    factors = generator.normal(size=(observations, 3))
    loadings = generator.normal(scale=0.5, size=(3, assets))
    idiosyncratic = generator.normal(scale=0.08, size=(observations, assets))
    return factors @ loadings + idiosyncratic


def test_oas_shrinkage_improves_conditioning_and_gmv_stability() -> None:
    result = oas_shrinkage_stability(_short_panel())

    assert 0.0 < result["intensity"] <= 1.0
    assert result["shrunk_condition_number"] < result["sample_condition_number"]
    assert result["shrunk_gross_leverage"] < result["sample_gross_leverage"]
    assert np.sum(result["shrunk_gmv_weights"]) == pytest.approx(1.0)
    assert result["intensity"] == pytest.approx(0.08931579914032077)


def test_oas_handles_singular_raw_covariance_without_forcing_gmv() -> None:
    returns = np.random.default_rng(3).normal(size=(6, 10))

    result = oas_shrinkage_stability(returns)

    assert result["sample_covariance_divisor"] == 6
    assert result["sample_gmv_status"] == "UNAVAILABLE"
    assert math.isinf(result["sample_condition_number"])
    assert result["shrunk_gmv_status"] == "AVAILABLE"
    assert np.sum(result["shrunk_gmv_weights"]) == pytest.approx(1.0)


def test_nearest_correlation_matches_higham_reference_fixture() -> None:
    invalid = np.array(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ]
    )

    result = nearest_correlation(invalid, tolerance=1e-11)
    repaired = np.asarray(result["correlation"])
    expected = np.array(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0, -0.5],
            [0.5, -0.5, 1.0],
        ]
    )

    assert result["converged"]
    assert result["status"] == "PASS"
    assert result["diagonal_error"] <= 1e-11
    assert result["psd_violation"] <= 1e-11
    assert np.diag(repaired) == pytest.approx(1.0)
    assert repaired == pytest.approx(expected, abs=2e-10)
    assert result["minimum_eigenvalue"] >= -2e-10


def test_nearest_correlation_exposes_nonconvergence_as_conditional() -> None:
    invalid = np.array(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ]
    )

    result = nearest_correlation(invalid, max_iterations=1)

    assert not result["converged"]
    assert result["status"] in {"CONDITIONAL_PASS", "REJECT"}
    assert result["iterations"] == 1


def test_markowitz_kkt_reconciles_residuals_and_feasible_perturbations() -> None:
    covariance = np.array(
        [
            [0.05, 0.01, 0.00, 0.00],
            [0.01, 0.08, 0.01, 0.00],
            [0.00, 0.01, 0.06, 0.01],
            [0.00, 0.00, 0.01, 0.10],
        ]
    )
    expected_returns = np.array([0.03, 0.05, 0.08, 0.11])
    constraints = np.column_stack([np.ones(4), expected_returns])
    targets = np.array([1.0, 0.07])

    result = markowitz_kkt(covariance, constraints, targets)
    weights = np.asarray(result["weights"])
    feasible_direction = np.array([1.0, -2.0, 1.0, 0.0])
    feasible_direction -= constraints @ np.linalg.solve(
        constraints.T @ constraints,
        constraints.T @ feasible_direction,
    )

    assert result["primal_residual"] < 1e-13
    assert result["stationarity_residual"] < 1e-13
    for scale in (-0.1, 0.1):
        alternative = weights + scale * feasible_direction
        alternative_objective = 0.5 * alternative @ covariance @ alternative
        assert alternative_objective >= result["objective"] - 1e-14


def test_markowitz_kkt_certificate_is_numerically_closed() -> None:
    covariance = np.diag([0.04, 0.06, 0.09])
    constraints = np.column_stack([np.ones(3), [0.02, 0.05, 0.09]])

    result = markowitz_kkt(covariance, constraints, [1.0, 0.05])

    assert result["primal_residual"] < 1e-13
    assert result["stationarity_residual"] < 1e-13
    assert constraints.T @ result["weights"] == pytest.approx([1.0, 0.05])


def test_long_only_markowitz_returns_full_kkt_audit() -> None:
    covariance = np.diag([0.04, 0.05, 0.07, 0.09])
    expected_returns = np.array([0.30, 0.09, 0.04, 0.01])
    factor = np.array([[-1.0], [-0.5], [0.5], [1.0]])

    result = long_only_markowitz_audit(
        covariance,
        expected_returns,
        risk_aversion=1.0,
        factor_loadings=factor,
        factor_targets=[0.0],
        upper_bound=0.55,
        tolerance=1e-7,
    )

    assert result["primal_residual"] < 1e-8
    assert result["stationarity_residual"] < 1e-7
    assert result["complementarity_residual"] < 1e-8
    assert result["dual_feasibility_violation"] < 1e-7
    assert result["active_lower_bounds"] == 2
    assert result["status"] == "PASS"


def test_turnover_regularization_distinguishes_l1_and_l2_contracts() -> None:
    covariance = np.diag([0.05, 0.07, 0.09, 0.12])
    expected_returns = np.array([0.08, 0.06, 0.03, 0.01])
    previous = np.full(4, 0.25)

    smooth = turnover_regularized(
        covariance,
        expected_returns,
        previous,
        risk_aversion=2.0,
        turnover_penalty=100.0,
        penalty="l2",
    )
    proportional = turnover_regularized(
        covariance,
        expected_returns,
        previous,
        risk_aversion=2.0,
        turnover_penalty=0.04,
        penalty="l1",
    )

    assert np.linalg.norm(np.asarray(smooth["weights"]) - previous) < 0.001
    assert proportional["budget_residual"] < 1e-10
    assert proportional["no_trade_positions"] >= 2


def test_l1_turnover_creates_exact_no_trade_positions() -> None:
    previous = np.full(4, 0.25)

    result = turnover_regularized(
        np.diag([0.05, 0.07, 0.09, 0.12]),
        [0.08, 0.06, 0.03, 0.01],
        previous,
        risk_aversion=2.0,
        turnover_penalty=0.04,
        penalty="l1",
    )

    assert result["budget_residual"] < 1e-10
    assert result["no_trade_positions"] >= 2
    assert result["subgradient_residual"] < 1e-6
    assert result["maximum_kkt_violation"] < 1e-6
    assert result["status"] == "PASS"


def test_risk_contributions_reconcile_euler_identity() -> None:
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    weights = np.array([0.6, 0.4])

    contributions = risk_contributions(weights, covariance)

    assert contributions.sum() == pytest.approx(
        np.sqrt(weights @ covariance @ weights)
    )


def test_risk_parity_solves_the_convex_log_barrier_contract() -> None:
    covariance = np.diag([0.04, 0.09, 0.16])
    budget = np.array([0.2, 0.3, 0.5])

    result = risk_parity(covariance, budget)

    assert np.sum(result["weights"]) == pytest.approx(1.0)
    assert result["risk_budget_shares"] == pytest.approx(budget, abs=1e-8)
    assert result["gradient_residual"] < 1e-7
    assert result["risk_budget_residual"] < 1e-8
    assert result["euler_residual"] < 1e-14
    assert result["status"] == "PASS"
    assert not result["lower_bound_touched"]


def test_risk_parity_correlated_fixture_has_governed_status() -> None:
    covariance = np.array(
        [
            [0.04, 0.012, 0.006],
            [0.012, 0.09, 0.018],
            [0.006, 0.018, 0.16],
        ]
    )

    result = risk_parity(covariance, [0.25, 0.35, 0.40])

    assert result["risk_budget_shares"] == pytest.approx(
        [0.25, 0.35, 0.40],
        abs=1e-8,
    )
    assert result["status"] == "PASS"


def test_risk_parity_rejects_invalid_controls() -> None:
    with pytest.raises(ValueError, match="positive tolerance"):
        risk_parity(np.eye(2), tolerance=0.0)
    with pytest.raises(ValueError, match="iteration"):
        risk_parity(np.eye(2), max_iterations=0)


def test_sherman_morrison_matches_dense_rank_one_update() -> None:
    matrix = np.array([[2.0, 0.2], [0.2, 1.0]])
    left = np.array([0.1, 0.3])
    right = np.array([0.2, -0.1])

    updated = sherman_morrison_inverse(
        np.linalg.inv(matrix),
        left,
        right,
    )

    assert updated == pytest.approx(
        np.linalg.inv(matrix + np.outer(left, right))
    )


def test_symmetric_rank_one_downdate_rejects_pd_boundary() -> None:
    with pytest.raises(ValueError, match="positive definiteness"):
        symmetric_rank_one_update_audit(
            np.eye(2),
            np.eye(2),
            [1.0, 0.0],
            -1.0,
        )


def test_rank_one_update_matches_full_refactorization() -> None:
    matrix = np.array([[2.0, 0.2], [0.2, 1.0]])
    direction = np.array([0.3, -0.1])

    result = symmetric_rank_one_update_audit(
        matrix,
        np.linalg.solve(matrix, np.eye(2)),
        direction,
        0.1,
    )

    assert result["inverse"] == pytest.approx(
        np.linalg.solve(result["matrix"], np.eye(2))
    )
    assert result["relative_inverse_residual"] < 1e-14
    assert result["pre_update_residual"] < 1e-14
    assert result["post_update_residual"] < 1e-14


def test_rank_one_update_rejects_stale_inverse_before_update() -> None:
    with pytest.raises(ValueError, match="does not match"):
        symmetric_rank_one_update_audit(
            np.eye(2),
            1.01 * np.eye(2),
            [0.2, -0.1],
            0.1,
        )


def test_symmetric_rank_one_audit_rejects_invalid_covariance_geometry() -> None:
    asymmetric = np.array([[2.0, 100.0], [0.0, 2.0]])
    with pytest.raises(ValueError, match="symmetric"):
        symmetric_rank_one_update_audit(
            asymmetric,
            np.linalg.inv(asymmetric),
            [0.0, 0.0],
            0.0,
        )
    with pytest.raises(ValueError, match="positive definite"):
        symmetric_rank_one_update_audit(
            np.diag([1.0, -1.0]),
            np.diag([1.0, -1.0]),
            [0.0, 0.0],
            0.0,
        )
    with pytest.raises(ValueError, match="inverse must be symmetric"):
        symmetric_rank_one_update_audit(
            np.eye(2),
            np.array([[1.0, 0.1], [0.0, 1.0]]),
            [0.0, 0.0],
            0.0,
        )


def test_rank_one_core_rejects_zero_denominator_before_division() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="denominator"):
            rank_one_inverse_core(
                np.eye(2),
                np.array([1.0, 0.0]),
                np.array([-1.0, 0.0]),
            )


def test_rank_one_sequence_monitors_drift_against_refactorization() -> None:
    generator = np.random.default_rng(11)
    matrix = np.eye(5)
    inverse = np.eye(5)
    maximum_residual = 0.0
    for _ in range(200):
        direction = generator.normal(size=5)
        evidence = symmetric_rank_one_update_audit(
            matrix,
            inverse,
            direction,
            0.002,
        )
        matrix = np.asarray(evidence["matrix"])
        inverse = np.asarray(evidence["inverse"])
        maximum_residual = max(
            maximum_residual,
            float(evidence["relative_inverse_residual"]),
        )

    assert maximum_residual < 2e-14
    assert inverse == pytest.approx(
        np.linalg.solve(matrix, np.eye(5)),
        abs=2e-14,
    )


def test_solve_audit_matches_normwise_backward_error_definition() -> None:
    matrix = hilbert(8)
    reference = np.ones(8)
    rhs = matrix @ reference

    result = solve_audit(
        matrix,
        rhs,
        reference_solution=reference,
    )
    solution = np.asarray(result["solution"])
    residual = rhs - matrix @ solution
    expected_backward_error = np.linalg.norm(residual) / (
        np.linalg.norm(matrix, ord=2) * np.linalg.norm(solution)
        + np.linalg.norm(rhs)
    )

    assert result["backward_error"] == pytest.approx(expected_backward_error)
    assert result["condition_number"] > 1e9
    assert result["forward_error"] <= result["inverse_forward_error"]


def test_solve_audit_reports_small_backward_error() -> None:
    matrix = hilbert(8)
    expected = np.ones(8)

    result = solve_audit(matrix, matrix @ expected, reference_solution=expected)

    assert result["condition_number"] > 1e9
    assert result["backward_error"] < 1e-15
    assert result["forward_error"] <= result["inverse_forward_error"]


def test_least_squares_svd_audit_exposes_rank_deficiency() -> None:
    design = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ]
    )
    rhs = np.array([1.0, 2.0, 3.0, 4.0])

    result = least_squares_svd_audit(design, rhs)

    assert result["rank"] == 1
    assert result["is_rank_deficient"]
    assert result["relative_residual"] < 1e-14
    assert math.isinf(result["condition_number"])
    assert result["raw_condition_number"] > 1e15
    assert result["retained_condition_number"] == pytest.approx(1.0)


def test_least_squares_svd_distinguishes_rank_from_identification() -> None:
    underdetermined = least_squares_svd_audit(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [1.0, 1.0],
    )
    identified = least_squares_svd_audit(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [1.0, 2.0, 3.0],
    )

    assert underdetermined["full_row_rank"]
    assert not underdetermined["full_column_rank"]
    assert underdetermined["is_underdetermined"]
    assert not underdetermined["is_parameter_identified"]
    assert underdetermined["identification_status"] == "NOT_IDENTIFIED"
    assert math.isinf(underdetermined["condition_number"])
    assert identified["full_column_rank"]
    assert identified["is_parameter_identified"]
    assert identified["identification_status"] == "IDENTIFIED"
    assert np.isfinite(identified["condition_number"])


def test_long_only_holdings_l1_penalty_is_constant() -> None:
    result = long_only_l1_constant_audit(
        np.diag([0.04, 0.06, 0.09]),
        [0.08, 0.05, 0.03],
        [0.0, 0.1, 1.0],
        risk_aversion=2.0,
    )

    assert result["l1_norms"] == pytest.approx([1.0, 1.0, 1.0])
    assert result["maximum_weight_difference"] < 1e-7


def test_long_only_l1_norm_validates_the_simplex_directly() -> None:
    assert long_only_l1_norm([0.2, 0.3, 0.5]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="long-only"):
        long_only_l1_norm([0.6, -0.1, 0.5])
    with pytest.raises(ValueError, match="fully invested"):
        long_only_l1_norm([0.2, 0.3, 0.4])


def test_gross_exposure_limit_changes_long_short_allocation() -> None:
    covariance = np.diag([0.04, 0.05, 0.06])
    expected = np.array([0.20, 0.05, -0.10])

    tight = gross_exposure_constrained_portfolio(
        covariance,
        expected,
        risk_aversion=0.5,
        gross_limit=1.1,
    )
    loose = gross_exposure_constrained_portfolio(
        covariance,
        expected,
        risk_aversion=0.5,
        gross_limit=2.0,
    )

    assert tight["gross_exposure"] <= 1.1 + 1e-8
    assert loose["gross_exposure"] > tight["gross_exposure"] + 0.5


def test_soft_threshold_and_l2_projection_have_distinct_geometry() -> None:
    thresholded = soft_threshold([-2.0, -0.2, 0.3, 2.0], 0.5)
    projected = project_l2_ball([3.0, 4.0], 2.0)

    assert thresholded == pytest.approx([-1.5, 0.0, 0.0, 1.5])
    assert np.linalg.norm(projected) == pytest.approx(2.0)
    assert projected / np.linalg.norm(projected) == pytest.approx([0.6, 0.8])


def test_portfolio_level_l1_l2_comparison_reports_geometry_and_stability() -> None:
    covariance = np.array(
        [
            [0.05, 0.01, 0.00, 0.00],
            [0.01, 0.07, 0.01, 0.00],
            [0.00, 0.01, 0.09, 0.02],
            [0.00, 0.00, 0.02, 0.12],
        ]
    )

    result = norm_constrained_portfolio_comparison(
        covariance,
        [0.10, 0.07, 0.02, -0.03],
        risk_aversion=0.8,
        gross_limit=1.4,
        l2_radius=0.75,
    )

    assert result["l1_gross_exposure"] <= 1.4 + 1e-8
    assert np.linalg.norm(result["l2_weights"]) <= 0.75 + 1e-8
    assert result["l1_perturbation_turnover"] >= 0.0
    assert result["l2_perturbation_turnover"] >= 0.0
    assert 0.0 <= result["l1_active_set_stability"] <= 1.0


def test_active_set_stability_is_jaccard_of_nonzero_positions() -> None:
    metrics = allocation_stability_metrics(
        np.array([0.0, 0.5, 0.5]),
        np.array([0.5, 0.0, 0.5]),
    )

    assert metrics["active_set_stability"] == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    "invalid_covariance",
    [
        np.array([[1.0, 0.2], [0.3, 1.0]]),
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
    ],
)
def test_sparse_portfolio_solvers_reject_invalid_covariance(
    invalid_covariance: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        gross_exposure_constrained_portfolio(
            invalid_covariance,
            [0.1, 0.0],
            risk_aversion=1.0,
            gross_limit=1.2,
        )
    with pytest.raises(ValueError):
        norm_constrained_portfolio_comparison(
            invalid_covariance,
            [0.1, 0.0],
            risk_aversion=1.0,
            gross_limit=1.2,
            l2_radius=0.8,
        )


@pytest.mark.parametrize("bad_control", [np.nan, np.inf, -np.inf])
def test_public_numerical_controls_reject_nonfinite_values(
    bad_control: float,
) -> None:
    with pytest.raises(ValueError):
        linear_shrinkage(np.eye(2), np.eye(2), bad_control)
    with pytest.raises(ValueError):
        woodbury_solve([1.0, 1.0], [[1.0], [0.0]], [[1.0]], [1.0, 1.0], tolerance=bad_control)
    with pytest.raises(ValueError):
        simulate_correlated(np.eye(2), 10, numerical_tolerance=bad_control)
    with pytest.raises(ValueError):
        nearest_correlation(np.eye(2), tolerance=bad_control)
    with pytest.raises(ValueError):
        sherman_morrison_inverse(np.eye(2), [0.1, 0.0], [0.0, 0.1], tolerance=bad_control)
    with pytest.raises(ValueError):
        least_squares_svd_audit(np.eye(2), [1.0, 1.0], relative_tolerance=bad_control)
    with pytest.raises(ValueError):
        soft_threshold([1.0], bad_control)
    with pytest.raises(ValueError):
        project_l2_ball([1.0], bad_control)


def test_governed_case_study_reconciles_the_published_decision_record() -> None:
    result = governed_risk_model_case()

    assert result["observations"] == 45
    assert result["assets"] == 25
    assert result["sample_rank"] == 25
    assert result["shrinkage_intensity"] == pytest.approx(0.08931579914032078)
    assert result["sample_condition_number"] == pytest.approx(10_599.3481)
    assert result["shrunk_condition_number"] == pytest.approx(122.42947405)
    assert result["sample_gross_leverage"] == pytest.approx(
        1.5354355548115273
    )
    assert result["shrunk_gross_leverage"] == pytest.approx(1.0)
    assert result["kkt_primal_residual"] < 1e-14
    assert result["kkt_stationarity_residual"] < 1e-8
    assert result["factor_neutrality_residual"] < 2e-16
    assert result["turnover"] == pytest.approx(0.279274107788, abs=2e-9)
    assert result["raw_target_turnover"] > result["turnover"]
    assert result["signed_turnover_limit"] == pytest.approx(0.20)
    assert result["no_trade_positions"] == 21
    assert result["turnover_kkt_violation"] < 1e-8
    assert len(result["snapshot_hash"]) == 64
    assert len(result["constraints_hash"]) == 64
    assert len(result["solver_configuration_hash"]) == 64
    assert result["threshold_policy"] == "QJ-RISK-MATRIX-2026.1"
    assert result["decision"] == "REJECT"
    assert "TURNOVER_LIMIT_EXCEEDED" in result["decision_reasons"]
