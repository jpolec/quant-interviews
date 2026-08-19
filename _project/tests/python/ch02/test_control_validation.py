"""Cross-module validation edges for every public Chapter 2 control."""

from __future__ import annotations

import numpy as np
import pytest

from quant_interviews.ch02.case_study import evaluate_case_decision
from quant_interviews.ch02.covariance import (
    covariance_psd_certificate,
    equicorrelation_eigenvalues,
    pca_risk_decomposition,
)
from quant_interviews.ch02.factor import (
    correlation_eigen_root,
    simulate_correlated,
    woodbury_solve,
)
from quant_interviews.ch02.optimization import (
    long_only_markowitz_audit,
    turnover_regularized,
)
from quant_interviews.ch02.regularization import (
    dykstra_nearest_correlation_core,
    linear_shrinkage,
    nearest_correlation,
    oas_intensity,
)
from quant_interviews.ch02.sparse import (
    gross_exposure_constrained_portfolio,
    long_only_l1_constant_audit,
    norm_constrained_portfolio_comparison,
)
from quant_interviews.ch02.updates import (
    least_squares_svd_audit,
    sherman_morrison_inverse,
    symmetric_rank_one_update_audit,
)


NONFINITE = [np.nan, np.inf, -np.inf]


@pytest.mark.parametrize("invalid", NONFINITE)
def test_covariance_and_factor_controls_reject_nonfinite_values(
    invalid: float,
) -> None:
    with pytest.raises(ValueError):
        covariance_psd_certificate(np.eye(2), absolute_tolerance=invalid)
    with pytest.raises(ValueError):
        pca_risk_decomposition(np.eye(2), [0.5, 0.5], relative_tolerance=invalid)
    with pytest.raises(ValueError):
        equicorrelation_eigenvalues(2, invalid)
    with pytest.raises(ValueError):
        woodbury_solve(
            [1.0, 1.0],
            [[1.0], [0.0]],
            [[1.0]],
            [1.0, 1.0],
            tolerance=invalid,
        )
    with pytest.raises(ValueError):
        correlation_eigen_root(np.eye(2), numerical_tolerance=invalid)
    with pytest.raises(ValueError):
        simulate_correlated(np.eye(2), 10, maximum_repair_norm=invalid)


@pytest.mark.parametrize("invalid", NONFINITE)
def test_regularization_and_update_controls_reject_nonfinite_values(
    invalid: float,
) -> None:
    with pytest.raises(ValueError):
        linear_shrinkage(np.eye(2), np.eye(2), invalid)
    with pytest.raises(ValueError):
        dykstra_nearest_correlation_core(
            np.eye(2),
            tolerance=invalid,
            max_iterations=10,
        )
    with pytest.raises(ValueError):
        nearest_correlation(np.eye(2), tolerance=invalid)
    with pytest.raises(ValueError):
        sherman_morrison_inverse(
            np.eye(2),
            [0.1, 0.0],
            [0.0, 0.1],
            tolerance=invalid,
        )
    with pytest.raises(ValueError):
        symmetric_rank_one_update_audit(
            np.eye(2),
            np.eye(2),
            [0.1, 0.0],
            invalid,
        )
    with pytest.raises(ValueError):
        symmetric_rank_one_update_audit(
            np.eye(2),
            np.eye(2),
            [0.1, 0.0],
            0.1,
            tolerance=invalid,
        )
    with pytest.raises(ValueError):
        least_squares_svd_audit(
            np.eye(2),
            [1.0, 1.0],
            relative_tolerance=invalid,
        )


@pytest.mark.parametrize("invalid", NONFINITE)
def test_optimization_and_sparse_controls_reject_nonfinite_values(
    invalid: float,
) -> None:
    with pytest.raises(ValueError):
        long_only_markowitz_audit(
            np.eye(2),
            [0.03, 0.04],
            risk_aversion=invalid,
        )
    with pytest.raises(ValueError):
        long_only_markowitz_audit(
            np.eye(2),
            [0.03, 0.04],
            risk_aversion=1.0,
            tolerance=invalid,
        )
    with pytest.raises(ValueError):
        turnover_regularized(
            np.eye(2),
            [0.03, 0.04],
            [0.5, 0.5],
            risk_aversion=1.0,
            turnover_penalty=invalid,
            penalty="l1",
        )
    with pytest.raises(ValueError):
        long_only_l1_constant_audit(
            np.eye(2),
            [0.03, 0.04],
            [0.0, 0.1],
            risk_aversion=invalid,
        )
    with pytest.raises(ValueError):
        gross_exposure_constrained_portfolio(
            np.eye(2),
            [0.03, 0.04],
            risk_aversion=1.0,
            gross_limit=invalid,
        )
    comparison = {
        "covariance": np.eye(2),
        "expected_returns": [0.03, 0.04],
        "risk_aversion": 1.0,
        "gross_limit": 1.2,
        "l2_radius": 0.8,
        "perturbation": 1e-4,
    }
    for control in (
        "risk_aversion",
        "gross_limit",
        "l2_radius",
        "perturbation",
    ):
        request = dict(comparison)
        request[control] = invalid
        with pytest.raises(ValueError):
            norm_constrained_portfolio_comparison(**request)


@pytest.mark.parametrize("invalid", [True, 2.5, np.nan])
def test_integer_controls_fail_with_stable_value_errors(invalid: object) -> None:
    with pytest.raises(ValueError):
        equicorrelation_eigenvalues(invalid, 0.2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        simulate_correlated(np.eye(2), invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        simulate_correlated(np.eye(2), 10, seed=invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        oas_intensity(np.eye(2), invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        nearest_correlation(np.eye(2), max_iterations=invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        dykstra_nearest_correlation_core(
            np.eye(2),
            tolerance=1e-10,
            max_iterations=invalid,  # type: ignore[arg-type]
        )


def test_simulation_rejects_negative_seed_before_every_execution_path() -> None:
    with pytest.raises(ValueError):
        simulate_correlated(np.eye(2), 10, seed=-1)
    with pytest.raises(ValueError):
        simulate_correlated(
            [[1.0, 1.0 + 5e-11], [1.0 + 5e-11, 1.0]],
            10,
            seed=-1,
            maximum_repair_norm=1e-12,
        )


@pytest.mark.parametrize("invalid", NONFINITE)
def test_case_threshold_controls_reject_nonfinite_values(invalid: float) -> None:
    base = {
        "condition_number": 10.0,
        "turnover": 0.1,
        "maximum_kkt_violation": 1e-12,
        "factor_neutrality_residual": 1e-12,
        "lineage_matches": True,
        "pit_windows": 3,
        "human_approved": True,
    }
    for control in (
        "conditioning_limit",
        "turnover_limit",
        "kkt_residual_limit",
        "factor_neutrality_limit",
    ):
        request = dict(base)
        request[control] = invalid
        with pytest.raises(ValueError):
            evaluate_case_decision(**request)
