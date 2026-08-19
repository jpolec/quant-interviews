from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from math import exp

import numpy as np
import pytest

from quant_interviews.ch04.correlated_diffusions import (
    correlated_normals,
    correlation_input_gate,
    nearest_psd_correlation,
)
from quant_interviews.ch04.exponential_martingale import (
    density_process_diagnostic,
    novikov_proxy,
    stochastic_exponential,
)
from quant_interviews.ch04.feynman_kac import (
    black_scholes_call_reference,
    discounted_gbm_call_mc,
    ou_terminal_and_running_payoff,
)
from quant_interviews.ch04.gbm import exact_transition_audit, ito_log_correction_audit
from quant_interviews.ch04.girsanov_importance import normal_tail_importance_sampling
from quant_interviews.ch04.hitting_probabilities import (
    brownian_hit_upper_before_lower,
    brownian_mean_exit_time,
    hitting_convergence_study,
    simulate_hit_probability,
    simulate_hitting_experiment,
)
from quant_interviews.ch04.integrated_case import (
    ReleaseGate,
    chapter_release_case,
    reduce_release_gates,
)
from quant_interviews.ch04.ito_integrals import (
    deterministic_ito_moments,
    simulate_deterministic_ito,
    stochastic_integral_martingale_study,
)
from quant_interviews.ch04.martingale_representation import (
    hedge_residual,
    least_squares_hedge,
    spanning_hedge_certificate,
)
from quant_interviews.ch04.ou_exact import ou_conditional_moments, simulate_ou_exact
from quant_interviews.ch04.predictability_guard import (
    TemporalValue,
    assert_predictable,
    select_as_known,
    temporal_snapshot_hash,
)
from quant_interviews.ch04.quadratic_variation import (
    brownian_paths,
    quadratic_variation_convergence,
    realized_quadratic_variation,
)
from quant_interviews.ch04.ratio_gbm import (
    ratio_gbm_parameter_audit,
    ratio_gbm_parameters,
)
from quant_interviews.ch04.sde_schemes import gbm_scheme_convergence


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def _row(
    *,
    revision: int,
    published: datetime,
    ingested: datetime,
    value: float,
    superseded: datetime | None = None,
    payload: str | None = None,
) -> TemporalValue:
    return TemporalValue(
        key="eps",
        valid_from=datetime(2025, 1, 1, tzinfo=UTC),
        valid_to=datetime(2026, 1, 1, tzinfo=UTC),
        published_at=published,
        ingested_at=ingested,
        value=value,
        revision_id=revision,
        superseded_at=superseded,
        source_priority=10,
        payload_hash=_sha(payload or f"eps-{revision}"),
    )


def test_point_in_time_selection_derives_knowledge_and_excludes_later_revision() -> None:
    first_known = datetime(2026, 1, 20, tzinfo=UTC)
    superseded = datetime(2026, 2, 15, tzinfo=UTC)
    rows = (
        _row(
            revision=1,
            published=first_known,
            ingested=first_known,
            value=4.2,
            superseded=superseded,
        ),
        _row(
            revision=2,
            published=superseded,
            ingested=superseded,
            value=3.9,
        ),
    )
    selected = select_as_known(
        rows,
        key="eps",
        valid_at=datetime(2025, 12, 31, tzinfo=UTC),
        decision_time=datetime(2026, 2, 1, tzinfo=UTC),
    )
    assert selected.value == 4.2
    assert selected.known_at == first_known
    assert len(temporal_snapshot_hash(rows)) == 64
    assert temporal_snapshot_hash(rows) == temporal_snapshot_hash(reversed(rows))
    mutated = (replace(rows[0], value=4.21, payload_hash=_sha("eps-r1-mutated")), rows[1])
    assert temporal_snapshot_hash(mutated) != temporal_snapshot_hash(rows)


def test_temporal_contract_rejects_forged_or_ambiguous_data() -> None:
    published = datetime(2026, 1, 5, tzinfo=UTC)
    with pytest.raises(ValueError, match="publication cannot follow ingestion"):
        _row(
            revision=1,
            published=published,
            ingested=datetime(2026, 1, 4, tzinfo=UTC),
            value=1.0,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        TemporalValue(
            "x",
            datetime(2025, 1, 1),
            None,
            datetime(2025, 1, 2),
            datetime(2025, 1, 2),
            1.0,
            payload_hash=_sha("x"),
        )
    tied = _row(
        revision=1,
        published=published,
        ingested=published,
        value=1.0,
        payload="same",
    )
    with pytest.raises(ValueError, match="ambiguous"):
        select_as_known(
            (tied, replace(tied, value=2.0)),
            key="eps",
            valid_at=datetime(2025, 12, 31, tzinfo=UTC),
            decision_time=datetime(2026, 1, 6, tzinfo=UTC),
        )


def test_predictability_guard_rejects_future_feature_and_naive_time() -> None:
    with pytest.raises(ValueError, match="anticipative"):
        assert_predictable(
            datetime(2026, 2, 2, tzinfo=UTC),
            datetime(2026, 2, 1, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        assert_predictable(datetime(2026, 2, 1), datetime(2026, 2, 2))


def test_quadratic_variation_rmse_has_half_order() -> None:
    study = quadratic_variation_convergence(
        partition_sizes=(16, 32, 64, 128), paths=12_000, seed=1
    )
    assert study.status == "PASS"
    assert study.rmse_mesh_slope == pytest.approx(0.5, abs=0.08)
    assert study.rows[-1].rmse < study.rows[0].rmse


def test_quadratic_variation_public_helpers_are_exercised() -> None:
    paths = brownian_paths(paths=5_000, steps=64, seed=31)
    qv = realized_quadratic_variation(paths)
    assert float(np.mean(qv)) == pytest.approx(1.0, abs=0.03)
    with pytest.raises(ValueError, match="finite paths"):
        realized_quadratic_variation(np.array([[0.0, np.nan]]))


def test_deterministic_ito_integral_matches_isometry() -> None:
    integrand = np.linspace(0.0, 1.0, 200)
    _, variance = deterministic_ito_moments(integrand, 1.0 / 200.0)
    sample = simulate_deterministic_ito(
        integrand, dt=1.0 / 200.0, paths=20_000, seed=2
    )
    assert float(np.mean(sample)) == pytest.approx(0.0, abs=0.012)
    assert float(np.var(sample)) == pytest.approx(variance, rel=0.04)


def test_predictable_and_anticipative_integrals_are_distinguished() -> None:
    study = stochastic_integral_martingale_study(steps=128, paths=20_000, seed=3)
    assert study.status == "PASS"
    assert study.predictable.relative_isometry_error < 0.05
    assert study.anticipative.mean_terminal > 20.0 * study.anticipative.standard_error


def test_ito_log_correction_and_exact_gbm_transition() -> None:
    correction = ito_log_correction_audit(paths=40_000, seed=4)
    transition = exact_transition_audit(paths=40_000, seed=5)
    assert correction.status == transition.status == "PASS"
    assert transition.minimum_terminal_value > 0.0


def test_discounted_gbm_call_matches_independent_black_scholes_reference() -> None:
    estimate, standard_error = discounted_gbm_call_mc(
        spot=100.0,
        strike=105.0,
        rate=0.02,
        volatility=0.25,
        maturity=1.25,
        paths=120_000,
        seed=44,
    )
    reference = black_scholes_call_reference(
        spot=100.0,
        strike=105.0,
        rate=0.02,
        volatility=0.25,
        maturity=1.25,
    )
    assert estimate == pytest.approx(reference, abs=5.0 * standard_error)


def test_ou_exact_supports_scalar_and_vector_state_and_rejects_nonfinite() -> None:
    mean, variance = ou_conditional_moments(
        1.2, kappa=1.4, theta=0.5, sigma=0.3, delta=0.25
    )
    vector_mean, vector_variance = ou_conditional_moments(
        np.array([1.2, 0.7]),
        kappa=1.4,
        theta=0.5,
        sigma=0.3,
        delta=0.25,
    )
    assert vector_mean[0] == pytest.approx(mean)
    assert vector_variance == variance
    paths = simulate_ou_exact(
        1.2,
        kappa=1.4,
        theta=0.5,
        sigma=0.3,
        dt=0.25,
        steps=2,
        paths=30_000,
        seed=6,
    )
    assert float(np.mean(paths[:, 1])) == pytest.approx(mean, abs=0.01)
    with pytest.raises(ValueError, match="finite"):
        simulate_ou_exact(
            float("nan"),
            kappa=1.0,
            theta=0.0,
            sigma=0.2,
            dt=0.1,
            steps=1,
            paths=2,
        )


def test_ratio_gbm_handles_degenerate_driver_tolerances_and_helper() -> None:
    audit = ratio_gbm_parameter_audit(
        mu_x=0.03,
        mu_y=0.03,
        sigma_x=0.2,
        sigma_y=0.2,
        correlation=1.0,
    )
    assert audit.status == "PASS"
    assert audit.volatility == 0.0
    assert ratio_gbm_parameters(
        mu_x=0.03,
        mu_y=0.03,
        sigma_x=0.2,
        sigma_y=0.2,
        correlation=1.0,
    ) == (audit.drift, 0.0)
    with pytest.raises(ValueError, match="finite"):
        ratio_gbm_parameter_audit(
            mu_x=0.0,
            mu_y=0.0,
            sigma_x=0.2,
            sigma_y=0.2,
            correlation=1.0,
            absolute_tolerance=float("nan"),
        )


def test_density_process_requires_predictable_timing_and_finite_diagnostics() -> None:
    rng = np.random.default_rng(7)
    dt = 0.01
    d_w = np.sqrt(dt) * rng.standard_normal((30_000, 100))
    theta = np.full_like(d_w, 0.3)
    diagnostic = density_process_diagnostic(
        theta,
        d_w,
        dt=dt,
        integrand_timing="left_endpoint_predictable",
    )
    assert diagnostic.status == "PASS"
    assert diagnostic.confidence_interval is not None
    assert diagnostic.confidence_interval[0] <= 1.0 <= diagnostic.confidence_interval[1]
    assert diagnostic.novikov_finite_sample_proxy == pytest.approx(exp(0.045), rel=0.02)
    with pytest.raises(ValueError, match="predictability"):
        stochastic_exponential(theta[:2], d_w[:2], dt, integrand_timing="contemporaneous")


def test_density_underflow_is_rejected_without_infinite_public_diagnostics() -> None:
    theta = np.full((2, 2), 1e3)
    d_w = np.zeros((2, 2))
    diagnostic = density_process_diagnostic(
        theta,
        d_w,
        dt=1.0,
        integrand_timing="left_endpoint_predictable",
    )
    assert diagnostic.status == "REJECT"
    assert diagnostic.mean_terminal_density is None
    assert diagnostic.novikov_finite_sample_proxy is None
    assert novikov_proxy(theta, 1.0) is None


def test_importance_sampling_reports_governed_tail_evidence() -> None:
    result = normal_tail_importance_sampling(3.0, shift=3.0, paths=100_000, seed=8)
    assert result.status == "PASS"
    assert result.estimate == pytest.approx(result.analytic_probability, abs=5.0 * result.standard_error)
    assert result.variance_reduction > 100.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": float("nan"), "shift": 3.0, "paths": 10},
        {"threshold": 3.0, "shift": 3.0, "paths": 1},
        {"threshold": 3.0, "shift": -3.0, "paths": 10},
    ],
)
def test_importance_sampling_rejects_invalid_or_wrong_direction(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        normal_tail_importance_sampling(**kwargs)


def test_spanning_certificate_discloses_redundant_holdings_policy() -> None:
    exposures = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    claim = np.array([1.2, -0.7])
    result = spanning_hedge_certificate(exposures, claim)
    assert result.status == "COMPLETE"
    assert result.market_complete and result.claim_spanned
    assert result.holdings_nullity == 1
    assert not result.holdings_unique
    assert result.hedge_solution_policy == "minimum_norm"
    alternative = result.hedge_positions + np.array([1.0, 0.0, -1.0])
    assert alternative @ exposures == pytest.approx(result.replicated_factor_exposure)


def test_regression_helpers_are_directly_exercised() -> None:
    basis = np.c_[np.ones(4), np.arange(4.0)]
    continuation = 1.0 + 2.0 * np.arange(4.0)
    beta = least_squares_hedge(basis, continuation)
    assert beta == pytest.approx([1.0, 2.0])
    assert hedge_residual(basis, continuation, beta) == pytest.approx(np.zeros(4))


def test_hitting_study_separates_single_grid_censoring_and_refinement() -> None:
    assert brownian_hit_upper_before_lower(0.25, 0.0, 1.0) == 0.25
    assert brownian_mean_exit_time(0.4, 0.0, 1.0) == pytest.approx(0.24)
    one = simulate_hitting_experiment(
        0.4, 0.0, 1.0, dt=0.002, paths=5_000, seed=9, maximum_steps=20_000
    )
    assert one.status == "EVIDENCE_ONLY"
    assert one.standard_error is not None and one.estimate is not None
    assert one.estimate == pytest.approx(
        one.analytic_probability, abs=4.0 * one.standard_error
    )
    assert simulate_hit_probability(0.4, 0.0, 1.0, dt=0.002, paths=2_000, seed=19) > 0.0
    study = hitting_convergence_study(
        0.4,
        0.0,
        1.0,
        step_sizes=(0.008, 0.004, 0.002),
        paths=8_000,
        seed=20,
    )
    assert study.status == "PASS"
    assert abs(study.rows[-1].mean_exit_time - 0.24) <= study.exit_time_bias_limit


def test_coarse_hitting_grid_cannot_receive_pass() -> None:
    study = hitting_convergence_study(
        0.4,
        0.0,
        1.0,
        step_sizes=(0.2, 0.1, 0.05),
        paths=5_000,
        seed=23,
    )
    assert study.status == "REJECT"


def test_hitting_truncation_reports_worst_case_probability_bounds() -> None:
    result = simulate_hitting_experiment(
        0.4, 0.0, 1.0, dt=1e-4, paths=1_000, seed=10, maximum_steps=1
    )
    assert result.unresolved_paths > 0
    assert result.status == "CENSORED"
    assert result.probability_lower_bound <= result.probability_upper_bound


def test_feynman_kac_discounts_running_payoff_at_payment_time() -> None:
    result = ou_terminal_and_running_payoff(
        state=0.7,
        kappa=1.2,
        theta=0.4,
        sigma=0.25,
        rate=0.03,
        maturity=1.0,
        running_weight=0.6,
        steps=128,
        paths=60_000,
        seed=11,
    )
    assert result.status == "PASS"
    assert result.analytic_value == pytest.approx(0.8158686581871775, abs=1e-12)
    assert result.confidence_interval[0] <= result.analytic_grid_value <= result.confidence_interval[1]
    assert abs(result.discretization_bias) <= 1e-4


def test_feynman_kac_quadrature_bias_refines_and_rate_limits_are_stable() -> None:
    coarse = ou_terminal_and_running_payoff(
        state=0.7,
        kappa=1.2,
        theta=0.4,
        sigma=0.0,
        rate=0.0,
        maturity=1.0,
        running_weight=0.6,
        steps=8,
        paths=2_000,
        seed=111,
        maximum_absolute_discretization_bias=1.0,
    )
    fine = ou_terminal_and_running_payoff(
        state=0.7,
        kappa=1.2,
        theta=0.4,
        sigma=0.0,
        rate=0.0,
        maturity=1.0,
        running_weight=0.6,
        steps=16,
        paths=2_000,
        seed=111,
        maximum_absolute_discretization_bias=1.0,
    )
    assert abs(fine.discretization_bias) < abs(coarse.discretization_bias)


def test_euler_and_milstein_recover_orders_with_empirical_weak_reconciliation() -> None:
    study = gbm_scheme_convergence(
        step_counts=(8, 16, 32, 64, 128), paths=30_000, seed=12
    )
    assert study.status == "PASS"
    assert study.euler_strong_slope == pytest.approx(0.5, abs=0.1)
    assert study.milstein_strong_slope == pytest.approx(1.0, abs=0.15)
    assert study.weak_order_status == "PASS"
    assert all(row.weak_reconciliation_z <= 5.0 for row in study.rows)


def test_zero_drift_is_exact_for_identity_weak_function_not_rejected() -> None:
    study = gbm_scheme_convergence(drift_rate=0.0, paths=12_000, seed=112)
    assert study.weak_order_status == "EXACT_FOR_TEST_FUNCTION"
    assert study.euler_weak_slope is None
    assert study.status == "PASS"


def test_zero_volatility_uses_deterministic_first_order_limit() -> None:
    study = gbm_scheme_convergence(volatility=0.0, paths=100, seed=113)
    assert study.status == "PASS"
    assert study.euler_strong_slope == pytest.approx(1.0, abs=0.1)


def test_correlation_gate_preserves_singular_psd_rank_and_consistency() -> None:
    result = correlation_input_gate(np.ones((3, 3)))
    assert result.status == "PASS"
    assert result.raw_rank == result.used_rank == 1
    assert result.repair_frobenius_norm == 0.0
    assert result.reconstruction_residual is not None
    assert result.reconstruction_residual < 1e-10


def test_near_psd_gate_records_actual_numerical_repair() -> None:
    raw = np.array([[1.0, 1.0 + 5e-11], [1.0 + 5e-11, 1.0]])
    result = correlation_input_gate(raw, relative_tolerance=1e-10)
    assert result.status == "PASS_NUMERICAL"
    assert result.used_correlation is not None and result.factor is not None
    assert result.repair_frobenius_norm is not None and result.repair_frobenius_norm > 0.0
    assert result.reconstruction_residual == pytest.approx(0.0, abs=1e-10)
    assert result.factor @ result.factor.T == pytest.approx(result.used_correlation, abs=1e-10)


def test_material_correlation_repair_requires_matching_owner_approved_gate() -> None:
    raw = np.array([[1.0, 1.02], [1.02, 1.0]])
    missing = correlation_input_gate(raw, maximum_repair_norm=0.04)
    assert missing.status == "REJECT"
    assert missing.reason == "REPAIR_APPROVAL_MISSING"
    approved = correlation_input_gate(
        raw, maximum_repair_norm=0.04, approval_owner="model-risk-owner"
    )
    assert approved.status == "CONDITIONAL_PASS"
    assert approved.repair_frobenius_norm == pytest.approx(np.sqrt(2.0) * 0.02, rel=1e-6)
    with pytest.raises(ValueError, match="governed gate"):
        correlated_normals(raw, draws=10, seed=1)
    sample = correlated_normals(raw, draws=2_000, seed=1, approved_gate=approved)
    assert sample.gate.approval_owner == "model-risk-owner"
    assert np.max(np.abs(sample.draws[:, 0] - sample.draws[:, 1])) < 1e-12


def test_correlated_normals_and_nearest_helper_expose_governance() -> None:
    spd = np.array([[1.0, 0.6], [0.6, 1.0]])
    sample = correlated_normals(spd, draws=30_000, seed=13)
    assert np.corrcoef(sample.draws.T)[0, 1] == pytest.approx(0.6, abs=0.02)
    repaired = nearest_psd_correlation(
        np.array([[1.0, 1.01], [1.01, 1.0]]), approval_owner="owner"
    )
    assert repaired.used_correlation is not None
    assert repaired.repair_frobenius_norm is not None
    assert np.min(np.linalg.eigvalsh(repaired.used_correlation)) >= -1e-9


def _synthetic_gate(gate_id: str, status: str) -> ReleaseGate:
    return ReleaseGate(
        gate_id=gate_id,
        value="1",
        operator="equals",
        limit="1",
        unit="state",
        owner="owner",
        evidence_sha256=_sha(gate_id),
        status=status,
        reason="test",
    )


def test_release_gate_reducer_has_pass_conditional_and_reject_paths() -> None:
    all_pass = tuple(_synthetic_gate(str(index), "PASS") for index in range(4))
    assert reduce_release_gates(all_pass) == "PASS"
    assert reduce_release_gates(all_pass[:-1] + (_synthetic_gate("c", "CONDITIONAL_PASS"),)) == "CONDITIONAL_PASS"
    for index in range(len(all_pass)):
        mutated = list(all_pass)
        mutated[index] = replace(mutated[index], status="REJECT")
        assert reduce_release_gates(tuple(mutated)) == "REJECT"
    assert reduce_release_gates((replace(all_pass[0], owner=""),)) == "REJECT"
    assert reduce_release_gates((replace(all_pass[0], evidence_sha256="g" * 64),)) == "REJECT"


def test_integrated_release_case_uses_real_hash_counts_and_external_gates() -> None:
    evidence = {
        name: ("PASS", _sha(name))
        for name in ("SQL_ASOF", "SQL_CORRELATION", "CPP_GBM", "CPP_SDE")
    }
    report = chapter_release_case(external_evidence=evidence)
    assert report["fixture_version"] == "ch04-release-v0.10"
    assert len(report["snapshot_hash"]) == 64
    assert report["input_count"] == 3
    assert report["excluded_revision_count"] == 1
    assert report["selected_volatility"] == 0.20
    assert report["final_decision"] == "CONDITIONAL_PASS"
    gates = {gate["gate_id"]: gate for gate in report["gates"]}
    assert all(len(gate["evidence_sha256"]) == 64 for gate in gates.values())
    assert gates["CORRELATION_REPAIR"]["owner"] == "model-risk-owner"


def test_integrated_case_rejects_missing_external_or_repair_approval() -> None:
    assert chapter_release_case()["final_decision"] == "REJECT"
    evidence = {
        name: ("PASS", _sha(name))
        for name in ("SQL_ASOF", "SQL_CORRELATION", "CPP_GBM", "CPP_SDE")
    }
    assert chapter_release_case(
        external_evidence=evidence, correlation_approval_owner=None
    )["final_decision"] == "REJECT"
