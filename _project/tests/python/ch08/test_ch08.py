from __future__ import annotations

from datetime import timedelta
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from quant_interviews.ch08.acf_pacf import correlogram
from quant_interviews.ch08.adf_power import adf_rejection_rate, simulate_ar1
from quant_interviews.ch08.ar1_mapping import map_ar1_persistence
from quant_interviews.ch08.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    calibration_metrics,
    temporal_calibration,
)
from quant_interviews.ch08.cointegration import engle_granger, fit_ecm
from quant_interviews.ch08.drift import DriftPolicy, drift_evidence
from quant_interviews.ch08.economic_alpha import (
    evaluate_threshold,
    nested_economic_evaluation,
    newey_west_long_run_variance,
    positions_from_probability,
    select_threshold,
)
from quant_interviews.ch08.feature_selection import blocked_stability_selection
from quant_interviews.ch08.hmm import (
    canonical_state_order,
    expected_state_durations,
    gaussian_hmm_filter,
    match_state_labels,
)
from quant_interviews.ch08.kalman import (
    dynamic_beta_filter,
    evaluate_dynamic_beta_holdout,
    fit_dynamic_beta_state_space,
)
from quant_interviews.ch08.stationarity import (
    fit_ar1,
    largest_mean_break,
    stationarity_report,
)
from quant_interviews.ch08.validation import (
    assert_no_interval_overlap,
    assert_nested_temporal_integrity,
    nested_walk_forward_splits,
    purged_embargo_split,
    walk_forward_splits,
)
from quant_interviews.ch08.volatility import (
    fit_garch11,
    fit_har,
    garch11_filter,
    har_frame,
    evaluate_har_holdout,
    qlike,
)


ROOT = Path(__file__).resolve().parents[4]


def random_walk_pair(seed: int = 2, observations: int = 600) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    x = np.cumsum(generator.normal(size=observations))
    spread = np.zeros(observations)
    for index in range(1, observations):
        spread[index] = 0.70 * spread[index - 1] + generator.normal(scale=0.35)
    y = 0.8 + 1.4 * x + spread
    return x, y


def test_ar1_half_life_stationarity_and_oscillation() -> None:
    generator = np.random.default_rng(1)
    values = np.zeros(500)
    for index in range(1, len(values)):
        values[index] = 0.8 * values[index - 1] + generator.normal(scale=0.5)
    fit = fit_ar1(values)
    assert 0.72 < fit.phi < 0.88
    assert fit.half_life is not None and 2.0 < fit.half_life < 5.0
    assert not fit.oscillatory
    assert fit.design_condition < 10.0
    timestamps = pd.date_range("2022-01-03", periods=len(values), freq="B", tz="UTC")
    report = stationarity_report(
        values,
        max_lag=2,
        timestamps=timestamps,
        calendar="XNYS-business-day",
        sampling_interval="1 business day",
    )
    assert report.n_obs == 500
    assert report.adf_pvalue < 0.05
    assert report.status in {"PASS", "CONDITIONAL_PASS"}
    assert fit.phi_standard_error > 0.0
    assert fit.phi_confidence_interval_95[0] < fit.phi < fit.phi_confidence_interval_95[1]

    alternating = np.array([(-0.75) ** index for index in range(40)])
    alternating += generator.normal(scale=0.005, size=alternating.size)
    negative = fit_ar1(alternating)
    assert negative.phi < 0.0 and negative.oscillatory


def test_ar1_rejects_missing_and_maps_zero_negative_cases() -> None:
    invalid = np.arange(30.0)
    invalid[4] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        fit_ar1(invalid)
    zero = map_ar1_persistence(0.0)
    assert zero.half_life_periods == 0.0 and zero.ou_mean_reversion is None
    negative = map_ar1_persistence(-0.8)
    assert negative.oscillatory and negative.half_life_periods is not None
    assert negative.ou_mean_reversion is None


def test_engle_granger_uses_cointegration_distribution_and_i1_gate() -> None:
    x, y = random_walk_pair()
    fit = engle_granger(x, y, adf_lag=1)
    assert abs(fit.alpha - 0.8) < 0.2
    assert abs(fit.beta - 1.4) < 0.05
    assert fit.coint_pvalue < 0.01
    assert fit.critical_values.shape == (3,)
    assert fit.design_rank == 2
    assert fit.cointegrated
    assert fit.x_level_adf_pvalue > 0.05
    assert fit.x_difference_adf_pvalue < 0.05

    generator = np.random.default_rng(20)
    stationary_x = generator.normal(size=300)
    stationary_y = 0.4 * stationary_x + generator.normal(size=300)
    with pytest.raises(ValueError, match=r"I\(1\)"):
        engle_granger(stationary_x, stationary_y)


def test_bivariate_ecm_uses_lagged_differences() -> None:
    x, y = random_walk_pair(observations=700)
    cointegration = engle_granger(x, y)
    result = fit_ecm(x, y, cointegration, lag_order=2)
    assert result.lag_order == 2
    assert result.y_equation.adjustment < 0.0
    assert result.y_equation.lagged_dy.shape == (2,)
    assert result.y_equation.lagged_dx.shape == (2,)
    assert result.x_equation.lagged_dy.shape == (2,)
    assert result.n_obs == len(x) - 1 - 2


def test_kalman_recovers_intercept_beta_and_preserves_psd() -> None:
    generator = np.random.default_rng(3)
    x = generator.normal(size=500)
    intercept = np.linspace(0.4, 0.6, x.size)
    beta = np.linspace(0.8, 1.2, x.size)
    y = intercept + beta * x + generator.normal(scale=0.20, size=x.size)
    result = dynamic_beta_filter(
        x,
        y,
        process_variance=1.0e-4,
        process_intercept_variance=5.0e-5,
        observation_variance=0.20**2,
        initial_intercept=0.4,
        initial_beta=0.8,
    )
    assert result.predicted_state.shape == (500, 2)
    assert result.predicted_covariance.shape == (500, 2, 2)
    assert np.allclose(result.predicted_state[0], [0.4, 0.8])
    assert abs(result.filtered_intercept[-1] - intercept[-1]) < 0.15
    assert abs(result.filtered_beta[-1] - beta[-1]) < 0.15
    assert np.all(result.innovation_variance > 0.0)
    assert np.min(np.linalg.eigvalsh(result.filtered_covariance)) >= -1.0e-12
    assert np.all(np.isfinite(result.standardized_innovations))


def test_hmm_log_space_handles_extreme_tail_and_labels_states() -> None:
    observations = np.array([0.0, 1.0e3, -1.0e3, 0.1])
    result = gaussian_hmm_filter(
        observations,
        [[0.95, 0.05], [0.1, 0.9]],
        [0.0, 0.0],
        [0.7, 1.5],
        [0.8, 0.2],
    )
    assert np.allclose(result.filtered.sum(axis=1), 1.0)
    assert np.isfinite(result.log_likelihood)
    assert np.array_equal(canonical_state_order([0, 0], [1.5, 0.7]), [1, 0])
    matched = match_state_labels([0, 2], [1, 2], [2.1, 0.1], [2.1, 0.9])
    assert matched.status == "PASS"
    assert np.array_equal(matched.permutation, [1, 0])
    ambiguous = match_state_labels([0, 0], [1, 1], [0, 0], [1, 1])
    assert ambiguous.status == "REJECT_UNIDENTIFIED"
    assert np.isinf(expected_state_durations([[1.0, 0.0], [0.2, 0.8]])[0])
    assert result.initial_probability_timing == "predicted_before_first_emission"


def test_garch_initial_state_is_training_owned_and_fit_is_valid() -> None:
    generator = np.random.default_rng(5)
    returns = generator.normal(scale=0.01, size=500)
    first = garch11_filter(returns, 1.0e-6, 0.08, 0.88)
    changed_future = returns.copy()
    changed_future[250:] *= 100.0
    second = garch11_filter(changed_future, 1.0e-6, 0.08, 0.88)
    assert first[0] == pytest.approx(1.0e-6 / (1.0 - 0.08 - 0.88))
    assert first[0] == second[0]
    explicit = garch11_filter(
        returns,
        1.0e-6,
        0.08,
        0.88,
        initial_variance=2.5e-4,
    )
    assert explicit[0] == pytest.approx(2.5e-4)
    fit = fit_garch11(returns)
    assert fit.success
    assert fit.alpha + fit.beta < 1.0 - fit.persistence_margin
    assert np.all(fit.variance > 0.0)


def test_qlike_is_exact_and_rejects_zero_realized_variance() -> None:
    realized = np.array([0.02, 0.04, 0.08])
    forecast = np.array([0.025, 0.035, 0.07])
    ratio = realized / forecast
    expected = np.mean(ratio - np.log(ratio) - 1.0)
    assert qlike(realized, forecast) == pytest.approx(expected)
    with pytest.raises(ValueError, match="strictly positive"):
        qlike([0.0, 0.1], [0.1, 0.1])


def test_har_gates_calendar_and_returns_positive_inverse_forecast() -> None:
    generator = np.random.default_rng(6)
    index = pd.date_range("2024-01-02", periods=220, freq="B", tz="UTC")
    realized = pd.Series(
        np.exp(generator.normal(-4.0, 0.25, size=index.size)),
        index=index,
    )
    fit = fit_har(realized, calendar="XNYS-business-day")
    assert fit.coefficients.shape == (4,)
    assert fit.smearing_factor > 0.0
    assert np.all(fit.variance_forecast > 0.0)
    target = realized.shift(-1).loc[fit.variance_forecast.index].to_numpy()
    assert qlike(target, fit.variance_forecast.to_numpy()) >= 0.0
    with pytest.raises(ValueError, match="increasing"):
        har_frame(realized.sort_index(ascending=False), calendar="XNYS-business-day")
    nonfinite = realized.copy()
    nonfinite.iloc[10] = np.inf
    with pytest.raises(ValueError, match="finite"):
        har_frame(nonfinite, calendar="XNYS-business-day")
    with pytest.raises(ValueError, match="DatetimeIndex"):
        har_frame(pd.Series(realized.to_numpy()), calendar="XNYS-business-day")
    holdout = evaluate_har_holdout(
        realized,
        training_end=index[169],
        calendar="XNYS-business-day",
    )
    assert len(holdout.records) == 50
    assert holdout.qlike >= 0.0
    assert all(row.forecast_origin < row.target_time for row in holdout.records)


def test_walk_forward_step_and_rolling_guards() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        walk_forward_splits(100, min_train=40, test_size=10, step=0)
    with pytest.raises(ValueError, match="shorter"):
        walk_forward_splits(
            100,
            min_train=40,
            test_size=10,
            rolling_window=30,
        )
    folds = walk_forward_splits(
        100,
        min_train=40,
        test_size=10,
        rolling_window=40,
    )
    assert len(folds) == 6
    assert all(max(fold.train_index) < min(fold.test_index) for fold in folds)


def test_nested_temporal_folds_never_tune_on_outer_test() -> None:
    starts = pd.date_range("2024-01-02", periods=140, freq="B", tz="UTC")
    intervals = pd.DataFrame(
        {"start": starts, "end": starts + pd.Timedelta(days=3)}
    )
    nested = nested_walk_forward_splits(
        140,
        outer_min_train=80,
        outer_test_size=20,
        inner_min_train=40,
        inner_validation_size=10,
        intervals=intervals,
        embargo=timedelta(days=2),
    )
    assert len(nested) == 3
    for fold in nested:
        outer_test = set(fold.outer.test_index)
        for inner in fold.inner:
            assert not outer_test.intersection(inner.train_index)
            assert not outer_test.intersection(inner.validation_index)
            assert max(inner.train_index) < min(inner.validation_index)
    assert_nested_temporal_integrity(intervals, nested)
    assert all(fold.outer.raw_train_index for fold in nested)


def test_purging_uses_exact_union_not_convex_hull() -> None:
    intervals = pd.DataFrame(
        {
            "start": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-03",
                    "2025-01-05",
                    "2025-01-10",
                ],
                utc=True,
            ),
            "end": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-04",
                    "2025-01-06",
                    "2025-01-11",
                ],
                utc=True,
            ),
        },
        index=["test-left", "boundary", "gap", "test-right"],
    )
    fold = purged_embargo_split(
        intervals,
        ["test-left", "test-right"],
        candidate_train_indices=["boundary", "gap"],
        mode="combinatorial",
    )
    assert fold.train_index == ("boundary", "gap")
    assert_no_interval_overlap(intervals, fold)


def test_walk_forward_purging_never_accepts_future_training() -> None:
    start = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
    intervals = pd.DataFrame(
        {"start": start, "end": start + pd.to_timedelta(3, unit="D")}
    )
    fold = purged_embargo_split(
        intervals,
        [8, 9],
        embargo=timedelta(days=2),
    )
    assert_no_interval_overlap(intervals, fold)
    assert all(index < 8 for index in fold.train_index)
    with pytest.raises(ValueError, match="precede"):
        purged_embargo_split(
            intervals,
            [8, 9],
            candidate_train_indices=[2, 15],
        )


def test_blocked_stability_scales_inside_resamples() -> None:
    generator = np.random.default_rng(7)
    features = generator.normal(size=(300, 6))
    features[:, 0] *= 1.0e6
    target = (
        1.8e-6 * features[:, 0]
        - 1.2 * features[:, 2]
        + generator.normal(scale=0.5, size=300)
    )
    result = blocked_stability_selection(
        features,
        target,
        block_size=20,
        n_resamples=30,
        alpha=0.03,
        seed=7,
    )
    assert result.selection_probability[0] > 0.8
    assert result.selection_probability[2] > 0.8
    assert result.convergence_failures == 0
    with pytest.raises(ValueError, match="configuration"):
        blocked_stability_selection(
            features,
            target,
            block_size=300,
            n_resamples=3,
            alpha=0.03,
        )


def test_calibrators_clip_and_validate_fit_state() -> None:
    generator = np.random.default_rng(8)
    scores = generator.normal(size=500)
    true_probability = 1.0 / (1.0 + np.exp(-scores))
    labels = generator.binomial(1, true_probability)
    for calibrator in [
        PlattCalibrator(regularization_c=2.0, epsilon=1.0e-4),
        IsotonicCalibrator(epsilon=1.0e-4),
    ]:
        with pytest.raises(RuntimeError, match="fitted"):
            calibrator.predict(scores[:5])
        probability = calibrator.fit(scores[:350], labels[:350]).predict(
            np.array([-1.0e6, 0.0, 1.0e6])
        )
        assert np.all((1.0e-4 <= probability) & (probability <= 1.0 - 1.0e-4))
        metrics = calibration_metrics(
            calibrator.predict(scores[350:]),
            labels[350:],
        )
        assert metrics.brier < 0.3 and np.isfinite(metrics.log_loss)


def test_drift_requires_magnitude_persistence_and_evidence() -> None:
    generator = np.random.default_rng(9)
    reference = np.clip(generator.normal(0.45, 0.08, size=300), 0, 1)
    live = np.clip(generator.normal(0.70, 0.08, size=300), 0, 1)
    policy = DriftPolicy(
        policy_id="drift-policy-v1",
        bandwidth=0.08,
        mmd_threshold=0.02,
        brier_threshold=0.03,
        minimum_persistence=2,
    )
    early = drift_evidence(
        reference,
        live,
        policy=policy,
        window_id="w1",
        window_start="2025-01-01T00:00:00Z",
        window_end="2025-01-02T00:00:00Z",
        model_hash="model-a",
        data_hash="data-1",
        evidence_id="drift-run-1",
    )
    assert early.magnitude_breach and not early.alert
    governed = drift_evidence(
        reference,
        live,
        policy=policy,
        window_id="w2",
        window_start="2025-01-02T00:00:00Z",
        window_end="2025-01-03T00:00:00Z",
        model_hash="model-a",
        data_hash="data-2",
        evidence_id="drift-run-2",
        history=[early.current_record],
    )
    assert governed.alert and governed.mean_shift > 0.15
    with pytest.raises(ValueError, match="probabilities"):
        drift_evidence(
            reference,
            live + 1.0,
            policy=policy,
            window_id="bad",
            window_start="2025-01-02T00:00:00Z",
            window_end="2025-01-03T00:00:00Z",
            model_hash="model-a",
            data_hash="bad",
            evidence_id="bad",
            history=[early.current_record],
        )


def test_economic_threshold_charges_entry_and_terminal_exit() -> None:
    probabilities = np.array([0.8, 0.7, 0.55, 0.4, 0.2, 0.9, 0.1])
    returns = np.array([0.02, 0.01, 0.005, -0.004, -0.02, 0.025, -0.015])
    result = evaluate_threshold(
        probabilities,
        returns,
        threshold=0.65,
        cost_per_unit_turnover=0.001,
        risk_aversion=1.0,
    )
    expected_position = np.array([1, 1, 0, 0, -1, 1, -1], dtype=float)
    expected_turnover = np.abs(
        np.diff(np.r_[0.0, expected_position, 0.0])
    ).sum()
    assert np.array_equal(result.position, expected_position)
    assert result.turnover == expected_turnover
    assert result.terminal_exit_turnover == 1.0
    assert result.cost == pytest.approx(0.001 * expected_turnover)
    assert result.net_pnl == pytest.approx(result.period_pnl.sum())


def test_threshold_selection_matches_manual_utilities() -> None:
    probability = np.array([0.9, 0.6, 0.1, 0.4, 0.8, 0.2])
    returns = np.array([0.03, -0.01, -0.02, 0.01, 0.02, -0.01])
    candidates = [0.55, 0.65, 0.75]
    manual = [
        evaluate_threshold(
            probability,
            returns,
            threshold=value,
            cost_per_unit_turnover=0.002,
            risk_aversion=0.5,
        )
        for value in candidates
    ]
    expected = max(
        manual,
        key=lambda result: (
            result.utility,
            -result.turnover,
            result.threshold,
        ),
    )
    selected = select_threshold(
        probability,
        returns,
        candidates,
        cost_per_unit_turnover=0.002,
        risk_aversion=0.5,
        selection_context="inner_validation",
    )
    assert selected.threshold == expected.threshold
    assert selected.utility == expected.utility


def test_correlogram_and_adf_power_baselines() -> None:
    generator = np.random.default_rng(10)
    values = np.zeros(600)
    for index in range(1, len(values)):
        values[index] = 0.7 * values[index - 1] + generator.normal(scale=0.7)
    result = correlogram(values, nlags=12)
    assert result.acf[1] > 0.55
    assert result.pacf[1] > 0.55
    stationary = adf_rejection_rate(0.85, 180, simulations=30, seed=11)
    near_unit = adf_rejection_rate(0.99, 180, simulations=30, seed=100)
    assert stationary.rejection_rate > near_unit.rejection_rate
    assert stationary.standard_error >= 0.0
    assert stationary.confidence_interval_95[0] <= stationary.rejection_rate
    assert stationary.rejection_rate <= stationary.confidence_interval_95[1]
    with pytest.raises(ValueError, match="one-dimensional"):
        correlogram(np.column_stack([values, values]), nlags=12)
    missing = values.copy()
    missing[100] = np.nan
    with pytest.raises(ValueError, match="calendar policy"):
        correlogram(missing, nlags=12)


def test_stationarity_diagnostic_failure_is_not_a_reassuring_pvalue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_interviews.ch08.stationarity as module

    generator = np.random.default_rng(101)
    values = generator.normal(size=120)
    timestamps = pd.date_range("2024-01-02", periods=120, freq="B", tz="UTC")

    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise ValueError("diagnostic unavailable")

    monkeypatch.setattr(module, "kpss", unavailable)
    result = stationarity_report(
        values,
        max_lag=1,
        timestamps=timestamps,
        calendar="XNYS",
        sampling_interval="1 business day",
    )
    assert result.kpss.status == "ERROR"
    assert result.kpss.statistic is None and result.kpss.pvalue is None
    assert result.status == "REJECT"


def test_ecm_rejects_foreign_or_rejected_certificate_and_gates_joint_root() -> None:
    x, y = random_walk_pair(seed=2, observations=500)
    certificate = engle_granger(x, y)
    result = fit_ecm(x, y, certificate, lag_order=2)
    assert result.spread_adjustment == pytest.approx(
        result.y_equation.adjustment - certificate.beta * result.x_equation.adjustment
    )
    assert result.scalar_root == pytest.approx(1.0 + result.spread_adjustment)
    assert result.scalar_stable == (abs(result.scalar_root) < 1.0)
    with pytest.raises(ValueError, match="not accepted"):
        fit_ecm(x, y, replace(certificate, cointegrated=False), lag_order=2)
    with pytest.raises(ValueError, match="another input pair"):
        fit_ecm(x + 1.0e-8, y, certificate, lag_order=2)
    with pytest.raises(ValueError, match="residual digest"):
        fit_ecm(
            x,
            y,
            replace(certificate, residual_digest="0" * 64),
            lag_order=2,
        )


def test_garch_nonconvergence_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_interviews.ch08.volatility as module

    class FailedFit:
        success = False
        nit = 4000
        message = "iteration limit"
        fun = 1.0
        x = np.array([1.0e-6, 0.05, 0.90])

    monkeypatch.setattr(module, "minimize", lambda *_args, **_kwargs: FailedFit())
    with pytest.raises(RuntimeError, match="did not converge"):
        module.fit_garch11(np.random.default_rng(1).normal(size=100))


def test_stability_selection_rejects_any_partial_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    import quant_interviews.ch08.feature_selection as module

    class Model:
        def __init__(self) -> None:
            self.named_steps = {"elasticnet": type("Fit", (), {"coef_": np.array([1.0, 0.0])})()}

        def fit(self, *_args: object, **_kwargs: object) -> "Model":
            warnings.warn("controlled failure", ConvergenceWarning)
            return self

    monkeypatch.setattr(module, "make_pipeline", lambda *_args, **_kwargs: Model())
    x = np.random.default_rng(2).normal(size=(80, 2))
    y = x[:, 0]
    with pytest.raises(RuntimeError, match="2/2"):
        module.blocked_stability_selection(
            x,
            y,
            block_size=10,
            n_resamples=2,
            alpha=0.1,
        )


def test_temporal_calibration_reports_murphy_terms_and_lineage() -> None:
    generator = np.random.default_rng(88)
    scores = generator.normal(size=220)
    labels = generator.binomial(1, 1.0 / (1.0 + np.exp(-scores)))
    score_times = pd.date_range("2024-01-01", periods=220, freq="D", tz="UTC")
    known_times = score_times + pd.Timedelta(hours=1)
    result = temporal_calibration(
        scores,
        labels,
        score_timestamps=score_times,
        label_known_timestamps=known_times,
        model_fit_ids=["oof-model-1"] * len(scores),
        calibration_indices=np.arange(20, 140),
        evaluation_indices=np.arange(160, 220),
        method="platt",
        bins=8,
    )
    metrics = result.calibrated_metrics
    assert abs(metrics.decomposition_residual) < 1.0e-12
    assert sum(row.count for row in metrics.reliability_bins) == 60
    assert len(result.lineage_digest) == 64
    with pytest.raises(ValueError, match="disjoint"):
        temporal_calibration(
            scores,
            labels,
            score_timestamps=score_times,
            label_known_timestamps=known_times,
            model_fit_ids=["oof"] * len(scores),
            calibration_indices=np.arange(20, 140),
            evaluation_indices=np.arange(130, 200),
        )


def test_kalman_qr_is_train_only_and_holdout_uses_predicted_state() -> None:
    generator = np.random.default_rng(123)
    x = generator.normal(size=180)
    beta = np.linspace(0.8, 1.1, len(x))
    y = 0.2 + beta * x + generator.normal(scale=0.15, size=len(x))
    fit = fit_dynamic_beta_state_space(
        x[:130],
        y[:130],
        initial_intercept=0.2,
        initial_beta=0.8,
    )
    evaluated = evaluate_dynamic_beta_holdout(fit, x[130:], y[130:])
    assert evaluated.path.predicted_state.shape == (50, 2)
    assert evaluated.one_step_rmse < 0.5
    assert evaluated.diagnostics.observations == 50
    mutated = y.copy()
    mutated[130:] += 100.0
    same_fit = fit_dynamic_beta_state_space(
        x[:130],
        mutated[:130],
        initial_intercept=0.2,
        initial_beta=0.8,
    )
    assert same_fit.training_input_digest == fit.training_input_digest


def test_half_open_boundaries_and_post_test_embargo_are_explicit() -> None:
    intervals = pd.DataFrame(
        {
            "start": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
                utc=True,
            ),
            "end": pd.to_datetime(
                ["2025-01-03", "2025-01-04", "2025-01-04", "2025-01-05"],
                utc=True,
            ),
        },
        index=["touching", "overlap", "test", "post-test"],
    )
    result = purged_embargo_split(
        intervals,
        ["test"],
        candidate_train_indices=["touching", "overlap", "post-test"],
        mode="combinatorial",
        embargo=timedelta(days=1),
    )
    assert "touching" in result.train_index
    assert "overlap" in result.purged_index
    assert "post-test" in result.embargo_index


def test_drift_history_rejects_caller_fabrication_and_nonfinite_scale() -> None:
    policy = DriftPolicy("p", 0.1, 0.0, 0.1, 2)
    reference = np.full(20, 0.5)
    live = np.linspace(0.2, 0.8, 20)
    first = drift_evidence(
        reference,
        live,
        policy=policy,
        window_id="1",
        window_start="2025-01-01T00:00:00Z",
        window_end="2025-01-02T00:00:00Z",
        model_hash="m",
        data_hash="d1",
        evidence_id="e1",
    )
    assert first.std_ratio is None and first.scale_status == "DEGENERATE_REFERENCE"
    assert not first.actionable
    with pytest.raises(ValueError, match="contiguous"):
        drift_evidence(
            reference,
            live,
            policy=policy,
            window_id="2",
            window_start="2025-01-03T00:00:00Z",
            window_end="2025-01-04T00:00:00Z",
            model_hash="m",
            data_hash="d2",
            evidence_id="e2",
            history=[first.current_record],
        )


def test_nested_economic_evaluation_freezes_inner_threshold_and_reconciles() -> None:
    generator = np.random.default_rng(19)
    observations = 140
    scores = generator.normal(size=observations)
    probability = 1.0 / (1.0 + np.exp(-scores))
    returns = 0.004 * np.sign(scores) + generator.normal(scale=0.003, size=observations)
    timestamps = pd.date_range("2024-01-01", periods=observations, freq="D", tz="UTC")
    folds = nested_walk_forward_splits(
        observations,
        outer_min_train=80,
        outer_test_size=20,
        inner_min_train=40,
        inner_validation_size=10,
    )
    result = nested_economic_evaluation(
        probability,
        returns,
        timestamps=timestamps,
        return_known_timestamps=timestamps + pd.Timedelta(minutes=1),
        nested_folds=folds,
        thresholds=[0.55, 0.65, 0.75],
        cost_per_unit_turnover=0.0002,
        risk_aversion=0.5,
        assets_under_management=[1.0, 2.0],
        impact_coefficient=1.0e-5,
    )
    assert len(result.folds) == 3
    assert result.attribution_residual == pytest.approx(0.0)
    assert result.capacity_curve[1].impact_cost > result.capacity_curve[0].impact_cost


def test_direct_public_helpers_have_boundary_evidence() -> None:
    values = simulate_ar1(0.8, 80, seed=22)
    assert values.shape == (80,) and np.all(np.isfinite(values))
    break_values = np.r_[np.zeros(60), np.ones(60)] + np.random.default_rng(3).normal(0, 0.01, 120)
    break_index, score = largest_mean_break(break_values)
    assert break_index is not None and 50 <= break_index <= 70 and score > 1.5
    assert np.array_equal(positions_from_probability([0.8, 0.5, 0.2], 0.7), [1, 0, -1])
    assert newey_west_long_run_variance([1.0, -1.0, 1.0, -1.0], max_lag=1) >= 0.0


def _python_probability(features: list[float]) -> float:
    score = -0.15 + sum(
        coefficient * feature
        for coefficient, feature in zip(
            [0.35, -0.20, 0.10, 0.05],
            features,
        )
    )
    if score >= 0.0:
        exponential = math.exp(-score)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(score)
    return exponential / (1.0 + exponential)


def test_cpp_online_scorer_strict_build_and_python_parity(tmp_path: Path) -> None:
    compiler = shutil.which("c++") or shutil.which("clang++")
    if compiler is None:
        pytest.fail("C++ compiler is required for the CH08 release suite")
    source = ROOT / "_book/cpp/ch08/online_scorer.cpp"
    binary = tmp_path / "online_scorer"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-pedantic",
            "-Werror",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fixtures = [
        [1.2, -0.4, 0.8, 0.1],
        [1000.0, 0.0, 0.0, 0.0],
        [-1000.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    canonical_schema = (
        '{"features":[{"name":"stationarity_score","transformation":"identity",'
        '"type":"float64","unit":"score"},{"name":"regime_probability",'
        '"transformation":"identity","type":"float64","unit":"probability"},'
        '{"name":"volatility_ratio","transformation":"identity","type":"float64",'
        '"unit":"ratio"},{"name":"calibration_residual","transformation":"identity",'
        '"type":"float64","unit":"probability"}],'
        '"schema_version":"ch08-feature-schema-v1"}'
    )
    expected_hash = "sha256:" + hashlib.sha256(canonical_schema.encode()).hexdigest()
    names = [
        "stationarity_score",
        "regime_probability",
        "volatility_ratio",
        "calibration_residual",
    ]
    for features in fixtures:
        completed = subprocess.run(
            [
                str(binary),
                expected_hash,
                *(f"{name}={value}" for name, value in zip(names, features)),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        version, schema_hash, probability_text = completed.stdout.split()
        assert version == "ch08-demo-v3"
        assert schema_hash == expected_hash
        assert float(probability_text) == pytest.approx(
            _python_probability(features),
            abs=1.0e-15,
            rel=1.0e-15,
        )
    for arguments, reason in [
        (["sha256:" + "0" * 64, *(f"{n}=0" for n in names)], "schema"),
        ([expected_hash, "regime_probability=0", "stationarity_score=0", "volatility_ratio=0", "calibration_residual=0"], "order"),
        ([expected_hash, "stationarity_score=nan", "regime_probability=0", "volatility_ratio=0", "calibration_residual=0"], "finite"),
        ([expected_hash, "stationarity_score=1.2x", "regime_probability=0", "volatility_ratio=0", "calibration_residual=0"], "finite"),
    ]:
        rejected = subprocess.run(
            [str(binary), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode == 2
        payload = json.loads(rejected.stderr)
        assert payload["status"] == "REJECT"
        assert reason in payload["reason"]
