"""Unit and simulation verification for Chapter 1."""

from __future__ import annotations

import numpy as np
import pytest

from quant_interviews.ch01.bayes import (
    beta_binomial_edge_update,
    gaussian_log_predictive_score,
    latent_mixture_expectation,
    latent_mixture_moments,
    normal_normal_posterior,
    positive_predictive_value,
)
from quant_interviews.ch01.dependence import (
    correlation_and_quadratic_dependence,
    empirical_lower_tail_dependence,
    simpson_group_rates,
    simpson_weighted_means,
)
from quant_interviews.ch01.regression import newey_west_covariance
from quant_interviews.ch01.robust import huber_location, median_mad
from quant_interviews.ch01.sampling import (
    annualized_sharpe_standard_error,
    bootstrap_sharpe,
    moving_block_bootstrap_indices,
    moving_block_bootstrap_mean_ci,
    normal_mle,
    stationary_block_bootstrap_indices,
)
from quant_interviews.ch01.testing import (
    alpha_spending_schedule,
    benjamini_hochberg,
    expected_max_standard_normal,
    gaussian_e_process_null_crossing_simulation,
    gaussian_mean_e_process,
    minimum_detectable_mean,
    winner_curse_gaps,
    winner_curse_distribution,
    winner_curse_simulation,
)
from quant_interviews.ch01.utility import (
    chapter_utility_examples,
    expected_log_growth,
    mean_variance_certainty_equivalent,
    research_utility_breakdown,
)


def test_ppv_matches_low_base_rate_reference_case() -> None:
    actual = positive_predictive_value(
        prevalence=0.005,
        sensitivity=0.90,
        specificity=0.90,
    )

    assert actual == pytest.approx(0.04327, rel=1e-4)


def test_ppv_rejects_zero_probability_conditioning() -> None:
    with pytest.raises(ValueError, match="zero probability"):
        positive_predictive_value(
            prevalence=0.0,
            sensitivity=0.0,
            specificity=1.0,
        )


def test_mixture_moments_include_between_regime_variance() -> None:
    result = latent_mixture_moments(
        weights=[0.5, 0.5],
        conditional_means=[0.005, 0.020],
        conditional_variances=[0.005**2, 0.020**2],
    )

    assert result["unconditional_mean"] == pytest.approx(0.0125)
    assert result["within_regime_variance"] == pytest.approx(0.0002125)
    assert result["between_regime_variance"] == pytest.approx(0.00005625)
    assert result["total_variance"] == pytest.approx(0.00026875)
    assert result["unconditional_sharpe"] == pytest.approx(0.7622, rel=1e-3)


def test_mixture_moments_reject_undefined_zero_variance_sharpe() -> None:
    with pytest.raises(ValueError, match="Sharpe is undefined"):
        latent_mixture_moments(
            weights=[0.5, 0.5],
            conditional_means=[0.0, 0.0],
            conditional_variances=[0.0, 0.0],
        )


def test_latent_mixture_expectation_does_not_require_positive_variance() -> None:
    assert latent_mixture_expectation([0.5, 0.5], [1.0, 1.0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("weights", "means", "message"),
    [
        ([0.5, 0.5], [1.0], "align"),
        ([0.5, np.nan], [1.0, 2.0], "finite"),
        ([0.4, 0.4], [1.0, 2.0], "sum to one"),
    ],
)
def test_latent_mixture_expectation_rejects_invalid_inputs(
    weights: list[float],
    means: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        latent_mixture_expectation(weights, means)


def _dependence_fixtures() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(20260729)
    correlation = np.array([[1.0, 0.5], [0.5, 1.0]])
    gaussian = generator.multivariate_normal(
        mean=[0.0, 0.0],
        cov=correlation,
        size=80_000,
    )
    scale = np.sqrt(
        4.0 / generator.chisquare(df=4, size=gaussian.shape[0])
    )
    return gaussian, gaussian * scale[:, None]


def test_t_copula_fixture_has_more_joint_downside_than_gaussian() -> None:
    gaussian, student = _dependence_fixtures()
    gaussian_diagnostic = empirical_lower_tail_dependence(
        gaussian[:, 0],
        gaussian[:, 1],
        quantile=0.025,
    )
    student_diagnostic = empirical_lower_tail_dependence(
        student[:, 0],
        student[:, 1],
        quantile=0.025,
    )

    assert gaussian_diagnostic["pearson_correlation"] == pytest.approx(
        student_diagnostic["pearson_correlation"],
        abs=0.04,
    )
    assert (
        student_diagnostic["lower_tail_dependence"]
        > gaussian_diagnostic["lower_tail_dependence"] + 0.08
    )


def test_quadratic_dependence_extension_detects_nonlinearity() -> None:
    x = np.linspace(-1.0, 1.0, 501)
    diagnostic = correlation_and_quadratic_dependence(x, x * x)

    assert diagnostic["linear_correlation"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostic["quadratic_dependence"] > 0.99


@pytest.mark.parametrize(
    "diagnostic",
    [
        lambda: empirical_lower_tail_dependence(np.ones(20), np.arange(20)),
        lambda: correlation_and_quadratic_dependence(
            [0.0, 1.0, np.nan],
            [0.0, 1.0, 2.0],
        ),
        lambda: correlation_and_quadratic_dependence(
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 2.0],
        ),
    ],
)
def test_dependence_diagnostics_reject_undefined_or_nonfinite_samples(
    diagnostic: object,
) -> None:
    with pytest.raises(ValueError):
        diagnostic()


def test_simpson_return_fixture_reverses_regime_rankings() -> None:
    return_sums = np.array([[0.2, -0.9], [0.9, -0.2]])
    observation_counts = np.array([[10.0, 90.0], [90.0, 10.0]])

    result = simpson_weighted_means(return_sums, observation_counts)
    regime_means = result["regime_means"]
    aggregate_means = result["aggregate_means"]

    np.testing.assert_allclose(
        regime_means,
        [[0.02, -0.01], [0.01, -0.02]],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        aggregate_means,
        [-0.007, 0.007],
        rtol=0.0,
        atol=1e-15,
    )
    assert regime_means[0, 0] > regime_means[1, 0]
    assert regime_means[0, 1] > regime_means[1, 1]
    assert aggregate_means[0] < aggregate_means[1]


def test_simpson_group_rates_is_a_checked_compatibility_wrapper() -> None:
    successes = [[20.0, 30.0], [10.0, 40.0]]
    totals = [[100.0, 100.0], [100.0, 100.0]]

    wrapped = simpson_group_rates(successes, totals)
    direct = simpson_weighted_means(successes, totals)

    assert wrapped["group_rates"] == pytest.approx(direct["regime_means"])
    assert wrapped["aggregate_rates"] == pytest.approx(direct["aggregate_means"])


def test_annualized_sharpe_standard_error_matches_reference_case() -> None:
    actual = annualized_sharpe_standard_error(
        annualized_sharpe=1.0,
        observations=756,
        periods_per_year=252.0,
        skewness=0.0,
        excess_kurtosis=0.0,
    )

    assert actual == pytest.approx(0.578, abs=0.001)
    assert 1.96 * actual == pytest.approx(1.133, abs=0.002)


def test_annualized_sharpe_15_reference_interval_is_locked() -> None:
    standard_error = annualized_sharpe_standard_error(
        annualized_sharpe=1.5,
        observations=756,
        periods_per_year=252.0,
    )

    assert standard_error == pytest.approx(0.57864, abs=1e-5)
    assert 1.5 - 1.96 * standard_error == pytest.approx(0.3659, abs=1e-4)
    assert 1.5 + 1.96 * standard_error == pytest.approx(2.6341, abs=1e-4)


def test_annualized_sharpe_se_scales_and_reflects_higher_moments() -> None:
    baseline = annualized_sharpe_standard_error(1.0, observations=252)

    assert annualized_sharpe_standard_error(
        1.0,
        observations=1_008,
    ) == pytest.approx(0.5 * baseline)
    assert annualized_sharpe_standard_error(
        1.0,
        observations=252,
        skewness=-1.0,
    ) > baseline
    assert annualized_sharpe_standard_error(
        1.0,
        observations=252,
        excess_kurtosis=4.0,
    ) > baseline
    with pytest.raises(ValueError, match="exceed one"):
        annualized_sharpe_standard_error(1.0, observations=1)


def test_moving_block_bootstrap_preserves_contiguous_blocks() -> None:
    first = moving_block_bootstrap_indices(12, block_length=3, draws=4, seed=7)
    second = moving_block_bootstrap_indices(12, block_length=3, draws=4, seed=7)

    assert np.array_equal(first, second)
    assert np.all((first >= 0) & (first < 12))
    assert np.all(np.diff(first.reshape(-1, 3), axis=1) == 1)


def test_moving_block_bootstrap_mean_ci_matches_reference_fixture() -> None:
    returns = np.array(
        [0.002, 0.003, -0.001, 0.004, 0.006, 0.002, -0.002, 0.001]
    )

    result = moving_block_bootstrap_mean_ci(
        returns,
        block_length=3,
        draws=2_000,
        confidence=0.95,
        periods_per_year=252.0,
        seed=11,
    )
    repeated = moving_block_bootstrap_mean_ci(
        returns,
        block_length=3,
        draws=2_000,
        confidence=0.95,
        periods_per_year=252.0,
        seed=11,
    )
    changed_seed = moving_block_bootstrap_mean_ci(
        returns,
        block_length=3,
        draws=2_000,
        confidence=0.95,
        periods_per_year=252.0,
        seed=12,
    )

    assert result["annualized_mean"] == pytest.approx(0.4725)
    assert result["lower"] == pytest.approx(0.1575)
    assert result["upper"] == pytest.approx(0.9135)
    assert np.asarray(result["bootstrap_draws"]).shape == (2_000,)
    assert np.array_equal(result["bootstrap_draws"], repeated["bootstrap_draws"])
    assert not np.array_equal(
        result["bootstrap_draws"],
        changed_seed["bootstrap_draws"],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"returns": [0.0, np.nan], "block_length": 1},
        {"returns": [0.0, 1.0], "block_length": 0},
        {"returns": [0.0, 1.0], "block_length": 1, "confidence": 1.0},
        {"returns": [0.0, 1.0], "block_length": 1, "draws": 0},
        {"returns": [0.0, 1.0], "block_length": 1, "periods_per_year": np.inf},
    ],
)
def test_moving_block_bootstrap_mean_ci_rejects_invalid_designs(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        moving_block_bootstrap_mean_ci(**kwargs)


def test_stationary_bootstrap_and_sharpe_are_deterministic_and_finite() -> None:
    first_rng = np.random.default_rng(31)
    second_rng = np.random.default_rng(31)
    first_indices = stationary_block_bootstrap_indices(40, 5.0, first_rng)
    second_indices = stationary_block_bootstrap_indices(40, 5.0, second_rng)
    returns = np.linspace(-0.03, 0.04, 80) + 0.002 * np.sin(np.arange(80))

    first = bootstrap_sharpe(returns, expected_block=4.0, draws=100, seed=9)
    second = bootstrap_sharpe(returns, expected_block=4.0, draws=100, seed=9)

    assert np.array_equal(first_indices, second_indices)
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))


@pytest.mark.parametrize("returns", [[1.0, 1.0, 1.0], [0.0, np.nan, 1.0]])
def test_bootstrap_sharpe_fails_closed_for_undefined_draws(
    returns: list[float],
) -> None:
    with pytest.raises(ValueError):
        bootstrap_sharpe(returns, expected_block=2.0, draws=10, seed=1)


def test_normal_mle_matches_closed_form_fixture() -> None:
    result = normal_mle([1.0, 2.0, 3.0])

    assert result["mean"] == pytest.approx(2.0)
    assert result["variance_mle"] == pytest.approx(2.0 / 3.0)
    assert result["fisher_information_mean"] == pytest.approx(4.5)


def test_normal_mle_rejects_zero_variance_boundary() -> None:
    with pytest.raises(ValueError, match="boundary"):
        normal_mle([2.0, 2.0, 2.0])


def test_robust_estimators_resist_single_large_contamination() -> None:
    observations = [0.0, 0.0, 1.0, 100.0]

    robust_scale = median_mad(observations)
    huber = huber_location(observations)

    assert robust_scale["median"] == pytest.approx(0.5)
    assert huber["estimate"] < 2.0
    assert huber["converged"] is True
    assert huber["final_step"] <= huber["tolerance"]


def test_huber_reports_scale_and_nonconvergence_without_silent_success() -> None:
    converged = huber_location(
        [0.0, 0.0, 1.0, 100.0],
        scale=1.0,
        tolerance=1e-10,
        iterations=100,
    )
    stopped = huber_location(
        [0.0, 0.0, 1.0, 100.0],
        scale=1.0,
        tolerance=1e-30,
        iterations=1,
    )

    assert converged["scale"] == pytest.approx(1.0)
    assert converged["scale_method"] == "caller_supplied"
    assert converged["converged"] is True
    assert stopped["iterations_used"] == 1
    assert stopped["converged"] is False
    assert stopped["final_step"] > stopped["tolerance"]


def test_benjamini_hochberg_returns_expected_discoveries() -> None:
    rejected = benjamini_hochberg([0.001, 0.01, 0.20, 0.50], alpha=0.05)

    assert rejected.tolist() == [True, True, False, False]


@pytest.mark.parametrize("pvalues", [[0.01, np.nan], [0.01, np.inf]])
def test_benjamini_hochberg_rejects_nonfinite_pvalues(
    pvalues: list[float],
) -> None:
    with pytest.raises(ValueError, match="valid p-values"):
        benjamini_hochberg(pvalues)


def test_expected_null_maximum_grows_with_search_family() -> None:
    assert expected_max_standard_normal(100) == pytest.approx(
        2.498590560962256,
    )
    assert expected_max_standard_normal(1_000) > expected_max_standard_normal(10)


def test_alpha_spending_matches_the_infinite_telescoping_schedule() -> None:
    schedule = alpha_spending_schedule(total_alpha=0.05, looks=5)

    assert schedule[0] == pytest.approx(0.025)
    assert schedule.sum() == pytest.approx(0.05 * 5.0 / 6.0)
    assert schedule.sum() < 0.05
    assert 0.05 - schedule.sum() == pytest.approx(0.05 / 6.0)


def test_gaussian_mean_e_process_matches_closed_form_and_crossing_rule() -> None:
    null_path = gaussian_mean_e_process(np.zeros(20), prior_variance=0.5)
    signal_path = gaussian_mean_e_process(np.full(20, 1.5), prior_variance=0.5)

    assert np.all(null_path > 0.0)
    assert np.all(null_path <= 1.0)
    assert signal_path[-1] > 20.0
    assert signal_path[-1] == pytest.approx(
        np.exp(0.25 * 30.0**2 / 11.0) / np.sqrt(11.0)
    )


def test_gaussian_mean_e_process_controls_null_crossing_in_simulation() -> None:
    evidence = gaussian_e_process_null_crossing_simulation(
        paths=20_000,
        horizon=250,
        threshold=20.0,
        prior_variance=0.5,
        confidence=0.95,
        seed=7,
    )

    assert evidence["crossing_rate"] <= 0.05
    assert evidence["upper_confidence_bound"] <= 0.05


def test_mde_matches_reference_value_and_inverse_root_n_scaling() -> None:
    mde_100 = minimum_detectable_mean(std=1.0, n=100)
    mde_400 = minimum_detectable_mean(std=1.0, n=400)

    assert mde_100 == pytest.approx(0.28)
    assert mde_400 == pytest.approx(0.5 * mde_100)
    with pytest.raises(ValueError, match="greater than one"):
        minimum_detectable_mean(std=1.0, n=1)


def test_winner_curse_is_positive_across_repeated_experiments() -> None:
    result = winner_curse_distribution(
        models=100,
        observations=50,
        experiments=2_000,
        seed=2,
    )

    assert result["mean_selection_gap"] > 0.0
    assert result["mean_selection_gap"] == pytest.approx(
        result["expected_null_maximum"],
        rel=0.15,
    )
    assert np.mean(np.asarray(result["selection_gaps"]) > 0.0) > 0.95


def test_winner_curse_wrappers_match_the_canonical_distribution() -> None:
    distribution = winner_curse_distribution(
        models=8,
        observations=40,
        experiments=1_000,
        true_mean=0.01,
        seed=17,
    )
    gaps = winner_curse_gaps(
        models=8,
        observations=40,
        experiments=1_000,
        true_mean=0.01,
        seed=17,
    )
    simulation = winner_curse_simulation(
        models=8,
        observations=40,
        true_mean=0.01,
        seed=17,
    )
    two_experiments = winner_curse_distribution(
        models=8,
        observations=40,
        experiments=2,
        true_mean=0.01,
        seed=17,
    )

    assert gaps == pytest.approx(distribution["selection_gaps"])
    assert simulation["selection_gap"] == pytest.approx(
        np.asarray(two_experiments["selection_gaps"])[0]
    )
    assert simulation["mean_selection_gap"] == pytest.approx(
        two_experiments["mean_selection_gap"]
    )


def test_integrated_case_expected_maximum_uses_120_models() -> None:
    assert 0.5 * expected_max_standard_normal(120) == pytest.approx(
        1.281202362699045,
    )


def test_beta_binomial_update_matches_the_derived_example() -> None:
    result = beta_binomial_edge_update(
        prior_alpha=8.0,
        prior_beta=8.0,
        wins=58,
        trials=100,
        lower_quantile=0.10,
        kelly_fraction=0.50,
    )

    assert result["posterior_alpha"] == 66.0
    assert result["posterior_beta"] == 50.0
    assert result["posterior_mean"] == pytest.approx(66.0 / 116.0)
    assert result["posterior_lower_quantile"] == pytest.approx(0.510, abs=0.002)
    assert result["conservative_kelly"] == pytest.approx(0.010, abs=0.002)


def test_normal_normal_extension_shrinks_toward_both_sources() -> None:
    result = normal_normal_posterior(
        prior_mean=0.0,
        prior_variance=1.0,
        sample_mean=1.0,
        observation_variance=1.0,
        n=4,
    )

    assert 0.0 < result["mean"] < 1.0
    assert result["variance"] < 1.0


def test_predictive_score_ranks_models_on_the_same_holdout() -> None:
    calibrated = gaussian_log_predictive_score([0.0], [0.0], [1.0])
    misspecified = gaussian_log_predictive_score([0.0], [2.0], [1.0])

    assert calibrated == pytest.approx(-0.5 * np.log(2.0 * np.pi))
    assert misspecified == pytest.approx(calibrated - 2.0)
    assert calibrated > misspecified


def test_newey_west_covariance_is_symmetric_positive_semidefinite() -> None:
    time = np.arange(50, dtype=float)
    design = np.column_stack([np.ones_like(time), time])
    residuals = np.sin(time)

    covariance = newey_west_covariance(design, residuals, max_lag=3)

    assert np.allclose(covariance, covariance.T, atol=1e-14)
    assert np.linalg.eigvalsh(covariance).min() > -1e-12


def test_newey_west_zero_lag_matches_manual_hc0_sandwich() -> None:
    design = np.array(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ]
    )
    residuals = np.array([0.5, -1.0, 0.25, 0.75])
    cross_product = design.T @ design
    bread = np.linalg.solve(cross_product, np.eye(cross_product.shape[0]))
    manual_meat = design.T @ np.diag(residuals**2) @ design
    expected = bread @ manual_meat @ bread

    actual = newey_west_covariance(design, residuals, max_lag=0)

    assert actual == pytest.approx(expected, abs=1e-14)


def test_newey_west_matches_independent_small_fixture() -> None:
    design = np.array(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    residuals = np.array([0.2, -0.4, 0.3, -0.1, 0.5])
    max_lag = 2
    scores = design * residuals[:, None]
    reference_meat = np.zeros((2, 2))
    for time_index in range(scores.shape[0]):
        reference_meat += np.outer(scores[time_index], scores[time_index])
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        for time_index in range(lag, scores.shape[0]):
            cross_term = np.outer(
                scores[time_index],
                scores[time_index - lag],
            )
            reference_meat += weight * (cross_term + cross_term.T)
    cross_product = design.T @ design
    reference_bread = np.linalg.solve(
        cross_product,
        np.eye(cross_product.shape[0]),
    )
    expected = reference_bread @ reference_meat @ reference_bread

    actual = newey_west_covariance(design, residuals, max_lag=max_lag)

    assert actual == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize(
    ("design", "residuals", "message"),
    [
        (
            [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
            [0.1, -0.2, 0.1],
            "full column rank",
        ),
        (
            [[1.0, 0.0], [1.0, np.nan], [1.0, 2.0]],
            [0.1, -0.2, 0.1],
            "finite",
        ),
        (
            [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
            [0.1, np.inf, 0.1],
            "finite",
        ),
    ],
)
def test_newey_west_rejects_unidentified_or_nonfinite_inputs(
    design: list[list[float]],
    residuals: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        newey_west_covariance(design, residuals, max_lag=1)


def test_research_utility_breakdown_reconciles_every_penalty() -> None:
    result = research_utility_breakdown(
        gross_alpha=0.12,
        trading_cost=0.09,
        variance=0.12**2,
        risk_aversion=3.0,
        tail_penalty=0.004,
        model_penalty=0.002,
        capacity_scale=0.25,
    )

    assert result["net_alpha"] == pytest.approx(0.03)
    assert result["variance_penalty"] == pytest.approx(0.0216)
    assert result["certainty_equivalent"] == pytest.approx(0.0024)
    assert result["capacity_adjusted_value"] == pytest.approx(0.0006)


def test_chapter_utility_examples_reconcile_table_and_figure_values() -> None:
    examples = chapter_utility_examples()

    expected = {
        "Signal A": {
            "gross_alpha": 0.06,
            "trading_cost": 0.01,
            "net_alpha": 0.05,
            "variance": 0.06**2,
            "risk_aversion": 3.0,
            "variance_penalty": 0.0054,
            "tail_penalty": 0.001,
            "model_penalty": 0.0005,
            "certainty_equivalent": 0.0431,
            "capacity_scale": 1.0,
            "capacity_adjusted_value": 0.0431,
        },
        "Signal B": {
            "gross_alpha": 0.12,
            "trading_cost": 0.09,
            "net_alpha": 0.03,
            "variance": 0.12**2,
            "risk_aversion": 3.0,
            "variance_penalty": 0.0216,
            "tail_penalty": 0.004,
            "model_penalty": 0.002,
            "certainty_equivalent": 0.0024,
            "capacity_scale": 0.25,
            "capacity_adjusted_value": 0.0006,
        },
    }
    for signal, fields in expected.items():
        for field, value in fields.items():
            assert examples[signal][field] == pytest.approx(value)


def test_positive_tail_penalty_reduces_certainty_equivalent_one_for_one() -> None:
    without_tail_penalty = research_utility_breakdown(0.05, 0.01, 0.02, 2.0)
    with_tail_penalty = research_utility_breakdown(
        0.05,
        0.01,
        0.02,
        2.0,
        tail_penalty=0.003,
    )

    assert (
        without_tail_penalty["certainty_equivalent"]
        - with_tail_penalty["certainty_equivalent"]
    ) == pytest.approx(0.003)


@pytest.mark.parametrize(
    "call",
    [
        lambda: latent_mixture_moments([1.0], [np.nan], [1.0]),
        lambda: gaussian_log_predictive_score([0.0], [np.inf], [1.0]),
        lambda: research_utility_breakdown(np.nan, 0.0, 1.0, 1.0),
    ],
)
def test_core_numeric_apis_reject_nonfinite_inputs(call: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        call()


def test_mean_variance_and_log_growth_extensions_remain_valid() -> None:
    certainty_equivalent = mean_variance_certainty_equivalent(
        mean_return=0.10,
        variance=0.04,
        risk_aversion=2.0,
        cost=0.01,
    )
    log_growth = expected_log_growth(
        win_probability=0.60,
        win_return=1.0,
        loss_return=-1.0,
        fraction=0.10,
    )

    assert certainty_equivalent == pytest.approx(0.05)
    assert log_growth > 0.0
