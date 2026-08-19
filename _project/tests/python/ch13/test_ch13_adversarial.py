from __future__ import annotations

import math

import numpy as np
import pytest

from quant_interviews.ch13.fermi_volume import (
    DecompositionEstimate,
    Factor,
    exact_trading_notional,
    point_estimate,
    reconcile_decompositions,
    reconcile_log_estimates,
    uncertainty_contributions,
)
from quant_interviews.ch13.gambler_ruin import (
    hit_upper_probability,
    simulate,
    simulate_certificate,
)
from quant_interviews.ch13.hash_collisions import (
    collision_certificate,
    required_buckets_certificate,
)
from quant_interviews.ch13.matrix_invariant import reduction_operations
from quant_interviews.ch13.one_bit_coordination import (
    Protocol,
    coordination_certificate,
    randomized_maximin,
)
from quant_interviews.ch13.sealed_bid import equilibrium_bid, simulate_revenue
from quant_interviews.ch13.search_stopping import (
    DiscreteDistribution,
    finite_horizon_decision,
    finite_horizon_value,
)
from quant_interviews.ch13.secretary import (
    simulate as simulate_secretary,
    simulate_classical_cutoff_under_noise,
)
from quant_interviews.ch13.selection_trap import (
    seeded_monte_carlo_cross_check,
    simulate as simulate_selection,
)
from quant_interviews.ch13.unbiased_bits import (
    VonNeumannExtractor,
    efficiency,
    iid_efficiency,
    von_neumann_extract,
)


def test_fermi_overflow_zero_turnover_and_duplicate_factors_fail_closed() -> None:
    with pytest.raises(OverflowError):
        point_estimate({"a": 1e200, "b": 1e200})
    with pytest.raises(OverflowError):
        exact_trading_notional([1e200], [1e200], [1.0])
    with pytest.raises(ValueError):
        exact_trading_notional([100.0], [10.0], [0.0])
    duplicate = (Factor("x", 2.0, 0.1), Factor("x", 3.0, 0.2))
    with pytest.raises(ValueError, match="unique"):
        uncertainty_contributions(duplicate)
    deterministic = (Factor("x", 2.0, 0.0),)
    with pytest.raises(ValueError, match="zero variance"):
        uncertainty_contributions(deterministic)


def test_fermi_covariance_and_typed_decomposition_evidence_are_enforced() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        reconcile_log_estimates(100.0, 105.0, -1.0, -1.0, -100.0)
    with pytest.raises(ValueError, match="Cauchy"):
        reconcile_log_estimates(100.0, 105.0, 0.1, 0.1, 0.02)
    first = DecompositionEstimate(
        100.0, 0.1, "USD/day", "US equities", "single-counted",
        "issuer aggregation", ("source-a",), "2026-08-02",
    )
    second = DecompositionEstimate(
        101.0, 0.1, "USD/day", "US equities", "single-counted",
        "market-cap turnover", ("source-b",), "2026-08-02",
    )
    assert reconcile_decompositions(first, second).status == "PASS"
    shared = DecompositionEstimate(
        101.0, 0.1, "USD/day", "US equities", "single-counted",
        "market-cap turnover", ("source-a",), "2026-08-02",
    )
    assert reconcile_decompositions(first, shared).status == "REJECT"
    incompatible = DecompositionEstimate(
        101.0, 0.1, "shares/day", "US equities", "single-counted",
        "market-cap turnover", ("source-b",), "2026-08-02",
    )
    with pytest.raises(ValueError, match="units"):
        reconcile_decompositions(first, incompatible)


def test_extractor_constructor_and_update_are_atomic() -> None:
    with pytest.raises(ValueError):
        VonNeumannExtractor(pending_bit=2)
    extractor = VonNeumannExtractor()
    extractor.update([0, 1])
    before = (
        extractor.pending_bit,
        extractor.total_tosses,
        extractor.accepted_pairs,
        extractor.rejected_pairs,
        tuple(extractor._outputs),
    )
    with pytest.raises(ValueError):
        extractor.update([0, 2])
    after = (
        extractor.pending_bit,
        extractor.total_tosses,
        extractor.accepted_pairs,
        extractor.rejected_pairs,
        tuple(extractor._outputs),
    )
    assert after == before


def test_secretary_certificate_distinguishes_reconciliation_from_estimation() -> None:
    insufficient = simulate_classical_cutoff_under_noise(10, 3, trials=1, seed=1)
    assert insufficient.status == "REJECT"
    assert not insufficient.evidence_sufficient
    exact = simulate_classical_cutoff_under_noise(20, 7, trials=5_000, seed=2)
    assert exact.status == "PASS"
    assert exact.analytic_probability is not None
    noisy = simulate_classical_cutoff_under_noise(
        20, 7, trials=5_000, seed=2, noise_sigma=0.05
    )
    assert noisy.status == "CONDITIONAL_PASS"
    assert noisy.analytic_probability is None


def test_bellman_certificate_covers_full_frozen_state_table() -> None:
    distribution = DiscreteDistribution((0.0, 0.5, 1.0), (0.25, 0.5, 0.25))
    result = finite_horizon_decision(distribution, 0.05, 4, 0.1)
    assert result.verified_states == 4 * 5
    assert result.bellman_residual <= 1e-12
    assert result.policy_consistency_residual <= 1e-12
    assert result.status == "PASS"


def test_auction_domain_and_versioned_tie_rule_are_enforced() -> None:
    with pytest.raises(ValueError):
        equilibrium_bid(0.8, 2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tie_rule"):
        simulate_revenue(4, trials=100, tie_rule="undefined")


def test_ruin_truncation_rejects_and_near_fair_value_is_stable() -> None:
    rejected = simulate_certificate(1, 3, 0.5, trials=2, seed=0, maximum_steps=1)
    assert rejected.truncated_paths > 0
    assert rejected.status == "REJECT"
    assert rejected.hit_upper_probability is None
    assert rejected.duration_standard_error is None
    assert hit_upper_probability(400, 1000, 0.500000000001) == pytest.approx(
        0.40000000048, abs=1e-12
    )


def test_selection_one_report_cannot_certify_analytic_posterior() -> None:
    result = seeded_monte_carlo_cross_check(
        0.9, 0.8, 0.05, 1, trials=1, seed=0
    )
    assert result.status == "REJECT"
    assert not result.evidence_sufficient
    assert math.isfinite(result.reconciliation_tolerance)


def test_hash_certificate_enforces_domain_assumptions_and_exact_minimality() -> None:
    with pytest.raises(ValueError):
        collision_certificate(10, 3)
    rejected = collision_certificate(10, 100, uniform_independent_hashes=False)
    assert rejected.status == "REJECT"
    sizing = required_buckets_certificate(100_000, 1e-7)
    assert sizing.status == "PASS"
    assert sizing.risk_at_buckets <= sizing.maximum_probability
    assert sizing.risk_at_previous_bucket > sizing.maximum_probability


def test_protocol_constructor_state_domain_hash_and_timing_are_governed() -> None:
    with pytest.raises(ValueError):
        Protocol((0, 1, 0), (0, 0, 0), (0, 0, 0, 0, 0, 0))
    protocol = Protocol((0, 0, 0), (1, 0, 0), (1, 2, 0, 2, 0, 1))
    with pytest.raises(ValueError):
        protocol.success(-1, 0)
    certificate = coordination_certificate()
    assert len(certificate.enumeration_sha256) == 64
    assert "after adversary commits" in certificate.shared_randomness_assumption
    assert "before the shared-seed" in certificate.adversary_assumption


def test_public_wrappers_match_their_certified_implementations() -> None:
    assert von_neumann_extract([0, 1, 1, 0]) == [0, 1]
    assert efficiency(0.3) == iid_efficiency(0.3)
    distribution = DiscreteDistribution((0.0, 1.0), (0.5, 0.5))
    assert finite_horizon_value(distribution, 0.1, 2, 0.0) == finite_horizon_decision(
        distribution, 0.1, 2, 0.0
    ).value
    assert 0.0 <= simulate_secretary(20, 7, trials=1_000, seed=3) <= 1.0
    ruin_probability, ruin_duration = simulate(4, 10, 0.55, trials=2_000, seed=4)
    assert 0.0 <= ruin_probability <= 1.0 and ruin_duration > 0.0
    assert 0.0 <= simulate_selection(0.02, 0.8, 0.05, 20, trials=2_000, seed=5) <= 1.0
    assert simulate_revenue(4, trials=2_000, seed=6) > 0.0
    value, weights = randomized_maximin()
    assert value == pytest.approx(0.5)
    assert np.sum(weights) == pytest.approx(1.0)
    assert reduction_operations(
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    ) == [(0, 2, 0, 2), (0, 2, 1, 2), (1, 2, 0, 2), (1, 2, 1, 2)]
