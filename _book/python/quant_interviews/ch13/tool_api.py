from __future__ import annotations

from hashlib import sha256
import json
from math import isclose, isfinite, log2
from typing import Any, Callable

import numpy as np

from .fermi_volume import Factor, analytic_lognormal_summary, point_estimate, uncertainty_contributions
from .gambler_ruin import expected_duration, hit_upper_probability
from .hash_collisions import (
    collision_certificate,
    probability_any_collision_exact,
    required_buckets_certificate,
)
from .matrix_invariant import reduction_certificate
from .one_bit_coordination import coordination_certificate
from .sealed_bid import equilibrium_bid
from .search_stopping import DiscreteDistribution, finite_horizon_decision
from .secretary import optimal_cutoff, success_probability
from .selection_trap import posterior_after_screen, posterior_prespecified


JSONDict = dict[str, Any]


def evidence_sha256(name: str, inputs: JSONDict, output_without_hash: JSONDict) -> str:
    payload = {
        "contractVersion": "0.10",
        "input": inputs,
        "output": output_without_hash,
        "tool": name,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def bind_evidence(name: str, inputs: JSONDict, output: JSONDict) -> JSONDict:
    result = dict(output)
    result["evidence_sha256"] = evidence_sha256(name, inputs, result)
    return result


def _close(actual: Any, expected: float, *, atol: float = 1e-10, rtol: float = 1e-9) -> bool:
    return (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isfinite(float(actual))
        and isclose(float(actual), expected, abs_tol=atol, rel_tol=rtol)
    )


def _estimate_fermi(inputs: JSONDict, output: JSONDict) -> list[str]:
    errors: list[str] = []
    factors = tuple(
        Factor(str(item["name"]), float(item["median"]), float(item["log_sigma"]))
        for item in inputs["factors"]
    )
    covariance = np.asarray(inputs["log_covariance"], dtype=float)
    analytic = analytic_lognormal_summary(factors, covariance)
    estimate = point_estimate({factor.name: factor.median for factor in factors})
    expected_contributions = uncertainty_contributions(factors, covariance)
    for key, expected in (("point_estimate", estimate), ("q025", analytic["q025"]),
                          ("median", analytic["median"]), ("q975", analytic["q975"])):
        if not _close(output.get(key), expected):
            errors.append(f"{key} is not the analytic factor-model result")
    contributions = output.get("variance_contributions", {})
    if set(contributions) != set(expected_contributions):
        errors.append("variance contribution identifiers do not match factors")
    elif any(not _close(contributions[name], value) for name, value in expected_contributions.items()):
        errors.append("variance contributions do not reconcile")
    if not output.get("q025", 1.0) <= output.get("median", 0.0) <= output.get("q975", -1.0):
        errors.append("quantiles are not ordered")
    if output.get("status") == "PASS" and output.get("reconciliation_residual", 1.0) > output.get("reconciliation_tolerance", 0.0):
        errors.append("PASS exceeds the reconciliation tolerance")
    return errors


def _extract_bits(inputs: JSONDict, output: JSONDict) -> list[str]:
    pending = inputs["pending_bit"]
    emitted: list[int] = []
    accepted = 0
    rejected = 0
    for bit in inputs["bits"]:
        if pending is None:
            pending = bit
        else:
            if pending == bit:
                rejected += 1
            else:
                emitted.append(pending)
                accepted += 1
            pending = None
    errors = []
    expected = {
        "output_bits": emitted,
        "consumed_tosses": len(inputs["bits"]),
        "accepted_pairs": accepted,
        "rejected_pairs": rejected,
        "pending_bit": pending,
    }
    if any(output.get(key) != value for key, value in expected.items()):
        errors.append("extractor output is not the atomic state transition")
    assumptions = {str(value).lower() for value in inputs["source_assumptions"]}
    if output.get("status") == "PASS" and not {"iid", "stationary"} <= assumptions:
        errors.append("PASS requires IID and stationary source assumptions")
    return errors


def _secretary(inputs: JSONDict, output: JSONDict) -> list[str]:
    expected_cutoff, expected_success = optimal_cutoff(int(inputs["n"]))
    errors = []
    if output.get("cutoff") != expected_cutoff:
        errors.append("cutoff is not the exact finite-n optimum")
    if not _close(output.get("exact_success"), expected_success):
        errors.append("exact_success is not P_n(cutoff)")
    if not (0 <= output.get("cutoff", -1) < inputs["n"]):
        errors.append("cutoff lies outside [0,n)")
    if output.get("wilson_lower", 1.0) > output.get("wilson_upper", 0.0):
        errors.append("Wilson interval is reversed")
    residual = output.get("analytic_residual")
    tolerance = output.get("reconciliation_tolerance")
    if output.get("status") == "PASS" and (
        inputs["noise_sigma_normalized"] != 0.0
        or not output.get("evidence_sufficient")
        or residual is None or tolerance is None or abs(residual) > tolerance
    ):
        errors.append("PASS requires sufficient zero-noise analytic reconciliation")
    return errors


def _search(inputs: JSONDict, output: JSONDict) -> list[str]:
    distribution = DiscreteDistribution(tuple(inputs["values"]), tuple(inputs["probabilities"]))
    decision = finite_horizon_decision(
        distribution,
        float(inputs["cost"]),
        int(inputs["draws_left"]),
        float(inputs["current_best"]),
    )
    errors = []
    for key, expected in (("value", decision.value), ("stop_value", decision.stop_value),
                          ("continuation_value", decision.continuation_value),
                          ("bellman_residual", decision.bellman_residual),
                          ("policy_consistency_residual", decision.policy_consistency_residual)):
        if not _close(output.get(key), expected):
            errors.append(f"{key} does not match the verified Bellman table")
    if output.get("action") != decision.action or output.get("verified_states") != decision.verified_states:
        errors.append("policy action/state coverage does not match the Bellman certificate")
    return errors


def _auction(inputs: JSONDict, output: JSONDict) -> list[str]:
    values = inputs["value_grid"]
    bids = output.get("bid_grid", [])
    expected = [
        equilibrium_bid(float(value), int(inputs["bidders"]), float(inputs["risk_aversion"]))
        for value in values
    ]
    errors = []
    if len(bids) != len(expected) or any(not _close(actual, target) for actual, target in zip(bids, expected)):
        errors.append("bid grid is not the declared symmetric equilibrium")
    if output.get("tie_rule") != inputs.get("tie_rule"):
        errors.append("tie rule is not preserved in the evidence")
    if output.get("status") == "PASS" and output.get("maximum_regret", 1.0) > output.get("regret_tolerance", 0.0):
        errors.append("PASS exceeds maximum utility-regret tolerance")
    if output.get("status") == "PASS" and abs(output.get("simulated_revenue", 0.0) - output.get("analytic_revenue", 0.0)) > 4.0 * output.get("revenue_standard_error", 0.0):
        errors.append("PASS fails analytic/simulation revenue reconciliation")
    return errors


def _ruin(inputs: JSONDict, output: JSONDict) -> list[str]:
    probability = hit_upper_probability(int(inputs["initial"]), int(inputs["upper"]), float(inputs["p"]))
    duration = expected_duration(int(inputs["initial"]), int(inputs["upper"]), float(inputs["p"]))
    errors = []
    if not _close(output.get("hit_probability"), probability) or not _close(output.get("expected_duration"), duration):
        errors.append("analytic ruin values are incorrect")
    if output.get("status") == "PASS":
        fields = ("simulated_probability", "probability_standard_error", "simulated_duration",
                  "duration_standard_error", "probability_residual", "probability_tolerance",
                  "duration_residual", "duration_tolerance")
        if output.get("truncated_paths") != 0 or output.get("completed_paths") != inputs["trials"]:
            errors.append("PASS cannot contain truncated or missing paths")
        if any(output.get(field) is None for field in fields):
            errors.append("PASS requires finite Monte Carlo reconciliation fields")
        elif (abs(output["probability_residual"]) > output["probability_tolerance"]
              or abs(output["duration_residual"]) > output["duration_tolerance"]):
            errors.append("PASS exceeds a ruin reconciliation tolerance")
    return errors


def _selection(inputs: JSONDict, output: JSONDict) -> list[str]:
    pre = posterior_prespecified(inputs["prior_prespecified"], inputs["power"], inputs["alpha"])
    screened = posterior_after_screen(inputs["prior_program"], inputs["power"], inputs["alpha"], inputs["candidates"])
    signal = 1.0 - (1.0 - inputs["power"]) * (1.0 - inputs["alpha"]) ** (inputs["candidates"] - 1)
    null = 1.0 - (1.0 - inputs["alpha"]) ** inputs["candidates"]
    errors = []
    for key, expected in (("prespecified_posterior", pre), ("screened_posterior", screened),
                          ("report_likelihood_signal", signal), ("report_likelihood_null", null)):
        if not _close(output.get(key), expected):
            errors.append(f"{key} is inconsistent with the reporting protocol")
    if output.get("status") == "PASS" and (
        not inputs["conditional_independence"]
        or not output.get("evidence_sufficient")
        or abs(output.get("posterior_residual", 1.0)) > output.get("reconciliation_tolerance", 0.0)
    ):
        errors.append("PASS lacks protocol assumptions or posterior reconciliation")
    return errors


def _matrix(inputs: JSONDict, output: JSONDict) -> list[str]:
    matrix = np.asarray(inputs["matrix"], dtype=int)
    errors = []
    try:
        certificate = reduction_certificate(matrix)
    except ValueError:
        if output.get("status") == "PASS":
            errors.append("irreducible/invalid matrix cannot receive PASS")
        return errors
    if output.get("reducible") is not True or not output.get("replay_is_zero"):
        errors.append("matrix reduction certificate does not replay to zero")
    if output.get("row_parity") != list(certificate.row_parity_before) or output.get("column_parity") != list(certificate.column_parity_before):
        errors.append("reported matrix parities are incorrect")
    if len(output.get("operations", [])) != len(certificate.operations):
        errors.append("operation certificate length is inconsistent")
    return errors


def _hashing(inputs: JSONDict, output: JSONDict) -> list[str]:
    certificate = collision_certificate(
        int(inputs["items"]), int(inputs["buckets"]),
        uniform_independent_hashes=bool(inputs["uniform_independent_hashes"]),
    )
    sizing = required_buckets_certificate(
        int(inputs["items"]), float(inputs["maximum_probability"]),
        uniform_independent_hashes=bool(inputs["uniform_independent_hashes"]),
    )
    errors = []
    for key, expected in (("expected_colliding_pairs", certificate.expected_colliding_pairs),
                          ("exact_any_collision_probability", certificate.exact_any_collision_probability),
                          ("poisson_any_collision_probability", certificate.poisson_any_collision_probability),
                          ("risk_at_previous_bucket", sizing.risk_at_previous_bucket)):
        if not _close(output.get(key), expected):
            errors.append(f"{key} is not the exact hash certificate value")
    if output.get("required_buckets_exact_postcheck") != sizing.buckets:
        errors.append("required bucket count failed exact post-check")
    expected_bits = int(log2(sizing.buckets - 1)) + 1
    if output.get("required_bits") != expected_bits:
        errors.append("required_bits does not encode the exact bucket count")
    if output.get("uniform_independent_hashes") != inputs["uniform_independent_hashes"]:
        errors.append("hash assumption evidence is inconsistent")
    if output.get("status") == "PASS" and not inputs["uniform_independent_hashes"]:
        errors.append("PASS requires uniform independent hashes")
    return errors


def _coordination(inputs: JSONDict, output: JSONDict) -> list[str]:
    certificate = coordination_certificate()
    errors = []
    if abs(sum(inputs["state_distribution"]) - 1.0) > 1e-12:
        errors.append("state distribution must sum to one")
    if inputs["shared_randomness"] != "drawn_after_adversary_commitment_before_private_observations" or inputs["adversary"] != "commits_state_before_shared_seed":
        errors.append("move timing does not support the claimed maximin value")
    expected_scalars = (
        ("deterministic_protocol_count", certificate.deterministic_protocol_count),
        ("best_deterministic_average", certificate.best_deterministic_average),
        ("best_deterministic_worst_case", certificate.best_deterministic_worst_case),
        ("randomized_maximin_value", certificate.randomized_maximin_value),
        ("analytic_upper_bound", certificate.analytic_upper_bound),
        ("analytic_upper_bound_gap", certificate.analytic_upper_bound_gap),
    )
    for key, expected in expected_scalars:
        if isinstance(expected, int):
            if output.get(key) != expected:
                errors.append(f"{key} is inconsistent with enumeration")
        elif not _close(output.get(key), expected):
            errors.append(f"{key} is inconsistent with enumeration")
    if output.get("enumeration_sha256") != certificate.enumeration_sha256:
        errors.append("enumeration hash does not bind the canonical protocol set")
    if len(output.get("statewise_success", [])) != 9 or any(not _close(value, 0.5) for value in output.get("statewise_success", [])):
        errors.append("statewise maximin vector is not uniformly one half")
    if output.get("status") == "PASS" and errors:
        errors.append("PASS is inconsistent with the coordination evidence")
    return errors


_VALIDATORS: dict[str, Callable[[JSONDict, JSONDict], list[str]]] = {
    "estimate_fermi_distribution": _estimate_fermi,
    "extract_unbiased_bits": _extract_bits,
    "solve_secretary_policy": _secretary,
    "solve_search_stopping": _search,
    "evaluate_first_price_equilibrium": _auction,
    "solve_gambler_ruin": _ruin,
    "condition_on_selection_protocol": _selection,
    "verify_matrix_invariant": _matrix,
    "size_hash_space": _hashing,
    "solve_one_bit_coordination": _coordination,
}


def validate_tool_result(name: str, inputs: JSONDict, output: JSONDict) -> tuple[str, ...]:
    """Recompute cross-field semantics; JSON Schema alone cannot do this."""
    if name not in _VALIDATORS:
        return (f"unknown CH13 tool: {name}",)
    errors: list[str] = []
    evidence = output.get("evidence_sha256")
    without_hash = {key: value for key, value in output.items() if key != "evidence_sha256"}
    if evidence != evidence_sha256(name, inputs, without_hash):
        errors.append("evidence_sha256 does not bind this input/output pair")
    try:
        errors.extend(_VALIDATORS[name](inputs, output))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        errors.append(f"semantic validation failed: {error}")
    if output.get("status") == "PASS" and errors:
        errors.append("PASS is forbidden while semantic errors remain")
    return tuple(errors)
