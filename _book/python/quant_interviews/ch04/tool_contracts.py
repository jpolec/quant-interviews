"""Executable adapters for the Chapter 4 deterministic tool contracts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import hashlib
import json
from typing import Any, Callable

import numpy as np

from quant_interviews.ch04.correlated_diffusions import correlation_input_gate
from quant_interviews.ch04.exponential_martingale import density_process_diagnostic
from quant_interviews.ch04.feynman_kac import ou_terminal_and_running_payoff
from quant_interviews.ch04.gbm import exact_transition_audit
from quant_interviews.ch04.hitting_probabilities import hitting_convergence_study
from quant_interviews.ch04.predictability_guard import TemporalValue, select_as_known
from quant_interviews.ch04.quadratic_variation import quadratic_variation_convergence
from quant_interviews.ch04.sde_schemes import gbm_scheme_convergence


VERSION = "v0.10"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(nested) for nested in value]
    return value


def _metric(name: str, value: Any, unit: str) -> dict[str, Any]:
    converted = _jsonable(value)
    if not isinstance(converted, (str, int, float, bool)) and converted is not None:
        converted = json.dumps(converted, sort_keys=True, separators=(",", ":"))
    return {"name": name, "value": converted, "unit": unit}


def _validated_execution(request: dict[str, Any]) -> dict[str, Any]:
    execution = request["execution"]
    payload = {key: value for key, value in request.items() if key != "execution"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    actual = hashlib.sha256(encoded).hexdigest()
    if execution["input_sha256"] != actual:
        raise ValueError("input_sha256 does not match the canonical request payload")
    return execution


def _response(
    execution: dict[str, Any],
    *,
    status: str,
    reason: str,
    evidence: Any,
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_json = json.dumps(
        _jsonable(evidence), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return {
        "request_id": execution["request_id"],
        "implementation_version": VERSION,
        "input_sha256": execution["input_sha256"],
        "evidence_sha256": hashlib.sha256(evidence_json.encode()).hexdigest(),
        "status": status,
        "reason": reason,
        "metrics": metrics,
    }


def select_predictable_feature(request: dict[str, Any]) -> dict[str, Any]:
    execution = _validated_execution(request)
    rows = tuple(
        TemporalValue(
            key=row["key"],
            valid_from=datetime.fromisoformat(row["valid_from"]),
            valid_to=(None if row["valid_to"] is None else datetime.fromisoformat(row["valid_to"])),
            published_at=datetime.fromisoformat(row["published_at"]),
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
            value=row["value"],
            revision_id=row["revision_id"],
            superseded_at=(
                None
                if row["superseded_at"] is None
                else datetime.fromisoformat(row["superseded_at"])
            ),
            source_priority=row["source_priority"],
            payload_hash=row["payload_hash"],
        )
        for row in request["rows"]
    )
    selected = select_as_known(
        rows,
        key=request["key"],
        valid_at=datetime.fromisoformat(request["valid_at"]),
        decision_time=datetime.fromisoformat(request["decision_time"]),
    )
    return _response(
        execution,
        status="PASS",
        reason="point-in-time revision selected",
        evidence=selected,
        metrics=[
            _metric("selected_value", selected.value, request["value_unit"]),
            _metric("revision_id", selected.revision_id, "count"),
            _metric("known_at", selected.known_at.isoformat(), "timestamp"),
        ],
    )


def estimate_quadratic_variation(request: dict[str, Any]) -> dict[str, Any]:
    result = quadratic_variation_convergence(
        partition_sizes=tuple(request["partition_sizes"]),
        paths=request["paths"],
        horizon=request["horizon"],
        seed=request["seed"],
    )
    return _response(
        _validated_execution(request),
        status=result.status,
        reason="nested-grid Brownian quadratic-variation study",
        evidence=result,
        metrics=[
            _metric("rmse_mesh_slope", result.rmse_mesh_slope, "slope"),
            _metric("finest_rmse", result.rows[-1].rmse, request["time_unit"]),
        ],
    )


def simulate_diffusion_exact(request: dict[str, Any]) -> dict[str, Any]:
    result = exact_transition_audit(
        spot=request["spot"],
        drift=request["drift"],
        volatility=request["volatility"],
        horizon=request["horizon"],
        paths=request["paths"],
        seed=request["seed"],
    )
    return _response(
        _validated_execution(request),
        status=result.status,
        reason="GBM exact moment, semigroup and positivity audit",
        evidence=result,
        metrics=[
            _metric("one_step_mean_error", result.one_step_mean_error, request["state_unit"]),
            _metric("minimum_terminal_value", result.minimum_terminal_value, request["state_unit"]),
        ],
    )


def evaluate_stochastic_exponential(request: dict[str, Any]) -> dict[str, Any]:
    dt = request["horizon"] / request["steps"]
    rng = np.random.default_rng(request["seed"])
    increments = np.sqrt(dt) * rng.standard_normal((request["paths"], request["steps"]))
    theta = np.full_like(increments, request["theta"])
    result = density_process_diagnostic(
        theta,
        increments,
        dt=dt,
        integrand_timing=request["integrand_timing"],
        confidence_z=request["confidence_z"],
        minimum_ess_fraction=request["minimum_ess_fraction"],
    )
    return _response(
        _validated_execution(request),
        status=result.status,
        reason=result.reason,
        evidence=result,
        metrics=[
            _metric("mean_terminal_density", result.mean_terminal_density, "density"),
            _metric("effective_sample_size_fraction", result.effective_sample_size_fraction, "fraction"),
        ],
    )


def compute_hitting_probability(request: dict[str, Any]) -> dict[str, Any]:
    result = hitting_convergence_study(
        request["start"],
        request["lower"],
        request["upper"],
        step_sizes=tuple(request["step_sizes"]),
        paths=request["paths"],
        seed=request["seed"],
        maximum_steps=request["maximum_steps"],
        confidence_z=request["confidence_z"],
        probability_bias_limit=request["probability_bias_limit"],
        exit_time_bias_limit=request["exit_time_bias_limit"],
        maximum_unresolved_fraction=request["maximum_unresolved_fraction"],
    )
    return _response(
        _validated_execution(request),
        status=result.status,
        reason="three-grid endpoint-monitoring and censoring gate",
        evidence=result,
        metrics=[
            _metric("finest_probability", result.rows[-1].estimate, "probability"),
            _metric("finest_mean_exit_time", result.rows[-1].mean_exit_time, request["time_unit"]),
        ],
    )


def price_feynman_kac_claim(request: dict[str, Any]) -> dict[str, Any]:
    result = ou_terminal_and_running_payoff(
        state=request["state"],
        kappa=request["kappa"],
        theta=request["theta"],
        sigma=request["sigma"],
        rate=request["rate"],
        maturity=request["maturity"],
        running_weight=request["running_weight"],
        steps=request["steps"],
        paths=request["paths"],
        seed=request["seed"],
        confidence_z=request["confidence_z"],
        maximum_absolute_discretization_bias=request["maximum_absolute_discretization_bias"],
    )
    return _response(
        _validated_execution(request),
        status=result.status,
        reason="running payoff discounted at payment time",
        evidence=result,
        metrics=[
            _metric("estimate", result.estimate, request["price_unit"]),
            _metric("analytic_value", result.analytic_value, request["price_unit"]),
            _metric("discretization_bias", result.discretization_bias, request["price_unit"]),
        ],
    )


def compare_sde_schemes(request: dict[str, Any]) -> dict[str, Any]:
    result = gbm_scheme_convergence(
        step_counts=tuple(request["step_counts"]),
        paths=request["paths"],
        initial_state=request["initial_state"],
        drift_rate=request["drift_rate"],
        volatility=request["volatility"],
        horizon=request["horizon"],
        seed=request["seed"],
    )
    return _response(
        _validated_execution(request),
        status=result.status,
        reason=result.weak_order_status,
        evidence=result,
        metrics=[
            _metric("euler_strong_slope", result.euler_strong_slope, "slope"),
            _metric("milstein_strong_slope", result.milstein_strong_slope, "slope"),
            _metric("euler_weak_slope", result.euler_weak_slope, "slope"),
        ],
    )


def validate_correlation_matrix(request: dict[str, Any]) -> dict[str, Any]:
    result = correlation_input_gate(
        np.asarray(request["matrix"], dtype=float),
        absolute_tolerance=request["absolute_tolerance"],
        relative_tolerance=request["relative_tolerance"],
        maximum_repair_norm=request["maximum_repair_norm"],
        approval_owner=request["approval_owner"],
        policy_id=request["policy_id"],
    )
    return _response(
        _validated_execution(request),
        status=result.status,
        reason=result.reason,
        evidence=result,
        metrics=[
            _metric("raw_minimum_eigenvalue", result.raw_eigenvalues[0], "eigenvalue"),
            _metric("repair_frobenius_norm", result.repair_frobenius_norm, "Frobenius norm"),
            _metric("reconstruction_residual", result.reconstruction_residual, "Frobenius norm"),
        ],
    )


TOOL_BINDINGS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "select_predictable_feature": select_predictable_feature,
    "estimate_quadratic_variation": estimate_quadratic_variation,
    "simulate_diffusion_exact": simulate_diffusion_exact,
    "evaluate_stochastic_exponential": evaluate_stochastic_exponential,
    "compute_hitting_probability": compute_hitting_probability,
    "price_feynman_kac_claim": price_feynman_kac_claim,
    "compare_sde_schemes": compare_sde_schemes,
    "validate_correlation_matrix": validate_correlation_matrix,
}
