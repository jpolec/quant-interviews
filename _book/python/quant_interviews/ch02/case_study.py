"""Deterministic integrated case study for Chapter 2."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from quant_interviews.ch02.optimization import (
    constrained_turnover_regularized,
)
from quant_interviews.ch02.regularization import oas_shrinkage_stability


_HARD_REJECTION_REASONS = frozenset(
    {
        "CONDITIONING_LIMIT_EXCEEDED",
        "TURNOVER_LIMIT_EXCEEDED",
        "KKT_LIMIT_EXCEEDED",
        "FACTOR_NEUTRALITY_BREACH",
        "LINEAGE_MISMATCH",
    }
)


def _finite_nonnegative(value: float, *, name: str) -> float:
    candidate = np.asarray(value)
    if candidate.ndim != 0 or candidate.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    scalar = float(candidate)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return scalar


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return integer


def evaluate_case_decision(
    *,
    condition_number: float,
    turnover: float,
    maximum_kkt_violation: float,
    factor_neutrality_residual: float,
    lineage_matches: bool,
    pit_windows: int,
    human_approved: bool,
    conditioning_limit: float = 150.0,
    turnover_limit: float = 0.20,
    kkt_residual_limit: float = 1e-8,
    factor_neutrality_limit: float = 1e-8,
    required_pit_windows: int = 3,
) -> dict[str, str | tuple[str, ...]]:
    """Evaluate deterministic numeric, lineage, temporal and human gates."""

    condition_number = _finite_nonnegative(
        condition_number,
        name="condition_number",
    )
    turnover = _finite_nonnegative(turnover, name="turnover")
    maximum_kkt_violation = _finite_nonnegative(
        maximum_kkt_violation,
        name="maximum_kkt_violation",
    )
    factor_neutrality_residual = _finite_nonnegative(
        factor_neutrality_residual,
        name="factor_neutrality_residual",
    )
    conditioning_limit = _finite_nonnegative(
        conditioning_limit,
        name="conditioning_limit",
    )
    turnover_limit = _finite_nonnegative(
        turnover_limit,
        name="turnover_limit",
    )
    kkt_residual_limit = _finite_nonnegative(
        kkt_residual_limit,
        name="kkt_residual_limit",
    )
    factor_neutrality_limit = _finite_nonnegative(
        factor_neutrality_limit,
        name="factor_neutrality_limit",
    )
    pit_windows = _nonnegative_integer(pit_windows, name="pit_windows")
    required_pit_windows = _nonnegative_integer(
        required_pit_windows,
        name="required_pit_windows",
    )
    if not isinstance(lineage_matches, (bool, np.bool_)):
        raise ValueError("lineage_matches must be boolean")
    if not isinstance(human_approved, (bool, np.bool_)):
        raise ValueError("human_approved must be boolean")

    reasons: list[str] = []
    if not bool(lineage_matches):
        reasons.append("LINEAGE_MISMATCH")
    if condition_number > conditioning_limit:
        reasons.append("CONDITIONING_LIMIT_EXCEEDED")
    if turnover > turnover_limit:
        reasons.append("TURNOVER_LIMIT_EXCEEDED")
    if maximum_kkt_violation > kkt_residual_limit:
        reasons.append("KKT_LIMIT_EXCEEDED")
    if factor_neutrality_residual > factor_neutrality_limit:
        reasons.append("FACTOR_NEUTRALITY_BREACH")
    if pit_windows < required_pit_windows:
        reasons.append("INSUFFICIENT_PIT_WINDOWS")
    if not bool(human_approved):
        reasons.append("HUMAN_APPROVAL_REQUIRED")

    if _HARD_REJECTION_REASONS.intersection(reasons):
        decision = "REJECT"
    elif reasons:
        decision = "CONDITIONAL_PASS"
    else:
        decision = "PASS"
    return {"decision": decision, "decision_reasons": tuple(reasons)}


def governed_risk_model_case(
    seed: int = 7,
) -> dict[str, float | int | bool | str | tuple[str, ...] | np.ndarray]:
    """Run the chapter's governed risk-model and allocation decision."""

    seed = _nonnegative_integer(seed, name="seed")
    generator = np.random.default_rng(seed)
    observation_count = 45
    asset_count = 25
    factor_count = 3
    factors = generator.normal(size=(observation_count, factor_count))
    loadings = generator.normal(
        scale=0.5,
        size=(factor_count, asset_count),
    )
    idiosyncratic = generator.normal(
        scale=0.08,
        size=(observation_count, asset_count),
    )
    returns = factors @ loadings + idiosyncratic

    stability = oas_shrinkage_stability(returns)
    sample_covariance = np.asarray(stability["sample_covariance"])
    shrunk_covariance = np.asarray(stability["shrunk_covariance"])
    constraints = np.column_stack(
        [np.ones(asset_count), loadings[0], loadings[1]]
    )
    targets = np.array([1.0, 0.0, 0.0])
    previous = np.full(asset_count, 1.0 / asset_count)
    allocation = constrained_turnover_regularized(
        shrunk_covariance,
        returns.mean(axis=0),
        previous,
        constraints,
        targets,
        risk_aversion=3.0,
        turnover_penalty=0.30,
    )
    raw_target = constrained_turnover_regularized(
        shrunk_covariance,
        returns.mean(axis=0),
        previous,
        constraints,
        targets,
        risk_aversion=3.0,
        turnover_penalty=0.0,
    )
    constraints_hash = hashlib.sha256(
        np.ascontiguousarray(constraints).tobytes()
    ).hexdigest()
    snapshot_hash = hashlib.sha256(
        np.ascontiguousarray(returns).tobytes()
    ).hexdigest()
    solver_configuration = {
        "risk_aversion": 3.0,
        "turnover_penalty": 0.30,
        "conditioning_limit": 150.0,
        "kkt_residual_limit": 1e-8,
        "factor_neutrality_limit": 1e-8,
        "turnover_limit": 0.20,
        "required_pit_windows": 3,
    }
    solver_hash = hashlib.sha256(
        json.dumps(
            solver_configuration,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    decision_record = evaluate_case_decision(
        condition_number=float(stability["shrunk_condition_number"]),
        turnover=float(allocation["turnover"]),
        maximum_kkt_violation=float(
            allocation["maximum_kkt_violation"]
        ),
        factor_neutrality_residual=float(
            allocation["factor_neutrality_residual"]
        ),
        lineage_matches=True,
        pit_windows=1,
        human_approved=False,
        conditioning_limit=solver_configuration["conditioning_limit"],
        turnover_limit=solver_configuration["turnover_limit"],
        kkt_residual_limit=solver_configuration["kkt_residual_limit"],
        factor_neutrality_limit=solver_configuration[
            "factor_neutrality_limit"
        ],
        required_pit_windows=solver_configuration["required_pit_windows"],
    )
    return {
        "observations": observation_count,
        "assets": asset_count,
        "sample_rank": int(np.linalg.matrix_rank(sample_covariance)),
        "sample_minimum_eigenvalue": float(
            np.linalg.eigvalsh(sample_covariance)[0]
        ),
        "shrinkage_intensity": float(stability["intensity"]),
        "sample_condition_number": float(
            stability["sample_condition_number"]
        ),
        "shrunk_condition_number": float(
            stability["shrunk_condition_number"]
        ),
        "sample_gross_leverage": float(stability["sample_gross_leverage"]),
        "shrunk_gross_leverage": float(stability["shrunk_gross_leverage"]),
        "weights": np.asarray(allocation["weights"]),
        "constraint_residuals": np.asarray(
            allocation["constraint_residuals"]
        ),
        "factor_residuals": np.asarray(allocation["factor_residuals"]),
        "factor_neutrality_residual": float(
            allocation["factor_neutrality_residual"]
        ),
        "kkt_primal_residual": float(allocation["primal_residual"]),
        "kkt_stationarity_residual": float(
            allocation["stationarity_residual"]
        ),
        "kkt_dual_feasibility_violation": float(
            allocation["dual_feasibility_violation"]
        ),
        "kkt_complementarity_residual": float(
            allocation["complementarity_residual"]
        ),
        "turnover": float(allocation["turnover"]),
        "turnover_cost": float(allocation["turnover_cost"]),
        "raw_target_turnover": float(raw_target["turnover"]),
        "signed_turnover_limit": solver_configuration["turnover_limit"],
        "no_trade_positions": int(allocation["no_trade_positions"]),
        "turnover_kkt_violation": float(
            allocation["maximum_kkt_violation"]
        ),
        "lineage_matches": True,
        "pit_windows": 1,
        "required_pit_windows": solver_configuration["required_pit_windows"],
        "human_approved": False,
        "seed": seed,
        "fixture_version": "ch02-governed-risk-model-v3",
        "snapshot_hash": snapshot_hash,
        "constraints_hash": constraints_hash,
        "solver_configuration_hash": solver_hash,
        "threshold_policy": "QJ-RISK-MATRIX-2026.1",
        "threshold_owner": "QuantJourney Model Risk",
        "threshold_signed_date": "2026-07-29",
        **decision_record,
    }
