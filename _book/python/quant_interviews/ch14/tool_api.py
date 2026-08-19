"""Versioned deterministic adapters for the ten CH14 tool contracts.

JSON Schema validates shape.  These adapters recompute the result and bind it to
the exact request, so a schema-valid but contradictory PASS cannot be accepted.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
import json
from math import isclose
import re
from typing import Any, Callable

import numpy as np

from .adaptive_interview import Evidence, Question, choose_next_question
from .ml_case import log_loss, population_stability_index
from .research_case import net_information_ratio, research_launch_gate, selection_adjusted_score
from .risk_case import historical_var_es, reverse_stress_scale, tail_risk_gate
from .scoring import EvidenceItem, ScoreCard, evidence_completeness, hiring_decision, weighted_score
from .takehome_case import score_takehome
from .trader_case import choose_order, fill_probability_threshold


JSONDict = dict[str, Any]
H0 = "0" * 64
H1 = "1" * 64
H2 = "2" * 64
CANONICAL_STATES = {
    "ACCEPT",
    "CONDITIONAL_ACCEPT",
    "PANEL_REVIEW",
    "REJECT",
    "INSUFFICIENT_EVIDENCE",
}


TOOL_PROMPTS = {
    "run_adaptive_mock_interview": "AI-CH14.01",
    "score_interview_evidence": "AI-CH14.12",
    "validate_research_case": "AI-CH14.02",
    "evaluate_trader_order_choice": "AI-CH14.03",
    "validate_pricing_hedging_case": "AI-CH14.04",
    "run_risk_case": "AI-CH14.07",
    "audit_code_task": "AI-CH14.05",
    "validate_ml_case": "AI-CH14.08",
    "audit_takehome_packet": "AI-CH14.10",
    "finalize_hiring_decision": "AI-CH14.15",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def evidence_sha256(name: str, inputs: JSONDict, output_without_hash: JSONDict) -> str:
    return sha256(
        _canonical_bytes(
            {
                "contractVersion": "0.10.0",
                "input": inputs,
                "output": output_without_hash,
                "tool": name,
            }
        )
    ).hexdigest()


def bind_evidence(name: str, inputs: JSONDict, output: JSONDict) -> JSONDict:
    bound = dict(output)
    bound["evidence_sha256"] = evidence_sha256(name, inputs, bound)
    return bound


def _base(prompt_id: str) -> JSONDict:
    return {
        "schema_version": "1.0",
        "request_id": "request-001",
        "prompt_id": prompt_id,
        "prompt_sha256": H0,
        "system_policy_sha256": H1,
        "tool_version": "0.10",
        "decision_time": "2026-01-15T12:00:00Z",
    }


def canonical_input(name: str) -> JSONDict:
    """Return the governed positive vector used by schema and conformance tests."""

    if name not in TOOL_PROMPTS:
        raise KeyError(f"unknown CH14 tool: {name}")
    value = _base(TOOL_PROMPTS[name])
    payloads: dict[str, JSONDict] = {
        "run_adaptive_mock_interview": {
            "role": "systematic-researcher",
            "remaining_minutes": 20,
            "targets": {"math": 4.0, "coding": 3.5},
            "evidence": [{"competency": "math", "score": 2.0, "confidence": 0.5}],
            "questions": [
                {"question_id": "q-math", "competency": "math", "difficulty": 3, "time_minutes": 10},
                {"question_id": "q-code", "competency": "coding", "difficulty": 3, "time_minutes": 10},
            ],
            "asked_question_ids": [],
        },
        "score_interview_evidence": {
            "scores": {"correctness": 4.2, "reasoning": 4.0, "implementation": 3.8, "judgment": 4.1, "communication": 3.9},
            "weights": {"correctness": 0.25, "reasoning": 0.25, "implementation": 0.2, "judgment": 0.2, "communication": 0.1},
            "hard_floors": {"correctness": 3.0, "judgment": 3.0},
            "required_quality": {"correctness": 0.8, "judgment": 0.8},
            "required_evidence_weights": {"correctness": 0.7, "judgment": 0.3},
            "evidence": [
                {"evidence_id": "e1", "competency": "correctness", "observation": "derived result", "quality": 0.9, "source_hash": H0, "observed_at": "2026-01-15T10:00:00Z", "scorer_id": "s1"},
                {"evidence_id": "e2", "competency": "judgment", "observation": "stated limits", "quality": 0.9, "source_hash": H1, "observed_at": "2026-01-15T10:05:00Z", "scorer_id": "s2"},
            ],
        },
        "validate_research_case": {
            "observed_sharpe": 1.5, "effective_trials": 4.0, "sample_years": 5.0,
            "sharpe_standard_error": 0.1, "standard_error_method": "block-bootstrap",
            "dependence_method": "effective-trials", "gross_alpha_annual": 0.111,
            "active_risk_annual": 0.08, "annual_one_way_turnover": 1.8,
            "cost_per_unit_turnover": 0.002, "nonlinear_impact_annual": 0.0324,
            "point_in_time_pass": True, "holdout_untouched": True,
            "capacity_pass": True, "operational_readiness_pass": True,
        },
        "evaluate_trader_order_choice": {
            "market_cost_bps": 3.0, "filled_limit_cost_bps": 1.0,
            "miss_cost_bps": 5.0, "fill_probability": 0.8,
            "no_trade_cost_bps": 9.0, "current_inventory": 0.0,
            "order_quantity": 1.0, "volatility": 0.2, "horizon_years": 0.01,
            "risk_aversion_bps_per_variance": 1.0, "proposed_risk": 10.0,
            "risk_limit": 100.0, "market_observed_at": "2026-01-15T11:59:55Z",
            "maximum_market_age_seconds": 10.0, "market_snapshot_hash": H0,
            "fill_model_hash": H1, "portfolio_snapshot_hash": H2,
            "limit_snapshot_hash": "3" * 64, "policy_version": "order-policy-v1",
        },
        "validate_pricing_hedging_case": {
            "theta": -4.86, "delta": 0.55, "gamma": 0.02, "spot": 100.0,
            "rate": 0.03, "dividend_yield": 0.01, "option_value": 8.0,
            "volatility": 0.2, "hedge_value_change": 1.2,
            "option_value_change": 1.1995, "financing": 0.0002,
            "dividend_cashflow": 0.0001, "transaction_cost": 0.0002,
            "hedge_tolerance": 0.01, "expiry_delta_convention": "symmetric_ATM_delta_0.5",
        },
        "run_risk_case": {
            "losses": [1.0, 2.0, 3.0, 4.0], "confidence": 0.75,
            "expected_shortfall_limit": 5.0, "coverage": 1.0,
            "minimum_coverage": 0.95, "tail_model_adequate": True,
            "base_stress_loss": 0.0, "loss_per_scale": 2.0,
            "target_loss": 6.0, "maximum_scale": 10.0,
        },
        "audit_code_task": {
            "language": "cpp", "source_sha256": H0,
            "compiler": "clang++", "language_standard": "c++20",
            "compiler_flags": ["-O2", "-Wall", "-Wextra", "-pedantic"],
            "test_run_id": "ch14-cpp-001", "tests_passed": 12,
            "tests_failed": 0, "branch_coverage": 1.0,
            "minimum_branch_coverage": 0.9,
        },
        "validate_ml_case": {
            "probabilities": [0.1, 0.8, 0.7, 0.2], "outcomes": [0, 1, 1, 0],
            "epsilon": 1e-6, "reference_scores": [0.1, 0.2, 0.3, 0.4],
            "current_scores": [0.15, 0.25, 0.35, 0.45], "psi_bins": 2,
            "psi_floor": 1e-6, "half_open_exact_union_purge_pass": True,
            "nonoverlapping_returns": True,
        },
        "audit_takehome_packet": {
            "candidate_id": "candidate-001", "manifest_schema_version": "1.0",
            "code_commit": "a" * 40, "data_hash": H0, "config_hash": H1,
            "environment_hash": H2, "result_hash": "3" * 64,
            "reproduced_result_hash": "3" * 64, "metric_residual": 0.0,
            "metric_tolerance": 1e-12, "reproducibility": 4.0,
            "research_quality": 4.0, "implementation": 4.0,
            "communication": 4.0, "leakage": False, "fabricated_result": False,
            "core_claim_reproduced": True, "decision_supported": True,
        },
        "finalize_hiring_decision": {
            "candidate_id": "candidate-001",
            "scores": {"correctness": 4.2, "reasoning": 4.0, "implementation": 3.8, "judgment": 4.1, "communication": 3.9},
            "weights": {"correctness": 0.25, "reasoning": 0.25, "implementation": 0.2, "judgment": 0.2, "communication": 0.1},
            "hard_floors": {"correctness": 3.0, "judgment": 3.0},
            "required_quality": {"correctness": 0.8, "judgment": 0.8},
            "required_evidence_weights": {"correctness": 0.7, "judgment": 0.3},
            "evidence": [
                {"evidence_id": "e1", "competency": "correctness", "observation": "derived result", "quality": 0.9, "source_hash": H0, "observed_at": "2026-01-15T10:00:00Z", "scorer_id": "s1"},
                {"evidence_id": "e2", "competency": "judgment", "observation": "stated limits", "quality": 0.9, "source_hash": H1, "observed_at": "2026-01-15T10:05:00Z", "scorer_id": "s2"},
            ],
            "scorer_ids": ["s1", "s2"], "policy_version": "hiring-v1",
            "dissent": [], "integrity_failures": [], "prior_packet_hash": H2,
        },
    }
    value.update(payloads[name])
    return value


def _common(decision: str, **values: Any) -> JSONDict:
    if decision not in CANONICAL_STATES:
        raise ValueError("unknown canonical decision")
    status = {
        "ACCEPT": "PASS", "CONDITIONAL_ACCEPT": "PASS",
        "PANEL_REVIEW": "PANEL_REVIEW", "REJECT": "REJECT",
        "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE",
    }[decision]
    return {"schema_version": "1.0", "status": status, "decision": decision, "errors": [], **values}


def _evidence(items: list[JSONDict]) -> tuple[EvidenceItem, ...]:
    return tuple(EvidenceItem(**item) for item in items)


def _score_card(values: JSONDict) -> ScoreCard:
    return ScoreCard(**values)


def _run_adaptive(inputs: JSONDict) -> JSONDict:
    question = choose_next_question(
        [Question(**item) for item in inputs["questions"]],
        set(inputs["asked_question_ids"]), inputs["targets"],
        [Evidence(**item) for item in inputs["evidence"]],
        inputs["remaining_minutes"],
    )
    if question is None:
        return _common("INSUFFICIENT_EVIDENCE", next_question_id="NONE", target_competency="NONE", time_minutes=0)
    return _common("ACCEPT", next_question_id=question.question_id,
                   target_competency=question.competency, time_minutes=question.time_minutes)


def _score_evidence(inputs: JSONDict) -> JSONDict:
    card = _score_card(inputs["scores"])
    completeness = evidence_completeness(
        inputs["required_quality"], inputs["required_evidence_weights"],
        _evidence(inputs["evidence"]),
    )
    floors = sorted(name for name, floor in inputs["hard_floors"].items()
                    if card.as_dict()[name] < floor)
    score = weighted_score(card, inputs["weights"])
    decision = "REJECT" if floors else (
        "INSUFFICIENT_EVIDENCE" if completeness.missing_evidence else
        "ACCEPT" if score >= 4.0 else "CONDITIONAL_ACCEPT" if score >= 3.5 else "REJECT"
    )
    return _common(decision, weighted_score=score,
                   weighted_completeness=completeness.weighted_completeness,
                   hard_floor_violations=floors,
                   missing_evidence=list(completeness.missing_evidence))


def _research(inputs: JSONDict) -> JSONDict:
    selection = selection_adjusted_score(
        inputs["observed_sharpe"], inputs["effective_trials"], inputs["sample_years"],
        sharpe_standard_error=inputs["sharpe_standard_error"],
        standard_error_method=inputs["standard_error_method"],
        dependence_method=inputs["dependence_method"],
    )
    economics = net_information_ratio(
        inputs["gross_alpha_annual"], inputs["active_risk_annual"],
        inputs["annual_one_way_turnover"], inputs["cost_per_unit_turnover"],
        inputs["nonlinear_impact_annual"],
    )
    gate = research_launch_gate(
        selection, economics, point_in_time_pass=inputs["point_in_time_pass"],
        holdout_untouched=inputs["holdout_untouched"], capacity_pass=inputs["capacity_pass"],
        operational_readiness_pass=inputs["operational_readiness_pass"],
    )
    return _common("ACCEPT" if gate.decision == "PASS" else "REJECT",
                   selection_lower_bound=selection.adjusted_lower_bound,
                   selection_status=selection.status, annual_linear_cost=economics.linear_cost,
                   annual_net_alpha=economics.net_alpha,
                   annual_net_information_ratio=economics.net_information_ratio,
                   failed_gates=list(gate.failed_gates))


def _trader(inputs: JSONDict) -> JSONDict:
    threshold = fill_probability_threshold(inputs["market_cost_bps"], inputs["filled_limit_cost_bps"], inputs["miss_cost_bps"])
    limit_cost = inputs["fill_probability"] * inputs["filled_limit_cost_bps"] + (1.0 - inputs["fill_probability"]) * inputs["miss_cost_bps"]
    decision = choose_order(
        market_cost_bps=inputs["market_cost_bps"], limit_expected_cost_bps=limit_cost,
        no_trade_cost_bps=inputs["no_trade_cost_bps"], current_inventory=inputs["current_inventory"],
        order_quantity=inputs["order_quantity"], fill_probability=inputs["fill_probability"],
        volatility=inputs["volatility"], horizon_years=inputs["horizon_years"],
        risk_aversion_bps_per_variance=inputs["risk_aversion_bps_per_variance"],
        proposed_risk=inputs["proposed_risk"], risk_limit=inputs["risk_limit"],
        decision_time=inputs["decision_time"], market_observed_at=inputs["market_observed_at"],
        maximum_market_age_seconds=inputs["maximum_market_age_seconds"],
        market_snapshot_hash=inputs["market_snapshot_hash"], fill_model_hash=inputs["fill_model_hash"],
        portfolio_snapshot_hash=inputs["portfolio_snapshot_hash"], limit_snapshot_hash=inputs["limit_snapshot_hash"],
        policy_version=inputs["policy_version"],
    )
    return _common("ACCEPT" if decision.status == "PASS" else "REJECT",
                   threshold_classification=threshold.classification,
                   threshold_probability=threshold.probability,
                   selected_action=decision.selected_action,
                   market_age_seconds=decision.market_age_seconds,
                   risk_utilization=decision.risk_utilization,
                   decision_packet_sha256=decision.decision_hash)


def _pricing(inputs: JSONDict) -> JSONDict:
    residual = (inputs["theta"] + (inputs["rate"] - inputs["dividend_yield"]) * inputs["spot"] * inputs["delta"]
                + 0.5 * inputs["volatility"] ** 2 * inputs["spot"] ** 2 * inputs["gamma"]
                - inputs["rate"] * inputs["option_value"])
    hedge = (inputs["option_value_change"] - inputs["hedge_value_change"]
             - inputs["financing"] - inputs["dividend_cashflow"] + inputs["transaction_cost"])
    passed = abs(residual) <= 1e-10 and abs(hedge) <= inputs["hedge_tolerance"]
    return _common("ACCEPT" if passed else "REJECT", pde_residual=residual,
                   hedge_residual=hedge, hedge_tolerance=inputs["hedge_tolerance"],
                   expiry_delta_convention=inputs["expiry_delta_convention"])


def _risk(inputs: JSONDict) -> JSONDict:
    certificate = historical_var_es(np.asarray(inputs["losses"], dtype=float), inputs["confidence"])
    gate = tail_risk_gate(certificate, limit=inputs["expected_shortfall_limit"],
                          coverage=inputs["coverage"], minimum_coverage=inputs["minimum_coverage"],
                          tail_model_adequate=inputs["tail_model_adequate"])
    reverse = reverse_stress_scale(
        lambda scale: inputs["base_stress_loss"] + inputs["loss_per_scale"] * scale,
        target_loss=inputs["target_loss"], maximum_upper=inputs["maximum_scale"],
    )
    passed = gate.decision == "PASS" and reverse.status == "PASS"
    return _common("ACCEPT" if passed else "REJECT", value_at_risk=certificate.value_at_risk,
                   expected_shortfall=certificate.expected_shortfall,
                   quantile_convention=certificate.quantile_convention,
                   tail_gate_status=gate.decision, reverse_stress_scale=reverse.scale,
                   reverse_stress_status=reverse.status)


def _code(inputs: JSONDict) -> JSONDict:
    required = {"-O2", "-Wall", "-Wextra", "-pedantic"}
    passed = (inputs["language"] == "cpp" and inputs["language_standard"] == "c++20"
              and required <= set(inputs["compiler_flags"]) and inputs["tests_failed"] == 0
              and inputs["tests_passed"] > 0 and inputs["branch_coverage"] >= inputs["minimum_branch_coverage"])
    findings = [] if passed else ["compiler, test, or coverage gate failed"]
    return _common("ACCEPT" if passed else "REJECT", compile_status="PASS" if passed else "FAIL",
                   test_status="PASS" if passed else "FAIL", coverage=inputs["branch_coverage"],
                   findings=findings)


def _ml(inputs: JSONDict) -> JSONDict:
    loss = log_loss(np.asarray(inputs["probabilities"]), np.asarray(inputs["outcomes"]), epsilon=inputs["epsilon"])
    psi = population_stability_index(np.asarray(inputs["reference_scores"]), np.asarray(inputs["current_scores"]),
                                     bins=inputs["psi_bins"], floor=inputs["psi_floor"])
    passed = inputs["half_open_exact_union_purge_pass"] and inputs["nonoverlapping_returns"]
    return _common("ACCEPT" if passed else "REJECT", log_loss=loss, psi=psi.value,
                   psi_policy=psi.smoothing_policy, leakage_status="PASS" if inputs["half_open_exact_union_purge_pass"] else "FAIL",
                   economic_interpretation="NONOVERLAPPING_TRADABLE" if inputs["nonoverlapping_returns"] else "PER_LABEL_ONLY")


def _takehome(inputs: JSONDict) -> JSONDict:
    hashes_valid = (len(inputs["code_commit"]) in (40, 64) and all(
        len(inputs[key]) == 64 for key in ("data_hash", "config_hash", "environment_hash", "result_hash", "reproduced_result_hash")
    ))
    reproduced = (hashes_valid and inputs["manifest_schema_version"] == "1.0"
                  and abs(inputs["metric_residual"]) <= inputs["metric_tolerance"])
    decision = score_takehome(
        reproducibility=inputs["reproducibility"], research_quality=inputs["research_quality"],
        implementation=inputs["implementation"], communication=inputs["communication"],
        leakage=inputs["leakage"], fabricated_result=inputs["fabricated_result"],
        core_claim_reproduced=inputs["core_claim_reproduced"] and reproduced,
        decision_supported=inputs["decision_supported"],
    )
    canonical = decision.status if decision.status in CANONICAL_STATES else "REJECT"
    return _common(canonical, manifest_valid=hashes_valid and inputs["manifest_schema_version"] == "1.0",
                   reproduction_status="BITWISE_REPRODUCED" if reproduced and inputs["result_hash"] == inputs["reproduced_result_hash"] else "MISMATCH",
                   weighted_score=decision.weighted_score, component_count=decision.component_count,
                   fatal_reasons=list(decision.fatal_reasons))


def _final(inputs: JSONDict) -> JSONDict:
    result = hiring_decision(
        card=_score_card(inputs["scores"]), weights=inputs["weights"], hard_floors=inputs["hard_floors"],
        required_evidence_quality=inputs["required_quality"], required_evidence_weights=inputs["required_evidence_weights"],
        evidence=_evidence(inputs["evidence"]), scorer_ids=tuple(inputs["scorer_ids"]),
        decision_time=inputs["decision_time"], policy_version=inputs["policy_version"],
        dissent=tuple(inputs["dissent"]), integrity_failures=tuple(inputs["integrity_failures"]),
        prior_packet_hash=inputs["prior_packet_hash"],
    )
    return _common(result.state, weighted_score=result.weighted_score, weighted_completeness=result.completeness,
                   recommendation=result.recommendation, decision_packet_sha256=result.decision_hash,
                   dissent=list(result.dissent), integrity_failures=list(result.integrity_failures),
                   hard_floor_violations=list(result.hard_floor_violations), missing_evidence=list(result.missing_evidence))


_EXECUTORS: dict[str, Callable[[JSONDict], JSONDict]] = {
    "run_adaptive_mock_interview": _run_adaptive,
    "score_interview_evidence": _score_evidence,
    "validate_research_case": _research,
    "evaluate_trader_order_choice": _trader,
    "validate_pricing_hedging_case": _pricing,
    "run_risk_case": _risk,
    "audit_code_task": _code,
    "validate_ml_case": _ml,
    "audit_takehome_packet": _takehome,
    "finalize_hiring_decision": _final,
}


def execute_tool(name: str, inputs: JSONDict) -> JSONDict:
    if name not in _EXECUTORS:
        raise KeyError(f"unknown CH14 tool: {name}")
    if inputs.get("schema_version") != "1.0" or inputs.get("tool_version") != "0.10":
        raise ValueError("unsupported schema or tool version")
    if inputs.get("prompt_id") != TOOL_PROMPTS[name]:
        raise ValueError("prompt/tool binding mismatch")
    for field in ("prompt_sha256", "system_policy_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(inputs.get(field, ""))) is None:
            raise ValueError(f"{field} must be a lowercase SHA-256")
    try:
        timestamp = datetime.fromisoformat(str(inputs.get("decision_time", "")).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("decision_time must be ISO date-time") from error
    if timestamp.tzinfo is None:
        raise ValueError("decision_time must include a timezone")
    return bind_evidence(name, inputs, _EXECUTORS[name](inputs))


def validate_tool_result(name: str, inputs: JSONDict, output: JSONDict) -> tuple[str, ...]:
    """Return semantic errors after exact deterministic recomputation."""

    try:
        expected = execute_tool(name, inputs)
    except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError) as error:
        return (f"semantic execution failed: {error}",)
    errors: list[str] = []
    if set(output) != set(expected):
        errors.append("output fields do not match the versioned adapter")
    for key, expected_value in expected.items():
        actual = output.get(key)
        if isinstance(expected_value, float) and isinstance(actual, (int, float)):
            if not isclose(float(actual), expected_value, rel_tol=1e-10, abs_tol=1e-12):
                errors.append(f"{key} does not match deterministic recomputation")
        elif actual != expected_value:
            errors.append(f"{key} does not match deterministic recomputation")
    if output.get("decision") not in CANONICAL_STATES:
        errors.append("decision is outside the canonical CH14 taxonomy")
    if output.get("status") == "PASS" and (output.get("errors") or errors):
        errors.append("PASS is forbidden while semantic errors remain")
    return tuple(dict.fromkeys(errors))


__all__ = [
    "CANONICAL_STATES", "TOOL_PROMPTS", "bind_evidence", "canonical_input",
    "evidence_sha256", "execute_tool", "validate_tool_result",
]
