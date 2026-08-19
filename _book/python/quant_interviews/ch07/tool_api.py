from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Callable

import numpy as np

from .callable_bond_tree import CallableBond, callable_bond_oas_certificate
from .cds_bootstrap import CDSQuote, bootstrap_cds_certificate
from .curve_bootstrap import DiscountCurve, bootstrap_curve_certificate
from .evidence import EvidenceCertificate, stable_manifest_hash
from .key_rate_hedge import solve_key_rate_hedge
from .multicurve_swap import MultiCurveSwap, NodeCurve, multicurve_risk_certificate
from .pnl_explain import PnLInputs, StateManifest, explain_certificate

IMPLEMENTATION_VERSION = "ch07.tool_api.v0.10"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def request_payload_hash(request: dict[str, Any]) -> str:
    payload = {key: value for key, value in request.items() if key != "execution"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return stable_manifest_hash(json.loads(encoded))


def _verify_request(request: dict[str, Any]) -> dict[str, Any]:
    execution = request["execution"]
    expected = request_payload_hash(request)
    if execution["input_sha256"] != expected:
        raise ValueError("execution input_sha256 does not match request payload")
    return execution


def _response(
    request: dict[str, Any],
    *,
    status: str,
    result: dict[str, Any],
    evidence: Any,
) -> dict[str, Any]:
    execution = _verify_request(request)
    json_result = _jsonable(result)
    return {
        "status": status,
        "request_id": execution["request_id"],
        "input_sha256": execution["input_sha256"],
        "evidence_sha256": stable_manifest_hash(
            {
                "request_id": execution["request_id"],
                "input_sha256": execution["input_sha256"],
                "evidence": _jsonable(evidence),
                "result": json_result,
                "implementation_version": IMPLEMENTATION_VERSION,
            }
        ),
        "implementation_version": IMPLEMENTATION_VERSION,
        "result": json_result,
    }


def _certificate_response(
    request: dict[str, Any],
    certificate: EvidenceCertificate[Any],
    result_adapter: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    return _response(
        request,
        status=certificate.status,
        result=result_adapter(certificate.result),
        evidence=certificate,
    )


def bootstrap_discount_curve(request: dict[str, Any]) -> dict[str, Any]:
    arguments = request["arguments"]
    deposits = {
        float(item["maturity"]): float(item["quote"])
        for item in arguments["instruments"]
        if item["instrument_type"] == "deposit"
    }
    swaps = {
        int(item["maturity"]): float(item["quote"])
        for item in arguments["instruments"]
        if item["instrument_type"] == "annual_swap"
    }
    certificate = bootstrap_curve_certificate(
        deposits,
        swaps,
        require_nonnegative_forwards=arguments["require_nonnegative_forwards"],
    )
    return _certificate_response(
        request,
        certificate,
        lambda result: {
            "curve_run_id": certificate.input_manifest_hash,
            "times": result.curve.times,
            "discount_factors": result.curve.discount_factors,
            "maximum_repricing_residual": result.maximum_repricing_residual,
            "minimum_interval_forward": result.minimum_interval_forward,
            "calibration_nodes": result.calibration_nodes,
        },
    )


def price_multicurve_swap(request: dict[str, Any]) -> dict[str, Any]:
    arguments = request["arguments"]
    trade = arguments["trade"]
    swap = MultiCurveSwap(
        np.asarray(trade["payment_times"], dtype=float),
        np.asarray(trade["accruals"], dtype=float),
        float(trade["notional"]),
    )
    discount = arguments["discount_curve"]
    projection = arguments["projection_curve"]
    certificate = multicurve_risk_certificate(
        swap,
        NodeCurve(
            np.asarray(discount["times"], dtype=float),
            np.asarray(discount["log_discount_factors"], dtype=float),
        ),
        NodeCurve(
            np.asarray(projection["times"], dtype=float),
            np.asarray(projection["forward_rates"], dtype=float),
        ),
        fixed_rate=trade.get("fixed_rate"),
        bump_bp=float(arguments["bump_bp"]),
    )
    return _certificate_response(
        request,
        certificate,
        lambda result: {
            "payer_pv": result.payer_pv,
            "par_rate": result.par_rate,
            "par_repricing_residual": result.par_repricing_residual,
            "discount_zero_node_dv01": result.discount_zero_node_dv01,
            "projection_forward_node_dv01": (
                result.projection_forward_node_dv01
            ),
            "node_coordinate_ids": result.node_coordinate_ids,
            "node_coordinate_units": result.node_coordinate_units,
            "risk_reconciliation_error": result.risk_reconciliation_error,
            "discount_curve_run_id": arguments["discount_curve_run_id"],
            "projection_curve_run_id": arguments["projection_curve_run_id"],
        },
    )


def compute_key_rate_hedge(request: dict[str, Any]) -> dict[str, Any]:
    arguments = request["arguments"]
    constraints = arguments["constraints"]
    result = solve_key_rate_hedge(
        np.asarray(arguments["portfolio_dv01"], dtype=float),
        np.asarray(arguments["hedge_matrix"], dtype=float),
        ridge_penalty=float(arguments["ridge_penalty"]),
        bucket_weights=np.asarray(arguments["bucket_weights"], dtype=float),
        lower_bounds=np.asarray(constraints["lower_bounds"], dtype=float),
        upper_bounds=np.asarray(constraints["upper_bounds"], dtype=float),
        tradable=np.asarray(constraints["tradable"], dtype=bool),
    )
    status = "PASS" if result.weighted_residual_norm <= arguments["residual_limit"] else "REJECT"
    return _response(
        request,
        status=status,
        result={
            "notionals": result.notionals,
            "residual_dv01": result.residual_dv01,
            "weighted_residual_norm": result.weighted_residual_norm,
            "condition_number": result.condition_number,
            "active_constraints": result.active_constraints,
            "solver": result.solver,
        },
        evidence=result,
    )


def bootstrap_cds_hazard_curve(request: dict[str, Any]) -> dict[str, Any]:
    arguments = request["arguments"]
    discount = arguments["discount_curve"]
    certificate = bootstrap_cds_certificate(
        tuple(
            CDSQuote(
                int(item["maturity_years"]),
                float(item["running_spread"]),
                float(item["upfront"]),
            )
            for item in arguments["quotes"]
        ),
        recovery=float(arguments["recovery"]),
        discount_curve=DiscountCurve(
            np.asarray(discount["times"], dtype=float),
            np.asarray(discount["discount_factors"], dtype=float),
        ),
        discount_curve_run_id=arguments["discount_curve_run_id"],
    )
    return _certificate_response(
        request,
        certificate,
        lambda result: {
            "discount_curve_run_id": result.discount_curve_run_id,
            "hazard_nodes": result.curve.hazards if result.curve else [],
            "node_end_times": result.curve.maturities if result.curve else [],
            "maximum_repricing_residual": result.maximum_repricing_residual,
            "calibration_nodes": result.calibration_nodes,
        },
    )


def price_callable_bond_oas(request: dict[str, Any]) -> dict[str, Any]:
    arguments = request["arguments"]
    bond = arguments["bond"]
    certificate = callable_bond_oas_certificate(
        bond=CallableBond(
            float(bond["face"]),
            float(bond["coupon_rate"]),
            int(bond["maturity_years"]),
            float(bond["call_price"]),
            int(bond["first_call_year"]),
        ),
        market_discount_factors=np.asarray(
            arguments["market_discount_factors"], dtype=float
        ),
        rate_step=float(arguments["rate_step"]),
        market_price=float(arguments["market_price"]),
        market_price_source=arguments["market_price_source"],
        market_price_evidence_hash=arguments.get(
            "market_price_evidence_hash"
        ),
        bump_bp=float(arguments["bump_bp"]),
    )
    return _certificate_response(
        request,
        certificate,
        lambda result: {
            "market_price": result.market_price,
            "straight_price": result.straight_price,
            "callable_price": result.callable_price,
            "oas": result.oas,
            "oas_repricing_residual": result.oas_repricing_residual,
            "effective_duration": result.effective_duration,
            "effective_convexity": result.effective_convexity,
            "exercise_boundary_rate_by_year": (
                result.exercise_boundary_rate_by_year
            ),
            "market_price_source": result.market_price_source,
        },
    )


def _state_manifest(value: dict[str, Any]) -> StateManifest:
    return StateManifest(
        value["artifact_type"],
        value["run_id"],
        value["state_hash"],
        tuple(value["dependency_hashes"]),
        int(value["record_count"]),
    )


def explain_fixed_income_pnl(request: dict[str, Any]) -> dict[str, Any]:
    arguments = request["arguments"]
    values = arguments["inputs"]
    certificate = explain_certificate(
        PnLInputs(
            carry=float(values["carry"]),
            roll_down=float(values["roll_down"]),
            curve_dv01=np.asarray(values["curve_dv01"], dtype=float),
            curve_move_bp=np.asarray(values["curve_move_bp"], dtype=float),
            curve_cross_gamma=np.asarray(
                values["curve_cross_gamma"], dtype=float
            ),
            basis_pnl=float(values["basis_pnl"]),
            cs01=float(values["cs01"]),
            credit_spread_move_bp=float(values["credit_spread_move_bp"]),
            credit_convexity=float(values["credit_convexity"]),
            volatility_pnl=float(values["volatility_pnl"]),
            cash_and_fixings_pnl=float(values["cash_and_fixings_pnl"]),
            exercise_pnl=float(values["exercise_pnl"]),
            model_change_pnl=float(values["model_change_pnl"]),
            data_correction_pnl=float(values["data_correction_pnl"]),
            actual_pnl=float(values["actual_pnl"]),
            position_manifest=_state_manifest(values["position_manifest"]),
            risk_manifest=_state_manifest(values["risk_manifest"]),
            market_manifest=_state_manifest(values["market_manifest"]),
            component_evidence_ids={
                key: tuple(identifiers)
                for key, identifiers in values["component_evidence_ids"].items()
            },
            not_applicable_reasons=values["not_applicable_reasons"],
            stale_input_count=int(values["stale_input_count"]),
        ),
        absolute_residual_limit=float(arguments["absolute_residual_limit"]),
        relative_residual_limit=float(arguments["relative_residual_limit"]),
    )
    return _certificate_response(
        request,
        certificate,
        lambda result: {
            "components": result.components,
            "actual_pnl": result.actual_pnl,
            "explained_pnl": result.explained_pnl,
            "residual": result.residual,
            "residual_fraction_of_actual": result.residual_fraction_of_actual,
            "snapshot_alignment": result.snapshot_alignment,
            "missing_component_count": result.missing_component_count,
            "stale_input_count": result.stale_input_count,
        },
    )
