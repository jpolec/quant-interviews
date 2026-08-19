from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import exp, isfinite
from typing import Any

import numpy as np

from .american_binomial import american_lattice_convergence
from .barrier_mc import barrier_monitoring_experiment
from .binomial_state_prices import two_state_replication_certificate
from .black_scholes import black_scholes_call
from .black_scholes_pde import black_scholes_pde_convergence_certificate
from .digital_replication import (
    analytic_cash_digital_call,
    digital_replication_convergence,
)
from .dupire import (
    dupire_local_variance_diagnostic,
    dupire_surface_sha256,
)
from .forward_start import forward_start_call_price
from .model_free_bounds import discrete_model_free_bounds
from .parity_audit import executable_put_call_parity_band
from .payoff_bounds import executable_call_bound_certificate
from .quanto import quanto_adjustment_certificate
from .risk_neutral import physical_probability_invariance_certificate
from .static_arbitrage import (
    SurfaceQuote,
    bid_ask_static_arbitrage_lp,
    butterfly_audit,
    calendar_surface_audit,
    vertical_spread_audit,
)


@dataclass(frozen=True)
class GateRecord:
    name: str
    required: bool
    value: float | int | str
    operator: str
    signed_limit: float | int | str
    unit: str
    owner: str
    evidence_sha256: str
    status: str
    reason: str


@dataclass(frozen=True)
class ReleasePolicy:
    version: str
    required_gate_names: tuple[str, ...]
    optional_gate_names: tuple[str, ...]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(nested) for nested in value]
    return value


def evidence_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def canonical_snapshot_hash(rows: tuple[dict[str, Any], ...]) -> str:
    ordered = tuple(sorted(rows, key=lambda row: (row["quote_id"], row["revision_id"])))
    return evidence_sha256(ordered)


def evaluate_release_gates(
    gates: tuple[GateRecord, ...],
    policy: ReleasePolicy,
) -> dict[str, Any]:
    """Reduce complete named gate evidence under a versioned policy."""

    names = tuple(gate.name for gate in gates)
    if len(set(names)) != len(names):
        raise ValueError("release gate names must be unique")
    known = set(policy.required_gate_names) | set(policy.optional_gate_names)
    if set(names) - known:
        raise ValueError("release evidence contains an ungoverned gate")
    by_name = {gate.name: gate for gate in gates}
    reasons: list[str] = []
    conditional: list[str] = []
    for name in policy.required_gate_names:
        gate = by_name.get(name)
        if gate is None:
            reasons.append(f"{name}:missing")
            continue
        if not gate.required:
            reasons.append(f"{name}:required_flag_false")
        if gate.status == "REJECT":
            reasons.append(f"{name}:{gate.reason}")
        elif gate.status == "CONDITIONAL_PASS":
            conditional.append(f"{name}:{gate.reason}")
        elif gate.status != "PASS":
            reasons.append(f"{name}:invalid_status")
        if len(gate.evidence_sha256) != 64:
            reasons.append(f"{name}:invalid_evidence_hash")
    if reasons:
        decision = "REJECT"
    elif conditional:
        decision = "CONDITIONAL_PASS"
    else:
        decision = "PASS"
    return {
        "policy_version": policy.version,
        "decision": decision,
        "rejection_reasons": tuple(reasons),
        "conditional_reasons": tuple(conditional),
    }


def _gate(
    *,
    name: str,
    required: bool,
    value: float | int | str,
    operator: str,
    signed_limit: float | int | str,
    unit: str,
    owner: str,
    status: str,
    reason: str,
    evidence: Any,
) -> GateRecord:
    if status not in {"PASS", "CONDITIONAL_PASS", "REJECT"}:
        raise ValueError("invalid gate status")
    return GateRecord(
        name=name,
        required=required,
        value=value,
        operator=operator,
        signed_limit=signed_limit,
        unit=unit,
        owner=owner,
        evidence_sha256=evidence_sha256(evidence),
        status=status,
        reason=reason,
    )


def _snapshot_fixture() -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    rows = (
        {
            "quote_id": "call-r1",
            "revision_id": 1,
            "bid": 8.8,
            "ask": 9.2,
            "market_event_time": "2026-07-25T10:00:00.000000Z",
            "known_at": "2026-07-25T10:00:01.000000Z",
            "payload_hash": "1" * 64,
            "eligible": False,
        },
        {
            "quote_id": "call-r2",
            "revision_id": 2,
            "bid": 8.9,
            "ask": 9.1,
            "market_event_time": "2026-07-25T10:00:00.000000Z",
            "known_at": "2026-07-25T10:00:03.000000Z",
            "payload_hash": "2" * 64,
            "eligible": True,
        },
        {
            "quote_id": "call-r3-future",
            "revision_id": 3,
            "bid": 9.0,
            "ask": 9.2,
            "market_event_time": "2026-07-25T10:00:00.000000Z",
            "known_at": "2026-07-25T10:00:06.000000Z",
            "payload_hash": "3" * 64,
            "eligible": False,
        },
        {
            "quote_id": "put-r1",
            "revision_id": 1,
            "bid": 6.1,
            "ask": 6.3,
            "market_event_time": "2026-07-25T10:00:01.000000Z",
            "known_at": "2026-07-25T10:00:02.000000Z",
            "payload_hash": "4" * 64,
            "eligible": True,
        },
    )
    selected = tuple(
        {key: value for key, value in row.items() if key != "eligible"}
        for row in rows
        if row["eligible"]
    )
    return rows, selected


def _surface_quotes() -> tuple[SurfaceQuote, ...]:
    return (
        SurfaceQuote("q90", "XYZ", 1.0, 90.0, 11.9, 12.1, 100.0, 0.97),
        SurfaceQuote("q100", "XYZ", 1.0, 100.0, 7.4, 8.1, 100.0, 0.97),
        SurfaceQuote("q110", "XYZ", 1.0, 110.0, 2.9, 3.1, 100.0, 0.97),
    )


def _dupire_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    strikes = np.linspace(70.0, 130.0, 81)
    maturities = np.linspace(0.25, 2.0, 15)
    forwards = 100.0 * np.exp(0.02 * maturities)
    discounts = np.exp(-0.02 * maturities)
    surface = np.array(
        [
            [
                black_scholes_call(
                    spot=100.0,
                    strike=float(strike),
                    rate=0.02,
                    dividend_yield=0.0,
                    volatility=0.2,
                    maturity=float(maturity),
                )
                for strike in strikes
            ]
            for maturity in maturities
        ]
    )
    surface[:, strikes.size // 2] += 0.8
    return strikes, maturities, surface, forwards, discounts


def chapter_release_case() -> dict[str, Any]:
    """Execute the deterministic, policy-complete CH05 rejection fixture."""

    all_rows, selected_rows = _snapshot_fixture()
    snapshot_hash = canonical_snapshot_hash(selected_rows)
    snapshot = {
        "valid_at": "2026-07-25T10:00:00.500000Z",
        "knowledge_cutoff": "2026-07-25T10:00:05.000000Z",
        "rows_received": len(all_rows),
        "rows_selected": len(selected_rows),
        "rows_excluded": len(all_rows) - len(selected_rows),
        "selected_rows": selected_rows,
        "snapshot_sha256": snapshot_hash,
    }

    payoff_bound = executable_call_bound_certificate(
        call_bid=8.9,
        call_ask=9.1,
        spot_bid=99.99,
        spot_ask=100.01,
        strike=100.0,
        discount_factor=exp(-0.02),
    )
    state_prices = two_state_replication_certificate(
        spot=100.0,
        up=120.0,
        down=80.0,
        payoff_up=20.0,
        payoff_down=0.0,
        gross_rate=1.02,
    )
    parity = executable_put_call_parity_band(
        call_bid=8.9,
        call_ask=9.1,
        put_bid=6.1,
        put_ask=6.3,
        spot_bid=99.99,
        spot_ask=100.01,
        strike=100.0,
        maturity=1.0,
        long_funding_rate=0.03,
        short_funding_rate=0.03,
    )
    quotes = _surface_quotes()
    strikes = np.array([quote.strike for quote in quotes])
    mids = np.array([0.5 * (quote.bid + quote.ask) for quote in quotes])
    quote_ids = tuple(quote.quote_id for quote in quotes)
    vertical = vertical_spread_audit(strikes, mids, 0.97, quote_ids=quote_ids)
    butterfly = butterfly_audit(strikes, mids, quote_ids=quote_ids)
    feasibility = bid_ask_static_arbitrage_lp(quotes)
    calendar = calendar_surface_audit(
        log_moneyness=np.array([-0.1, 0.0, 0.1]),
        maturities=np.array([0.5, 1.0]),
        total_variance=np.array([[0.04, 0.04, 0.04], [0.05, 0.03, 0.05]]),
        forwards=np.array([101.0, 102.0]),
        discount_factors=np.array([0.99, 0.97]),
    )
    digital_parameters = {
        "spot": 105.0,
        "rate": 0.03,
        "dividend_yield": 0.01,
        "volatility": 0.2,
        "maturity": 1.0,
    }
    digital_reference = analytic_cash_digital_call(strike=100.0, **digital_parameters)
    digital = digital_replication_convergence(
        lambda digital_strike: black_scholes_call(
            strike=digital_strike, **digital_parameters
        ),
        strike=100.0,
        widths=(10.0, 5.0, 2.5, 1.25),
        analytic_digital=digital_reference,
        call_half_spread=0.02,
    )
    american = american_lattice_convergence(
        spot=90.0,
        strike=100.0,
        rate=0.05,
        dividend_yield=0.0,
        volatility=0.25,
        maturity=1.0,
        base_steps=(100, 200, 400),
    )
    risk_neutral = physical_probability_invariance_certificate(
        payoffs=np.array([0.0, 10.0]),
        state_prices=np.array([0.45, 0.50]),
        physical_probabilities_a=np.array([0.8, 0.2]),
        physical_probabilities_b=np.array([0.2, 0.8]),
        cash_price=0.95,
        cash_payoff=1.0,
        traded_payoffs=np.array([[80.0], [120.0]]),
        traded_prices=np.array([96.0]),
    )
    pde = black_scholes_pde_convergence_certificate(
        spot=100.0,
        strike=100.0,
        rate=0.03,
        dividend_yield=0.01,
        volatility=0.2,
        maturity=1.0,
        maximum_spot=400.0,
        grids=((80, 160), (160, 320), (320, 640)),
    )
    forward_start_100 = forward_start_call_price(
        spot=100.0, alpha=1.0, rate=0.03, dividend_yield=0.01,
        volatility=0.2, start=0.5, maturity=1.5,
    )
    forward_start_150 = forward_start_call_price(
        spot=150.0, alpha=1.0, rate=0.03, dividend_yield=0.01,
        volatility=0.2, start=0.5, maturity=1.5,
    )
    homogeneity_residual = forward_start_150 - 1.5 * forward_start_100
    quanto_domestic = quanto_adjustment_certificate(
        asset_currency="JPY", payoff_currency="USD",
        collateral_currency="USD", pricing_numeraire="USD",
        fx_quote_direction="DOMESTIC_PER_FOREIGN",
        asset_volatility=0.2, fx_volatility=0.1, correlation=0.5,
    )
    quanto_inverse = quanto_adjustment_certificate(
        asset_currency="JPY", payoff_currency="USD",
        collateral_currency="USD", pricing_numeraire="USD",
        fx_quote_direction="FOREIGN_PER_DOMESTIC",
        asset_volatility=0.2, fx_volatility=0.1, correlation=0.5,
    )
    quanto_residual = (
        quanto_domestic["drift_adjustment"] - quanto_inverse["drift_adjustment"]
    )
    barrier = barrier_monitoring_experiment(
        spot=100.0,
        strike=100.0,
        barrier=80.0,
        rate=0.03,
        dividend_yield=0.0,
        volatility=0.2,
        maturity=1.0,
        fine_steps=252,
        paths=6000,
        monitoring_indices={
            "monthly": tuple(range(21, 253, 21)),
            "weekly": tuple(range(5, 251, 5)) + (252,),
            "daily": tuple(range(1, 253)),
        },
        seed=1705,
        batch_size=1500,
    )
    dupire_strikes, dupire_maturities, dupire_surface, forwards, discounts = _dupire_fixture()
    dupire_hash = dupire_surface_sha256(
        strike=dupire_strikes,
        maturity=dupire_maturities,
        call_surface=dupire_surface,
        forwards=forwards,
        discount_factors=discounts,
        rate=0.02,
        dividend_yield=0.0,
    )
    dupire = dupire_local_variance_diagnostic(
        strike=dupire_strikes,
        maturity=dupire_maturities,
        call_surface=dupire_surface,
        rate=0.02,
        dividend_yield=0.0,
        forwards=forwards,
        discount_factors=discounts,
        upstream_surface_sha256=dupire_hash,
        upstream_static_arbitrage_status="PASS",
        expected_local_variance=0.04,
    )
    states = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    hedge_payoffs = np.column_stack(
        (
            np.ones(states.size), states,
            np.maximum(states - 90.0, 0.0),
            np.maximum(states - 110.0, 0.0),
        )
    )
    reference_state_prices = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
    robust = discrete_model_free_bounds(
        states,
        (states >= 105.0).astype(float),
        hedge_payoffs,
        reference_state_prices @ hedge_payoffs,
        target_breakpoints=np.array([105.0]),
    )

    gates = (
        _gate(name="snapshot_lineage", required=True, value=len(selected_rows), operator=">=", signed_limit=2, unit="selected quotes", owner="market-data-control", status="PASS", reason="bitemporal snapshot and hash present", evidence=snapshot),
        _gate(name="payoff_bounds", required=True, value=payoff_bound["status"], operator="=", signed_limit="PASS", unit="state", owner="derivatives-validation", status=payoff_bound["status"], reason="executable call bounds", evidence=payoff_bound),
        _gate(name="state_price_replication", required=True, value=state_prices["status"], operator="=", signed_limit="PASS", unit="state", owner="derivatives-validation", status=state_prices["status"], reason="replication identities", evidence=state_prices),
        _gate(name="executable_parity", required=True, value=parity["status"], operator="=", signed_limit="PASS", unit="state", owner="market-arbitrage-control", status=parity["status"], reason="executable parity band", evidence=parity),
        _gate(name="vertical_slopes", required=True, value=len(vertical), operator="=", signed_limit=0, unit="violations", owner="surface-control", status="PASS" if not vertical else "REJECT", reason="vertical slope violations", evidence=vertical),
        _gate(name="surface_feasibility", required=True, value=feasibility["minimum_quote_repair"], operator="=", signed_limit=0.0, unit="price units", owner="surface-control", status=feasibility["status"], reason="strike-slice quote-box feasibility", evidence=feasibility),
        _gate(name="calendar_total_variance", required=True, value=calendar["violation_count"], operator="=", signed_limit=0, unit="violations", owner="surface-control", status=calendar["status"], reason="calendar total variance decreases", evidence=calendar),
        _gate(name="digital_replication", required=True, value=digital.final_observed_order or 0.0, operator="in", signed_limit="[1.5,2.5]", unit="observed order", owner="numerical-methods", status=digital.status, reason="digital convergence/cost frontier", evidence=digital),
        _gate(name="american_lattice", required=True, value=american.pair_spreads[-1], operator="<=", signed_limit=american.price_tolerance, unit="price", owner="lattice-model-owner", status=american.status, reason="paired-grid and policy certificate", evidence=american),
        _gate(name="risk_neutral_market", required=True, value=risk_neutral["maximum_pricing_residual"], operator="<=", signed_limit=risk_neutral["pricing_tolerance"], unit="price", owner="pricing-theory-control", status=risk_neutral["status"], reason="state-price market and physical contrast", evidence=risk_neutral),
        _gate(name="pde_convergence", required=True, value=abs(pde.price_errors[-1]), operator="<=", signed_limit=pde.price_tolerance, unit="price", owner="pde-model-owner", status=pde.status, reason="nested grid, Greeks and far boundary", evidence=pde),
        _gate(name="forward_start_homogeneity", required=True, value=abs(homogeneity_residual), operator="<=", signed_limit=1e-12, unit="price", owner="exotics-validation", status="PASS" if abs(homogeneity_residual) <= 1e-12 else "REJECT", reason="constant-volatility homogeneity", evidence={"price_100": forward_start_100, "price_150": forward_start_150}),
        _gate(name="quanto_requotation", required=True, value=abs(quanto_residual), operator="<=", signed_limit=1e-15, unit="drift", owner="quanto-model-owner", status="PASS" if abs(quanto_residual) <= 1e-15 else "REJECT", reason="economic invariance under consistent FX re-quotation", evidence={"domestic": quanto_domestic, "inverse": quanto_inverse}),
        _gate(name="barrier_monitoring", required=True, value=abs(barrier["bridge_reference_error"]), operator="<=", signed_limit=barrier["bridge_reference_limit"], unit="price", owner="barrier-model-owner", status=barrier["status"], reason="reference, pathwise parity and schedule ordering", evidence=barrier),
        _gate(name="dupire_density", required=True, value=max(dupire.negative_fraction, dupire.nonfinite_fraction), operator="=", signed_limit=0.0, unit="fraction", owner="local-vol-model-owner", status=dupire.status, reason="surface preconditions or local-variance recovery failed", evidence=dupire),
        _gate(name="discrete_robust_bounds", required=True, value=max(abs(robust.lower_duality_gap), abs(robust.upper_duality_gap)), operator="<=", signed_limit=1e-9, unit="price", owner="robust-pricing-owner", status=robust.solver_status, reason="finite-state primal-dual certificate only", evidence=robust),
        _gate(name="butterfly_midpoint_diagnostic", required=False, value=len(butterfly), operator="diagnostic", signed_limit=0, unit="mid violations", owner="surface-control", status="CONDITIONAL_PASS" if butterfly else "PASS", reason="midpoints violate but quote box is feasible", evidence=butterfly),
    )
    policy = ReleasePolicy(
        version="ch05-release-policy-v0.10",
        required_gate_names=tuple(gate.name for gate in gates if gate.required),
        optional_gate_names=tuple(gate.name for gate in gates if not gate.required),
    )
    decision = evaluate_release_gates(gates, policy)
    bundle_hash = evidence_sha256(
        {
            "snapshot": snapshot,
            "gates": gates,
            "policy": policy,
            "decision": decision,
        }
    )
    return {
        "fixture_type": "synthetic governed full-path rejection fixture",
        "fixture_version": "ch05-release-v0.10",
        "seed": 1705,
        "snapshot": snapshot,
        "gates": tuple(asdict(gate) for gate in gates),
        "policy": asdict(policy),
        "decision": decision["decision"],
        "rejection_reasons": decision["rejection_reasons"],
        "conditional_reasons": decision["conditional_reasons"],
        "evidence_bundle_sha256": bundle_hash,
    }


def rejected_gate_mutation(
    gates: tuple[GateRecord, ...],
    gate_name: str,
) -> tuple[GateRecord, ...]:
    """Deterministic test helper: reject exactly one named gate."""

    if gate_name not in {gate.name for gate in gates}:
        raise ValueError("unknown gate")
    return tuple(
        replace(gate, status="REJECT", reason="mutation-test rejection")
        if gate.name == gate_name
        else gate
        for gate in gates
    )
