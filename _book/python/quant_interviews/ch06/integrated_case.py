from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from math import exp, isfinite
from typing import Any, Mapping

import numpy as np

from .black_scholes_greeks import call_greeks, call_price
from .gamma_vega import long_gamma_short_vega_calendar
from .heston import heston_diagnostics
from .implied_vol import implied_vol_bid_ask
from .jump_hedging import hedge_frequency_study
from .roughness import roughness_recovery_study
from .surface import surface_arbitrage_audit
from .variance_products import variance_swap_audit


@dataclass(frozen=True)
class ReleasePolicyGate:
    gate_id: str
    metric: str
    required: bool
    operator: str
    limit: float
    unit: str
    owner: str

    def __post_init__(self) -> None:
        if not self.gate_id or not self.metric or not self.unit or not self.owner:
            raise ValueError("release gates require identifiers, units and owners")
        if self.operator not in {"<=", ">="}:
            raise ValueError("release gate operator must be <= or >=")
        if not isfinite(self.limit):
            raise ValueError("release gate limits must be finite")


@dataclass(frozen=True)
class ReleasePolicy:
    policy_version: str
    gates: tuple[ReleasePolicyGate, ...]

    def __post_init__(self) -> None:
        gate_ids = [gate.gate_id for gate in self.gates]
        if not self.policy_version or not self.gates:
            raise ValueError("release policy must be versioned and non-empty")
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("release gate identifiers must be unique")


@dataclass(frozen=True)
class ReleaseEvidence:
    gate_id: str
    metric: str
    observed: float
    required: bool
    operator: str
    signed_limit: float
    unit: str
    owner: str
    status: str
    evidence_hash: str


@dataclass(frozen=True)
class ReleaseDecision:
    policy_version: str
    policy_hash: str
    input_snapshot_hash: str
    result_bundle_hash: str
    release_evidence_hash: str
    gate_evidence: tuple[ReleaseEvidence, ...]
    final_decision: str
    decision_reason: str


def _canonical_payload(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical_payload(asdict(value))
    if isinstance(value, np.ndarray):
        return [_canonical_payload(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_payload(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_payload(item) for item in value]
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("hashed evidence cannot contain non-finite floats")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def _sha256_payload(payload: Any) -> str:
    serialised = json.dumps(
        _canonical_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def evaluate_release(
    *,
    metrics: Mapping[str, float],
    policy: ReleasePolicy | Mapping[str, Any],
    input_snapshot_hash: str,
    result_bundle_hash: str | None = None,
) -> ReleaseDecision:
    if isinstance(policy, Mapping):
        if set(policy) != {"policy_version", "gates"}:
            raise ValueError("policy must contain exactly policy_version and gates")
        raw_gates = policy["gates"]
        if not isinstance(raw_gates, (list, tuple)):
            raise TypeError("policy gates must be a list or tuple")
        gates: list[ReleasePolicyGate] = []
        for raw_gate in raw_gates:
            if not isinstance(raw_gate, Mapping):
                raise TypeError("each policy gate must be an object")
            gates.append(ReleasePolicyGate(**dict(raw_gate)))
        policy = ReleasePolicy(
            policy_version=str(policy["policy_version"]),
            gates=tuple(gates),
        )
    elif not isinstance(policy, ReleasePolicy):
        raise TypeError("policy must be a ReleasePolicy or policy object")

    hexadecimal = set("0123456789abcdef")
    if (
        len(input_snapshot_hash) != 64
        or any(character not in hexadecimal for character in input_snapshot_hash)
    ):
        raise ValueError("input_snapshot_hash must be a SHA-256 hex digest")
    if result_bundle_hash is None:
        result_bundle_hash = _sha256_payload(metrics)
    if (
        len(result_bundle_hash) != 64
        or any(character not in hexadecimal for character in result_bundle_hash)
    ):
        raise ValueError("result_bundle_hash must be a SHA-256 hex digest")
    policy_hash = _sha256_payload(policy)
    evidence: list[ReleaseEvidence] = []
    for gate in policy.gates:
        if gate.metric not in metrics:
            raise ValueError(f"missing release metric: {gate.metric}")
        observed = float(metrics[gate.metric])
        if not isfinite(observed):
            raise ValueError(f"release metric {gate.metric} must be finite")
        passes = (
            observed <= gate.limit
            if gate.operator == "<="
            else observed >= gate.limit
        )
        status = "PASS" if passes else "REJECT"
        evidence_payload = {
            "gate": gate,
            "input_snapshot_hash": input_snapshot_hash,
            "result_bundle_hash": result_bundle_hash,
            "observed": observed,
            "status": status,
        }
        evidence.append(
            ReleaseEvidence(
                gate_id=gate.gate_id,
                metric=gate.metric,
                observed=observed,
                required=gate.required,
                operator=gate.operator,
                signed_limit=gate.limit,
                unit=gate.unit,
                owner=gate.owner,
                status=status,
                evidence_hash=_sha256_payload(evidence_payload),
            )
        )

    required_failures = [
        item.gate_id for item in evidence if item.required and item.status == "REJECT"
    ]
    optional_failures = [
        item.gate_id
        for item in evidence
        if not item.required and item.status == "REJECT"
    ]
    if required_failures:
        final_decision = "REJECT"
        reason = "required gates failed: " + ", ".join(required_failures)
    elif optional_failures:
        final_decision = "CONDITIONAL_PASS"
        reason = "optional gates require review: " + ", ".join(optional_failures)
    else:
        final_decision = "PASS"
        reason = "all required and optional gates passed"
    release_evidence_hash = _sha256_payload(
        {
            "input_snapshot_hash": input_snapshot_hash,
            "result_bundle_hash": result_bundle_hash,
            "policy_hash": policy_hash,
            "evidence": evidence,
            "final_decision": final_decision,
        }
    )
    return ReleaseDecision(
        policy_version=policy.policy_version,
        policy_hash=policy_hash,
        input_snapshot_hash=input_snapshot_hash,
        result_bundle_hash=result_bundle_hash,
        release_evidence_hash=release_evidence_hash,
        gate_evidence=tuple(evidence),
        final_decision=final_decision,
        decision_reason=reason,
    )


def _surface_fixture() -> tuple[np.ndarray, ...]:
    log_moneyness = np.array([-0.20, -0.10, 0.0, 0.10, 0.20])
    maturities = np.array([0.25, 0.75])
    forwards = np.array([100.0, 100.0])
    discounts = np.exp(-0.02 * maturities)
    implied_vols = np.array(
        [
            [0.30, 0.265, 0.24, 0.225, 0.22],
            [0.31, 0.275, 0.25, 0.235, 0.23],
        ]
    )
    calls = np.empty_like(implied_vols)
    for row, maturity in enumerate(maturities):
        spot = forwards[row] * exp(-0.02 * maturity)
        for column, coordinate in enumerate(log_moneyness):
            strike = forwards[row] * exp(coordinate)
            calls[row, column] = call_price(
                spot,
                strike,
                0.02,
                0.0,
                implied_vols[row, column],
                maturity,
            )
    return (
        log_moneyness,
        maturities,
        forwards,
        discounts,
        calls,
        implied_vols,
    )


def _release_policy(
    limit_overrides: Mapping[str, float] | None,
) -> ReleasePolicy:
    limits = {
        "surface_violation_count": 0.0,
        "surface_box_feasible": 1.0,
        "calendar_greek_feasible": 1.0,
        "calendar_economic": 1.0,
        "iv_interval_usable": 1.0,
        "variance_tail_fraction": 0.25,
        "heston_martingale_error": 0.01,
        "roughness_absolute_bias": 0.12,
        "diffusion_cvar_95": 3.0,
        "jump_cvar_95": 7.0,
        "event_gap_loss": 5.0,
    }
    if limit_overrides:
        unknown = set(limit_overrides) - set(limits)
        if unknown:
            raise ValueError(f"unknown release limit overrides: {sorted(unknown)}")
        limits.update({key: float(value) for key, value in limit_overrides.items()})
    return ReleasePolicy(
        policy_version="ch06-release-policy-v0.10",
        gates=(
            ReleasePolicyGate("surface.geometry", "surface_violation_count", True, "<=", limits["surface_violation_count"], "count", "volatility-control"),
            ReleasePolicyGate("surface.executable", "surface_box_feasible", True, ">=", limits["surface_box_feasible"], "boolean-flag", "market-data-control"),
            ReleasePolicyGate("calendar.greeks", "calendar_greek_feasible", True, ">=", limits["calendar_greek_feasible"], "boolean-flag", "options-risk"),
            ReleasePolicyGate("calendar.economics", "calendar_economic", False, ">=", limits["calendar_economic"], "boolean-flag", "options-trading"),
            ReleasePolicyGate("iv.interval", "iv_interval_usable", True, ">=", limits["iv_interval_usable"], "boolean-flag", "volatility-control"),
            ReleasePolicyGate("variance.tails", "variance_tail_fraction", True, "<=", limits["variance_tail_fraction"], "fraction", "variance-products"),
            ReleasePolicyGate("heston.martingale", "heston_martingale_error", True, "<=", limits["heston_martingale_error"], "relative-error", "model-validation"),
            ReleasePolicyGate("roughness.bias", "roughness_absolute_bias", False, "<=", limits["roughness_absolute_bias"], "Hurst-exponent", "research-validation"),
            ReleasePolicyGate("hedge.diffusion", "diffusion_cvar_95", True, "<=", limits["diffusion_cvar_95"], "currency-per-option", "options-risk"),
            ReleasePolicyGate("hedge.jump", "jump_cvar_95", True, "<=", limits["jump_cvar_95"], "currency-per-option", "options-risk"),
            ReleasePolicyGate("hedge.event", "event_gap_loss", True, "<=", limits["event_gap_loss"], "currency-per-option", "event-risk"),
        ),
    )


def chapter_release_case(
    *,
    surface_mid_shift: float = 2.0,
    limit_overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if not isfinite(surface_mid_shift):
        raise ValueError("surface_mid_shift must be finite")
    fixture_manifest = {
        "fixture_type": "synthetic governed release fixture",
        "fixture_version": "ch06-release-v0.10",
        "surface_version": "surface-v2026.08.02",
        "greek_convention_version": "greeks-v0.10",
        "seed_manifest": [601, 602, 603],
    }
    policy = _release_policy(limit_overrides)

    greeks = call_greeks(100.0, 100.0, 0.02, 0.0, 0.242, 0.5)
    calendar = long_gamma_short_vega_calendar(
        spot=100.0,
        rate=0.02,
        dividend=0.0,
        vol=0.242,
        short_horizon=0.08,
        long_horizon=0.75,
        log_forward_moneyness=0.0,
        volatility_point_scenario=-1.0,
    )
    bid_price = call_price(100.0, 100.0, 0.02, 0.0, 0.237, 0.5)
    ask_price = call_price(100.0, 100.0, 0.02, 0.0, 0.248, 0.5)
    implied_interval = implied_vol_bid_ask(
        bid_price, ask_price, 100.0, 100.0, 0.02, 0.0, 0.5
    )

    k, t, f, d, clean_calls, vols = _surface_fixture()
    reviewed_calls = clean_calls.copy()
    reviewed_calls[0, 2] += surface_mid_shift
    bid_prices = reviewed_calls - 0.03
    ask_prices = reviewed_calls + 0.03
    surface = surface_arbitrage_audit(
        log_forward_moneyness=k,
        maturities=t,
        forwards=f,
        discounts=d,
        call_prices=reviewed_calls,
        implied_vols=vols,
        bid_prices=bid_prices,
        ask_prices=ask_prices,
    )

    variance_forward = 100.0
    variance_horizon = 1.0
    variance_discount = exp(-0.02)
    variance_spot = variance_forward * variance_discount
    strikes = np.linspace(25.0, 250.0, 901)
    strip_calls = np.asarray(
        [
            call_price(
                variance_spot,
                float(strike),
                0.02,
                0.0,
                0.242,
                variance_horizon,
            )
            for strike in strikes
        ]
    )
    strip_puts = strip_calls - variance_discount * (variance_forward - strikes)
    variance = variance_swap_audit(
        forward=variance_forward,
        strikes=strikes,
        puts=strip_puts,
        calls=strip_calls,
        discount=variance_discount,
        horizon=variance_horizon,
    )

    heston_inputs = {
        "v0": 0.05,
        "kappa": 1.5,
        "theta": 0.04,
        "xi": 0.30,
        "rho": -0.75,
        "horizon": 1.0,
        "step_counts": (32, 64),
        "paths": 12_000,
        "seed": 601,
        "martingale_limit": 0.01,
    }
    heston = heston_diagnostics(**heston_inputs)
    roughness_inputs = {
        "true_hurst": 0.14,
        "points": 256,
        "paths": 12,
        "max_lag": 12,
        "seed": 602,
        "tolerance": 0.12,
    }
    roughness = roughness_recovery_study(**roughness_inputs)
    hedge_inputs = {
        "rebalance_counts": (4, 8, 16, 32),
        "paths": 1_500,
        "transaction_cost_bps": 2.5,
        "jump_intensity": 2.0,
        "jump_return": -0.15,
        "seed": 603,
    }
    hedge = hedge_frequency_study(**hedge_inputs)

    full_inputs = {
        "manifest": fixture_manifest,
        "surface": {
            "log_moneyness": k,
            "maturities": t,
            "forwards": f,
            "discounts": d,
            "raw_mid_prices": reviewed_calls,
            "bid_prices": bid_prices,
            "ask_prices": ask_prices,
            "implied_vols": vols,
        },
        "variance_strip": {
            "forward": variance_forward,
            "discount": variance_discount,
            "horizon": variance_horizon,
            "strikes": strikes,
            "puts": strip_puts,
            "calls": strip_calls,
        },
        "heston_inputs": heston_inputs,
        "roughness_inputs": roughness_inputs,
        "hedge_inputs": hedge_inputs,
        "policy": policy,
    }
    input_snapshot_hash = _sha256_payload(full_inputs)
    raw_surface_hash = _sha256_payload(full_inputs["surface"])
    repaired_surface_hash = (
        _sha256_payload(surface.executable_prices)
        if surface.executable_prices is not None
        else None
    )

    metrics = {
        "surface_violation_count": float(len(surface.violations)),
        "surface_box_feasible": float(surface.bid_ask_feasible is True),
        "calendar_greek_feasible": float(calendar.greek_feasibility_status == "PASS"),
        "calendar_economic": float(calendar.economic_scenario_status == "PASS"),
        "iv_interval_usable": float(implied_interval.status == "PASS"),
        "variance_tail_fraction": max(
            variance.lower_tail_fraction, variance.upper_tail_fraction
        ),
        "heston_martingale_error": heston.rows[-1].forward_martingale_error,
        "roughness_absolute_bias": abs(
            roughness.estimated_hurst_mean - roughness.true_hurst
        ),
        "diffusion_cvar_95": hedge.selected_diffusion_cvar_95,
        "jump_cvar_95": hedge.selected_jump_cvar_95,
        "event_gap_loss": hedge.event_gap_loss,
    }
    result_bundle_hash = _sha256_payload(
        {
            "greeks": greeks,
            "calendar": calendar,
            "implied_interval": implied_interval,
            "surface": surface,
            "variance": variance,
            "heston": heston,
            "roughness": roughness,
            "hedge": hedge,
            "metrics": metrics,
        }
    )
    decision = evaluate_release(
        metrics=metrics,
        policy=policy,
        input_snapshot_hash=input_snapshot_hash,
        result_bundle_hash=result_bundle_hash,
    )
    return {
        **fixture_manifest,
        "snapshot_hash": input_snapshot_hash,
        "input_snapshot_hash": input_snapshot_hash,
        "raw_surface_hash": raw_surface_hash,
        "repaired_surface_hash": repaired_surface_hash,
        "policy_hash": decision.policy_hash,
        "result_bundle_hash": decision.result_bundle_hash,
        "release_evidence_hash": decision.release_evidence_hash,
        "raw_vega_per_unit_vol": greeks.raw_vega_per_unit_vol,
        "vega_per_vol_point": greeks.vega_per_vol_point,
        "calendar_status": calendar.status,
        "calendar_greek_status": calendar.greek_feasibility_status,
        "calendar_economic_status": calendar.economic_scenario_status,
        "calendar_net_gamma": calendar.net_gamma,
        "calendar_net_vega_per_vol_point": calendar.net_vega_per_vol_point,
        "implied_vol_bid": implied_interval.bid.implied_vol,
        "implied_vol_ask": implied_interval.ask.implied_vol,
        "surface_violation_count": len(surface.violations),
        "surface_bid_ask_feasible": surface.bid_ask_feasible,
        "surface_repair_l1_norm": surface.repair_l1_norm,
        "surface_status": surface.status,
        "variance_strike": variance.fair_variance,
        "variance_lower_tail_fraction": variance.lower_tail_fraction,
        "heston_feller": heston.feller_condition,
        "heston_martingale_error": heston.rows[-1].forward_martingale_error,
        "heston_status": heston.status,
        "roughness_estimate": roughness.estimated_hurst_mean,
        "roughness_ci": roughness.confidence_interval,
        "roughness_status": roughness.status,
        "selected_rebalances": hedge.selected_rebalances,
        "diffusion_cvar": hedge.selected_diffusion_cvar_95,
        "selected_jump_cvar": hedge.selected_jump_cvar_95,
        "jump_loss_floor": hedge.jump_loss_floor,
        "event_short_option_jump_loss": hedge.event_gap_loss,
        "gate_evidence": [asdict(item) for item in decision.gate_evidence],
        "final_decision": decision.final_decision,
        "decision_reason": decision.decision_reason,
    }
