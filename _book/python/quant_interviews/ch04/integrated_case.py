from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import re
from typing import Any, Mapping

import numpy as np

from quant_interviews.ch04.correlated_diffusions import correlation_input_gate
from quant_interviews.ch04.exponential_martingale import density_process_diagnostic
from quant_interviews.ch04.feynman_kac import ou_terminal_and_running_payoff
from quant_interviews.ch04.gbm import exact_transition_audit
from quant_interviews.ch04.girsanov_importance import normal_tail_importance_sampling
from quant_interviews.ch04.hitting_probabilities import hitting_convergence_study
from quant_interviews.ch04.ito_integrals import stochastic_integral_martingale_study
from quant_interviews.ch04.martingale_representation import spanning_hedge_certificate
from quant_interviews.ch04.ou_exact import ou_conditional_moments, simulate_ou_exact
from quant_interviews.ch04.predictability_guard import (
    TemporalValue,
    select_as_known,
    temporal_snapshot_hash,
)
from quant_interviews.ch04.quadratic_variation import quadratic_variation_convergence
from quant_interviews.ch04.ratio_gbm import ratio_gbm_parameter_audit
from quant_interviews.ch04.sde_schemes import gbm_scheme_convergence


@dataclass(frozen=True)
class ReleaseGate:
    gate_id: str
    value: str
    operator: str
    limit: str
    unit: str
    owner: str
    evidence_sha256: str
    status: str
    reason: str
    required: bool = True


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(nested) for nested in value]
    return value


def _hash(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _gate(
    gate_id: str,
    value: str,
    operator: str,
    limit: str,
    unit: str,
    owner: str,
    evidence: Any,
    status: str,
    reason: str,
) -> ReleaseGate:
    return ReleaseGate(
        gate_id=gate_id,
        value=value,
        operator=operator,
        limit=limit,
        unit=unit,
        owner=owner,
        evidence_sha256=_hash(evidence),
        status=status,
        reason=reason,
    )


def reduce_release_gates(gates: tuple[ReleaseGate, ...]) -> str:
    required = tuple(gate for gate in gates if gate.required)
    if not required or any(
        gate.status == "REJECT"
        or not gate.owner
        or _SHA256_RE.fullmatch(gate.evidence_sha256) is None
        for gate in required
    ):
        return "REJECT"
    if any(gate.status == "CONDITIONAL_PASS" for gate in required):
        return "CONDITIONAL_PASS"
    if all(gate.status == "PASS" for gate in required):
        return "PASS"
    return "REJECT"


def _source_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def chapter_release_case(
    *,
    external_evidence: Mapping[str, tuple[str, str]] | None = None,
    correlation_approval_owner: str | None = "model-risk-owner",
) -> dict[str, object]:
    """Run the governed CH04 fixture, including explicit SQL/C++ evidence inputs."""

    external = {} if external_evidence is None else dict(external_evidence)
    valid_from = datetime(2026, 7, 29, 0, 0, tzinfo=UTC)
    valid_to = datetime(2026, 7, 30, 0, 0, tzinfo=UTC)
    rows = (
        TemporalValue(
            key="volatility",
            valid_from=valid_from,
            valid_to=valid_to,
            published_at=datetime(2026, 7, 29, 16, 0, tzinfo=UTC),
            ingested_at=datetime(2026, 7, 29, 16, 1, tzinfo=UTC),
            value=0.20,
            revision_id=1,
            superseded_at=datetime(2026, 7, 30, 9, 0, tzinfo=UTC),
            source_priority=10,
            payload_hash=_source_hash("volatility-r1"),
        ),
        TemporalValue(
            key="volatility",
            valid_from=valid_from,
            valid_to=valid_to,
            published_at=datetime(2026, 7, 30, 8, 58, tzinfo=UTC),
            ingested_at=datetime(2026, 7, 30, 9, 0, tzinfo=UTC),
            value=0.23,
            revision_id=2,
            source_priority=10,
            payload_hash=_source_hash("volatility-r2"),
        ),
        TemporalValue(
            key="drift",
            valid_from=valid_from,
            valid_to=valid_to,
            published_at=datetime(2026, 7, 29, 15, 58, tzinfo=UTC),
            ingested_at=datetime(2026, 7, 29, 16, 2, tzinfo=UTC),
            value=0.03,
            revision_id=1,
            source_priority=10,
            payload_hash=_source_hash("drift-r1"),
        ),
    )
    decision_time = datetime(2026, 7, 29, 16, 30, tzinfo=UTC)
    selected = select_as_known(
        rows,
        key="volatility",
        valid_at=datetime(2026, 7, 29, 16, 0, tzinfo=UTC),
        decision_time=decision_time,
    )
    snapshot_hash = temporal_snapshot_hash(rows)

    qv = quadratic_variation_convergence(paths=6_000, seed=404)
    ito = stochastic_integral_martingale_study(paths=10_000, seed=405)
    exact_transition = exact_transition_audit(paths=20_000, seed=406)
    ou_mean, ou_variance = ou_conditional_moments(
        1.2, kappa=1.4, theta=0.5, sigma=0.3, delta=0.25
    )
    ou_paths = simulate_ou_exact(
        1.2,
        kappa=1.4,
        theta=0.5,
        sigma=0.3,
        dt=0.25,
        steps=1,
        paths=15_000,
        seed=407,
    )
    ou_mean_error = abs(float(np.mean(ou_paths[:, 1])) - ou_mean)
    ou_mean_se = math.sqrt(ou_variance / ou_paths.shape[0])
    ou_status = "PASS" if ou_mean_error <= 5.0 * ou_mean_se else "REJECT"
    ratio = ratio_gbm_parameter_audit(
        mu_x=0.04,
        mu_y=0.02,
        sigma_x=0.2,
        sigma_y=0.15,
        correlation=0.4,
    )

    paths = 20_000
    steps = 64
    dt = 1.0 / steps
    rng = np.random.default_rng(408)
    d_w = math.sqrt(dt) * rng.standard_normal((paths, steps))
    theta = np.full_like(d_w, 0.3)
    density = density_process_diagnostic(
        theta,
        d_w,
        dt=dt,
        integrand_timing="left_endpoint_predictable",
        minimum_ess_fraction=0.8,
    )
    importance = normal_tail_importance_sampling(
        3.0, shift=3.0, paths=60_000, seed=409
    )
    hedge = spanning_hedge_certificate(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        np.array([1.2, -0.7]),
    )
    hitting = hitting_convergence_study(
        0.6,
        0.0,
        1.0,
        step_sizes=(0.008, 0.004, 0.002),
        paths=8_000,
        seed=410,
    )
    feynman_kac = ou_terminal_and_running_payoff(
        state=0.7,
        kappa=1.2,
        theta=0.4,
        sigma=0.25,
        rate=0.03,
        maturity=1.0,
        running_weight=0.6,
        steps=128,
        paths=30_000,
        seed=413,
    )
    schemes = gbm_scheme_convergence(paths=12_000, seed=414)
    correlation = correlation_input_gate(
        np.array([[1.0, 1.02], [1.02, 1.0]]),
        maximum_repair_norm=0.04,
        approval_owner=correlation_approval_owner,
    )

    gates: list[ReleaseGate] = [
        _gate("PIT_SNAPSHOT", snapshot_hash, "matches", snapshot_hash, "sha256", "data-owner", rows, "PASS", "canonical bitemporal hash"),
        _gate("QUADRATIC_VARIATION_ORDER", f"{qv.rmse_mesh_slope:.6f}", "in", "[0.4,0.6]", "slope", "quant-validation", qv, qv.status, "nested Brownian refinement"),
        _gate("ITO_PREDICTABILITY", ito.status, "equals", "PASS", "state", "quant-validation", ito, ito.status, "predictable sum passes; anticipative control fails"),
        _gate("GBM_EXACT_TRANSITION", exact_transition.status, "equals", "PASS", "state", "quant-validation", exact_transition, exact_transition.status, "moments, positivity and semigroup"),
        _gate("OU_EXACT_TRANSITION", f"{ou_mean_error:.6g}", "<=", f"{5.0 * ou_mean_se:.6g}", "state", "quant-validation", {"mean_error": ou_mean_error, "limit": 5.0 * ou_mean_se}, ou_status, "independent exact-moment check"),
        _gate("RATIO_DIFFUSION", ratio.status, "equals", "PASS", "state", "model-owner", ratio, ratio.status, "finite nondegenerate variance"),
        _gate("DENSITY_PROCESS", f"{density.effective_sample_size_fraction:.6f}", ">=", "0.8", "fraction", "model-risk", density, density.status, density.reason),
        _gate("IMPORTANCE_SAMPLING", f"{importance.variance_reduction:.3f}", ">", "1", "ratio", "quant-validation", importance, importance.status, "analytic tail lies in sampling interval"),
        _gate("MARKET_SPANNING", f"rank={hedge.factor_rank};nullity={hedge.holdings_nullity}", "rank", "factor_count", "count", "model-owner", hedge, "PASS" if hedge.market_complete and hedge.claim_spanned else "REJECT", "price unique; minimum-norm holdings disclosed"),
        _gate("HITTING_REFINEMENT", f"{hitting.rows[-1].estimate:.6f}", "signed budgets", "prob=0.025;time=0.055", "probability/time", "numerical-risk", hitting, hitting.status, "three-grid endpoint-monitoring study"),
        _gate("FEYNMAN_KAC", f"{feynman_kac.analytic_value:.10f}", "sampling+grid", "CI and 1e-4", "price", "quant-validation", feynman_kac, feynman_kac.status, "time-of-payment discounting"),
        _gate("SDE_SCHEMES", f"{schemes.euler_strong_slope:.4f}/{schemes.milstein_strong_slope:.4f}", "in", "[0.4,0.65]/[0.85,1.15]", "slope", "numerical-risk", schemes, schemes.status, schemes.weak_order_status),
        _gate("CORRELATION_REPAIR", str(correlation.repair_frobenius_norm), "<=", "0.04", "Frobenius norm", correlation_approval_owner or "", correlation, correlation.status, correlation.reason),
    ]
    for gate_id in ("SQL_ASOF", "SQL_CORRELATION", "CPP_GBM", "CPP_SDE"):
        status, evidence_hash = external.get(gate_id, ("REJECT", ""))
        gates.append(
            ReleaseGate(
                gate_id=gate_id,
                value=status,
                operator="equals",
                limit="PASS",
                unit="state",
                owner="release-engineering",
                evidence_sha256=evidence_hash,
                status=status,
                reason="external executable evidence required",
            )
        )
    gate_tuple = tuple(gates)
    final = reduce_release_gates(gate_tuple)
    return {
        "fixture_type": "synthetic governed release fixture",
        "fixture_version": "ch04-release-v0.10",
        "seed_manifest": {
            "qv": 404,
            "ito": 405,
            "gbm": 406,
            "ou": 407,
            "density": 408,
            "importance": 409,
            "hitting": [410, 411, 412],
            "feynman_kac": 413,
            "schemes": 414,
        },
        "snapshot_hash": snapshot_hash,
        "input_count": len(rows),
        "excluded_revision_count": sum(
            row.key == "volatility" and row.known_at > decision_time for row in rows
        ),
        "selected_volatility": selected.value,
        "gates": tuple(asdict(gate) for gate in gate_tuple),
        "final_decision": final,
    }
