from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
import json
from math import isfinite


class Severity(str, Enum):
    SOFT = "SOFT"
    HARD = "HARD"


class DecisionState(str, Enum):
    PASS = "PASS"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"
    BLOCK = "BLOCK"
    INVALID_OR_STALE_INPUT = "INVALID_OR_STALE_INPUT"


@dataclass(frozen=True)
class RiskSnapshot:
    snapshot_id: str
    known_at: datetime
    model_version: str
    gross: float
    net: float
    single_name_max: float
    var_99: float
    liquidity_days: float

    def __post_init__(self) -> None:
        if not self.snapshot_id or not self.model_version:
            raise ValueError("risk snapshot lineage is required")
        _require_aware_utc(self.known_at, "known_at")
        values = (
            self.gross,
            self.net,
            self.single_name_max,
            self.var_99,
            self.liquidity_days,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("risk snapshot must be finite")
        if min(
            self.gross,
            self.single_name_max,
            self.var_99,
            self.liquidity_days,
        ) < 0.0:
            raise ValueError("unsigned risk measures must be non-negative")


@dataclass(frozen=True)
class Limits:
    policy_version: str
    gross: float
    abs_net: float
    single_name_max: float
    var_99: float
    liquidity_days: float
    soft_metrics: frozenset[str] = frozenset({"liquidity_days"})
    metric_tolerances: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        values = (
            self.gross,
            self.abs_net,
            self.single_name_max,
            self.var_99,
            self.liquidity_days,
        )
        if not self.policy_version or not all(
            isfinite(value) and value >= 0.0 for value in values
        ):
            raise ValueError("invalid limit policy")
        valid = {
            "gross",
            "abs_net",
            "single_name_max",
            "var_99",
            "liquidity_days",
        }
        if not self.soft_metrics <= valid:
            raise ValueError("unknown soft limit metric")
        tolerance_names = [name for name, _ in self.metric_tolerances]
        if len(tolerance_names) != len(set(tolerance_names)):
            raise ValueError("duplicate metric tolerance")
        if set(tolerance_names) - valid or any(
            not isfinite(value) or value < 0.0
            for _, value in self.metric_tolerances
        ):
            raise ValueError("invalid metric tolerance")

    def tolerance(self, metric: str) -> float:
        return dict(self.metric_tolerances).get(metric, 0.0)


@dataclass(frozen=True)
class Breach:
    metric: str
    observed: float
    limit: float
    severity: Severity
    existing: bool
    worsened: bool

    @property
    def excess(self) -> float:
        return self.observed - self.limit


@dataclass(frozen=True)
class PretradeCertificate:
    status: DecisionState
    before_snapshot_id: str
    after_snapshot_id: str
    policy_version: str
    existing_breaches: tuple[Breach, ...]
    new_breaches: tuple[Breach, ...]
    resolved_breaches: tuple[str, ...]
    input_hash: str
    reason: str


def _metric_values(
    snapshot: RiskSnapshot,
    limits: Limits,
) -> dict[str, tuple[float, float]]:
    return {
        "gross": (snapshot.gross, limits.gross),
        "abs_net": (abs(snapshot.net), limits.abs_net),
        "single_name_max": (
            snapshot.single_name_max,
            limits.single_name_max,
        ),
        "var_99": (snapshot.var_99, limits.var_99),
        "liquidity_days": (
            snapshot.liquidity_days,
            limits.liquidity_days,
        ),
    }


def evaluate(
    snapshot: RiskSnapshot,
    limits: Limits,
) -> dict[str, tuple[float, float, Severity]]:
    return {
        metric: (
            observed,
            limit,
            Severity.SOFT
            if metric in limits.soft_metrics
            else Severity.HARD,
        )
        for metric, (observed, limit) in _metric_values(
            snapshot, limits
        ).items()
        if observed > limit
    }


def pretrade_decision(
    snapshot_before: RiskSnapshot,
    snapshot_after: RiskSnapshot,
    limits: Limits,
    *,
    decision_time: datetime,
    maximum_age: timedelta,
    portfolio_id: str,
    order_id: str,
    risk_data_hash: str,
) -> PretradeCertificate:
    _require_aware_utc(decision_time, "decision_time")
    if maximum_age < timedelta(0):
        raise ValueError("maximum_age must be non-negative")
    if not portfolio_id or not order_id:
        raise ValueError("portfolio_id and order_id are required")
    _require_digest(risk_data_hash, "risk_data_hash")
    input_payload = {
        "before": asdict(snapshot_before),
        "after": asdict(snapshot_after),
        "limits": {
            **asdict(limits),
            "soft_metrics": sorted(limits.soft_metrics),
            "metric_tolerances": limits.metric_tolerances,
        },
        "decision_time": decision_time.isoformat(),
        "maximum_age_seconds": maximum_age.total_seconds(),
        "portfolio_id": portfolio_id,
        "order_id": order_id,
        "risk_data_hash": risk_data_hash,
    }
    input_hash = sha256(
        json.dumps(
            input_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    invalid_lineage = snapshot_before.model_version != snapshot_after.model_version
    if (
        invalid_lineage
        or
        snapshot_before.known_at > decision_time
        or snapshot_after.known_at > decision_time
        or decision_time - snapshot_before.known_at > maximum_age
        or decision_time - snapshot_after.known_at > maximum_age
    ):
        return PretradeCertificate(
            DecisionState.BLOCK,
            snapshot_before.snapshot_id,
            snapshot_after.snapshot_id,
            limits.policy_version,
            (),
            (),
            (),
            input_hash,
            "model mismatch or risk snapshot is future-dated/stale; fail closed",
        )
    before = evaluate(snapshot_before, limits)
    after = evaluate(snapshot_after, limits)
    existing: list[Breach] = []
    new: list[Breach] = []
    for metric, (observed, limit, severity) in after.items():
        if metric in before:
            prior_observed = before[metric][0]
            existing.append(
                Breach(
                    metric,
                    observed,
                    limit,
                    severity,
                    True,
                    observed > prior_observed + limits.tolerance(metric),
                )
            )
        else:
            new.append(
                Breach(
                    metric,
                    observed,
                    limit,
                    severity,
                    False,
                    True,
                )
            )
    resolved = tuple(sorted(set(before) - set(after)))
    if any(
        breach.severity is Severity.HARD
        for breach in new
    ) or any(
        breach.severity is Severity.HARD and breach.worsened
        for breach in existing
    ):
        status = DecisionState.BLOCK
        reason = "new or worsened hard breach"
    elif new or any(breach.worsened for breach in existing):
        status = DecisionState.REQUIRES_APPROVAL
        reason = "new or worsened soft breach"
    else:
        status = DecisionState.PASS
        reason = "no new or worsened breach"
    return PretradeCertificate(
        status,
        snapshot_before.snapshot_id,
        snapshot_after.snapshot_id,
        limits.policy_version,
        tuple(sorted(existing, key=lambda item: item.metric)),
        tuple(sorted(new, key=lambda item: item.metric)),
        resolved,
        input_hash,
        reason,
    )


def require_approval(
    snapshot_before: RiskSnapshot,
    snapshot_after: RiskSnapshot,
    limits: Limits,
    *,
    decision_time: datetime,
    maximum_age: timedelta,
    portfolio_id: str,
    order_id: str,
    risk_data_hash: str,
) -> PretradeCertificate:
    return pretrade_decision(
        snapshot_before,
        snapshot_after,
        limits,
        decision_time=decision_time,
        maximum_age=maximum_age,
        portfolio_id=portfolio_id,
        order_id=order_id,
        risk_data_hash=risk_data_hash,
    )


def _require_aware_utc(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{field} must be normalized to UTC")


def _require_digest(value: str, field: str) -> None:
    if len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
