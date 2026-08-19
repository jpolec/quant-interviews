from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
import re
from typing import Literal


HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FillThreshold:
    classification: Literal[
        "INTERIOR",
        "LIMIT_DOMINATES",
        "MARKET_DOMINATES",
        "EQUAL",
        "INVALID_MODEL",
    ]
    probability: float | None
    limit_cost_at_zero: float
    limit_cost_at_one: float
    market_cost: float


@dataclass(frozen=True)
class OrderDecision:
    market_cost_bps: float
    limit_expected_cost_bps: float
    no_trade_cost_bps: float
    market_inventory_penalty_bps: float
    limit_inventory_penalty_bps: float
    market_objective_bps: float
    limit_objective_bps: float
    no_trade_objective_bps: float
    selected_action: Literal[
        "MARKET",
        "LIMIT",
        "NO_TRADE",
        "BLOCK_STALE_MARKET",
        "BLOCK_RISK",
        "REJECT",
    ]
    market_age_seconds: float
    risk_utilization: float
    decision_time: str
    policy_version: str
    decision_hash: str
    status: Literal["PASS", "REJECT"]


def _finite(*values: float) -> bool:
    return all(isfinite(value) for value in values)


def _utc(value: str, *, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _digest(value: dict[str, object]) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _hash(value: str, *, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def market_order_cost(
    half_spread_bps: float,
    impact_bps: float,
    fees_bps: float,
) -> float:
    if not _finite(half_spread_bps, impact_bps, fees_bps):
        raise ValueError("market-order costs must be finite")
    if min(half_spread_bps, impact_bps, fees_bps) < 0.0:
        raise ValueError("market-order costs must be non-negative")
    result = half_spread_bps + impact_bps + fees_bps
    if not isfinite(result):
        raise FloatingPointError("market-order cost overflow")
    return result


def limit_order_expected_cost(
    fill_probability: float,
    rebate_bps: float,
    adverse_selection_bps: float,
    miss_cost_bps: float,
) -> float:
    if not _finite(
        fill_probability,
        rebate_bps,
        adverse_selection_bps,
        miss_cost_bps,
    ):
        raise ValueError("limit-order inputs must be finite")
    if not 0.0 <= fill_probability <= 1.0:
        raise ValueError("fill_probability must lie in [0, 1]")
    if min(rebate_bps, adverse_selection_bps, miss_cost_bps) < 0.0:
        raise ValueError("cost inputs must be non-negative")
    filled_cost = adverse_selection_bps - rebate_bps
    result = fill_probability * filled_cost + (1.0 - fill_probability) * miss_cost_bps
    if not isfinite(result):
        raise FloatingPointError("limit-order expected cost overflow")
    return result


def fill_probability_threshold(
    market_cost_bps: float,
    filled_limit_cost_bps: float,
    miss_cost_bps: float,
) -> FillThreshold:
    if not _finite(market_cost_bps, filled_limit_cost_bps, miss_cost_bps):
        raise ValueError("threshold inputs must be finite")
    if min(market_cost_bps, filled_limit_cost_bps, miss_cost_bps) < 0.0:
        return FillThreshold(
            "INVALID_MODEL",
            None,
            miss_cost_bps,
            filled_limit_cost_bps,
            market_cost_bps,
        )
    endpoint_zero = miss_cost_bps
    endpoint_one = filled_limit_cost_bps
    if endpoint_zero == market_cost_bps and endpoint_one == market_cost_bps:
        classification = "EQUAL"
        probability = None
    elif max(endpoint_zero, endpoint_one) <= market_cost_bps:
        classification = "LIMIT_DOMINATES"
        probability = None
    elif min(endpoint_zero, endpoint_one) >= market_cost_bps:
        classification = "MARKET_DOMINATES"
        probability = None
    else:
        denominator = miss_cost_bps - filled_limit_cost_bps
        threshold = (miss_cost_bps - market_cost_bps) / denominator
        if not 0.0 < threshold < 1.0:
            raise AssertionError("crossing endpoints require an interior threshold")
        classification = "INTERIOR"
        probability = threshold
    return FillThreshold(
        classification,  # type: ignore[arg-type]
        probability,
        endpoint_zero,
        endpoint_one,
        market_cost_bps,
    )


def inventory_risk_penalty(
    inventory: float,
    volatility: float,
    horizon_years: float,
    risk_aversion_bps_per_variance: float,
) -> float:
    if not _finite(inventory, volatility, horizon_years, risk_aversion_bps_per_variance):
        raise ValueError("inventory-risk inputs must be finite")
    if volatility < 0.0 or horizon_years < 0.0 or risk_aversion_bps_per_variance < 0.0:
        raise ValueError("volatility, horizon and risk aversion must be non-negative")
    result = (
        0.5
        * risk_aversion_bps_per_variance
        * inventory
        * inventory
        * volatility
        * volatility
        * horizon_years
    )
    if not isfinite(result):
        raise FloatingPointError("inventory-risk penalty overflow")
    return result


def choose_order(
    *,
    market_cost_bps: float,
    limit_expected_cost_bps: float,
    no_trade_cost_bps: float,
    current_inventory: float,
    order_quantity: float,
    fill_probability: float,
    volatility: float,
    horizon_years: float,
    risk_aversion_bps_per_variance: float,
    proposed_risk: float,
    risk_limit: float,
    decision_time: str,
    market_observed_at: str,
    maximum_market_age_seconds: float,
    market_snapshot_hash: str,
    fill_model_hash: str,
    portfolio_snapshot_hash: str,
    limit_snapshot_hash: str,
    policy_version: str,
) -> OrderDecision:
    numeric = (
        market_cost_bps,
        limit_expected_cost_bps,
        no_trade_cost_bps,
        current_inventory,
        order_quantity,
        fill_probability,
        proposed_risk,
        risk_limit,
        maximum_market_age_seconds,
    )
    if not _finite(*numeric):
        raise ValueError("order-decision inputs must be finite")
    if min(market_cost_bps, limit_expected_cost_bps, no_trade_cost_bps) < 0.0:
        raise ValueError("order costs must be non-negative")
    if not 0.0 <= fill_probability <= 1.0:
        raise ValueError("fill_probability must lie in [0, 1]")
    if proposed_risk < 0.0 or risk_limit <= 0.0 or maximum_market_age_seconds < 0.0:
        raise ValueError("risk and freshness policy are invalid")
    if not policy_version.strip():
        raise ValueError("policy_version is required")
    hashes = {
        "market_snapshot_hash": _hash(market_snapshot_hash, name="market_snapshot_hash"),
        "fill_model_hash": _hash(fill_model_hash, name="fill_model_hash"),
        "portfolio_snapshot_hash": _hash(
            portfolio_snapshot_hash, name="portfolio_snapshot_hash"
        ),
        "limit_snapshot_hash": _hash(limit_snapshot_hash, name="limit_snapshot_hash"),
    }
    decision_at = _utc(decision_time, name="decision_time")
    market_at = _utc(market_observed_at, name="market_observed_at")
    age = (decision_at - market_at).total_seconds()

    post_fill = current_inventory + order_quantity
    market_penalty = inventory_risk_penalty(
        post_fill, volatility, horizon_years, risk_aversion_bps_per_variance
    )
    miss_penalty = inventory_risk_penalty(
        current_inventory, volatility, horizon_years, risk_aversion_bps_per_variance
    )
    limit_penalty = (
        fill_probability * market_penalty + (1.0 - fill_probability) * miss_penalty
    )
    market_objective = market_cost_bps + market_penalty
    limit_objective = limit_expected_cost_bps + limit_penalty
    no_trade_objective = no_trade_cost_bps + miss_penalty
    if not _finite(
        market_objective, limit_objective, no_trade_objective, limit_penalty
    ):
        raise FloatingPointError("order objective is non-finite")

    if age < 0.0 or age > maximum_market_age_seconds:
        selected: Literal[
            "MARKET",
            "LIMIT",
            "NO_TRADE",
            "BLOCK_STALE_MARKET",
            "BLOCK_RISK",
            "REJECT",
        ] = "BLOCK_STALE_MARKET"
        status: Literal["PASS", "REJECT"] = "REJECT"
    elif proposed_risk > risk_limit:
        selected = "BLOCK_RISK"
        status = "REJECT"
    else:
        selected = min(
            (
                (market_objective, "MARKET"),
                (limit_objective, "LIMIT"),
                (no_trade_objective, "NO_TRADE"),
            ),
            key=lambda item: (item[0], item[1]),
        )[1]  # type: ignore[assignment]
        status = "PASS"

    payload: dict[str, object] = {
        "costs": {
            "market": market_cost_bps,
            "limit": limit_expected_cost_bps,
            "no_trade": no_trade_cost_bps,
        },
        "decision_time": decision_at.isoformat(),
        "fill_probability": fill_probability,
        "freshness": {
            "market_age_seconds": age,
            "maximum_market_age_seconds": maximum_market_age_seconds,
        },
        "hashes": hashes,
        "inventory": {
            "current": current_inventory,
            "order": order_quantity,
        },
        "objectives": {
            "market": market_objective,
            "limit": limit_objective,
            "no_trade": no_trade_objective,
        },
        "policy_version": policy_version,
        "risk": {"proposed": proposed_risk, "limit": risk_limit},
        "selected_action": selected,
        "status": status,
    }
    return OrderDecision(
        market_cost_bps=market_cost_bps,
        limit_expected_cost_bps=limit_expected_cost_bps,
        no_trade_cost_bps=no_trade_cost_bps,
        market_inventory_penalty_bps=market_penalty,
        limit_inventory_penalty_bps=limit_penalty,
        market_objective_bps=market_objective,
        limit_objective_bps=limit_objective,
        no_trade_objective_bps=no_trade_objective,
        selected_action=selected,
        market_age_seconds=age,
        risk_utilization=proposed_risk / risk_limit,
        decision_time=decision_at.isoformat(),
        policy_version=policy_version,
        decision_hash=_digest(payload),
        status=status,
    )
