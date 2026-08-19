from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TriggerDecision:
    triggered: bool
    score: float
    reason_code: str
    decision_time: str
    prior_triggered: bool


def captured_alpha(
    interval: FloatArray | float,
    alpha0: float,
    decay: float,
) -> FloatArray:
    delta = np.asarray(interval, dtype=float)
    if (
        not np.isfinite(delta).all()
        or np.any(delta <= 0.0)
        or not np.isfinite(alpha0)
        or not np.isfinite(decay)
        or decay <= 0.0
    ):
        raise ValueError("interval and decay must be positive and finite")
    ratio = -np.expm1(-decay * delta) / (decay * delta)
    return alpha0 * ratio


def policy_value(
    interval: FloatArray | float,
    alpha0: float,
    decay: float,
    fixed: float,
    impact: float,
    risk_drift: float,
) -> FloatArray:
    delta = np.asarray(interval, dtype=float)
    if (
        not np.isfinite(delta).all()
        or np.any(delta <= 0.0)
        or not np.isfinite(
            np.asarray((alpha0, decay, fixed, impact, risk_drift), dtype=float)
        ).all()
        or decay <= 0.0
        or min(fixed, impact, risk_drift) < 0.0
    ):
        raise ValueError("invalid rebalancing policy inputs")
    return (
        captured_alpha(delta, alpha0, decay)
        - fixed / delta
        - impact * np.sqrt(delta)
        - risk_drift * delta * delta
    )


def event_trigger_decision(
    normalized_alpha_change: float,
    normalized_exposure_change: float,
    breach_probability: float,
    liquidity_cost_penalty: float,
    weights: tuple[float, float, float, float],
    threshold: float,
    *,
    decision_time: str,
    prior_triggered: bool = False,
    maximum_staleness_breached: bool = False,
    hard_risk_breached: bool = False,
) -> TriggerDecision:
    values = np.asarray(
        (
            normalized_alpha_change,
            normalized_exposure_change,
            breach_probability,
            liquidity_cost_penalty,
            *weights,
            threshold,
        ),
        dtype=float,
    )
    if not np.isfinite(values).all() or not decision_time:
        raise ValueError("trigger inputs and decision time are required")
    if not 0.0 <= breach_probability <= 1.0:
        raise ValueError("breach probability must lie in [0, 1]")
    if liquidity_cost_penalty < 0.0:
        raise ValueError("liquidity cost penalty must be nonnegative")
    if any(weight < 0.0 for weight in weights) or threshold < 0.0:
        raise ValueError("trigger weights and threshold must be nonnegative")
    alpha_weight, exposure_weight, risk_weight, liquidity_weight = weights
    score = (
        alpha_weight * abs(normalized_alpha_change)
        + exposure_weight * abs(normalized_exposure_change)
        + risk_weight * breach_probability
        - liquidity_weight * liquidity_cost_penalty
    )
    if hard_risk_breached:
        triggered = True
        reason = "HARD_RISK_OVERRIDE"
    elif maximum_staleness_breached:
        triggered = True
        reason = "MAX_STALENESS"
    elif score > threshold:
        triggered = True
        reason = "SCORE_BREACH"
    elif prior_triggered and score > 0.8 * threshold:
        triggered = True
        reason = "PERSISTENT_TRIGGER"
    else:
        triggered = False
        reason = "HOLD"
    return TriggerDecision(
        triggered=triggered,
        score=float(score),
        reason_code=reason,
        decision_time=decision_time,
        prior_triggered=prior_triggered,
    )


def event_trigger(
    alpha_change: float,
    exposure_change: float,
    breach_risk: float,
    liquidity_cost_penalty: float,
    weights: tuple[float, float, float, float],
    threshold: float,
    *,
    decision_time: str,
    hard_risk_breached: bool = False,
) -> bool:
    return event_trigger_decision(
        alpha_change,
        exposure_change,
        breach_risk,
        liquidity_cost_penalty,
        weights,
        threshold,
        decision_time=decision_time,
        hard_risk_breached=hard_risk_breached,
    ).triggered
