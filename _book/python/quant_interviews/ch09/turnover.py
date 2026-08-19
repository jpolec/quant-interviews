from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HysteresisDecision:
    prior_position: int
    new_position: int
    signal: float
    reason_code: str
    decision_time: str


def soft_threshold(
    target_trade: FloatArray,
    linear_cost: FloatArray,
    curvature: FloatArray,
) -> FloatArray:
    target = np.asarray(target_trade, dtype=float)
    linear = np.asarray(linear_cost, dtype=float)
    quadratic = np.asarray(curvature, dtype=float)
    if target.shape != linear.shape or target.shape != quadratic.shape:
        raise ValueError("shape mismatch")
    if not (
        np.isfinite(target).all()
        and np.isfinite(linear).all()
        and np.isfinite(quadratic).all()
    ):
        raise ValueError("trade inputs must be finite")
    if np.any(quadratic <= 0.0) or np.any(linear < 0.0):
        raise ValueError("invalid costs")
    return np.sign(target) * np.maximum(
        np.abs(target) - linear / quadratic,
        0.0,
    )


def realized_cost(
    trade: FloatArray,
    linear_cost: FloatArray,
    impact: FloatArray,
) -> float:
    vector = np.asarray(trade, dtype=float)
    linear = np.asarray(linear_cost, dtype=float)
    matrix = np.asarray(impact, dtype=float)
    if linear.shape != vector.shape or matrix.shape != (vector.size, vector.size):
        raise ValueError("cost-model shape mismatch")
    if not (
        np.isfinite(vector).all()
        and np.isfinite(linear).all()
        and np.isfinite(matrix).all()
    ):
        raise ValueError("cost-model inputs must be finite")
    if np.any(linear < 0.0) or not np.allclose(
        matrix,
        matrix.T,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("linear cost must be nonnegative and impact symmetric")
    if float(np.linalg.eigvalsh(matrix).min()) < -1e-12:
        raise ValueError("impact matrix must be positive semidefinite")
    return float(linear @ np.abs(vector) + 0.5 * vector @ matrix @ vector)


def hysteresis_action(
    signal: float,
    position: int,
    enter: float,
    exit: float,
) -> int:
    if position not in (-1, 0, 1):
        raise ValueError("position must be -1, 0 or 1")
    if not np.isfinite(signal):
        raise ValueError("signal must be finite")
    if not 0.0 <= exit < enter:
        raise ValueError("require 0 <= exit < enter")
    new_side = 1 if signal > 0.0 else -1 if signal < 0.0 else 0
    if position == 0:
        return new_side if abs(signal) > enter else 0
    if new_side != position:
        return new_side if abs(signal) > enter else 0
    return 0 if abs(signal) < exit else position


def hysteresis_decision(
    signal: float,
    position: int,
    enter: float,
    exit: float,
    decision_time: str,
) -> HysteresisDecision:
    if not decision_time:
        raise ValueError("decision time is required for replay")
    new_position = hysteresis_action(signal, position, enter, exit)
    if new_position == position:
        reason = "HOLD"
    elif new_position == 0:
        reason = "EXIT"
    elif position == 0:
        reason = "ENTER"
    else:
        reason = "REVERSE"
    return HysteresisDecision(
        prior_position=position,
        new_position=new_position,
        signal=float(signal),
        reason_code=reason,
        decision_time=decision_time,
    )
