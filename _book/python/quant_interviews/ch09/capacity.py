from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CapacityResult:
    coefficient: float
    break_even_aum: float
    profit_optimal_aum: float


@dataclass(frozen=True)
class CapacityDiagnostics:
    candidate_aum: float
    participation: FloatArray
    stressed_participation: FloatArray
    liquidation_days: FloatArray
    binding_instrument_index: int
    maximum_stressed_participation: float
    participation_limit: float
    volume_stress_multiplier: float
    status: str


def capacity_coefficient(
    trade_notional_per_aum: FloatArray,
    volatility: FloatArray,
    dollar_volume: FloatArray,
    impact: FloatArray,
) -> float:
    trade_rate = np.abs(np.asarray(trade_notional_per_aum, dtype=float))
    sigma = np.asarray(volatility, dtype=float)
    volume = np.asarray(dollar_volume, dtype=float)
    coefficient = np.asarray(impact, dtype=float)
    if (
        trade_rate.shape != sigma.shape
        or trade_rate.shape != volume.shape
        or trade_rate.shape != coefficient.shape
        or not np.isfinite(trade_rate).all()
        or not np.isfinite(sigma).all()
        or not np.isfinite(volume).all()
        or not np.isfinite(coefficient).all()
        or np.any(volume <= 0.0)
        or np.any(sigma < 0.0)
        or np.any(coefficient < 0.0)
    ):
        raise ValueError("invalid capacity inputs")
    return float(
        np.sum(
            coefficient
            * sigma
            * trade_rate**1.5
            / np.sqrt(volume)
        )
    )


def capacity(
    alpha: float,
    fixed_cost: float,
    coefficient: float,
) -> CapacityResult:
    edge = alpha - fixed_cost
    if (
        not np.isfinite(edge)
        or not np.isfinite(coefficient)
        or edge <= 0.0
        or coefficient <= 0.0
    ):
        raise ValueError("positive finite edge and coefficient required")
    return CapacityResult(
        coefficient=coefficient,
        break_even_aum=(edge / coefficient) ** 2,
        profit_optimal_aum=(2.0 * edge / (3.0 * coefficient)) ** 2,
    )


def net_alpha(
    aum: FloatArray | float,
    alpha: float,
    fixed_cost: float,
    coefficient: float,
) -> FloatArray:
    scale = np.asarray(aum, dtype=float)
    if (
        not np.isfinite(scale).all()
        or np.any(scale < 0.0)
        or not np.isfinite(
            np.asarray((alpha, fixed_cost, coefficient), dtype=float)
        ).all()
        or fixed_cost < 0.0
        or coefficient < 0.0
    ):
        raise ValueError("AUM and coefficient must be nonnegative")
    return alpha - fixed_cost - coefficient * np.sqrt(scale)


def capacity_diagnostics(
    trade_notional_per_aum: FloatArray,
    candidate_aum: float,
    dollar_adv: FloatArray,
    execution_horizon_days: float,
    maximum_participation: float,
    volume_stress_multiplier: float = 0.7,
) -> CapacityDiagnostics:
    trade_rate = np.abs(np.asarray(trade_notional_per_aum, dtype=float))
    adv = np.asarray(dollar_adv, dtype=float)
    scalars = np.asarray(
        (
            candidate_aum,
            execution_horizon_days,
            maximum_participation,
            volume_stress_multiplier,
        ),
        dtype=float,
    )
    if (
        trade_rate.ndim != 1
        or trade_rate.size == 0
        or trade_rate.shape != adv.shape
        or not np.isfinite(trade_rate).all()
        or not np.isfinite(adv).all()
        or not np.isfinite(scalars).all()
        or candidate_aum <= 0.0
        or execution_horizon_days <= 0.0
        or not 0.0 < maximum_participation <= 1.0
        or not 0.0 < volume_stress_multiplier <= 1.0
        or np.any(adv <= 0.0)
    ):
        raise ValueError("invalid capacity-diagnostic inputs")
    notional = trade_rate * candidate_aum
    participation = notional / (adv * execution_horizon_days)
    stressed = participation / volume_stress_multiplier
    liquidation_days = notional / (
        maximum_participation * adv * volume_stress_multiplier
    )
    binding = int(np.argmax(stressed))
    maximum = float(stressed[binding])
    return CapacityDiagnostics(
        candidate_aum=float(candidate_aum),
        participation=participation,
        stressed_participation=stressed,
        liquidation_days=liquidation_days,
        binding_instrument_index=binding,
        maximum_stressed_participation=maximum,
        participation_limit=float(maximum_participation),
        volume_stress_multiplier=float(volume_stress_multiplier),
        status="PASS" if maximum <= maximum_participation else "REJECT",
    )
