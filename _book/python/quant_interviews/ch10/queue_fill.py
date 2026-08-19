from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class QueueModel:
    queue_ahead_lots: int
    marketable_lots_per_second: float
    cancellation_ahead_lots_per_second: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.queue_ahead_lots, bool)
            or not isinstance(self.queue_ahead_lots, int)
            or self.queue_ahead_lots < 0
        ):
            raise ValueError("queue_ahead_lots must be a non-negative integer")
        rates = (
            self.marketable_lots_per_second,
            self.cancellation_ahead_lots_per_second,
        )
        if not all(np.isfinite(rate) and rate >= 0.0 for rate in rates):
            raise ValueError("flow rates must be finite and non-negative")

    def fill_probability(self, horizon_seconds: float) -> float:
        if horizon_seconds < 0.0 or not np.isfinite(horizon_seconds):
            raise ValueError("horizon must be finite and non-negative")
        if horizon_seconds == 0.0 or self.marketable_lots_per_second == 0.0:
            return 0.0

        queue_states = self.queue_ahead_lots + 1
        generator = np.zeros((queue_states, queue_states), dtype=float)
        depletion_rate = (
            self.marketable_lots_per_second
            + self.cancellation_ahead_lots_per_second
        )
        for state in range(self.queue_ahead_lots):
            generator[state, state] = -depletion_rate
            generator[state, state + 1] = depletion_rate
        # Once no lots remain ahead, only marketable flow can fill our lot.
        generator[-1, -1] = -self.marketable_lots_per_second
        initial = np.zeros(queue_states, dtype=float)
        initial[0] = 1.0
        survival = float(np.sum(initial @ expm(horizon_seconds * generator)))
        return float(np.clip(1.0 - survival, 0.0, 1.0))

    def simulate_fill_probability(
        self,
        horizon_seconds: float,
        simulations: int = 50_000,
        seed: int = 7,
    ) -> float:
        if horizon_seconds < 0.0 or not np.isfinite(horizon_seconds):
            raise ValueError("horizon must be finite and non-negative")
        if (
            isinstance(simulations, bool)
            or not isinstance(simulations, int)
            or simulations <= 0
        ):
            raise ValueError("simulations must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if horizon_seconds == 0.0 or self.marketable_lots_per_second == 0.0:
            return 0.0
        rng = np.random.default_rng(seed)
        depletion_rate = (
            self.marketable_lots_per_second
            + self.cancellation_ahead_lots_per_second
        )
        if self.queue_ahead_lots:
            queue_clear_time = rng.gamma(
                shape=self.queue_ahead_lots,
                scale=1.0 / depletion_rate,
                size=simulations,
            )
        else:
            queue_clear_time = np.zeros(simulations, dtype=float)
        own_fill_wait = rng.exponential(
            1.0 / self.marketable_lots_per_second,
            size=simulations,
        )
        fills = queue_clear_time + own_fill_wait <= horizon_seconds
        return float(np.mean(fills))


@dataclass(frozen=True)
class CompetingRiskResult:
    fill_first_probability: float
    cancel_first_probability: float
    adverse_move_first_probability: float
    survival_probability: float


def competing_risks(
    fill_rate: float,
    cancel_rate: float,
    adverse_move_rate: float,
    horizon_seconds: float,
) -> CompetingRiskResult:
    rates = np.asarray([fill_rate, cancel_rate, adverse_move_rate], dtype=float)
    if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("competing-risk rates must be finite and non-negative")
    if not np.isfinite(horizon_seconds) or horizon_seconds < 0.0:
        raise ValueError("horizon must be finite and non-negative")
    total = float(rates.sum())
    if total == 0.0:
        return CompetingRiskResult(0.0, 0.0, 0.0, 1.0)
    survival = float(np.exp(-total * horizon_seconds))
    event_probability = 1.0 - survival
    probabilities = event_probability * rates / total
    return CompetingRiskResult(
        fill_first_probability=float(probabilities[0]),
        cancel_first_probability=float(probabilities[1]),
        adverse_move_first_probability=float(probabilities[2]),
        survival_probability=survival,
    )
