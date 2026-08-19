from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DrawdownStats:
    maximum_drawdown: float
    peak_index: int
    trough_index: int
    recovery_index: int | None
    recovery_time: int | None
    maximum_drawdown_duration: int
    current_drawdown_duration: int
    total_underwater_periods: int
    ulcer_index: float

    @property
    def time_under_water(self) -> int:
        """Deprecated alias for total underwater periods."""
        return self.total_underwater_periods


def wealth_from_returns(returns: FloatArray) -> FloatArray:
    values = np.asarray(returns, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or np.any(values <= -1.0)
    ):
        raise ValueError("returns must be finite and greater than -100%")
    return np.concatenate(([1.0], np.cumprod(1.0 + values)))


def _maximum_underwater_run(drawdown: FloatArray) -> tuple[int, int]:
    maximum = 0
    current = 0
    for value in drawdown[1:]:
        if value > 0.0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum, current


def drawdown_stats(returns: FloatArray) -> DrawdownStats:
    wealth = wealth_from_returns(returns)
    running_peak = np.maximum.accumulate(wealth)
    drawdown = 1.0 - wealth / running_peak
    trough = int(np.argmax(drawdown))
    relevant_peak = running_peak[trough]
    peak_candidates = np.flatnonzero(
        np.isclose(wealth[: trough + 1], relevant_peak, atol=0.0, rtol=1e-14)
    )
    peak_index = int(peak_candidates[-1])
    recovery_candidates = np.flatnonzero(
        wealth[trough + 1 :] >= relevant_peak
    )
    recovery_index = (
        None
        if recovery_candidates.size == 0
        else int(trough + 1 + recovery_candidates[0])
    )
    maximum_duration, current_duration = _maximum_underwater_run(drawdown)
    return DrawdownStats(
        maximum_drawdown=float(drawdown[trough]),
        peak_index=peak_index,
        trough_index=trough,
        recovery_index=recovery_index,
        recovery_time=(
            None
            if recovery_index is None
            else int(recovery_index - peak_index)
        ),
        maximum_drawdown_duration=maximum_duration,
        current_drawdown_duration=current_duration,
        total_underwater_periods=int(np.sum(drawdown > 0.0)),
        ulcer_index=float(np.sqrt(np.mean(drawdown[1:] * drawdown[1:]))),
    )


def conditional_drawdown_at_risk(
    returns: FloatArray,
    alpha: float = 0.9,
) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    wealth = wealth_from_returns(returns)
    drawdown = 1.0 - wealth / np.maximum.accumulate(wealth)
    drawdown = drawdown[1:]
    ordered = np.sort(drawdown)[::-1]
    tail_mass = (1.0 - alpha) * ordered.size
    full_observations = int(np.floor(tail_mass))
    fractional_mass = tail_mass - full_observations
    total = float(np.sum(ordered[:full_observations]))
    if fractional_mass > 0.0:
        total += fractional_mass * float(ordered[full_observations])
    return total / tail_mass


def moving_block_paths(
    returns: FloatArray,
    block: int,
    paths: int,
    seed: int = 0,
) -> FloatArray:
    values = np.asarray(returns, dtype=float)
    if (
        values.ndim != 1
        or values.size < 1
        or not np.isfinite(values).all()
        or np.any(values <= -1.0)
        or not 1 <= block <= values.size
        or paths < 1
    ):
        raise ValueError("invalid moving-block bootstrap inputs")
    rng = np.random.default_rng(seed)
    output = np.empty((paths, values.size))
    for path in range(paths):
        resampled: list[float] = []
        while len(resampled) < values.size:
            start = int(rng.integers(0, values.size - block + 1))
            resampled.extend(values[start : start + block])
        output[path] = resampled[: values.size]
    return output
