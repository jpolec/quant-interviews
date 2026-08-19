from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, sqrt
from typing import Callable

import numpy as np

from .black_scholes_greeks import call_greeks, call_price


@dataclass(frozen=True)
class JumpGapResult:
    spot_change: float
    exact_delta_hedged_gap: float
    gamma_approximation: float
    approximation_error: float


@dataclass(frozen=True)
class HedgeFrequencyRow:
    rebalances: int
    hedge_error_rmse: float
    hedge_loss_cvar_95: float
    mean_transaction_cost: float
    model_fair_value_mismatch: float
    governed_objective: float


@dataclass(frozen=True)
class HedgeFrequencyStudy:
    diffusion_rows: tuple[HedgeFrequencyRow, ...]
    jump_rows: tuple[HedgeFrequencyRow, ...]
    selected_rebalances: int
    jump_loss_floor: float
    selected_diffusion_cvar_95: float
    selected_jump_cvar_95: float
    event_gap_loss: float
    maximum_diffusion_cvar_95: float
    maximum_jump_cvar_95: float
    maximum_event_gap_loss: float
    combined_governed_objective: float
    diffusion_status: str
    jump_status: str
    event_status: str
    status: str


def jump_gap_exposure(
    *,
    price_fn: Callable[[float], float],
    price_before_jump: float,
    delta_before: float,
    gamma_before: float,
    spot: float,
    jump_return: float,
) -> JumpGapResult:
    values = (
        price_before_jump,
        delta_before,
        gamma_before,
        spot,
        jump_return,
    )
    if not all(isfinite(value) for value in values) or spot <= 0.0:
        raise ValueError("jump-gap inputs must be finite and spot positive")
    spot_change = spot * jump_return
    if spot + spot_change <= 0.0:
        raise ValueError("jump must leave a strictly positive spot")
    price_after = float(price_fn(spot + spot_change))
    if not isfinite(price_after):
        raise ValueError("price function returned a non-finite value")
    exact_gap = price_after - price_before_jump - delta_before * spot_change
    gamma_approximation = 0.5 * gamma_before * spot_change * spot_change
    return JumpGapResult(
        spot_change=spot_change,
        exact_delta_hedged_gap=exact_gap,
        gamma_approximation=gamma_approximation,
        approximation_error=gamma_approximation - exact_gap,
    )


def discrete_delta_hedge_pnl(
    option_values: np.ndarray,
    deltas: np.ndarray,
    spots: np.ndarray,
    fees_per_unit: float = 0.0,
) -> float:
    values = np.asarray(option_values, dtype=float)
    delta = np.asarray(deltas, dtype=float)
    spot = np.asarray(spots, dtype=float)
    if (
        values.ndim != 1
        or spot.ndim != 1
        or delta.ndim != 1
        or values.size != spot.size
        or delta.size != spot.size - 1
        or not np.isfinite(values).all()
        or not np.isfinite(delta).all()
        or not np.isfinite(spot).all()
    ):
        raise ValueError("shape or finiteness mismatch")
    if not isfinite(fees_per_unit) or fees_per_unit < 0.0:
        raise ValueError("fees_per_unit cannot be negative")
    if np.any(spot <= 0.0):
        raise ValueError("spots must be positive")
    option_change = values[-1] - values[0]
    hedge_change = float(np.sum(delta * np.diff(spot)))
    hedge_turnover = float(
        np.sum(np.abs(np.diff(np.r_[0.0, delta, 0.0])))
    )
    return float(option_change - hedge_change - fees_per_unit * hedge_turnover)


def _cvar_loss(samples: np.ndarray, quantile: float = 0.95) -> float:
    losses = -samples
    threshold = float(np.quantile(losses, quantile))
    return float(np.mean(losses[losses >= threshold]))


def _hedge_one_grid(
    *,
    path: np.ndarray,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    horizon: float,
    transaction_cost_bps: float,
    option_position: float,
    transacted_option_premium: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if path.ndim != 2:
        raise ValueError("hedge paths must be two-dimensional")
    paths, steps_plus_one = path.shape
    if (
        paths < 2
        or steps_plus_one < 2
        or not np.isfinite(path).all()
        or np.any(path <= 0.0)
    ):
        raise ValueError("hedge paths must be finite, positive and two-dimensional")
    steps = steps_plus_one - 1
    dt = horizon / steps
    initial_price = call_price(
        float(path[0, 0]),
        strike,
        rate,
        dividend,
        vol,
        horizon,
    )
    initial_delta = call_greeks(
        float(path[0, 0]),
        strike,
        rate,
        dividend,
        vol,
        horizon,
    ).delta
    stock_position = np.full(paths, -option_position * initial_delta)
    transaction_premium = (
        initial_price
        if transacted_option_premium is None
        else transacted_option_premium
    )
    fair_value_mismatch = -option_position * (
        transaction_premium - initial_price
    )
    initial_cost = (
        transaction_cost_bps
        * 1e-4
        * np.abs(stock_position)
        * path[:, 0]
    )
    cash = np.full(
        paths,
        -option_position * transaction_premium,
    ) - stock_position * path[:, 0] - initial_cost
    costs = initial_cost.copy()
    for step in range(steps):
        cash *= exp(rate * dt)
        cash += stock_position * path[:, step] * (exp(dividend * dt) - 1.0)
        if step < steps - 1:
            remaining = horizon - (step + 1) * dt
            new_delta = np.asarray(
                [
                    call_greeks(
                        float(spot),
                        strike,
                        rate,
                        dividend,
                        vol,
                        remaining,
                    ).delta
                    for spot in path[:, step + 1]
                ]
            )
            new_position = -option_position * new_delta
            trade = new_position - stock_position
            trade_cost = (
                transaction_cost_bps
                * 1e-4
                * np.abs(trade)
                * path[:, step + 1]
            )
            cash -= trade * path[:, step + 1] + trade_cost
            costs += trade_cost
            stock_position = new_position
    payoff = np.maximum(path[:, -1] - strike, 0.0)
    terminal_unwind_cost = (
        transaction_cost_bps
        * 1e-4
        * np.abs(stock_position)
        * path[:, -1]
    )
    costs += terminal_unwind_cost
    terminal_pnl = (
        option_position * payoff
        + stock_position * path[:, -1]
        + cash
        - terminal_unwind_cost
    )
    return terminal_pnl, costs, fair_value_mismatch


def hedge_frequency_study(
    *,
    spot: float = 100.0,
    strike: float = 100.0,
    rate: float = 0.02,
    dividend: float = 0.0,
    vol: float = 0.20,
    horizon: float = 30.0 / 365.0,
    rebalance_counts: tuple[int, ...] = (4, 8, 16, 32),
    paths: int = 4_000,
    transaction_cost_bps: float = 2.0,
    jump_intensity: float = 2.0,
    jump_return: float = -0.12,
    cost_weight: float = 1.0,
    option_position: float = -1.0,
    transacted_option_premium: float | None = None,
    maximum_diffusion_cvar_95: float = 3.0,
    maximum_jump_cvar_95: float = 7.0,
    maximum_event_gap_loss: float = 5.0,
    diffusion_objective_weight: float = 1.0,
    jump_objective_weight: float = 1.0,
    seed: int = 0,
) -> HedgeFrequencyStudy:
    values = (
        spot,
        strike,
        rate,
        dividend,
        vol,
        horizon,
        transaction_cost_bps,
        jump_intensity,
        jump_return,
        cost_weight,
        option_position,
        maximum_diffusion_cvar_95,
        maximum_jump_cvar_95,
        maximum_event_gap_loss,
        diffusion_objective_weight,
        jump_objective_weight,
    )
    if transacted_option_premium is not None:
        values = (*values, transacted_option_premium)
    if not all(isfinite(value) for value in values):
        raise ValueError("hedge-study inputs must be finite")
    if (
        spot <= 0.0
        or strike <= 0.0
        or vol <= 0.0
        or horizon <= 0.0
        or isinstance(paths, bool)
        or not isinstance(paths, int)
        or paths < 2
        or transaction_cost_bps < 0.0
        or jump_intensity < 0.0
        or jump_return <= -1.0
        or cost_weight < 0.0
        or option_position == 0.0
        or maximum_diffusion_cvar_95 < 0.0
        or maximum_jump_cvar_95 < 0.0
        or maximum_event_gap_loss < 0.0
        or diffusion_objective_weight < 0.0
        or jump_objective_weight < 0.0
        or diffusion_objective_weight + jump_objective_weight <= 0.0
        or (transacted_option_premium is not None and transacted_option_premium < 0.0)
    ):
        raise ValueError("invalid hedge-study parameters")
    if (
        not rebalance_counts
        or any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for count in rebalance_counts
        )
        or tuple(sorted(set(rebalance_counts))) != rebalance_counts
    ):
        raise ValueError("rebalance_counts must be unique and increasing")
    finest_steps = rebalance_counts[-1]
    if any(finest_steps % count != 0 for count in rebalance_counts):
        raise ValueError("rebalance grids must divide the finest grid")
    if jump_intensity * horizon / finest_steps > 1.0:
        raise ValueError("jump probability per finest step cannot exceed one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    dt = horizon / finest_steps
    normal = rng.standard_normal((paths, finest_steps))
    jump_uniform = rng.random((paths, finest_steps))
    jump_event = jump_uniform < jump_intensity * dt
    diffusion_log_increment = (
        (rate - dividend - 0.5 * vol * vol) * dt
        + vol * sqrt(dt) * normal
    )
    jump_log_increment = (
        (
            rate
            - dividend
            - 0.5 * vol * vol
            - jump_intensity * jump_return
        )
        * dt
        + vol * sqrt(dt) * normal
        + np.log1p(jump_return) * jump_event
    )
    diffusion_fine = spot * np.exp(
        np.c_[np.zeros(paths), np.cumsum(diffusion_log_increment, axis=1)]
    )
    jump_fine = spot * np.exp(
        np.c_[np.zeros(paths), np.cumsum(jump_log_increment, axis=1)]
    )

    diffusion_rows: list[HedgeFrequencyRow] = []
    jump_rows: list[HedgeFrequencyRow] = []
    for count in rebalance_counts:
        stride = finest_steps // count
        for source, target_rows in (
            (diffusion_fine, diffusion_rows),
            (jump_fine, jump_rows),
        ):
            sampled = source[:, ::stride]
            pnl, costs, fair_value_mismatch = _hedge_one_grid(
                path=sampled,
                strike=strike,
                rate=rate,
                dividend=dividend,
                vol=vol,
                horizon=horizon,
                transaction_cost_bps=transaction_cost_bps,
                option_position=option_position,
                transacted_option_premium=transacted_option_premium,
            )
            cvar = _cvar_loss(pnl)
            mean_cost = float(np.mean(costs))
            target_rows.append(
                HedgeFrequencyRow(
                    rebalances=count,
                    hedge_error_rmse=float(np.sqrt(np.mean(pnl * pnl))),
                    hedge_loss_cvar_95=cvar,
                    mean_transaction_cost=mean_cost,
                    model_fair_value_mismatch=fair_value_mismatch,
                    governed_objective=cvar + cost_weight * mean_cost,
                )
            )
    combined_objectives = [
        diffusion_objective_weight * diffusion_row.hedge_loss_cvar_95
        + jump_objective_weight * jump_row.hedge_loss_cvar_95
        + cost_weight
        * 0.5
        * (
            diffusion_row.mean_transaction_cost
            + jump_row.mean_transaction_cost
        )
        for diffusion_row, jump_row in zip(
            diffusion_rows, jump_rows, strict=True
        )
    ]
    selected_index = int(np.argmin(combined_objectives))
    selected_diffusion = diffusion_rows[selected_index]
    selected_jump = jump_rows[selected_index]
    jump_floor = min(row.hedge_loss_cvar_95 for row in jump_rows)
    event_horizon = max(horizon, 1e-12)
    event_price = call_price(spot, strike, rate, dividend, vol, event_horizon)
    event_greeks = call_greeks(
        spot, strike, rate, dividend, vol, event_horizon
    )
    event_gap = jump_gap_exposure(
        price_fn=lambda moved_spot: call_price(
            moved_spot,
            strike,
            rate,
            dividend,
            vol,
            event_horizon,
        ),
        price_before_jump=event_price,
        delta_before=event_greeks.delta,
        gamma_before=event_greeks.gamma,
        spot=spot,
        jump_return=jump_return,
    )
    event_gap_loss = -option_position * event_gap.exact_delta_hedged_gap
    diffusion_status = (
        "PASS"
        if selected_diffusion.hedge_loss_cvar_95
        <= maximum_diffusion_cvar_95
        else "REJECT"
    )
    jump_status = (
        "PASS"
        if selected_jump.hedge_loss_cvar_95 <= maximum_jump_cvar_95
        else "REJECT"
    )
    event_status = (
        "PASS" if event_gap_loss <= maximum_event_gap_loss else "REJECT"
    )
    return HedgeFrequencyStudy(
        diffusion_rows=tuple(diffusion_rows),
        jump_rows=tuple(jump_rows),
        selected_rebalances=selected_diffusion.rebalances,
        jump_loss_floor=jump_floor,
        selected_diffusion_cvar_95=selected_diffusion.hedge_loss_cvar_95,
        selected_jump_cvar_95=selected_jump.hedge_loss_cvar_95,
        event_gap_loss=event_gap_loss,
        maximum_diffusion_cvar_95=maximum_diffusion_cvar_95,
        maximum_jump_cvar_95=maximum_jump_cvar_95,
        maximum_event_gap_loss=maximum_event_gap_loss,
        combined_governed_objective=combined_objectives[selected_index],
        diffusion_status=diffusion_status,
        jump_status=jump_status,
        event_status=event_status,
        status=(
            "PASS"
            if {diffusion_status, jump_status, event_status} == {"PASS"}
            else "REJECT"
        ),
    )
