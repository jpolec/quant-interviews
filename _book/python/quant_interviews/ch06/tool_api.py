from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .black_scholes_greeks import GreekConventions, call_greeks
from .gamma_vega import long_gamma_short_vega_calendar
from .hedging_pnl import self_financing_hedge_reconciliation
from .heston import heston_diagnostics
from .implied_vol import implied_vol_call
from .integrated_case import evaluate_release
from .jump_hedging import hedge_frequency_study
from .roughness import roughness_recovery_study
from .sabr import sabr_smile_audit
from .skew import skew_fit
from .sticky import sticky_delta_scenario
from .surface import surface_arbitrage_audit
from .variance_products import variance_swap_audit


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _select(record: Any, fields: Sequence[str]) -> dict[str, Any]:
    return {field: _json_value(getattr(record, field)) for field in fields}


def compute_option_greeks(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    vol: float,
    horizon: float,
    currency: str,
) -> dict[str, Any]:
    result = call_greeks(
        spot,
        strike,
        rate,
        dividend,
        vol,
        horizon,
        conventions=GreekConventions(currency=currency),
    )
    output = _select(
        result,
        (
            "delta",
            "gamma",
            "cash_gamma",
            "raw_vega_per_unit_vol",
            "vega_per_vol_point",
            "theta_per_year",
            "theta_per_calendar_day",
            "raw_rho_per_unit_rate",
            "rho_per_basis_point",
        ),
    )
    output["status"] = "PASS"
    return output


def reconcile_delta_hedged_pnl(
    *,
    option_pnl: Sequence[float],
    delta_hedge_pnl: Sequence[float],
    financing_pnl: Sequence[float],
    dividend_pnl: Sequence[float],
    execution_cost: Sequence[float],
    accounting_pnl: Sequence[float],
    tolerance: float,
) -> dict[str, Any]:
    result = self_financing_hedge_reconciliation(
        option_pnl=np.asarray(option_pnl, dtype=float),
        delta_hedge_pnl=np.asarray(delta_hedge_pnl, dtype=float),
        financing_pnl=np.asarray(financing_pnl, dtype=float),
        dividend_pnl=np.asarray(dividend_pnl, dtype=float),
        execution_cost=np.asarray(execution_cost, dtype=float),
        accounting_pnl=np.asarray(accounting_pnl, dtype=float),
        tolerance=tolerance,
    )
    return _select(
        result,
        (
            "attributed_pnl",
            "accounting_pnl",
            "event_residuals",
            "max_abs_event_residual",
            "residual",
            "status",
        ),
    )


def design_gamma_vega_calendar(
    *,
    spot: float,
    rate: float,
    dividend: float,
    vol: float,
    short_horizon: float,
    long_horizon: float,
    spot_return_scenario: float,
    volatility_point_scenario: float,
    holding_days: float,
) -> dict[str, Any]:
    result = long_gamma_short_vega_calendar(
        spot=spot,
        rate=rate,
        dividend=dividend,
        vol=vol,
        short_horizon=short_horizon,
        long_horizon=long_horizon,
        spot_return_scenario=spot_return_scenario,
        volatility_point_scenario=volatility_point_scenario,
        holding_days=holding_days,
    )
    return _select(
        result,
        (
            "selected_short_long_weight",
            "net_gamma",
            "net_vega_per_vol_point",
            "scenario_pnl",
            "estimated_crossing_cost",
            "greek_feasibility_status",
            "economic_scenario_status",
            "status",
        ),
    )


def analyze_sticky_delta(
    *,
    spot: float,
    spot_move: float,
    target_delta: float,
    rate: float,
    dividend: float,
    horizon: float,
    smile_delta_nodes: Sequence[float],
    smile_vols: Sequence[float],
    delta_convention: str,
    maximum_iterations: int,
    tolerance: float,
) -> dict[str, Any]:
    nodes = np.asarray(smile_delta_nodes, dtype=float)
    vols = np.asarray(smile_vols, dtype=float)
    if (
        nodes.ndim != 1
        or vols.shape != nodes.shape
        or nodes.size < 3
        or not np.isfinite(nodes).all()
        or not np.isfinite(vols).all()
        or np.any(np.diff(nodes) <= 0.0)
        or np.any(nodes <= 0.0)
        or np.any(nodes >= 1.0)
        or np.any(vols <= 0.0)
    ):
        raise ValueError("smile nodes must be finite, ordered and domain-valid")

    def smile_by_delta(delta: float) -> float:
        if delta < nodes[0] or delta > nodes[-1]:
            raise ValueError("requested delta lies outside the supplied smile grid")
        return float(np.interp(delta, nodes, vols))

    result = sticky_delta_scenario(
        spot=spot,
        spot_move=spot_move,
        target_delta=target_delta,
        rate=rate,
        dividend=dividend,
        horizon=horizon,
        smile_by_delta=smile_by_delta,
        delta_convention=delta_convention,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )
    return _select(
        result,
        (
            "initial_strike",
            "sticky_strike_vol",
            "sticky_delta_vol",
            "finite_move_average_delta",
            "local_sticky_delta",
            "iterations",
            "status",
        ),
    )


def fit_equity_skew(
    *,
    strikes: Sequence[float],
    forward: float,
    vols: Sequence[float],
    confidence_level: float,
    maximum_slope: float,
) -> dict[str, Any]:
    result = skew_fit(
        strikes=np.asarray(strikes, dtype=float),
        forward=forward,
        vols=np.asarray(vols, dtype=float),
        confidence_level=confidence_level,
        maximum_slope=maximum_slope,
    )
    return _select(
        result,
        (
            "intercept",
            "slope",
            "slope_standard_error",
            "slope_confidence_interval",
            "degrees_of_freedom",
            "confidence_interval_method",
            "status",
        ),
    )


def invert_implied_volatility(
    *,
    price: float,
    s: float,
    k: float,
    r: float,
    q: float,
    t: float,
    price_tolerance: float,
    volatility_tolerance: float,
    vega_floor: float,
    max_iterations: int,
) -> dict[str, Any]:
    result = implied_vol_call(
        price,
        s,
        k,
        r,
        q,
        t,
        price_tolerance=price_tolerance,
        volatility_tolerance=volatility_tolerance,
        vega_floor=vega_floor,
        max_iterations=max_iterations,
    )
    return _select(
        result,
        (
            "status",
            "implied_vol",
            "repricing_residual",
            "lower_vol",
            "upper_vol",
            "bracket_width",
            "vega",
            "iterations",
        ),
    )


def validate_volatility_surface(
    *,
    log_forward_moneyness: Sequence[float],
    maturities: Sequence[float],
    forwards: Sequence[float],
    discounts: Sequence[float],
    call_prices: Sequence[Sequence[float]],
    implied_vols: Sequence[Sequence[float]],
    bid_prices: Sequence[Sequence[float]] | None,
    ask_prices: Sequence[Sequence[float]] | None,
    tolerance: float,
) -> dict[str, Any]:
    result = surface_arbitrage_audit(
        log_forward_moneyness=np.asarray(log_forward_moneyness, dtype=float),
        maturities=np.asarray(maturities, dtype=float),
        forwards=np.asarray(forwards, dtype=float),
        discounts=np.asarray(discounts, dtype=float),
        call_prices=np.asarray(call_prices, dtype=float),
        implied_vols=np.asarray(implied_vols, dtype=float),
        bid_prices=None if bid_prices is None else np.asarray(bid_prices, dtype=float),
        ask_prices=None if ask_prices is None else np.asarray(ask_prices, dtype=float),
        tolerance=tolerance,
    )
    return _select(
        result,
        (
            "violations",
            "minimum_call_slope_residual",
            "minimum_convexity_residual",
            "minimum_calendar_total_variance_increment",
            "minimum_normalized_calendar_call_increment",
            "bid_ask_feasible",
            "executable_prices",
            "implied_vol_reprices",
            "repair_l1_norm",
            "maximum_constraint_violation",
            "status",
        ),
    )


def replicate_variance_product(
    *,
    forward: float,
    strikes: Sequence[float],
    puts: Sequence[float],
    calls: Sequence[float],
    discount: float,
    horizon: float,
    parity_tolerance: float,
    maximum_tail_fraction: float,
) -> dict[str, Any]:
    result = variance_swap_audit(
        forward=forward,
        strikes=np.asarray(strikes, dtype=float),
        puts=np.asarray(puts, dtype=float),
        calls=np.asarray(calls, dtype=float),
        discount=discount,
        horizon=horizon,
        parity_tolerance=parity_tolerance,
        maximum_tail_fraction=maximum_tail_fraction,
    )
    return _select(
        result,
        (
            "fair_variance",
            "integral_component",
            "forward_correction",
            "k0",
            "included_strikes",
            "lower_tail_fraction",
            "upper_tail_fraction",
            "maximum_put_call_parity_residual",
            "status",
        ),
    )


def validate_heston_model(
    *,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    horizon: float,
    step_counts: Sequence[int],
    paths: int,
    seed: int,
    require_feller: bool,
) -> dict[str, Any]:
    result = heston_diagnostics(
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        horizon=horizon,
        step_counts=tuple(step_counts),
        paths=paths,
        seed=seed,
        require_feller=require_feller,
    )
    rows = [
        _select(
            row,
            (
                "steps",
                "variance_mean_z_score",
                "forward_martingale_error",
                "negative_pre_truncation_fraction",
                "zero_terminal_fraction",
            ),
        )
        for row in result.rows
    ]
    return {
        "feller_condition": result.feller_condition,
        "rows": rows,
        "status": result.status,
    }


def validate_sabr_smile(
    *,
    forward: float,
    strikes: Sequence[float],
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    horizon: float,
    shift: float,
    maximum_absolute_log_moneyness_limit: float,
    maximum_nu_squared_horizon: float,
) -> dict[str, Any]:
    result = sabr_smile_audit(
        forward=forward,
        strikes=np.asarray(strikes, dtype=float),
        alpha=alpha,
        beta=beta,
        rho=rho,
        nu=nu,
        horizon=horizon,
        shift=shift,
        maximum_absolute_log_moneyness_limit=maximum_absolute_log_moneyness_limit,
        maximum_nu_squared_horizon=maximum_nu_squared_horizon,
    )
    return _select(
        result,
        (
            "implied_vols",
            "atm_vol",
            "nearest_grid_strike",
            "downside_skew",
            "curvature",
            "maximum_absolute_log_moneyness",
            "nu_squared_horizon",
            "invalid_volatility_count",
            "status",
        ),
    )


def estimate_roughness(
    *,
    true_hurst: float,
    points: int,
    paths: int,
    max_lag: int,
    microstructure_noise_std: float,
    seed: int,
    tolerance: float,
    confidence_level: float,
) -> dict[str, Any]:
    result = roughness_recovery_study(
        true_hurst=true_hurst,
        points=points,
        paths=paths,
        max_lag=max_lag,
        microstructure_noise_std=microstructure_noise_std,
        seed=seed,
        tolerance=tolerance,
        confidence_level=confidence_level,
    )
    return _select(
        result,
        (
            "estimated_hurst_mean",
            "confidence_interval",
            "estimate_standard_error",
            "confidence_interval_method",
            "coverage_status",
            "status",
        ),
    )


def stress_jump_hedging(
    *,
    spot: float,
    strike: float,
    vol: float,
    horizon: float,
    rebalance_counts: Sequence[int],
    paths: int,
    transaction_cost_bps: float,
    jump_intensity: float,
    jump_return: float,
    maximum_diffusion_cvar_95: float,
    maximum_jump_cvar_95: float,
    maximum_event_gap_loss: float,
    seed: int,
) -> dict[str, Any]:
    result = hedge_frequency_study(
        spot=spot,
        strike=strike,
        vol=vol,
        horizon=horizon,
        rebalance_counts=tuple(rebalance_counts),
        paths=paths,
        transaction_cost_bps=transaction_cost_bps,
        jump_intensity=jump_intensity,
        jump_return=jump_return,
        maximum_diffusion_cvar_95=maximum_diffusion_cvar_95,
        maximum_jump_cvar_95=maximum_jump_cvar_95,
        maximum_event_gap_loss=maximum_event_gap_loss,
        seed=seed,
    )
    return _select(
        result,
        (
            "selected_rebalances",
            "selected_diffusion_cvar_95",
            "selected_jump_cvar_95",
            "event_gap_loss",
            "combined_governed_objective",
            "diffusion_status",
            "jump_status",
            "event_status",
            "status",
        ),
    )


def evaluate_chapter_release(
    *,
    metrics: Mapping[str, float],
    policy: Mapping[str, Any],
    input_snapshot_hash: str,
    result_bundle_hash: str | None,
) -> dict[str, Any]:
    result = evaluate_release(
        metrics=metrics,
        policy=policy,
        input_snapshot_hash=input_snapshot_hash,
        result_bundle_hash=result_bundle_hash,
    )
    return _select(
        result,
        (
            "policy_version",
            "policy_hash",
            "input_snapshot_hash",
            "result_bundle_hash",
            "release_evidence_hash",
            "gate_evidence",
            "final_decision",
            "decision_reason",
        ),
    )
