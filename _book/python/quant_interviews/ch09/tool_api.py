from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from typing import Any, Callable

import numpy as np

from .attribution import reconcile
from .black_litterman import ViewLineage, posterior
from .capacity import capacity, capacity_coefficient, capacity_diagnostics
from .covariance import perturbation_diagnostics
from .drawdown import conditional_drawdown_at_risk, drawdown_stats
from .factor_neutral import constrained_neutral_portfolio
from .kelly import multivariate_kelly, risk_adjusted_certainty_equivalent
from .performance import sharpe_ratio, spectral_effective_rank
from .portfolio import portfolio_certificate
from .rebalancing import event_trigger_decision
from .risk_parity import risk_contributions
from .robust import robust_mean_variance
from .stress import coverage, stress_pnl
from .tail_risk import historical_var_es
from .var_backtest import (
    VaRForecastRecord,
    point_in_time_exceedance_summary,
)

TOOL_VERSION = "0.10"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _envelope(
    payload: dict[str, Any],
    result: dict[str, Any],
    *,
    status: str = "PASS",
    units: dict[str, str] | None = None,
    residuals: dict[str, float] | None = None,
    tolerances: dict[str, float] | None = None,
    diagnostics: dict[str, Any] | None = None,
    reason_codes: list[str] | None = None,
) -> dict[str, Any]:
    normalized = _jsonable(result)
    return {
        "status": status,
        "tool_version": TOOL_VERSION,
        "input_sha256": payload["input_sha256"],
        "result_sha256": _digest(normalized),
        "units": units or {"primary": "dimensionless"},
        "residuals": residuals or {"certificate": 0.0},
        "tolerances": tolerances or {"certificate": 1.0e-10},
        "diagnostics": _jsonable(diagnostics or {}),
        "result": normalized,
        "reason_codes": reason_codes or [f"{status}_DETERMINISTIC"],
    }


def optimize_mean_variance_portfolio(payload: dict[str, Any]) -> dict[str, Any]:
    result = portfolio_certificate(
        np.asarray(payload["expected_returns"], dtype=float),
        np.asarray(payload["covariance"], dtype=float),
        payload["portfolio_type"],
        target=payload["target_return"],
        risk_free=float(payload["risk_free"]),
        condition_number_limit=float(payload["condition_number_limit"]),
        tolerance=float(payload["tolerance"]),
    )
    return _envelope(
        payload,
        asdict(result),
        status=result.status,
        units={"expected_return": "annual_return_fraction", "volatility": "annual_volatility_fraction"},
        residuals={"budget": result.budget_residual, "optimizer_weight_l2": result.optimizer_weight_residual_l2},
        tolerances={"optimizer": result.tolerance},
        diagnostics={"instrument_ids": payload["instrument_ids"]},
    )


def construct_black_litterman_posterior(payload: dict[str, Any]) -> dict[str, Any]:
    result = posterior(
        np.asarray(payload["covariance"], dtype=float),
        np.asarray(payload["prior"], dtype=float),
        float(payload["tau"]),
        np.asarray(payload["P"], dtype=float),
        np.asarray(payload["q"], dtype=float),
        np.asarray(payload["omega"], dtype=float),
        tuple(ViewLineage(**record) for record in payload["view_lineage"]),
        decision_time=payload["decision_time"],
    )
    return _envelope(
        payload,
        asdict(result),
        status=result.status,
        units={"posterior_mean": "annual_return_fraction"},
        residuals={"maximum_view": float(np.max(np.abs(result.posterior_view_residual)))},
        diagnostics={"surprise_degrees_of_freedom": result.surprise_degrees_of_freedom},
    )


def compute_risk_contributions(payload: dict[str, Any]) -> dict[str, Any]:
    weights = np.asarray(payload["weights"], dtype=float)
    covariance = np.asarray(payload["covariance"], dtype=float)
    contributions = risk_contributions(weights, covariance)
    volatility = float(np.sqrt(weights @ covariance @ weights))
    residual = abs(float(contributions.sum()) - volatility)
    return _envelope(payload, {"volatility": volatility, "contributions": contributions}, units={"volatility": "return_fraction"}, residuals={"euler": residual})


def evaluate_turnover_policy(payload: dict[str, Any]) -> dict[str, Any]:
    result = event_trigger_decision(
        float(payload["normalized_alpha_change"]),
        float(payload["normalized_exposure_change"]),
        float(payload["breach_probability"]),
        float(payload["liquidity_cost_penalty"]),
        tuple(float(value) for value in payload["weights"]),
        float(payload["threshold"]),
        decision_time=payload["decision_time"],
        hard_risk_breached=bool(payload["hard_risk_breached"]),
    )
    return _envelope(payload, asdict(result), units={"score": "normalized_score"})


def compute_var_es(payload: dict[str, Any]) -> dict[str, Any]:
    value_at_risk, expected_shortfall = historical_var_es(
        np.asarray(payload["losses"], dtype=float),
        float(payload["alpha"]),
    )
    return _envelope(payload, {"value_at_risk": value_at_risk, "expected_shortfall": expected_shortfall, "quantile_convention": "INVERTED_CDF"}, units={"loss": "input_loss_units"}, residuals={"es_minus_var": expected_shortfall - value_at_risk})


def run_portfolio_stress(payload: dict[str, Any]) -> dict[str, Any]:
    result = stress_pnl(
        np.asarray(payload["delta"], dtype=float),
        np.asarray(payload["gamma"], dtype=float),
        np.asarray(payload["shock"], dtype=float),
    )
    covered = coverage(
        np.asarray(payload["market_values"], dtype=float),
        np.asarray(payload["coverage_fractions"], dtype=float),
    )
    return _envelope(payload, {**asdict(result), "coverage": covered}, status=result.status, units={"pnl": "input_currency", "coverage": "fraction"})


def estimate_strategy_capacity(payload: dict[str, Any]) -> dict[str, Any]:
    coefficient = capacity_coefficient(
        np.asarray(payload["trade_notional_per_aum"], dtype=float),
        np.asarray(payload["volatility"], dtype=float),
        np.asarray(payload["dollar_adv"], dtype=float),
        np.asarray(payload["impact"], dtype=float),
    )
    result = capacity(float(payload["alpha"]), float(payload["fixed_cost"]), coefficient)
    diagnostics = capacity_diagnostics(
        np.asarray(payload["trade_notional_per_aum"], dtype=float),
        float(payload["candidate_aum"]),
        np.asarray(payload["dollar_adv"], dtype=float),
        float(payload["execution_horizon_days"]),
        float(payload["maximum_participation"]),
        float(payload["volume_stress_multiplier"]),
    )
    return _envelope(payload, asdict(result), status=diagnostics.status, units={"aum": "USD", "participation": "fraction"}, diagnostics=asdict(diagnostics))


def reconcile_performance_attribution(payload: dict[str, Any]) -> dict[str, Any]:
    result = reconcile(
        float(payload["total"]),
        {str(key): float(value) for key, value in payload["components"].items()},
        nav=float(payload["nav"]),
        absolute_tolerance=float(payload["absolute_tolerance"]),
        nav_tolerance=float(payload["nav_tolerance"]),
        gross_tolerance=float(payload["gross_tolerance"]),
    )
    return _envelope(payload, asdict(result), status=result.status, units={"total": "base_currency"}, residuals={"pnl": result.residual}, tolerances={"absolute": float(payload["absolute_tolerance"]), "nav": float(payload["nav_tolerance"]), "gross": float(payload["gross_tolerance"])})


def optimize_robust_portfolio(payload: dict[str, Any]) -> dict[str, Any]:
    result = robust_mean_variance(
        np.asarray(payload["mean"], dtype=float),
        np.asarray(payload["covariance"], dtype=float),
        np.asarray(payload["uncertainty"], dtype=float),
        float(payload["radius"]),
        float(payload["risk_aversion"]),
        uncertainty_set_version=payload["uncertainty_set_version"],
        optimality_tolerance=float(payload["optimality_tolerance"]),
    )
    return _envelope(payload, asdict(result), status=result.status, units={"expected_return": "annual_return_fraction"}, residuals={"support": result.support_function_residual, "stationarity": result.stationarity_residual}, tolerances={"optimality": float(payload["optimality_tolerance"])})


def compute_drawdown_path(payload: dict[str, Any]) -> dict[str, Any]:
    returns = np.asarray(payload["returns"], dtype=float)
    result = drawdown_stats(returns)
    cdar = conditional_drawdown_at_risk(returns, float(payload["alpha"]))
    return _envelope(payload, {**asdict(result), "conditional_drawdown_at_risk": cdar}, units={"drawdown": "fraction"})


def compute_kelly_allocation(payload: dict[str, Any]) -> dict[str, Any]:
    mean = np.asarray(payload["mean"], dtype=float)
    covariance = np.asarray(payload["covariance"], dtype=float)
    uncertainty = np.asarray(payload["posterior_mean_covariance"], dtype=float)
    weights = multivariate_kelly(mean, covariance, uncertainty, float(payload["fraction"]))
    certainty_equivalent = risk_adjusted_certainty_equivalent(mean, covariance + uncertainty)
    return _envelope(payload, {"weights": weights, "certainty_equivalent": certainty_equivalent}, units={"weights": "capital_fraction", "certainty_equivalent": "log_growth_approximation"})


def compute_performance_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    ratio = sharpe_ratio(np.asarray(payload["returns"], dtype=float), int(payload["periods"]), int(payload["hac_lags"]))
    rank = spectral_effective_rank(np.asarray(payload["correlation"], dtype=float))
    return _envelope(payload, {"hac_sharpe": ratio, "spectral_effective_rank": rank}, units={"hac_sharpe": "annualized_ratio"})


def estimate_covariance_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    result = perturbation_diagnostics(np.asarray(payload["signal"], dtype=float), np.asarray(payload["covariance"], dtype=float), np.asarray(payload["perturbed_covariance"], dtype=float), payload["normalization"])
    return _envelope(payload, asdict(result), units={"condition_number": "ratio"})


def neutralize_portfolio_exposures(payload: dict[str, Any]) -> dict[str, Any]:
    result = constrained_neutral_portfolio(np.asarray(payload["alpha"], dtype=float), np.asarray(payload["covariance"], dtype=float), np.asarray(payload["constraints"], dtype=float), np.asarray(payload["targets"], dtype=float))
    maximum = float(np.max(np.abs(result.constraint_residual), initial=0.0))
    return _envelope(payload, asdict(result), units={"weights": "portfolio_weight"}, residuals={"maximum_constraint": maximum})


def point_in_time_portfolio_state(payload: dict[str, Any]) -> dict[str, Any]:
    eligible = [record for record in payload["records"] if record["valid_time"] <= payload["decision_time"] and record["knowledge_time"] <= payload["knowledge_cutoff"]]
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for record in sorted(eligible, key=lambda item: (item["valid_time"], item["knowledge_time"], item["revision_id"]), reverse=True):
        key = (record["portfolio_id"], record["instrument_id"])
        if key not in selected:
            selected[key] = record
        elif all(record[field] == selected[key][field] for field in ("valid_time", "knowledge_time", "revision_id")):
            raise ValueError("full ordering-key tie in PIT holdings")
    if not selected:
        raise ValueError("no eligible point-in-time holdings")
    return _envelope(payload, {"positions": list(selected.values()), "selected_count": len(selected)}, units={"market_value": "record_currency"})


def portfolio_risk_kernel(payload: dict[str, Any]) -> dict[str, Any]:
    weights = np.asarray(payload["weights"], dtype=float)
    covariance = np.asarray(payload["covariance"], dtype=float)
    if covariance.shape != (weights.size, weights.size) or not np.isfinite(weights).all() or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10) or float(np.linalg.eigvalsh(covariance).min()) < -1e-12:
        raise ValueError("invalid PSD covariance kernel inputs")
    variance = max(float(weights @ covariance @ weights), 0.0)
    return _envelope(payload, {"variance": variance, "volatility": float(np.sqrt(variance))}, units={"variance": "return_squared"})


def pretrade_constraint_gate(payload: dict[str, Any]) -> dict[str, Any]:
    weights = np.asarray(payload["weights"], dtype=float)
    gross = float(np.abs(weights).sum())
    maximum = float(np.max(np.abs(weights)))
    passed = gross <= float(payload["gross_limit"]) + 1e-12 and maximum <= float(payload["single_name_limit"]) + 1e-12
    return _envelope(payload, {"passed": passed, "gross": gross, "maximum_absolute_weight": maximum}, status="PASS" if passed else "REJECT", units={"weights": "portfolio_weight"})


def run_var_backtest(payload: dict[str, Any]) -> dict[str, Any]:
    result = point_in_time_exceedance_summary(tuple(VaRForecastRecord(**record) for record in payload["records"]), evaluation_time=payload["evaluation_time"], alpha=float(payload["alpha"]))
    return _envelope(payload, asdict(result), status=result.status, units={"loss": "record_loss_units"})


TOOLS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    name: value
    for name, value in globals().copy().items()
    if callable(value) and name in {
        "optimize_mean_variance_portfolio", "construct_black_litterman_posterior",
        "compute_risk_contributions", "evaluate_turnover_policy", "compute_var_es",
        "run_portfolio_stress", "estimate_strategy_capacity",
        "reconcile_performance_attribution", "optimize_robust_portfolio",
        "compute_drawdown_path", "compute_kelly_allocation",
        "compute_performance_metrics", "estimate_covariance_diagnostics",
        "neutralize_portfolio_exposures", "point_in_time_portfolio_state",
        "portfolio_risk_kernel", "pretrade_constraint_gate", "run_var_backtest",
    }
}


def execute_tool(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        function = TOOLS[name]
    except KeyError as error:
        raise ValueError(f"unknown CH09 tool: {name}") from error
    return function(payload)
